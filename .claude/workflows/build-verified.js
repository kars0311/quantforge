export const meta = {
  name: 'build-verified',
  description: 'Builder agent with background research agents; independent verifier writes proving tests at each milestone',
  whenToUse: 'Implementing a QuantForge component (or any multi-milestone task) where each milestone must be independently verified with tests before the next one starts',
  phases: [
    { title: 'Plan', detail: 'break the task into serial milestones' },
    { title: 'Build', detail: 'one builder per milestone; research delegated to background Explore agents' },
    { title: 'Verify', detail: 'independent verifier checks code vs spec and writes pytest cases' },
    { title: 'Clean', detail: 'cleaner lints and tidies the touched files without changing behavior' },
  ],
}

// ---- input ----------------------------------------------------------------
// Invoke with: Workflow({name: 'build-verified', args: 'Implement interchange.py per docs/components/01-interchange.md'})
// or:          args: {task: '...', maxFixRounds: 2}
const task = typeof args === 'string' ? args : args && args.task
if (!task) throw new Error("Pass the task as args, e.g. args: 'Implement src/quantforge/interchange.py per docs/components/01-interchange.md'")
const MAX_FIX_ROUNDS = (args && args.maxFixRounds) || 2

// ---- schemas --------------------------------------------------------------
const PLAN_SCHEMA = {
  type: 'object',
  required: ['milestones'],
  properties: {
    milestones: {
      type: 'array', minItems: 1, maxItems: 8,
      items: {
        type: 'object',
        required: ['title', 'spec', 'done_when'],
        properties: {
          title: { type: 'string', description: 'short name, e.g. "validate_frame + SCHEMAS"' },
          spec: { type: 'string', description: 'what to build: functions, signatures, behavior, edge cases' },
          done_when: { type: 'string', description: 'objective acceptance criteria the verifier will check' },
        },
      },
    },
  },
}

const BUILD_SCHEMA = {
  type: 'object',
  required: ['summary', 'files_changed'],
  properties: {
    summary: { type: 'string', description: 'what was built and any decisions made' },
    files_changed: { type: 'array', items: { type: 'string' } },
    caveats: { type: 'array', items: { type: 'string' }, description: 'known limitations or things the verifier should scrutinize' },
  },
}

const CLEAN_SCHEMA = {
  type: 'object',
  required: ['summary', 'files_cleaned', 'suite_green'],
  properties: {
    summary: { type: 'string', description: 'what was cleaned and what was deliberately left alone' },
    files_cleaned: { type: 'array', items: { type: 'string' } },
    suite_green: { type: 'boolean', description: 'true only if ruff check AND the full pytest suite pass after cleaning' },
    skipped: { type: 'array', items: { type: 'string' }, description: 'cleanups considered but skipped because they would change behavior — for the user to decide' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['passed', 'findings', 'tests_added'],
  properties: {
    passed: { type: 'boolean', description: 'true only if code matches spec, rigor rules hold, and the full pytest suite is green including your new tests' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'description'],
        properties: {
          severity: { type: 'string', enum: ['blocker', 'minor'] },
          description: { type: 'string' },
          file: { type: 'string' },
        },
      },
    },
    tests_added: { type: 'array', items: { type: 'string' }, description: 'test files/functions you wrote' },
  },
}

// ---- shared prompt fragments ----------------------------------------------
const REPO_RULES = `
Ground rules (from AGENTS.md — read it, plus the relevant docs/components/*.md design doc, before coding):
- No look-ahead bias: positions.shift(1); a signal may never see same-day data it could not have known.
- Transaction costs on turnover; metrics/performance.py is the single source of truth for metrics.
- Frozen interfaces in engine/base.py; cross-stage hand-offs via the interchange Parquet contract.
- All Claude API calls go through ai/budget.py; PUBLIC_MODE means parameter-only, never execute LLM-generated code.
- Keep 'ruff check .' and 'pytest' green — never finish with either failing.
- Docstrings explain WHY; clear conventional code over cleverness (the author must defend it in an interview).`

const RESEARCH_DELEGATION = `
Research delegation — IMPORTANT: you have the Agent tool. When you hit a question that is a LOOKUP
(external library behavior, an API convention, where something lives in this repo, how a reference
implementation handles a case) rather than a design decision, do NOT stop building to chase it:
spawn an Explore agent (subagent_type: "Explore", read-only) with a precise question and keep
working on parts that don't depend on the answer; fold the answer in when it returns. You may have
at most TWO research agents running at any time. Design decisions are yours, not theirs.`

// ---- phase 1: plan --------------------------------------------------------
phase('Plan')
log(`Planning milestones for: ${task}`)
const plan = await agent(
  `You are planning implementation milestones inside the repo at the current working directory (QuantForge).
Task: ${task}

Read AGENTS.md, docs/product.md, and the relevant docs/components/*.md design doc(s) for this task.
Break the task into 1–8 SERIAL milestones. Each milestone is one component or one integral
function/method cluster that can be built and then independently verified. Order them by dependency.
For each: title, a precise build spec (functions, signatures, behavior, edge cases — lift these from
the design doc), and objective done_when criteria a skeptical verifier can check.
Do not write any code. Return only the structured plan.`,
  { label: 'plan-milestones', schema: PLAN_SCHEMA },
)
if (!plan || !plan.milestones.length) throw new Error('Planner returned no milestones')
log(`${plan.milestones.length} milestone(s): ${plan.milestones.map(m => m.title).join(' → ')}`)

// ---- phases 2+3: build → verify (→ fix → re-verify) per milestone, serial --
const results = []
const priorSummaries = []

for (let i = 0; i < plan.milestones.length; i++) {
  const m = plan.milestones[i]
  const tag = `${i + 1}/${plan.milestones.length}`

  const build = await agent(
    `You are the BUILDER for milestone ${tag} of: ${task}
Milestone: ${m.title}
Spec: ${m.spec}
Done when: ${m.done_when}
${priorSummaries.length ? `Already completed this run:\n${priorSummaries.join('\n')}` : ''}
${REPO_RULES}
${RESEARCH_DELEGATION}

Build exactly this milestone (no scope creep into later milestones). Write clean code with
why-docstrings. Before finishing, run 'ruff check .' and 'pytest' and fix anything red.
Return the structured build report.`,
    { label: `build:${m.title}`, phase: 'Build', agentType: 'general-purpose', schema: BUILD_SCHEMA },
  )
  if (!build) { results.push({ milestone: m.title, status: 'builder-failed' }); break }
  log(`Built ${tag} "${m.title}": ${build.files_changed.join(', ')}`)

  let verdict = null
  let round = 0
  while (round <= MAX_FIX_ROUNDS) {
    verdict = await agent(
      `You are an INDEPENDENT, ADVERSARIAL VERIFIER. Another agent claims to have completed a
milestone; your job is to try to prove it wrong, then lock in correctness with tests.
Task context: ${task}
Milestone: ${m.title}
Spec: ${m.spec}
Done when: ${m.done_when}
Builder's report: ${JSON.stringify({ summary: build.summary, files: build.files_changed, caveats: build.caveats || [] })}
${REPO_RULES}

Do all of the following:
1. Read the changed files AND the relevant docs/components/*.md design doc yourself — do not
   trust the builder's summary.
2. Check the code against the spec, the done_when criteria, and the rigor rules above
   (look-ahead bias and cost accounting especially, where relevant).
3. WRITE pytest test cases in tests/ that prove the milestone's behavior — including at least
   one adversarial case (e.g. a prescient signal earning ~0, a hand-computed expected value,
   malformed input rejected). Match the style of tests/test_smoke_vertical_slice.py.
4. Run 'ruff check .' and the FULL 'pytest' suite.
Set passed=true only if everything above holds and the whole suite is green. List every defect
as a finding (severity 'blocker' if it must be fixed before the next milestone).`,
      { label: `verify:${m.title}${round ? ` (round ${round + 1})` : ''}`, phase: 'Verify', agentType: 'general-purpose', schema: VERDICT_SCHEMA },
    )
    if (!verdict) { verdict = { passed: false, findings: [{ severity: 'blocker', description: 'verifier agent failed to return' }], tests_added: [] }; break }
    if (verdict.passed) break

    const blockers = verdict.findings.filter(f => f.severity === 'blocker')
    log(`Verifier rejected "${m.title}" (round ${round + 1}): ${blockers.length} blocker(s)`)
    if (round === MAX_FIX_ROUNDS) break

    await agent(
      `You are the FIXER for milestone "${m.title}" of: ${task}
An independent verifier found these defects — fix every blocker (and minors where cheap):
${JSON.stringify(verdict.findings)}
Spec: ${m.spec}
${REPO_RULES}
${RESEARCH_DELEGATION}
Do not delete or weaken the verifier's tests to make them pass — fix the code. If a test itself
is wrong, say so explicitly in your summary with the reasoning. Run 'ruff check .' and 'pytest'
before finishing.`,
      { label: `fix:${m.title} (round ${round + 1})`, phase: 'Build', agentType: 'general-purpose', schema: BUILD_SCHEMA },
    )
    round++
  }

  results.push({
    milestone: m.title,
    status: verdict.passed ? 'verified' : 'failed-verification',
    files_changed: build.files_changed,
    tests_added: verdict.tests_added,
    findings: verdict.findings,
  })
  priorSummaries.push(`- ${m.title}: ${build.summary}`)

  if (!verdict.passed) {
    log(`Stopping: "${m.title}" failed verification after ${MAX_FIX_ROUNDS + 1} round(s) — later milestones depend on it.`)
    break
  }
}

// ---- phase 4: clean -------------------------------------------------------
// One pass over everything the run touched, AFTER all building/verifying is done,
// so the cleaner never races a builder and its no-behavior-change rule is checkable
// against a finished, green suite.
let clean = null
const touched = [...new Set(results.flatMap(r => [...(r.files_changed || []), ...(r.tests_added || [])]))]
if (touched.length) {
  phase('Clean')
  clean = await agent(
    `You are the CLEANER, the final pass of a build workflow for: ${task}
Every milestone is already built and verified; the test suite is the contract. Files touched this
run (clean these and nothing else): ${JSON.stringify(touched)}
${REPO_RULES}

Your job — polish, with ZERO behavior change:
1. Run 'ruff check --fix .' then 'ruff format' on the touched files; fix any remaining lint by hand.
2. Tidy what build/fix rounds left behind IN THE TOUCHED FILES: dead imports and unused helpers,
   duplicated logic, leftover debug code, inconsistent naming between milestones, misplaced or
   redundant comments. Make docstrings say WHY, consistently, in the style of metrics/performance.py.
3. HARD LIMITS: do not change any public signature, any return value, any raised exception type or
   message that a test asserts on, and do not edit, delete, or weaken any test's assertions
   (renaming/reordering for clarity is fine if the suite stays green). If a cleanup you want
   requires a behavior change, SKIP it and list it in 'skipped' instead.
4. After cleaning, run 'ruff check .' and the FULL 'pytest' suite. If anything is red, fix it or
   revert your own change — never finish red. Set suite_green accordingly and be honest.`,
    { label: 'clean', phase: 'Clean', agentType: 'general-purpose', schema: CLEAN_SCHEMA },
  )
  if (clean) {
    log(`Cleaned ${clean.files_cleaned.length} file(s); suite ${clean.suite_green ? 'green' : 'RED — needs attention'}`)
  } else {
    log('Cleaner agent failed to return — code is as the verifiers left it (already green, just unpolished)')
  }
}

const verified = results.filter(r => r.status === 'verified').length
log(`Done: ${verified}/${plan.milestones.length} milestones verified`)
return { task, milestones: results, allVerified: verified === plan.milestones.length, clean }
