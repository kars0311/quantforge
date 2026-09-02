# tearsheet.R — R analytics/risk layer (CORE): first concrete polyglot interop.
#
# WHAT THIS SCRIPT IS
# -------------------
# A headless batch script (run with `Rscript`, never interactively) that reads the Parquet
# hand-off written by the Python pipeline, recomputes the six headline performance metrics
# independently in R, and writes them back as Parquet so a Python test can assert that both
# languages agree. The Parquet *file* is the contract (see src/quantforge/interchange.py —
# SCHEMAS); there is no in-process Python<->R bridge to break.
#
# WHY RECOMPUTE METRICS IN A SECOND LANGUAGE AT ALL?
# Because agreement between two independent implementations is strong evidence that neither
# has a silent bug (RG-6 in the design docs: agreement within ~1%; since the formulas below
# are transcribed exactly, we actually expect agreement near machine precision).
#
# CONTRACT (docs/components/08-r-tearsheet.md)
#   Rscript analytics_r/tearsheet.R [data_dir=data_cache] [out_dir=analytics_r/output]
#   reads:  data_dir/returns.parquet       (interchange `returns` kind: date, ret)
#           data_dir/asset_returns.parquet (interchange `asset_returns` kind: date, ticker,
#                                           ret — REQUIRED; feeds the optimizer cross-check)
#   writes: out_dir/tearsheet.png          (charts.PerformanceSummary — the visual deliverable)
#           data_dir/metrics_r.parquet     (interchange `metrics` kind: name, value)
#           data_dir/weights_r.parquet     (interchange `weights` kind: ticker, weight —
#                                           PortfolioAnalytics' independent min-variance optimum)
#   stdout: table.AnnualizedReturns + table.Drawdowns (institutional-style display tables)
#   exit:   0 on success; non-zero + message on ANY failure, via stop() — Rscript turns an
#           uncaught stop() into exit status 1, which is what pytest's subprocess check needs.
#
# READING GUIDE (Kent): each numbered STEP below maps 1:1 to a Python concept you already
# built. R differences that matter are called out where they occur — the big one is that
# R's sd() divides by n-1 (sample std) while the Python reference uses ddof=0 (population
# std); see STEP 5. Both inputs are read AND validated up front (STEPs 1-2) before any
# output file is produced, so a broken hand-off can never leave partial artifacts behind.

# --- Libraries ---------------------------------------------------------------------------
# library(...) in R ~ `import ...` in Python. suppressPackageStartupMessages keeps the
# stderr of a *batch* script clean so that when it fails, the only thing on stderr is the
# actual error message (pytest surfaces stderr on failure — noise there hides the signal).
# The extra suppressMessages layer catches load-time chatter that is NOT a "package startup
# message" in R's technical sense (e.g. quantmod's "Registered S3 method overwritten" note,
# emitted while tidyquant pulls in its dependencies) — same goal, different message channel.
suppressMessages(suppressPackageStartupMessages({
  library(arrow)     # Parquet I/O — same Arrow memory format pyarrow uses on the Python side
  library(tidyquant) # the R finance meta-package (FR-6): attaches the whole
                     # PerformanceAnalytics/quantmod/xts/zoo stack we draw on below —
                     # xts for time-indexed series (~ pd.Series with a DatetimeIndex),
                     # PerformanceAnalytics for the tearsheet chart + display tables.
  library(PortfolioAnalytics) # the second, independent optimizer (~ PyPortfolioOpt's
                              # EfficientFrontier on the Python side) for the RG-6 cross-check.
  library(ROI)                # R Optimization Infrastructure: the solver front-end that
                              # optimize_method = "ROI" dispatches through (~ cvxpy's role
                              # under PyPortfolioOpt).
  library(ROI.plugin.quadprog) # registers the quadprog QP solver with ROI. Loaded explicitly
                               # because ROI discovers solvers by loaded plugin, not by what
                               # happens to be installed — without this line the StdDev QP
                               # below would have no solver to dispatch to.
}))

# --- STEP 0: command-line arguments ------------------------------------------------------
# commandArgs(trailingOnly = TRUE) ~ Python's sys.argv[1:] — just the user's arguments,
# without the interpreter/script paths. Both arguments are optional positionals with the
# defaults fixed by the component contract.
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[[1]] else "data_cache"
out_dir  <- if (length(args) >= 2) args[[2]] else "analytics_r/output"

# dir.create ~ os.makedirs(out_dir, exist_ok=TRUE). showWarnings=FALSE makes it silent when
# the directory already exists (recursive=TRUE is the exist_ok/parents combination).
# out_dir holds the tearsheet PNG written in STEP 4; created up front so the contract
# ("out_dir exists after any successful run") holds even if STEP 4 ever fails mid-render.
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# --- STEP 1: read returns.parquet and validate it against the interchange contract -------
# Mirrors quantforge.interchange.read_frame(path, "returns"): read, then validate loudly at
# the boundary. R is a *consumer* of the contract here, so it re-checks rather than trusting
# that the writer was our own Python code — a malformed file must fail with a message naming
# the problem, not surface as NaNs in the metrics comparison.
returns_path <- file.path(data_dir, "returns.parquet") # ~ os.path.join
if (!file.exists(returns_path)) {
  stop("missing input file: ", returns_path,
       " — run the Python pipeline first to produce the interchange hand-off")
}
returns_df <- arrow::read_parquet(returns_path)

# Columns must be exactly `date, ret`, in that order — column ORDER is part of the contract
# (interchange.py validates it too), so identical() on the full names vector is the right
# check, not a set comparison. ~ validate_frame's "expected columns" branch.
expected_cols <- c("date", "ret")
if (!identical(names(returns_df), expected_cols)) {
  stop("returns.parquet has wrong columns: expected exactly [",
       paste(expected_cols, collapse = ", "), "] in that order, got [",
       paste(names(returns_df), collapse = ", "), "]")
}

# Dtype checks ~ validate_frame's per-column checks. Arrow reads the contract's
# timestamp[ns, UTC] column as POSIXct (R's tz-aware timestamp type, ~ pd.Timestamp).
if (!inherits(returns_df$date, "POSIXct")) {
  stop("returns.parquet column 'date' must be a timestamp (POSIXct), got class: ",
       paste(class(returns_df$date), collapse = "/"))
}
if (!is.double(returns_df$ret)) {
  stop("returns.parquet column 'ret' must be double (float64), got type: ",
       typeof(returns_df$ret))
}

# Pin the display timezone to UTC. For a POSIXct this is a lossless view change (the value
# is seconds-since-epoch underneath; ~ .dt.tz_convert("UTC") on an aware pandas column) —
# it just guarantees every date printed in the tearsheet matches the file's UTC convention.
attr(returns_df$date, "tzone") <- "UTC"

# --- STEP 2: read asset_returns.parquet, validate, pivot to a wide panel -----------------
# The per-asset input for the optimizer cross-check (component doc 08, internal step 4).
# Mirrors read_frame(path, "asset_returns"): long format, one row per (date, ticker) pair.
# It is REQUIRED — the week-6 deliverable includes the second-optimizer weights, so a run
# without this file is a broken hand-off, not a partial success. Validated HERE, before any
# output artifact is written, for the same reason as STEP 1: a malformed file must fail
# loudly with nothing left on disk, never surface as a half-produced tearsheet.
asset_path <- file.path(data_dir, "asset_returns.parquet")
if (!file.exists(asset_path)) {
  stop("missing input file: ", asset_path,
       " — run the Python pipeline first to produce the per-asset interchange hand-off")
}
asset_df <- arrow::read_parquet(asset_path)

# Exact columns in exact order, same identical() discipline as STEP 1 (~ validate_frame).
expected_asset_cols <- c("date", "ticker", "ret")
if (!identical(names(asset_df), expected_asset_cols)) {
  stop("asset_returns.parquet has wrong columns: expected exactly [",
       paste(expected_asset_cols, collapse = ", "), "] in that order, got [",
       paste(names(asset_df), collapse = ", "), "]")
}
if (!inherits(asset_df$date, "POSIXct")) {
  stop("asset_returns.parquet column 'date' must be a timestamp (POSIXct), got class: ",
       paste(class(asset_df$date), collapse = "/"))
}
if (!is.character(asset_df$ticker)) {
  stop("asset_returns.parquet column 'ticker' must be string (character), got type: ",
       typeof(asset_df$ticker))
}
if (!is.double(asset_df$ret)) {
  stop("asset_returns.parquet column 'ret' must be double (float64), got type: ",
       typeof(asset_df$ret))
}
attr(asset_df$date, "tzone") <- "UTC" # same lossless UTC pin as STEP 1

# A duplicate (date, ticker) pair cannot be pivoted — two candidate values for one cell —
# and silently keeping one would hide an upstream data bug (~ interchange.to_wide's
# duplicate guard, which raises for the identical reason).
if (anyDuplicated(asset_df[, c("date", "ticker")]) > 0) {
  stop("asset_returns.parquet has duplicate (date, ticker) rows — ",
       "each pair must appear at most once to pivot to a wide panel")
}

# Pivot long -> wide (~ interchange.to_wide / df.pivot): one column per ticker, dates as
# the index. Built by merging one single-ticker xts per ticker — merge.xts outer-joins on
# the date index, so a (date, ticker) pair missing from the file becomes an NA cell,
# exactly like pandas' pivot. Tickers are sorted so downstream order is deterministic
# regardless of the file's row order.
tickers <- sort(unique(asset_df$ticker))
asset_xts <- do.call(merge, lapply(tickers, function(tk) {
  rows <- asset_df$ticker == tk
  xts::xts(asset_df$ret[rows], order.by = asset_df$date[rows])
}))
colnames(asset_xts) <- tickers

# The optimizer contract (mirroring _validate_returns_panel in portfolio/optimize.py —
# the Python side of this cross-check): no NAs (a ragged panel is a data-loading problem,
# not something an optimizer should paper over), at least 2 assets (nothing to allocate
# across otherwise), at least 2 rows (a covariance needs them). NA cells cover both a
# missing (date, ticker) pair and a literal NA return in the file.
if (any(is.na(asset_xts))) {
  stop("asset_returns.parquet has NA cells after pivoting to wide — every ticker must ",
       "have a return for every date (ragged or NA-holed panels are rejected, as in the ",
       "Python optimizer)")
}
if (length(tickers) < 2) {
  stop("asset_returns.parquet must contain at least 2 tickers to optimize over, got ",
       length(tickers))
}
if (nrow(asset_xts) < 2) {
  stop("asset_returns.parquet needs at least 2 dates to estimate a covariance, got ",
       nrow(asset_xts))
}

# --- STEP 3: build the xts series --------------------------------------------------------
# xts = "eXtensible Time Series": a numeric vector ordered by a time index — the R finance
# ecosystem's equivalent of pd.Series(values, index=DatetimeIndex). PerformanceAnalytics
# (the chart + tables in STEP 4) takes xts natively, so we adopt R's convention at the
# boundary instead of passing bare vectors around.
returns_xts <- xts::xts(returns_df$ret, order.by = returns_df$date)
colnames(returns_xts) <- "ret"

# A series with zero usable observations means the hand-off is broken. Check it HERE, before
# the tearsheet step, so an empty/all-NA file fails with this diagnostic instead of whatever
# PerformanceAnalytics would say about a chart of nothing. (all() of an empty vector is TRUE
# in R — same convention as Python's all([]) — so this one condition covers both the
# zero-row file and the all-NA file.)
if (all(is.na(returns_df$ret))) {
  stop("returns.parquet contains no non-NA returns — nothing to compute metrics on")
}

# --- STEP 4: tearsheet — PNG chart + display tables (component doc 08, internal step 2) ---
# The visual deliverable. png(...) opens a GRAPHICS DEVICE: from here until dev.off(), every
# plot call renders into the file instead of a window (~ matplotlib's Agg backend +
# savefig). This is what makes the script truly headless — the png device needs no display,
# unlike quartz()/X11(), which must never appear in a batch script (they'd fail on a server
# with no DISPLAY, exactly where this runs in CI/Docker).
png(file.path(out_dir, "tearsheet.png"), width = 1200, height = 900)
# charts.PerformanceSummary draws the standard institutional tearsheet in three stacked
# panels sharing one date axis: cumulative (compounded) return, per-period returns, and
# drawdown — growth, day-to-day risk, and pain, in one picture.
PerformanceAnalytics::charts.PerformanceSummary(returns_xts)
# dev.off() closes the device — flushes and finalizes the PNG file (~ fig.savefig + close).
# It RETURNS the now-active device (visibly!), and Rscript auto-prints top-level values, so
# a bare dev.off() would leak "null device 1" into stdout; invisible() keeps stdout clean
# for the tables below.
invisible(dev.off())

# Display tables to stdout, so a headless run still shows the institutional-style summary
# in the terminal/log. print() forces the table to render even under Rscript (~ Python's
# print(df); at an interactive prompt R auto-prints, in a batch script it does not).
#
# CONVENTIONS CAVEAT (AR-3): these tables use PerformanceAnalytics' OWN conventions —
# e.g. SAMPLE std dev (n-1) and geometric annualization of the mean return — which differ
# from performance.py's definitions (population std, (1+total)^(252/n)-1). That's fine
# because they are DISPLAY-ONLY: nothing here crosses the Parquet boundary or feeds the
# cross-language comparison. The numbers that do cross are compute_metrics_r's (STEP 5),
# which transcribe Python's definitions exactly — we match definitions, not invent variants.
cat("\n== Annualized returns (PerformanceAnalytics conventions — display only) ==\n")
print(PerformanceAnalytics::table.AnnualizedReturns(returns_xts))
cat("\n== Worst drawdowns (PerformanceAnalytics conventions — display only) ==\n")
# table.Drawdowns warns on stderr when the series has fewer than the 5 drawdowns it shows
# by default ("Only N available..."). That's informational, not a failure — suppress it so
# stderr stays reserved for real errors (the invariant the Libraries comment establishes).
print(suppressWarnings(PerformanceAnalytics::table.Drawdowns(returns_xts)))

# --- STEP 5: compute_metrics_r — reimplements Python's compute_metrics -------------------
# Reference: src/quantforge/metrics/performance.py (the single source of truth for metric
# DEFINITIONS — this function transcribes those definitions, it does not invent variants).
# Same conventions, stated explicitly:
ANN <- 252        # trading days per year (annualization factor)
RISK_FREE <- 0.0  # risk-free rate assumption; Python documents the same simplification

compute_metrics_r <- function(returns) {
  # Accept the xts (or any numeric vector) and work on plain numbers, like Python's
  # `r = pd.Series(returns).dropna()`. as.numeric(coredata(...)) strips the time index —
  # none of the six formulas need dates, only the ordered values.
  r <- as.numeric(zoo::coredata(returns))
  r <- r[!is.na(r)] # ~ .dropna(); an all-NA file must not silently become metrics of nothing
  n <- length(r)
  if (n == 0) {
    # Python returns a dict of NaNs here; for the batch script an empty series means the
    # hand-off is broken, so fail loudly instead of writing a file of NaNs that a tolerance
    # comparison could never catch. (stop() => non-zero exit for pytest.) The linear flow
    # above already rejected this case before the tearsheet step; this guard stays so the
    # function is safe on its own, not only in this script's call order.
    stop("returns.parquet contains no non-NA returns — nothing to compute metrics on")
  }

  # total_return: compound the whole series. prod(1 + r) - 1 ~ (1 + r).prod() - 1.
  total_return <- prod(1 + r) - 1

  # cagr: annualize the total by the number of OBSERVATIONS n (not calendar days) — the
  # exact convention performance.py uses: (1 + total)^(252/n) - 1.
  cagr <- (1 + total_return)^(ANN / n) - 1

  # POPULATION standard deviation — THE one real Python/R divergence in this file:
  # Python uses r.std(ddof=0) (divide by n); R's sd() is SAMPLE std (divide by n-1).
  # We state the difference and correct it explicitly — sd(r) * sqrt((n-1)/n) rescales
  # sample std to population std — rather than absorbing it silently into the ~1%
  # cross-check tolerance, where it would mask genuine bugs of similar size.
  # n == 1 needs its own branch: R's sd() of a single value is NA (0/0 sample variance),
  # and NA * 0 stays NA, whereas numpy's ddof=0 std of one value is exactly 0.
  std <- if (n < 2) 0.0 else sd(r) * sqrt((n - 1) / n)

  # ann_vol: scale daily std to annual by sqrt(time) — the iid-returns convention.
  ann_vol <- std * sqrt(ANN)

  # sharpe: annualized mean excess return over annualized vol, risk-free 0.
  # mean(r)/std * sqrt(252) ~ Python's excess / std * np.sqrt(ANN). Guard std == 0
  # (constant returns) with NaN exactly as Python does — 0/0 would poison the comparison.
  excess <- mean(r) - RISK_FREE / ANN
  sharpe <- if (std > 0) excess / std * sqrt(ANN) else NaN

  # max_drawdown: worst peak-to-trough loss of the compounded equity curve.
  # cumprod ~ .cumprod(), cummax ~ .cummax(); the ratio equity/peak - 1 is the drawdown
  # at each date, and its min is the (negative) maximum drawdown.
  equity <- cumprod(1 + r)
  max_drawdown <- min(equity / cummax(equity) - 1)

  # hit_rate: fraction of STRICTLY positive days. mean(r > 0) ~ (r > 0).mean() — R, like
  # Python, treats TRUE as 1 in arithmetic, so mean of a logical vector is a proportion.
  hit_rate <- mean(r > 0)

  # Return a two-column data.frame shaped like the interchange `metrics` kind
  # (~ pd.DataFrame({"name": [...], "value": [...]})). Row order matches Python's _KEYS
  # exactly — order is part of what the cross-check test asserts.
  data.frame(
    name = c("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "hit_rate"),
    value = c(total_return, cagr, ann_vol, sharpe, max_drawdown, hit_rate),
    stringsAsFactors = FALSE # keep name as character, not factor (contract says string)
  )
}

metrics_df <- compute_metrics_r(returns_xts)

# --- STEP 6: write metrics_r.parquet -----------------------------------------------------
# ~ quantforge.interchange.write_frame(metrics_df, path, "metrics"). arrow maps R's
# character -> Arrow string and double -> float64, which is exactly the `metrics` schema
# (name: string, value: float64), so Python's read_frame(..., "metrics") validates it.
# Written into data_dir (not out_dir): metrics_r.parquet is a machine-readable interchange
# artifact that lives next to its Python inputs; out_dir is for human-facing tearsheet
# outputs (the STEP 4 PNG).
metrics_path <- file.path(data_dir, "metrics_r.parquet")
arrow::write_parquet(metrics_df, metrics_path)

# --- STEP 7: PortfolioAnalytics — the independent second optimizer (RG-6) ----------------
# The Python counterpart is optimize_weights(returns, {"objective": "min_volatility"}) in
# src/quantforge/portfolio/optimize.py (component 07). Two libraries, two languages, one
# convex problem — if both land on the same weights, neither optimizer setup has a silent
# bug (same logic as the metrics cross-check above).
#
# WHY the cross-check objective is MINIMUM-VARIANCE, not max-Sharpe: min-vol is a convex
# quadratic program — minimize w' S w subject to sum(w) = 1, 0 <= w <= 1 — with a UNIQUE
# global optimum (for a positive-definite S) that any correct QP solver must find, so the
# two libraries are obligated to agree to solver precision. Max-Sharpe is not a plain QP:
# each library reaches it through its own internal transformation/search, and two correct
# implementations can legitimately return slightly different portfolios — a cross-check
# there would test solver internals, not our setup.
#
# ANNUALIZATION NOTE: Python minimizes over sample_cov * 252 (annualized), while
# PortfolioAnalytics uses the plain sample covariance of the daily returns. Scaling S by a
# positive constant scales the objective without moving its argmin, so the WEIGHTS are
# directly comparable; only the reported volatilities would differ by sqrt(252).
pspec <- portfolio.spec(assets = tickers) # ~ EfficientFrontier(mu, cov, ...): declare assets
# Same constraint set as the Python optimizer's defaults (component 07):
#   full_investment: sum(weights) = 1     ~ pypfopt's implicit fully-invested budget
#   box 0..1:        long-only            ~ weight_bounds=(0.0, 1.0)
pspec <- add.constraint(pspec, type = "full_investment")
pspec <- add.constraint(pspec, type = "box", min = 0, max = 1)
# The objective: minimize portfolio risk. name = "StdDev" — minimizing std dev and
# minimizing variance pick the same portfolio (sqrt is monotone), and with the ROI method
# this is solved as the QP above. ~ ef.min_volatility().
pspec <- add.objective(pspec, type = "risk", name = "StdDev")
# optimize_method = "ROI" dispatches to quadprog — a DETERMINISTIC global QP solve, unlike
# PortfolioAnalytics' random/DEoptim search methods, whose stochastic answers would make
# the cross-check flaky and the tolerance meaningless.
opt <- optimize.portfolio(R = asset_xts, portfolio = pspec, optimize_method = "ROI")
w <- extractWeights(opt)[tickers] # reindex by sorted tickers ~ .reindex(returns.columns)

# Sanity-check the solve before anything touches disk (same spirit as write_frame
# validating BEFORE writing): the budget must hold to solver precision and no weight may
# leave the long-only box by more than numerical dust. A violation means the solver setup
# is wrong, and a wrong weights file must never be written for the cross-check to "pass".
if (any(is.na(w)) || abs(sum(w) - 1) > 1e-6 || any(w < -1e-8) || any(w > 1 + 1e-8)) {
  stop("optimizer produced invalid weights (sum=", sum(w),
       ", range=[", min(w), ", ", max(w), "]) — expected sum 1 and every weight in [0, 1]")
}

# --- STEP 8: write weights_r.parquet -----------------------------------------------------
# The interchange `weights` kind (ticker: string, weight: float64) — arrow maps R's
# character/double to exactly those Arrow types, so Python's read_frame(..., "weights")
# validates it. One row per asset INCLUDING near-zero weights (an absent row would be
# ambiguous: excluded asset or lost row?), in sorted-ticker order so repeated runs produce
# byte-identical, diffable files. Written to data_dir like metrics_r.parquet: it is a
# machine-readable cross-check artifact, not a human-facing plot.
weights_df <- data.frame(
  ticker = tickers,
  weight = as.numeric(w),
  stringsAsFactors = FALSE
)
weights_path <- file.path(data_dir, "weights_r.parquet")
arrow::write_parquet(weights_df, weights_path)

# A tiny success breadcrumb on stderr (message() writes to stderr, like print(...,
# file=sys.stderr)) so a human running the script by hand sees it worked; pytest keys off
# the exit status, not this text. Reaching this line at all means every step above
# succeeded — any failure would have stop()ped with a non-zero exit first.
message("tearsheet.R: wrote ", file.path(out_dir, "tearsheet.png"))
message("tearsheet.R: wrote ", metrics_path, " (", nrow(metrics_df), " metrics)")
message("tearsheet.R: wrote ", weights_path, " (", nrow(weights_df), " weights)")
