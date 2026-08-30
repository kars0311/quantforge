"""Mean-reversion: fade short-term extremes. Strategy #2 (week 4).

Signal: rolling z-score of price vs its own trailing mean,

    z_t = (P_t - rolling_mean(P, lookback)_t) / rolling_std(P, lookback)_t

with every rolling window strictly backward-looking (window=lookback, min_periods=lookback), so a
row's weight uses information through that row's close only; the engine's one-day shift handles
execution timing. Entries/exits use hysteresis (enter at a wide threshold, exit at a narrow one)
so the book does not churn every time z wiggles across a single line.

params: {lookback:int=20, entry_z:float=2.0, exit_z:float=0.5, mode:str='long_flat'}
  mode 'long_flat'  -> long oversold names (z < -entry_z), exit when z > -exit_z.
  mode 'long_short' -> additionally short overbought names (z > entry_z), cover when z < exit_z.

**Honesty caveat (short side):** the backtest charges only symmetric transaction costs on turnover.
Real short positions also pay borrow fees, face locate constraints (the shares may not be available
to borrow at all), and carry squeeze risk — none of which are modeled here. long_short results are
therefore optimistic on the short leg; this is stated rather than hidden (product.md, 05-strategies).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..engine.base import Strategy


def _hysteresis_state(entry: pd.DataFrame, exit_: pd.DataFrame, z: pd.DataFrame) -> pd.DataFrame:
    """One-sided hysteresis as a vectorized state machine: 1.0 while 'in', 0.0 while 'out'.

    Why event-frame + ffill instead of a Python loop: mark entry rows 1.0 and exit rows 0.0 in an
    otherwise-NaN frame, then forward-fill — the last event before each row determines its state,
    which is exactly what a sequential loop would compute, at vectorized speed.

    Precedence (documented behavior): where entry and exit are simultaneously true — possible only
    when exit_z > entry_z, so the bands overlap — **entry wins** (the 1.0 is written after the 0.0).

    NaN policy: any row where z is NaN (warmup, or a NaN price poisoning the window) is forced flat
    (0.0) and *breaks* state — a position never forward-fills across a gap in the data, because we
    could not have known at the time whether the exit condition fired inside it. Comparisons with
    NaN are False, so entry can never fire on such a row; writing the 0.0 last makes the override
    explicit.
    """
    state = pd.DataFrame(float("nan"), index=z.index, columns=z.columns)
    state[exit_] = 0.0
    state[entry] = 1.0  # after exit_: entry wins where both fire
    state[z.isna()] = 0.0  # NaN z overrides everything and severs the ffill chain
    return state.ffill().fillna(0.0)


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def generate_signals(self, prices, params: dict[str, Any] | None = None) -> pd.DataFrame:
        params = params or {}
        lookback = int(params.get("lookback", 20))
        entry_z = float(params.get("entry_z", 2.0))
        exit_z = float(params.get("exit_z", 0.5))
        mode = str(params.get("mode", "long_flat"))
        if mode not in ("long_flat", "long_short"):
            raise ValueError(f"mode must be 'long_flat' or 'long_short', got {mode!r}")

        prices = prices.sort_index()  # sort_index returns a new frame; the caller's is untouched

        # min_periods=lookback: z is NaN until a full window exists — no partial-window peeking.
        # rolling_std uses pandas' default ddof=1 (sample std): the window is a sample of the
        # price process, not the full population, and n-1 keeps the estimator unbiased; with
        # lookback >= 5 the practical difference vs ddof=0 is a slightly wider band.
        rolling = prices.rolling(window=lookback, min_periods=lookback)
        z = (prices - rolling.mean()) / rolling.std()

        # Long book: enter deeply oversold, hold until z recovers past the narrow exit band.
        state = _hysteresis_state(entry=z < -entry_z, exit_=z > -exit_z, z=z)

        if mode == "long_short":
            # Two independent one-sided machines, then summed. This is safe (states are mutually
            # exclusive) because the entry bands are disjoint: entry_z >= 0.5 > 0 >= -exit_z, so
            # the row that enters one side necessarily satisfies the other side's exit condition
            # and closes it in the same bar.
            short = _hysteresis_state(entry=z > entry_z, exit_=z < exit_z, z=z)
            state = state - short  # per-name state in {-1, 0, +1}

        # Equal weight per active name: w_i = state_i / n_active, so gross sum|w| = 1 whenever any
        # name is active (Sigma w = 1 for long_flat) and rows with no active names are all-zero.
        # The 1/n arithmetic lands within a few ULPs of 1.0, inside the engine's 1e-9 tolerance.
        n_active = (state != 0.0).sum(axis=1)
        return state.div(n_active.where(n_active != 0), axis=0).fillna(0.0)
