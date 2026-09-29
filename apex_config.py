"""apex_config — validation for everything a client can write.

Extracted from apex_dashboard.py. Pure functions with no project imports, so the
rules that guard the trading ledger can be tested directly and cannot drift into
the request handler.

Background: these endpoints used to cast and store client values unchecked. A
negative quantity made execute_sell subtract cash and inflate realised_pnl; a
risk_per_trade of 50 was consumed verbatim by the position-sizing arithmetic.
"""
from __future__ import annotations

# ── /api/config ──────────────────────────────────────────────────────────────
# key: (cast, minimum, maximum)
CONFIG_BOUNDS: dict = {
    "risk_per_trade":           (float, 0.0001, 0.25),
    "stop_loss_pct":             (float, 0.001,  0.50),
    "target_pct":                (float, 0.001,  1.00),
    "confidence_threshold":      (int,   1,       100),
    "check_interval_min":        (int,   1,       240),
    "idle_interval_min":         (int,   1,       1440),
    "india_max_positions":       (int,   1,       200),
    "us_max_positions":          (int,   1,       200),
    "eod_harvest_min":           (int,   0,       120),
    "eod_exit_min":              (int,   0,       120),
    "rl_exit_confidence":        (int,   1,       100),
    "short_confidence_threshold": (int,  1,       100),
    "atr_sl_mult":               (float, 0.1,     10.0),
    "atr_tp_mult":               (float, 0.1,     20.0),
    "daily_loss_limit_pct":      (float, 0.001,   0.50),
    "max_drawdown_pct":          (float, 0.001,   0.90),
    "open_filter_min":           (int,   0,       240),
    "cooldown_after_sl_min":     (int,   0,       1440),
    "commission_pct":            (float, 0.0,     0.05),
    "max_position_pct":          (float, 0.01,    1.0),
}

# Turning settings off bypasses the confidence gate, the position cap, the ADX
# and index filters and the ATR SL/TP clamps. It needs an explicit confirmation
# so it cannot be a one-keypress toggle on a live account.
UNSAFE_KEYS = frozenset({"settings_enabled"})
UNSAFE_CONFIRM_FIELD = "confirm_disable_settings"


def _coerce(field: str, value, cast, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    fv = float(value)
    if fv != fv or fv in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be a finite number")
    if not (lo <= fv <= hi):
        raise ValueError(f"{field}={value} out of range [{lo}, {hi}]")
    return cast(fv)


def validate_config_payload(data) -> tuple[dict, str]:
    """Coerce and bound-check a /api/config body. Returns (clean, error)."""
    if not isinstance(data, dict):
        return {}, "Body must be a JSON object"
    clean: dict = {}
    for k, v in data.items():
        if k == UNSAFE_CONFIRM_FIELD:
            continue
        if k in UNSAFE_KEYS:
            if v is not False and v != 0:
                return {}, f"{k} may only be set to false"
            if data.get(UNSAFE_CONFIRM_FIELD) is not True:
                return {}, (
                    f"Disabling {k} turns off the risk gates. "
                    f"Send {k}=false with {UNSAFE_CONFIRM_FIELD}=true to confirm."
                )
            clean[k] = False
            continue
        if k not in CONFIG_BOUNDS:
            return {}, f"Unknown setting: {k}"
        try:
            clean[k] = _coerce(k, v, *CONFIG_BOUNDS[k])
        except ValueError as e:
            return {}, str(e)
    return clean, ""


# ── Manual ledger corrections ────────────────────────────────────────────────
# realised_pnl is deliberately the only field allowed to go negative.
STATE_EDIT_SPEC: dict = {
    "cash":           (float, 0.0,   1e12),
    "realised_pnl":   (float, -1e12, 1e12),
    "wins":           (int,   0,     1_000_000),
    "losses":         (int,   0,     1_000_000),
    "peak_portfolio": (float, 0.0,   1e12),
    "max_drawdown":   (float, 0.0,   1e12),
}

# qty is strictly positive: a negative one made execute_sell subtract cash.
POSITION_EDIT_SPEC: dict = {
    "qty":       (int,   1,      100_000_000),
    "entry":     (float, 1e-6,   1e9),
    "stop_loss": (float, 1e-6,   1e9),
    "target":    (float, 1e-6,   1e9),
}


def clean_numeric(data, spec: dict) -> tuple[dict, str]:
    """Coerce and range-check an edit payload. Returns (clean, error)."""
    if not isinstance(data, dict):
        return {}, "Body must be a JSON object"
    clean: dict = {}
    for field, (cast, lo, hi) in spec.items():
        if field not in data:
            continue
        try:
            clean[field] = _coerce(field, data[field], cast, lo, hi)
        except ValueError as e:
            return {}, str(e)
    return clean, ""
