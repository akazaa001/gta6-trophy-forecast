"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Model (own design, standard library only, deterministic, no network):

For each trophy we work on the logit scale, y = logit(rate), against
log-time x = ln(1 + t / 24h) where t is hours since the game's release
(2026-11-19T00:00Z). Earn rates grow roughly linearly in log-time while
players progress through the game, then saturate; a line in log-time
captures the early-to-mid growth and extrapolates gently, unlike a line
in raw time (which explodes) or flat persistence (which ignores growth).

* History is a step function: Exophase refreshes its counts on its own
  schedule, so consecutive reads are often identical. We keep only the
  change points (each distinct rate, at the first read showing it).
* A weighted least-squares line is fitted per trophy on the change
  points, with weights that favour recent reads (24 h half-life). The
  slope is shrunk toward the median slope of its tier (platinum/gold/
  silver/bronze), because a single trophy's staircase is noisy and
  trophies of a tier progress at similar speeds.
* The line is read at an effective target time 3 h before the stated
  target: the answer is the page's rates at the target hour, and the
  page only shows Exophase's last refresh before that read, so the
  answer lags true progress by roughly one refresh interval.
* Spread is honest: residual scale of the fit, plus a random-walk term
  growing with the horizon, plus a fixed refresh-lag term. With too
  little history to fit (e.g. the practice cases, where every read is
  identical), we fall back to persistence with a deliberately wide
  spread, because an unseen refresh can move every rate at once.
* Quantiles are the median logit plus z_tau * sigma, converted back to
  percent and rounded to 4 decimals (half up, as the house baseline).

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
LOG_TIME_SCALE_H = 24.0      # x = ln(1 + t / 24h)
RECENCY_HALF_LIFE_H = 24.0   # fit weights favour recent change points
REFRESH_LAG_H = 3.0          # answer lags true progress by ~one refresh
RW_SIGMA_PER_SQRT_DAY = 0.35  # random-walk spread, logit units / sqrt(day)
REFRESH_SIGMA = 0.30          # extra spread for refresh-lag uncertainty
MIN_SPREAD = 0.35
MAX_SPREAD = 2.5
DEGENERATE_SPREAD_BASE = 0.9  # wide spread when no trend can be fitted
MAX_LOGIT_MOVE = 4.0          # cap extrapolated move from the last rate


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(dt: datetime) -> float:
    return (dt - RELEASE).total_seconds() / 3600.0


def log_time(t_h: float) -> float:
    return math.log1p(max(t_h, 0.0) / LOG_TIME_SCALE_H)


def change_points(case: dict, idx: int):
    """Distinct (t_hours, rate) points for trophy idx, first read per value."""
    pts = []
    prev = None
    for read in case["history"]:
        rate = read["rates"][idx]
        if rate is None:
            continue
        if prev is not None and rate == prev:
            continue
        prev = rate
        pts.append((hours_since_release(parse_time(read["at"])), float(rate)))
    return pts


def wls_line(pts, t_ref_h):
    """Weighted least squares y = a + b*x on change points.

    Returns (a, b, resid_rms, slope_se, n) or None when the fit is not
    identified. Weights decay with age from the last point (24 h half-life).
    """
    if len(pts) < 3:
        return None
    xs = [log_time(t) for t, _ in pts]
    ys = [logit(r) for _, r in pts]
    if max(xs) - min(xs) < 0.05:
        return None
    ws = [0.5 ** ((t_ref_h - t) / RECENCY_HALF_LIFE_H) for t, _ in pts]
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw)
    # Standard error of the slope: how far the extrapolated line can be
    # off because the slope itself is uncertain. It grows with the
    # distance extrapolated, so it belongs in the spread, not the median.
    slope_se = rms / math.sqrt(sxx) if sxx > 0 else 0.0
    return a, b, rms, slope_se, len(pts)


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    horizon_days = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])
    t_target = hours_since_release(target_dt) - REFRESH_LAG_H
    x_target = log_time(t_target)
    t_last = hours_since_release(parse_time(case["history"][-1]["at"]))

    trophies = case["trophies"]
    n = len(trophies)

    # First pass: per-trophy fits and tier slope pools.
    fits = []
    tier_slopes: dict[str, list[float]] = {}
    for i, trophy in enumerate(trophies):
        pts = change_points(case, i)
        fit = wls_line(pts, t_last)
        fits.append((pts, fit))
        if fit is not None:
            tier_slopes.setdefault(trophy["tier"], []).append(fit[1])
    tier_median_slope = {t: median(s) for t, s in tier_slopes.items()}

    rw = RW_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        pts, fit = fits[i]
        last_rate = pts[-1][1] if pts else 0.5
        last_logit = logit(last_rate)

        if fit is not None:
            a, b, rms, slope_se, npts = fit
            # Shrink the slope toward the tier median: a staircase fit on
            # few change points is noisy, and tiers progress together.
            tier_b = tier_median_slope.get(trophy["tier"], b)
            w = npts / (npts + 3.0)
            b_shrunk = w * b + (1.0 - w) * tier_b
            # Slopes should not be wildly negative: rates rarely collapse
            # over a week except by dilution, which the fit already sees.
            b_shrunk = max(b_shrunk, -1.5)
            # Anchor the line at the last observed rate, then extrapolate
            # with the shrunk slope: the anchor keeps the forecast honest
            # to the most recent page, the slope carries the trend.
            dx = x_target - log_time(pts[-1][0])
            med_logit = last_logit + b_shrunk * dx
            med_logit = last_logit + max(-MAX_LOGIT_MOVE,
                                         min(MAX_LOGIT_MOVE,
                                             med_logit - last_logit))
            # Spread: residual scale (floored, because a staircase fit
            # understates true noise), slope uncertainty growing with the
            # extrapolation distance, a random-walk term over the horizon,
            # and a fixed term for refresh-lag uncertainty.
            extrap = min(slope_se * abs(dx), 1.0)
            sigma = math.sqrt(max(rms, 0.25) ** 2 + extrap ** 2
                              + rw ** 2 + REFRESH_SIGMA ** 2)
        else:
            # No identifiable trend: persist, with a wide spread. An unseen
            # Exophase refresh can shift every rate at once (the practice
            # cases are exactly this: flat history, refreshed answer).
            med_logit = last_logit
            sigma = DEGENERATE_SPREAD_BASE + 0.5 * math.sqrt(max(horizon_days, 0.0))

        sigma = min(max(sigma, MIN_SPREAD), MAX_SPREAD)
        q = []
        for z in Z:
            v = round4(clamp_rate(inv_logit(med_logit + z * sigma)))
            q.append(v)
        # Enforce non-decreasing after rounding.
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
