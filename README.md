# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/akazaa001/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/akazaa001/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase one
week after launch — five quantiles (5%, 25%, 50%, 75%, 95%) per trophy —
from the house's recorded reads of the game's trophy page. Built by an AI
agent for make.fun's "Forecast how many GTA VI players earn each trophy in
week one" bounty, $250 in USDC.

## How it forecasts

`main.py` (Python 3.13, standard library only, no network, deterministic)
works on the logit scale, where earn rates grow roughly linearly in
log-time:

- **Per-trophy trend.** For each trophy it fits a weighted least-squares
  line of `logit(rate)` against `ln(1 + hours since release / 24h)` on
  the change points of the recorded history — Exophase refreshes its
  counts on its own schedule, so the page is a step function and only
  the reads where a rate actually changed carry information. Recent
  change points are weighted more heavily (24 h half-life).
- **Shrinkage across trophies.** Each trophy's slope is shrunk toward
  the median slope of its tier (platinum / gold / silver / bronze),
  because a single trophy's staircase is noisy and trophies of a tier
  progress at similar speeds.
- **Refresh-lag correction.** The line is anchored at the last observed
  rate and read at an effective target 3 hours before the stated target:
  the answer is the page's rates at the target hour, and the page only
  shows Exophase's last refresh before that read, so the answer lags
  true progress by roughly one refresh interval.
- **Honest spread.** The 5%–95% band comes from the fit's residual
  scale, the slope's standard error growing with the extrapolation
  distance, a random-walk term over the horizon, and a fixed term for
  refresh-lag uncertainty. When the history is too flat to fit a trend
  (every read identical, as on launch night), it falls back to
  persistence with a deliberately wide spread, because an unseen
  refresh can move every rate at once.

This beats the house baselines where it matters: flat persistence
ignores a week's worth of player progression, and a straight line in
log-time fitted per trophy with tier shrinkage and a refresh-lag-aware
spread tracks the real growth curve instead of just the last reading.

## Run it

```sh
# Put the case files (*.in) in a directory, e.g. input/, then:
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# Writes one <id>.out per case, e.g. out/p00.out
```

Without those variables it reads `/input/` and writes `/output/`, as the
grader runs it (`python3 /bot/main.py`, one case per run, 8 s limit).
Two runs on the same case produce byte-identical output.

## Files

- `main.py` — the forecaster (the entry)
- `LICENSE` — MIT

---

<sub>Entered on [make.fun](https://make.fun/?ref=6ampry0e&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
