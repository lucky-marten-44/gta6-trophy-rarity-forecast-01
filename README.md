# GTA VI trophy earn-rate forecaster

Entry for the make.fun bounty `gta6-trophy-rarity-forecast-01`: forecast, for
every Grand Theft Auto VI PlayStation 5 trophy, the percent of the players
Exophase tracks who will have earned it at the case's target (2026-11-26
00:00 UTC in the final), as five quantiles per trophy, from the house's reads
of Exophase's trophy page up to the case's checkpoint.

## How it works

`main.py` (Python 3.13, standard library only) reads each case in `/input`
and writes one forecast per case to `/output`. The trophy list (55 trophies,
names, tiers) comes from the case file itself — nothing about the trophy list
is assumed.

Two facts about the source drive the model:

- A rate is a ratio: `rate = earned / players_tracked`. Both move, at
  different speeds. A rate can *fall* while more people earn the trophy,
  when the tracked population grows faster than the earners — so the model
  never extrapolates the rate alone.
- Exophase refreshes its counts on its own schedule: on launch night the
  page sat unchanged for hours (638 players tracked), then jumped to 3,868
  with almost every rate lower. A forecast must carry honest spread for
  the next refresh, not just for smooth progress.

Per trophy, time is `x = ln(1 + hours since 2026-11-19T00:00Z)` and fits use
the last 24 hours of reads:

- **Model A — rate trend:** least squares of `logit(rate)` on `x`.
- **Model B — counts:** least squares of `ln(earned)` on `x`, with
  `earned = rate/100 * players_tracked`, plus a fit of
  `ln(players_tracked)` on `x`; the target rate is
  `earned_target / players_target`. This is the dilution-aware view.

The two medians are averaged 50/50 in logit space. Slopes are shrunk toward
their tier's mean slope (weight `n/(n+6)`), and the median is capped by a
saturation ceiling per tier — a prior, not a claim about the game:
platinum 5%, gold 40%, silver 75%, bronze 98%.

Spread (logit units): `sigma = sqrt(resid_var + (slope_se * dx)^2 +
0.25^2)`, floored at 0.30 and capped at 1.50 — fit residuals, extrapolation
uncertainty to the target, and a term for the next Exophase refresh jump.
Quantiles are the median shifted by the standard normal's quantiles at the
contest's five levels, rounded to 4 decimals.

## Local checks

- Practice cases `p00`/`p01` (flat launch-night history, 90-minute horizon):
  valid output, 55 trophies each, medians equal to the last read rates with
  a 0.29-logit spread — wider than the house persistence baseline's 0.125,
  which matters because both practice answers are the post-refresh page.
- A synthetic final-like case (120 reads over 2.6 days, tracked players
  638 → ~40k, chapter-1 rate 1.25% → 55% by the deadline, 4.5-day horizon):
  medians come out sane — chapter 1 ~84%, chapter 2 ~65%, final mission
  ~13%, platinum ~1.0% — with no trophy breaching its tier ceiling.

## Assumptions (stated, not hidden)

- The trophy list and rates are taken from the case file as given; where
  the real trophy list is unconfirmed, this entry forecasts exactly the 55
  trophies the case lists, nothing more.
- Tier ceilings are priors chosen by the author, not Rockstar/Exophase
  data; they only bind when a fitted trend would otherwise run past them.
- No outside data is used at run time: the program runs offline on the
  case file alone.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same input always produces byte-identical output, in
well under a second per case.
