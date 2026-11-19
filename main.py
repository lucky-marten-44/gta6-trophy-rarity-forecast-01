"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Forecast, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of
the players Exophase tracks who will have earned it at the case's target
(2026-11-26 00:00 UTC in the final), as five quantiles per trophy, from the
house's reads of Exophase's trophy page up to the case's checkpoint.

What the data are
-----------------
Each read gives, per trophy, the earn rate in percent (2 decimals) and the
number of players tracked. The trophy list itself (55 trophies, names, tiers)
comes from the case file -- nothing about the trophy list is assumed or
invented here. Two facts about the source drive the whole model:

1. A rate is a ratio: rate = earned / players_tracked. Both the numerator
   (players earning the trophy as they play) and the denominator (new
   players syncing their profiles) move, and they move at different speeds.
   A rate can *fall* while more people earn the trophy, when the tracked
   population grows faster than the earners. Any model that extrapolates
   the rate alone misses that dilution.
2. Exophase refreshes its counts on its own schedule, not continuously:
   on launch night the page sat unchanged for hours (638 players tracked,
   every rate identical), then jumped at 04:30 UTC to 3,868 players with
   almost every rate lower. So a read shows the rates as of Exophase's
   last refresh, and a forecast must carry honest spread for the next
   refresh, not just for smooth progress.

Model (per trophy, all in natural logs)
---------------------------------------
Time is x = ln(1 + hours since 2026-11-19T00:00Z), the launch's midnight.
Log time is used because launch curves are concave: the first hours move
fast, later hours move slower. Fits use only the last 24 hours of reads
(the current regime); if fewer than 3 reads fall in that window, all reads.

Two medians are fitted and averaged 50/50 in logit space:

* Model A -- rate trend: least squares of logit(rate) on x. The direct
  trend of the rate itself.
* Model B -- counts: least squares of ln(earned) on x, where
  earned = rate/100 * players_tracked, plus a fit of ln(players_tracked)
  on x over the whole history; the target rate is earned_target /
  players_target. This is the dilution-aware view: it separates "more
  people earned it" from "more people joined the tracked set".

Both slopes are shrunk toward their tier's mean slope (platinum / gold /
silver / bronze), weight n/(n+6) on the trophy's own slope, because a
single trophy's 2-decimal rates are noisy and trophies of a tier progress
at similar speeds. Slopes are capped at |b| <= 4 in logit/log units.

The median is capped by a saturation ceiling per tier -- a prior, not a
claim about the game: platinum 5%, gold 40%, silver 75%, bronze 98%.
A trophy's rate cannot run away past what its tier plausibly reaches in
week one, however steep its last day looked. Quantiles are clamped to
[0.01, 99.99] percent, the range the scorer's logit accepts.

Spread (logit units): sigma = sqrt(resid_var + (slope_se * dx)^2
+ refresh_var), floored at 0.30 and capped at 1.50, where resid_var is
the residual variance of Model A's fit (floored at 0.02, so a perfectly
flat history -- as on launch night, before the first refresh -- does not
claim zero uncertainty), slope_se * dx is the standard error of the
extrapolation to the target, and refresh_var = 0.25^2 covers the next
Exophase refresh moving rates in a jump. Quantiles are the median shifted
by the standard normal's quantiles at the contest's five levels.

Fallbacks keep every case valid: a trophy with no usable reads forecasts
its last known rate (or 0.01% with the widest spread if never seen);
a missing players_tracked disables Model B for that case and Model A is
used alone; a rate of 0 or 100 is clamped before the logit, exactly as
the scorer clamps it.

Standard library only, no network, no clock, no randomness: the same
input always produces the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

ORIGIN = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
WINDOW_HOURS = 24.0
SHRINK_N = 6.0
MAX_SLOPE = 4.0
RESID_VAR_FLOOR = 0.02
REFRESH_SD = 0.25
SIGMA_FLOOR = 0.30
SIGMA_CAP = 1.50

# Saturation ceilings per tier (percent): priors on how far a trophy of
# that tier can plausibly get in week one, not claims about the game.
TIER_CEILING = {"platinum": 5.0, "gold": 40.0, "silver": 75.0, "bronze": 98.0}


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does, so a
    whole number comes back as an int (100, not 100.0)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_origin(moment):
    return (moment - ORIGIN).total_seconds() / 3600.0


def fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_var, slope_se)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return mean_y, 0.0, 0.0, 0.0
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / sxx
    intercept = mean_y - slope * mean_x
    resid_sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid_var = resid_sq / n
    slope_se = math.sqrt(resid_var / sxx) if resid_var > 0 else 0.0
    return intercept, slope, resid_var, slope_se


def shrink(slope, tier_mean, n):
    w = n / (n + SHRINK_N)
    return w * slope + (1.0 - w) * tier_mean


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    target = parse_time(case["target"])
    x_target = math.log1p(max(0.0, hours_since_origin(target)))

    # Per-read x and players-tracked, oldest first.
    reads = []
    for row in history:
        moment = parse_time(row["slot"])
        reads.append({
            "x": math.log1p(max(0.0, hours_since_origin(moment))),
            "n": row.get("players_tracked"),
            "rates": row["rates"],
        })
    x_last = reads[-1]["x"]

    # Window: last 24h of reads; all reads if fewer than 3 fall inside.
    slots = [parse_time(row["slot"]) for row in history]
    window = [r for r, s in zip(reads, slots)
              if (slots[-1] - s).total_seconds() <= WINDOW_HOURS * 3600]
    if len(window) < 3:
        window = reads

    # Players-tracked trend over the whole history (Model B's denominator).
    n_fit = None
    n_pts = [(r["x"], math.log(r["n"])) for r in reads
             if r["n"] is not None and r["n"] > 0]
    if len(n_pts) >= 2:
        xs, ys = zip(*n_pts)
        n_fit = fit_line(list(xs), list(ys))

    # First pass: raw per-trophy fits, to get tier-mean slopes.
    per_trophy = []
    for i, trophy in enumerate(trophies):
        a_pts = [(r["x"], logit(r["rates"][i])) for r in window
                 if r["rates"][i] is not None]
        b_pts = []
        for r in window:
            rate = r["rates"][i]
            if rate is None or rate <= 0 or r["n"] is None or r["n"] <= 0:
                continue
            b_pts.append((r["x"], math.log(rate / 100.0 * r["n"])))
        last_rate = None
        for r in reversed(reads):
            if r["rates"][i] is not None:
                last_rate = r["rates"][i]
                break
        per_trophy.append({"trophy": trophy, "a_pts": a_pts, "b_pts": b_pts,
                           "last_rate": last_rate})

    tier_slopes_a = {}
    tier_slopes_b = {}
    for tier in TIER_CEILING:
        slopes_a = []
        slopes_b = []
        for t in per_trophy:
            if t["trophy"]["tier"] != tier:
                continue
            if len(t["a_pts"]) >= 2:
                _, b, _, _ = fit_line([p[0] for p in t["a_pts"]],
                                      [p[1] for p in t["a_pts"]])
                slopes_a.append(max(-MAX_SLOPE, min(MAX_SLOPE, b)))
            if len(t["b_pts"]) >= 2:
                _, b, _, _ = fit_line([p[0] for p in t["b_pts"]],
                                      [p[1] for p in t["b_pts"]])
                slopes_b.append(max(-MAX_SLOPE, min(MAX_SLOPE, b)))
        tier_slopes_a[tier] = (sum(slopes_a) / len(slopes_a)) if slopes_a else 0.0
        tier_slopes_b[tier] = (sum(slopes_b) / len(slopes_b)) if slopes_b else 0.0

    rows = []
    for t in per_trophy:
        trophy = t["trophy"]
        tier = trophy["tier"]
        ceiling = TIER_CEILING.get(tier, 98.0)
        last_rate = t["last_rate"]
        if last_rate is None:
            # Never observed: widest spread around the floor.
            med_logit = logit(0.01)
            sigma = SIGMA_CAP
            rows.append({"id": trophy["id"],
                         "q": quantiles(med_logit, sigma)})
            continue

        medians = []
        resid_var = RESID_VAR_FLOOR
        slope_se = 0.0
        dx = max(0.0, x_target - x_last)

        # Model A: logit(rate) trend.
        if len(t["a_pts"]) >= 2:
            xs = [p[0] for p in t["a_pts"]]
            ys = [p[1] for p in t["a_pts"]]
            a, b, rv, se = fit_line(xs, ys)
            b = max(-MAX_SLOPE, min(MAX_SLOPE, b))
            b = shrink(b, tier_slopes_a[tier], len(xs))
            medians.append(a + b * x_target)
            resid_var = max(RESID_VAR_FLOOR, rv)
            slope_se = se
        elif len(t["a_pts"]) == 1:
            medians.append(t["a_pts"][0][1])
        else:
            medians.append(logit(last_rate))

        # Model B: earned-count growth over tracked-player growth.
        if len(t["b_pts"]) >= 2 and n_fit is not None:
            xs = [p[0] for p in t["b_pts"]]
            ys = [p[1] for p in t["b_pts"]]
            a, b, _, _ = fit_line(xs, ys)
            b = max(-MAX_SLOPE, min(MAX_SLOPE, b))
            b = shrink(b, tier_slopes_b[tier], len(xs))
            ln_e_target = a + b * x_target
            ln_n_target = n_fit[0] + n_fit[1] * x_target
            rate_b = math.exp(ln_e_target - ln_n_target) * 100.0
            medians.append(logit(rate_b))

        med_logit = sum(medians) / len(medians)
        # Saturation ceiling, in logit space.
        med_logit = min(med_logit, logit(ceiling))
        med_logit = max(logit(0.01), min(logit(99.99), med_logit))

        var = resid_var + (slope_se * dx) ** 2 + REFRESH_SD ** 2
        sigma = math.sqrt(var)
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(med_logit, sigma)})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def quantiles(med_logit, sigma):
    out = []
    prev = 0.0
    for z in Z:
        value = round4(clamp_rate(inv_logit(med_logit + z * sigma)))
        if value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
