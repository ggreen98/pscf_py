# Uncertainty & Dispersion Extensions — Reference Notes

Synthesis of a design discussion on (1) quantifying uncertainty in the multi-site
PSCF result and (2) whether/how to represent pollutant dispersion, which doesn't
exist in the base trajectory-endpoint method at all. This is exploratory/future
work, not yet implemented. For the core multi-site PSCF math (criterion,
summation, weighting), see `multi_site_pscf_notes.md` — this doc builds on top
of that.

---

## 1. Why PSCF has no dispersion built in

A back trajectory is a single deterministic line (one parcel's path), but real
pollutant transport is a spreading, turbulently-mixed plume. PSCF/CWT as
originally defined don't model this at all — a trajectory endpoint either
lands in a grid cell or it doesn't, with zero width. This is a known, accepted
simplification in the method, not a bug.

traPSA's own partial mitigation for this is **3×3 grid smoothing**
(`G_ij = mean of non-missing neighbors`) — listed in traPSA's parameters but not
yet implemented in this project's pipeline. It's a crude way of widening every
thin trajectory line's footprint after the fact; not a real dispersion model,
but the field's standard cheap answer to the same gap being discussed here.

---

## 2. Taxonomy of uncertainty sources

Discussed as four candidate sources; resolves cleanly into three, because two
of the original four behave the same way:

| Source | Behavior | Shrinks with more data? |
|---|---|---|
| Receptor measurement error | Negligible given equipment precision — only a few boundary-case hours near the threshold would misclassify | N/A (assumed negligible) |
| Endpoint density / sampling uncertainty | Statistical — how many independent trajectory-hours support a cell's `m_ij/n_ij` ratio | **Yes** — random/sampling uncertainty, shrinks as endpoint count grows |
| Trajectory positional error + dispersion (no distinct measurement of pollutant spread) | Systematic — every trajectory shares the same zero-width-line idealization | **No** — more thin lines through a cell doesn't make any one of them closer to representing the true plume width |

Endpoint density is *already* partially encoded via the existing trailing-effect
`weighting_function`, but that's a heuristic proxy, not a real statistical
uncertainty estimate.

### Statistical (binomial) framing of endpoint-density uncertainty

Since `PSCF_ij = m_ij / n_ij` is a proportion, treat `m_ij` as Binomial(`n_ij`,
`PSCF_ij`) and use the standard error of a proportion as a per-cell continuous
uncertainty surface:

```
SE_ij = sqrt( PSCF_ij * (1 - PSCF_ij) / n_ij )
```

Computable directly from data already produced by the pipeline (`n_ij`,
`m_ij`) — no new inputs needed. Caveat: this treats each endpoint as an
independent Bernoulli trial, which isn't strictly true (endpoints within one
trajectory are highly spatially/temporally correlated) — see §7 for a way to
get a more defensible "effective N."

### Prefer the Wilson score interval over the plain SE formula

The plain `SE = sqrt(p(1-p)/n)` formula breaks down exactly where this grid
lives most: at `p=0` (a common case — a cell with zero flagged endpoints) it
reports `SE=0`, implying perfect confidence, which is wrong. This is a known
failure mode of the normal approximation at small `n`/extreme `p` — the
textbook-recommended fix is the **Wilson score interval**, still a simple
closed-form calculation (no simulation needed), but specifically designed to
stay well-behaved at small `n` and `p` near 0 or 1. Use Wilson as the default,
not the plain SE formula.

**Implementation dependency**: both this and the endpoint-density heatmap idea
(`multi_site_pscf_notes.md` §9) need `PSCF.pscf()` to expose the raw
`n_ij`/`m_ij` (currently local variables, discarded after the final weighted
result is returned) — one fix serves both.

### Fuller list of assumptions this violates, beyond independence

Independence (endpoints within a trajectory aren't independent trials) is the
one already called out above. Three more, roughly in order of how much they
actually matter here:

1. **Identical distribution across pooled samples — the same issue as the
   multi-site dilution problem in `multi_site_pscf_notes.md` §9, seen from a
   different angle.** The binomial model assumes every trial shares one true
   `p`. A cell's `n_ij` can be a mix of endpoints from *different sites*, each
   with its own genuinely different true flagged-rate (`C_s` is per-site).
   Near a boundary between two sites' footprints, there may not be one single
   `p` being estimated at all — pooling those endpoints into one binomial
   calculation bakes in the same conflation that causes dilution in the point
   estimate itself. This is spatially localized (worst near inter-site
   overlap, negligible for cells dominated by one site), not uniform.
2. **Large-sample validity of the normal approximation.** Rule of thumb:
   `n·p > 5` and `n·(1-p) > 5`. Small-`n`, low-`p` cells — exactly the ones the
   trailing-effect weighting already flags as low-confidence — routinely fail
   this. This is the deeper reason to prefer Wilson over the plain formula,
   not just that it handles `p=0` more gracefully.
3. **Stationarity over time.** Binomial SE assumes a single fixed true `p`
   being estimated by repeated sampling of the same thing. Pooling multiple
   years (2015–2025) assumes conditions didn't meaningfully change over that
   span — new emissions sources, changes to HYSPLIT's driving meteorological
   model/resolution, land-use change, etc. would all violate this.
4. **Minor, not a real practical concern**: classical binomial SE technically
   assumes `n` is fixed in advance rather than itself an empirical outcome —
   standard practice in applied statistics is to condition on the observed
   `n` and proceed anyway, which is fine here too.

### Is it still worth calculating, given all of this?

Yes, but frame precisely what it does and doesn't capture rather than
presenting it as *the* uncertainty:

- Wilson directly fixes #2 above (that's what it's for).
- It does **not** fix #1 (mixed-site pooling) or the independence issue —
  those mean the result should be read as a **floor on true uncertainty**,
  systematically optimistic, especially near inter-site boundaries.
- It's cheap, needs no new data (once `n_ij`/`m_ij` are exposed), and gives a
  continuous companion to the existing discrete trailing-effect tiers — a
  real improvement over having no per-cell uncertainty signal at all.
- Suggested caption/label for any figure: "estimated standard error from
  endpoint sample size (Wilson score); does not account for trajectory-to-
  trajectory correlation (likely underestimates true uncertainty) or
  mixed-site pooling near inter-site boundaries." Same scoping discipline
  already used for the Monte Carlo jitter and CWT-vs-GPR decisions elsewhere
  in this doc — build the useful partial answer now, document its limits
  precisely, rather than waiting for a perfect method.

### Implemented — `PSCF.pscf()` and `ResultMap`, plus what real output taught us

`PSCF.pscf()` now computes `self.lower_ci`/`self.upper_ci` (via
`pscf_cell_sample_size_error_estimation()`, using `statsmodels.stats.proportion.
proportion_confint(method='wilson')`, verified against its actual source
earlier) and returns `(result, lower_ci, upper_ci)`. `ResultMap` takes
`lower_ci`/`upper_ci` as optional constructor args and, if both are given,
adds the interval width (`upper_ci - lower_ci`) as a **toggleable** overlay
layer ("Uncertainty (Wilson CI width)") via `folium.LayerControl` — hidden by
default, shown on demand, rather than always cluttering the main view. CWT has
no equivalent yet (still an open item, §8) — omitting `lower_ci`/`upper_ci`
when constructing `ResultMap` for a CWT result leaves the map exactly as
before, no toggle shown.

**Computed on the raw (unweighted) `m_ij`/`n_ij`, not the final weighted
result — confirmed by re-reading the actual code, not assumed.**
`pscf_cell_sample_size_error_estimation()` reads `self.numerator`/
`self.denominator` directly, set *before* `weighting_function()` runs — the
weighted `result` (`unscaled_result * w_ij`) is never touched by the CI at
all. This matches the earlier design decision (Wilson's math assumes
`m_ij/n_ij` is a genuine proportion, which is true of the raw ratio but not of
the weighted result) and means **the choice of weighting scheme
(`'standard'`/`'binary'`/anything else) has zero effect on the CI** — it's
fully decoupled from whatever weighting is applied to the displayed map.

**Real gap worth being precise about in any write-up**: the map plots the
*weighted* result, but the CI describes uncertainty around the *unweighted*
ratio for that same cell — the CI bounds don't bracket the exact number being
displayed, they bracket the raw `PSCF_ij` before the trailing-effect discount.
In practice these move in the same direction for low-confidence cells (low
`n_ij` → both a heavier weighting discount *and* a wider CI), so they're not
contradicting each other, but label it precisely: "confidence interval on the
raw endpoint ratio, prior to trailing-effect weighting," not "a band around
the plotted value."

**Originally observed in real output, now fixed rather than just described**:
the CI-width overlay used to light up *more* cells than the main PSCF result —
any cell with `n_ij > 0` (real trajectory traffic) but `m_ij = 0` (none of
those hours were ever flagged) gives `PSCF_ij = 0` exactly, and `pscf()`'s
divide (`np.divide(..., out=np.zeros_like(...), where=denominator!=0)`) and
`render_overlay`'s alpha rule (`np.where(data > 0, 0.85, 0.0)`) both collapsed
this onto the *same* value/transparency as a cell with `n_ij = 0` (no data at
all) — conflating "well-sampled, confirmed low" with "never sampled, unknown."
Wilson still reported a genuine nonzero-width interval for the former, because
"zero flagged hours out of a small-to-moderate sample" isn't the same
statistical claim as "we're certain the true probability is zero" (the same
`p̂=0` failure mode Wilson was chosen to avoid) — which is what made the
discrepancy visible in the first place.

**Fix**: `pscf()`'s divide now seeds its `out` array with `np.nan` instead of
`0` (`out=np.full_like(self.numerator, np.nan)`), so only genuinely-unsampled
cells (`n_ij == 0`) come out as `NaN` — a cell with `n_ij > 0, m_ij == 0` now
correctly computes to a real `0.0`. `render_overlay`'s alpha rule now keys off
`np.isnan(data)` instead of `data > 0`, so that real zero gets plotted (bottom
of the color scale) instead of vanishing like a no-data cell. The CI overlay's
own masking (`mask = n_ij > 0` in `pscf_cell_sample_size_error_estimation`)
already made this distinction correctly — the main PSCF map was the one that
needed to catch up, not the CI overlay.

---

## 3. Monte Carlo uncertainty estimation (the main plan)

**Core idea**: classify once, then treat the *spatial attribution* of already-
classified hours as the uncertain part, and resample it.

1. Compute `C_s` and flag `p_s,l` exactly once, from real (undispersed) receptor
   measurements — this never changes across Monte Carlo draws (see §6 for why
   this must stay fixed).
2. For each draw, perturb every trajectory's endpoints via **directional
   jitter that grows with `traj_age`** (matching HYSPLIT's own documented
   positional error, ~15–20% of travel distance/day per the traPSA guide).
3. Rebin the perturbed endpoints into `n_ij`/`m_ij`, recompute
   `PSCF_ij = m_ij/n_ij` for that draw (same flags, same formula — only the
   spatial landing positions changed).
4. Repeat for many draws (~100s). At each cell, take the spread (std dev or a
   percentile interval) of `PSCF_ij` across draws — that spread is the
   uncertainty map.

**Critical implementation detail**: jitter must be applied **coherently along
each trajectory, not independently per endpoint**. Independent per-point noise
produces physically implausible jagged zig-zag paths. A real alternate
trajectory under slightly different meteorology is a smooth alternate route —
so each draw should perturb a whole trajectory with one smoothly-growing
displacement (e.g. a single random direction/magnitude per trajectory, scaled
by `|traj_age|`), not fresh independent noise at every hour.

**Nice emergent property**: because jitter reshuffles which cell endpoints
land in, cells with only a few real trajectories passing through will show
much more variance across draws than cells with thousands — so this one
procedure captures *both* the positional/dispersion uncertainty and much of
the endpoint-density sampling uncertainty simultaneously, without needing the
binomial SE formula as a separate calculation (though it remains a useful
cheap cross-check).

**Region-specific caveat for the writeup**: HYSPLIT's ~15–20%/day error figure
is typically derived from flat-terrain case studies. The Front Range's complex
mountain/valley flow is a known regime where trajectory error tends to run
higher than that baseline — so even the positional-uncertainty component of
this estimate should be described as a floor, not a ceiling, in any paper.

### Reassessment: Monte Carlo may now be preferable to SQTBA, not just a fallback

Revisit after digging into SQTBA's actual assumptions (§5): SQTBA locks in one
fixed, rigid analytic kernel (isotropic, `σ = a·t'`, a single time-based rate
constant with no way to represent the real ambiguity in what should actually
drive dispersion — elapsed time vs. distance traveled vs. atmospheric
stability, see §5). Monte Carlo doesn't require committing to one closed-form
model at all — whatever perturbation model you think is most defensible (or
several, compared against each other) can be encoded directly into how a draw
is generated, and the spread across draws is an honest empirical answer rather
than one fixed surface built from one fixed set of assumptions already known
to be shaky.

**Compute concern, addressed with real numbers rather than a guess**: timing
instrumentation (`main.py`) showed trajectory *reading* dominates runtime
(3.93s of 5.06s total even for a small 2-site test) — the actual PSCF
computation step is cheap (0.13s), because it's vectorized `histogram2d` work
that doesn't care much about dataset size. The implication for Monte Carlo:
read the trajectory archive from disk **once**, then each draw only needs to
(1) generate one random perturbation per trajectory (vectorized, cheap —
trajectory count, not endpoint count), (2) apply it to that trajectory's
already-in-memory endpoints, (3) rebin + recompute (the cheap part already
timed). So the cost structure is one expensive one-time read plus many cheap
incremental passes, not "redo the expensive part hundreds of times."
**Practical next step**: time a small batch (10–20 draws) on real data first,
extrapolate to the full draw count, rather than assuming it's intractable.

---

## 4. Directional jitter refinement

Two ways to make the jitter direction non-isotropic (real transport isn't
equally likely to be misdirected in every compass direction):

1. **Trajectory's own heading** — the vector between consecutive endpoints
   (e.g. `traj_age=0` to `traj_age=-1`) already encodes a local direction of
   travel, at no cost of additional data.
2. **Station-observed wind-direction ratios** — derive empirical directional
   ratios from real ground-station wind data, ideally time-matched to each
   specific trajectory's hour (more physically precise than a climatological/
   aggregate wind rose, since it ties the perturbation cone to that day's
   actual synoptic regime rather than an average that's wrong on any given
   day).

**Decision**: build the simpler version first (trajectory's own heading), then
empirically check whether its implied directional distribution already tracks
real station wind-direction ratios. If it does, the extra complexity of
pulling in and time-matching a second data source isn't needed.

**Validation method**: wind direction is circular (0°/360° wraparound) — bin
both distributions into the same angular bins (e.g. 16-point compass rose) and
compare via circular statistics (e.g. mean resultant vector / circular
variance), not a standard linear-data test like KS, which would be misleading
across the wraparound point.

---

## 5. SQTBA — an empirically-grounded alternative to ad hoc smoothing

Verified directly from the installed `openair` 2.17.0 R source
(`getAnywhere("calc_SQTBA")` → `make_grid` → `pred_Q`), **and** directly from
the traPSA guide's own text (§3.6, extracted via `pdftotext`) — not from
memory. SQTBA (Simplified Quantitative Transport Bias Analysis) is a
**published, already-implemented dispersion-kernel method** — not something to
reinvent.

### The kernel, as traPSA's guide actually states it

```
Q(x,t|x',t') = 1/(2π·σx(t')·σy(t')) · exp[-½{((X-x'(t'))/σx(t'))² + ((Y-y'(t'))/σy(t'))²}]
```

- `x'(t'), y'(t')` — that specific trajectory's own position at age `t'` (its
  "centerline"). Fixed per trajectory point, *not* the receptor except at
  `t'=0`.
- `X, Y` — the grid cell being evaluated (its center, in practice). The free
  variable you sweep across the whole grid for one fixed trajectory point.
- `σx(t'), σy(t')` — spread in each direction, allowed to differ (this is a
  **diagonal-covariance** Gaussian: `σx ≠ σy` permitted, but no correlation/
  rotation term, so the ellipse can stretch but can't tilt off the grid's own
  axes). Stricter than this would be a full covariance matrix (tilt allowed,
  needs a `ρ` term); looser (`σx = σy` forced) is the **spherical/isotropic**
  case — which is what both openair's actual code *and*, per below, traPSA's
  own worked default *actually* use, despite the equation's more general form
  allowing anisotropy.
- Normalization `1/(2π·σx·σy)`: scales the puff's *height*, not its shape —
  bigger spread means the same total probability mass spread over more area,
  so the peak has to come down proportionally. Has no X/Y dependence, so it
  cancels out of the final `Σ(Q_c)/Σ(Q)` ratio regardless of its exact value.

### traPSA's actual specified σ — verified from the guide text directly

```
σx(t') = σy(t') = a · t'          where a = 5.4 km/h ("dispersion speed")
```

So despite the equation's general form allowing `σx ≠ σy`, **traPSA's own
reference recipe sets them equal** — a plain isotropic circle that grows
linearly with elapsed time, matching openair's actual implementation
(`sigma_i <- sigma_base * abs(hour.inc)`). Neither the published method nor
its real implementation exploits the anisotropy the equation's shape permits —
the rotated/elongated-ellipse idea (§4) would be a genuine enhancement beyond
both, not a more-granular version of something either already does.

### Critical limitation, not previously flagged: time-based, not distance-based

`a = 5.4 km/h` is a **flat rate per hour of elapsed time**, completely
independent of how far the parcel actually traveled in that time. A
trajectory that covered 150 km in 3 hours (fast wind) and one that covered
15 km in 3 hours (calm wind) get an *identical* `σ = 5.4 × 3 = 16.2 km` — the
formula only counts hours elapsed, ignoring information already present in
the trajectory data itself (every hourly endpoint's lat/lon directly gives
you actual distance traveled per step).

This matters because it contradicts a convention already used elsewhere in
this project: HYSPLIT's own commonly-cited positional uncertainty is
"~15-20% of *travel distance*" (§3, CLAUDE.md), i.e. distance-based, not
time-based. But **swapping to distance-based scaling isn't a clean fix
either** — there are two different physical mechanisms being conflated:
1. **Positional/modeling error** ("is the computed path itself correct") —
   plausibly *does* scale with distance/integration steps, since small
   per-step errors compound as the calculation proceeds. This is what
   HYSPLIT's own %-of-distance convention is actually about.
2. **Real turbulent dispersion** ("given the path is right, how much does the
   actual plume spread around it") — does *not* cleanly track distance or
   speed, and can run the *opposite* direction: strong, fast, well-organized
   synoptic wind is often associated with *stable* conditions and comparatively
   *less* turbulent mixing, while calm, light-wind conditions are often when
   local convective turbulence and complex-terrain effects (relevant for the
   Front Range specifically) actually dominate — producing *more* real
   dispersion despite less distance covered.

So `a·t'` isn't "wrong because it should have been distance-based" — it's a
simplification that avoids picking apart two mechanisms with genuinely
different (and for #2, not well-established) relationships to wind speed.
Consistent with the method's own name — it's the *Simplified* QTBA. No
citable precedent found for a published SQTBA/QTBA variant that resolves this
by incorporating real per-trajectory travel distance — this would be a
genuine open methodological question, not a known/available fix.

### Full aggregation pipeline (guide §3.6), and its relationship to the simplified `Σ(Q_c)/Σ(Q)` shown earlier

The guide breaks aggregation into two explicit steps beyond `Q` itself:

```
T̄_l(x|x') = [∫ Q(x,t|x',t') dt' from t-τ to t] / [∫ dt' from t-τ to t]   (avg over ONE trajectory's own points)
T̃(x|x')   = Σ_l T̄_l(x|x') · c_l                                          (concentration-weighted sum ACROSS trajectories)
SQTBA(x|x') = T̃(x|x') / Σ_l T̄_l(x|x')                                    (final ratio — same shape as CWT: Σ(weight·value)/Σ(weight))
```

The `∫dt'` is a genuine calculus integral in the guide's continuous-time
formulation, but trajectories are only known at discrete hourly points in
practice, so it's approximated as a discrete average over one trajectory's
own hourly `Q` values — the same "sum approximates integral" discretization
used throughout this project.

**This two-step version is mathematically equivalent to the flat
`SQTBA_ij = Σ(Q_c)/Σ(Q)` shown earlier — but only because all trajectories in
this project have the same number of points** (uniformly 6 or 12 hours, never
mixed, per `multi_site_pscf_notes.md` §2). The per-trajectory averaging in
step 1 divides every trajectory by the same constant `n` (its point count),
which cancels identically between numerator and denominator once summed
across trajectories in step 3. **If trajectory lengths ever varied, this
equivalence would break** — the guide's careful two-step version would
correctly give every trajectory equal say regardless of length, while the
flattened one-big-sum version would let longer trajectories quietly dominate
just by contributing more raw points. This is the exact same "record length
as implicit weight" problem already documented in `multi_site_pscf_notes.md`
§9 — worth building the careful two-step version rather than the flattened
shortcut, since it's more robust to a data-scope change later, even though
both give identical answers under the current uniform-length constraint.

**Suggested cross-check for a paper**: run `trajLevel(statistic = "sqtba")` on
the same data and compare against the custom Monte Carlo jitter approach —
if they produce visually/statistically similar footprints, that's good
corroborating evidence for both methods.

---

## 6. Where dispersion fits in the pipeline — classification vs. spatial attribution

**Key structural rule**: `C_s` / `p_s,l` classification and dispersion
weighting are two separate pipeline stages and must not be conflated.

- **Classification** (`C_s`, `p_s,l`) answers "was this an hour of genuinely
  high measured pollution at the receptor?" — a property of the *time-series*
  concentration data alone (still just `traj_age == 0` rows, exactly as
  already implemented). Dispersion has nothing to do with this step and
  should **not** change it — recomputing a percentile on some
  dispersion-weighted combined-endpoint population doesn't make conceptual
  sense, since that would mix a time-domain classification with a
  spatial-domain weighting.
- **Spatial attribution** (`n_ij`/`m_ij` binning) answers "given this hour's
  classification, how should its trajectory's mass be distributed across the
  grid?" — this is where dispersion kernels/jitter belong.

So: classify first (unweighted, as now), bin/weight second.

### Risk: dispersion kernel + trailing-effect weighting can double-penalize distant real sources

If the same kernel/decay applies identically regardless of a trajectory's
flag status, the *ratio* `m_ij/n_ij` at a distant cell isn't inherently
biased — a real, consistently-flagged distant source should still asymptote
toward the correct proportion given enough samples. But the *absolute*
accumulated weight at distant cells will be lower purely because the kernel
decays with distance/age — and that interacts with the existing endpoint-
density `weighting_function`, which *also* penalizes low-`n_ij` cells. Stack
both and a real distant source gets discounted twice for the same underlying
reason.

**Practical implication**: if dispersion weighting is added, reconsider
whether to still apply the original trailing-effect tiers unchanged — those
tiers (`2×avg`, `1×avg`, etc.) were calibrated assuming `n_ij` is a raw
endpoint *count*; once `n_ij` becomes a continuous kernel-weighted sum, "average
endpoints per cell" means something numerically different. Check empirically:
plot kernel-weighted `n_ij` alongside raw-count `n_ij` for a few known-distant
cells and see how hard the trailing-effect tiers actually bite once dispersion
has already discounted them.

---

## 7. Magnitude/concentration mapping (CWT) as a complement to PSCF

PSCF discards magnitude (binarizes to above/below threshold). **CWT
(Concentration Weighted Trajectory)** — listed as a distinct method in
traPSA's guide (§3.4) and already implemented in `openair`'s `trajLevel` — keeps
the actual `ppb` value and computes a weighted average per cell instead of a
count ratio:

```
CWT_ij = Σ(weight · ppb) / Σ(weight)
```

Verified from the `openair` source, plain CWT (`statistic == "cwt"`) has
**no distance term at all** — weighting is purely a function of `N` (endpoint
count in that cell), using fixed absolute-count tiers (`N > 80` → full,
`20 < N <= 80` → ×0.7, `10 < N <= 20` → ×0.42, `N <= 10` → ×0.05). This differs
in form from PSCF's own tiers, which are relative to the grid average
(`2×avg`, `avg`, `avg/2`) rather than fixed absolute counts. So plain CWT's
only "distance sensitivity" is indirect — cells far from the receptor
naturally accumulate fewer endpoints, which triggers the same low-`N`
downweighting any sparse cell would get, near or far.

Adding the SQTBA kernel from §5 as the `weight` term turns plain CWT into a
dispersion-aware magnitude estimate (`Σ Q·c / Σ Q`) — this is exactly what
SQTBA already is.

**Different uncertainty framework needed**: since CWT/SQTBA output is a
magnitude (ppb), not a proportion, its uncertainty is a classical weighted-mean
standard error (propagating measurement precision + spread from the dispersion
weights), not the binomial-proportion framing that fits PSCF (§2–3). Don't
reuse the PSCF uncertainty approach unchanged for this — it's a genuinely
different quantity being estimated.

### Multi-site CWT/SQTBA — structurally simpler than multi-site PSCF, but needs its own normalization

No threshold/classification step exists for CWT at all — no `C_s`, no per-site
criterion, no `p_s,l` flag. Each site's paired `(ppb, lat, lon, hour.inc)`
trajectories feed directly into the same kernel-weighted accumulation
(`Σ Q·c`, `Σ Q`), summed across sites into one shared grid exactly like
`n_ij`/`m_ij` already accumulate for PSCF, then divided once at the end.

**But**: pooling raw `ppb` across sites with different baseline pollution
levels risks letting whichever site has the highest absolute concentrations
dominate the combined magnitude map near it — not because that region is a
stronger real source, but just because that site's numbers run bigger. This is
the exact same problem that forced multi-site PSCF to need a per-site
criterion in the first place (traPSA's guide, §3.7: "The PSCF model can't be
directly applied into multi-site analysis as a fixed criterion value is
used"). Consider normalizing each site's `ppb` (e.g. relative to that site's
own mean/median, or a z-score) before pooling into the shared kernel-weighted
sum, so the result reflects relative enhancement above each site's own
baseline rather than raw absolute levels.

---

## 8. `ss_scaled` weighting — CI-based (inverse-variance) alternative to the discrete tiers

The `'standard'` trailing-effect weighting (§ multi_site_pscf_notes.md) uses
arbitrary, hand-picked tier bounds/weights. `ss_scaled` replaces that with a
continuous correction grounded in the Wilson CI machinery already built for
§2: **inverse-variance weighting** — `w_ij ∝ 1/CI_width_ij²` — the standard
statistical convention for discounting an estimate in proportion to how
imprecise it is (meta-analysis, weighted least squares). Getting there
correctly took several wrong turns worth documenting, since each one is a
real, generalizable pitfall, not just a one-off bug.

### Normalization: divide-by-max, and why not divide-by-max alone

Raw `1/width²` is unbounded (blows up as `width → 0`) and needs rescaling into
something comparable to the tier system's `[min_weight, 1.0]` range. `w_ij =
raw / np.nanmax(raw)` anchors the single most-precise cell in the grid at
weight 1.0 and scales everything else relative to it — standard
inverse-variance-weighting normalization, and consistent with the fact that
the existing `'relative'` tier comparison is *already* dataset-relative
(tiers are multiples of *this grid's* average endpoint count, not a fixed
constant). Two implementation pitfalls found along the way:

- **`np.max` vs `np.nanmax`**: `ci_width` is `NaN` wherever `n_ij == 0`
  (§2's masking). Plain `np.max` propagates `NaN` across the *entire* result
  (verified directly: `np.max([1,2,nan,4])` → `nan`), which would silently
  turn the whole normalized `w_ij` array into `NaN`. Needs `np.nanmax`.
- **Divide-by-zero guard**: add a tiny `eps` to `ci_width` before squaring/
  inverting, so a cell with `width == 0` exactly doesn't produce `inf`.

### Structural pitfall: `traj_age == 0` guarantees an artificial max

Even with `nanmax` correct, an early real-data test showed the whole map
crushed to near-uniform low weight except right around monitoring-site
clusters. Root cause, confirmed directly: `traj_age == 0` marks each
trajectory's *receptor* row — always at that site's own fixed lat/lon. Every
single hourly trajectory (thousands, over a multi-year record) contributes
exactly one endpoint stacked into that same cell, while endpoints at other
`traj_age` values spread spatially across many different cells as the parcel
actually traveled. This gives each monitoring site's own cell a guaranteed,
structural `n_ij` pileup — extreme, but unrelated to genuine "well-sampled
source evidence." That pileup cell becomes the `nanmax` anchor, and since
divide-by-max is an *uncapped* ratio (unlike the tier system, which caps out
at `1.0` for anything beyond its top bound), every other cell's weight gets
crushed by however extreme that one structural artifact happens to be.

A percentile-based anchor (95th/99th) was considered as a more robust
alternative, but rejected: with enough clustered sites (e.g. several sites
near Denver, visible in the same real-data test), the inflated pileup cells
can themselves populate the top percentile band rather than sit above it —
percentile trimming only helps against a *minority* outlier, not a systematic
one shared by every site. The robust fix instead: **exclude `traj_age == 0`
rows from the local `combine_df` inside `PSCF.pscf()`** (`combine_df =
combine_df.loc[combine_df['traj_age'] != 0]`), before building
`self.numerator`/`self.denominator` — removing the artifact at its source
rather than statistically working around it. This is scoped as a *local*
variable inside `pscf()`, not a mutation of `self.combine_data` or
`TrajectoryDataset.data` — both `get_site_locations()` and
`create_criterion_dict()` read `traj_age == 0` rows for their own,
legitimate purposes (site coordinates; per-site criterion `C_s`) and would
break if those rows were removed at the shared-dataset level instead.
**Not yet applied to `cwt()`** — same pileup would affect CWT's own `n_ij` if
`ss_scaled`-style weighting is ever extended there (open item below).

### Ordering dependency: CI must be computed before the weight that reads it

`weighting_function`'s `ss_scaled` branch needs `self.upper_ci`/
`self.lower_ci` (originally) or `self.upper_scaling_ci`/`self.lower_scaling_ci`
(current) to already exist. `PSCF.__init__` never initializes them at all, and
`pscf()` originally called `weighting_function()` *before*
`pscf_cell_sample_size_error_estimation()` — meaning the very first call would
raise `AttributeError` (attribute doesn't exist yet, not even as `None`), and
any later call would read *stale* values left over from a previous call, not
this call's own numerator/denominator. Fixed by reordering `pscf()`: compute
`self.numerator`/`self.denominator`, then immediately call
`pscf_cell_sample_size_error_estimation()`, *then* call `weighting_function()`.

### The key statistical fix: decouple the weight from the cell's own p̂

Using the *real* Wilson CI (real `m_ij`) as the weighting input has a serious
problem: Wilson variance is `p̂(1-p̂)/n` — maximized at `p̂=0.5`, shrinking
toward 0 as `p̂` approaches either extreme. Verified directly with
`statsmodels`:

| Case | width | raw weight (`1/width²`) |
|---|---|---|
| n=50, p̂=0.0 (0/50) | 0.0713 | 196.4 |
| n=50, p̂=0.5 (25/50) | 0.2671 | 14.0 |
| n=10, p̂=0.0 (0/10) | 0.2775 | 13.0 |

Two cells with the *identical sample size* (n=50) get a 14× difference in
weight purely from where `p̂` landed. Worse: a cell with only 10 trajectories
matches the trust-weight of a 50-trajectory cell, just by having an extreme
measured proportion. This is backwards for what the trailing-effect
correction is *for* — small samples are exactly the ones prone to landing on
a spuriously extreme `p̂` by chance, so rewarding extremeness with higher
trust amplifies the very artifact the correction exists to catch. The
resolution: the trailing-effect weight and the CI overlay are answering
different questions ("do we have enough data volume to trust this cell at
all" vs. "how precisely is this specific estimate pinned down") and only one
of them should depend on `p̂`.

**Implementation**: a second, separate CI is computed in
`pscf_cell_sample_size_error_estimation()`, using a **fixed `p̂ = 0.5`**
(`count = n_ij/2`, i.e. `smp.proportion_confint(count=n_ij/2, nobs=n_ij, ...)`)
instead of the real `m_ij` — verified this produces a width that depends on
`n` alone (10 → 0.53, 50 → 0.27, 200 → 0.14), with zero sensitivity to the
actual measured proportion. Stored as `self.lower_scaling_ci`/
`self.upper_scaling_ci`, used only by `ss_scaled`. The original
`self.lower_ci`/`self.upper_ci` (real `m_ij`) are untouched and still drive
the CI-width map overlay, since genuine `p̂`-dependent precision *is* the
correct thing to show there — extreme measured proportions genuinely do carry
more information at the same `n`, and that's legitimate, useful information
for that specific display, just not for this weight.

**Sanity check, confirmed after the fix**: holding `n_ij` fixed and increasing
`m_ij` correctly increases the final weighted result, proportionally and
without distortion — since `ss_scaled`'s weight no longer moves at all when
`m_ij` changes (it's a pure function of `n_ij`), the basic PSCF signal (more
qualifying trajectories through a cell → higher source likelihood) comes
through cleanly instead of being compounded by a simultaneously-shifting
weight.

### Open: multi-site dilution of the `'standard'` tier reference point

A related, *not yet fixed* issue found while investigating this: adding more
sites to a multi-site run measurably lowers `PSCF` even for cells nowhere
near the new site, under the `'standard'` (tier-based) weighting. Root cause:
`avg_endpoints = total_endpoints / occupied_cells` in `pscf()` is computed
**network-wide, across every pooled site** — adding a distant site inflates
`total_endpoints` substantially while only modestly growing the occupied-cell
count, raising `avg_endpoints` for the whole grid. A cell whose own `n_ij`
never changed can get pushed into a lower tier purely because the reference
point moved, not because its own evidence weakened. traPSA has no native
answer here — it's a single-receptor tool with no concept of a network-wide
average at all (`avg` in its own formula only ever means "this one site's own
average"); the multi-site accumulation this project already does is an
extension beyond traPSA's own design, and this is the piece that extension
didn't resolve. Likely fix, mirroring the already-per-site criterion `C_s`
convention (`multi_site_pscf_notes.md` §2): compute `avg_endpoints` **per
site** and weight each site's own contribution to a cell against its own
average, before pooling — not a one-line change, since it requires per-site
histograms prior to accumulation rather than weighting the already-pooled
total. Not yet implemented.

---

## 9. Open questions / next steps

- [ ] Build trajectory-heading-based directional jitter first; validate
      against real station wind-direction ratios before adding that as a
      second data source (§4).
- [ ] Decide on jitter magnitude scaling (HYSPLIT's generic %/day figure vs.
      something region-specific for Front Range complex terrain) (§3).
- [ ] Run `openair::trajLevel(statistic = "sqtba")` on the same dataset as a
      cross-check against the custom Monte Carlo approach (§5).
- [ ] Check empirically whether dispersion-kernel-weighted `n_ij` and the
      existing trailing-effect tiers double-penalize distant cells; decide
      whether to recalibrate tiers or drop one mechanism (§6).
- [ ] If building CWT/SQTBA for magnitude estimates, decide on a per-site
      normalization scheme before pooling across sites (§7).
- [ ] Work out the weighted-mean SE formulation for CWT/SQTBA uncertainty,
      separate from PSCF's binomial-proportion approach (§7).
- [x] Expose `n_ij`/`m_ij` from `PSCF.pscf()` (`self.numerator`/
      `self.denominator`) — done; also used by the Wilson calculation below.
- [x] Build the Wilson score interval for sampling-uncertainty (§2) —
      implemented (`pscf_cell_sample_size_error_estimation()`), returned from
      `pscf()`, and wired into `ResultMap` as a toggleable overlay layer. Still
      label any figure precisely per §2's suggested caption (sampling
      uncertainty only, floor not ceiling).
- [x] The two stacked `branca` legends (main result + CI width) turned out not
      to stack cleanly at all — confirmed directly from generated output that
      both `LinearColormap` instances' internal scripts use the identical,
      non-unique `d3.select(".legend.leaflet-control")` selector (and the
      same `id="legend"` on their own `<svg>`), so both legends' content
      collided into the *same* first-matching container, which is why
      toggling one legend's visibility toggled both. Fixed by dropping
      `branca.colormap.LinearColormap` for these two legends entirely and
      replacing them with hand-built HTML/CSS gradient-bar divs, each with
      its own unique `id` (`main-legend-box`/`ci-legend-box`), referenced
      directly via `getElementById` in the toggle JS — no ambiguity possible.
- [x] Fixed a related conflation in the main PSCF map: cells with `n_ij == 0`
      (no data) and cells with `n_ij > 0, m_ij == 0` (well-sampled,
      legitimately zero) both rendered identically as fully transparent.
      `pscf()`'s divide now seeds `NaN` for the former only, and
      `render_overlay`'s alpha rule keys off `np.isnan(data)` instead of
      `data > 0`, so a real zero is now visibly plotted instead of vanishing
      like a no-data cell (§2).
- [ ] Per-site (not network-wide) reference point for the `'standard'`
      trailing-effect weighting's `avg_endpoints` — adding an unrelated,
      distant site currently lowers PSCF for cells that site's trajectories
      never touch, since `avg_endpoints` is pooled across all sites rather
      than computed per site (§8).
- [ ] Apply the same `traj_age == 0` exclusion used in `pscf()` to `cwt()`,
      for consistency, if/when CWT adopts sample-size-based weighting (§8).
- [ ] Same binomial-SE limitations apply if a CWT-side sampling-uncertainty
      measure is ever built (§7) — worth deciding whether it's even meaningful
      there given CWT has no `p`/proportion at all, only a weighted mean.
- [ ] Time a small batch of Monte Carlo draws (10-20) on real data to get an
      actual per-draw cost estimate, now that MC looks preferable to SQTBA
      given SQTBA's rigid, time-only dispersion assumption (§3, §5).
- [ ] Open methodological question, no known fix: whether/how to make SQTBA's
      `σ` responsive to real per-trajectory travel distance or atmospheric
      stability instead of a flat time-based rate — and whether that's even
      well-motivated, given distance doesn't cleanly track true dispersion
      either (§5).
