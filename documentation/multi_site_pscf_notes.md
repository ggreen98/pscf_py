# Multi-Site PSCF — Reference Notes

Reference doc for refactoring `single_site_pscf` (in `pscf.ipynb`) into a multi-site
version, following traPSA's convention. See `CLAUDE.md` for the full domain context
this is drawn from.

---

## 1. Variables

| Symbol | Meaning |
|---|---|
| `s` | site index |
| `l` | trajectory (hour) index, within a site |
| `i, j` | grid cell indices (lat, lon) |
| `c_{s,l}` | measured concentration at site `s`, hour `l` |
| `C_s` | site `s`'s criterion (threshold) concentration |
| `p_{s,l}` | 1 if `c_{s,l} >= C_s`, else 0 — "high pollution episode" flag |
| `tau_{ij,s,l}` | endpoint count trajectory `l` (site `s`) contributes to cell `(i,j)` |
| `n_{ij}` | total endpoints in cell `(i,j)`, summed across **all** sites/trajectories |
| `m_{ij}` | endpoints in cell `(i,j)` from flagged (`p=1`) trajectories, summed across all sites |

---

## 2. Per-site criterion `C_s`

Each site gets its **own** threshold — not a network-wide one.

```
C_s = 80th percentile of site s's own concentration distribution
p_s,l = 1  if c_s,l >= C_s
      = 0  otherwise
```

- traPSA default quantile: **0.8** (range cited in the guide: 50%–90%).
- `C_s` is computed **after** merging that site's concentration data with its
  trajectory endpoints (i.e., on `combine_df`, not the raw concentration record),
  and after subsetting to the analysis scope (summer, daytime).

**Endpoint-level tables and `C_s` — use the hour-0 point only (recommended):**
a trajectory's `ppb` value is broadcast to every one of its endpoint rows
(hour 0 = the receptor/start point, hour -1, -2, ... back to -6 or -12).
Filtering to just the `hour == 0` row per trajectory before computing the
quantile gives an exact one-row-per-trajectory table by construction —
`groupby('site')['ppb'].quantile(0.80)` on that filtered subset is guaranteed
correct regardless of trajectory length, with no reliance on lengths staying
uniform. This is safer than computing on the full endpoint-exploded table, so
prefer it: filter to hour 0 → compute `C_s` per site → flag `p` on that
filtered subset → merge/broadcast `p` back onto the full endpoint table
(join on site + trajectory id) before binning.

(Aside: because trajectory length is in fact uniform within a run — all 6-hr
or all 12-hr, never mixed — computing the quantile on the *full* exploded
table would also happen to give the same answer, per the equal-replication
argument. But filtering to hour 0 is simpler to reason about and doesn't
depend on that assumption holding, so it's the preferred method.)

Practically, this means the criterion calculation and the binning step read
from two different slices of the same combined table: `C_s`/flagging uses
only the `hour == 0` rows, while `n_ij`/`m_ij` binning (§3) uses **all** rows
(every hour offset) — the flag just needs to be merged onto all of a
trajectory's rows first.

**Pipeline order (per site, before any cross-site accumulation):**

```
raw concentration data
  -> filter to summer + daytime hours
  -> inner join with trajectory endpoints  (this is combine_df)
  -> C_s = combine_df['ppb'].quantile(0.80)
  -> flag p_s,l on combine_df
```

Why post-merge, not pre-merge: see `CLAUDE.md` §"Decision: compute C_s post-merge."
Short version — only hours with a matching trajectory file should inform the
threshold, matches openair convention, and the summer/daytime subsetting is
intentional scoping that must happen upstream of the quantile calculation either way.

---

## 3. Long-format design — one combined endpoint table

Rather than looping over sites and accumulating separate arrays, build a single
long dataframe with one row per trajectory endpoint, carrying a `site` column:

```
columns: site, trajectory_id (or timestamp), lat, lon, ppb
```

Where `ppb` is that endpoint's trajectory's paired concentration (broadcast
across all endpoints of the same trajectory, as it already is in `combine_df`),
and `hour` is the trajectory step offset (0 = receptor/start point, negative
going back in time).

Steps:

```
1. C_s per site:  df[df.hour == 0].groupby('site')['ppb'].quantile(0.80)
2. flag:          p = 1 if ppb >= C_s else 0   (computed on hour==0 rows,
                    then merged back onto every row of that trajectory)
3. n_ij:          histogram2d(ALL rows' lat/lon, grid edges)  <- all hours
4. m_ij:          histogram2d(rows where p == 1, grid edges)  <- all hours
```

Because `histogram2d` in steps 3–4 is run on the *entire* combined table without
filtering by site, it implicitly performs the cross-site sum — every site's
endpoints land in the same shared grid automatically. This is equivalent to,
and simpler than, explicitly looping over sites and accumulating `n_ij_s`/`m_ij_s`
into `grid.n_ij`/`grid.m_ij` one site at a time — same result, fewer moving parts.

No dividing, no weighting yet at this stage — `n_ij`/`m_ij` are still raw counts.

---

## 4. Cross-site accumulation

With the long-format design (§3), accumulation is **implicit** — a single pass
over the combined table already produces the fully-summed `n_ij`/`m_ij`:

```
n_ij = sum_s( n_ij_s )     <- done automatically by histogram2d over all rows
m_ij = sum_s( m_ij_s )     <- done automatically by histogram2d over p==1 rows
```

(If instead processing sites one at a time — e.g. because each site's trajectory
files are loaded separately — the equivalent explicit form is a loop doing
`grid.n_ij += n_ij_s`, `grid.m_ij += m_ij_s` per site. `PSCFGrid.n_ij` / `.m_ij`
exist for exactly that case. Same math either way — pick whichever fits how the
data actually gets loaded.)

---

## 5. Multi-site PSCF formula

Computed **once**, after all sites have been accumulated:

```
PSCF_ij = m_ij / n_ij

         = sum_s( sum_l( p_s,l * tau_ij,s,l ) )
           ---------------------------------------
           sum_s( sum_l( tau_ij,s,l ) )
```

Cells with `n_ij == 0` → `PSCF_ij = 0` (avoid divide-by-zero), same convention
as the current single-site code.

---

## 6. Weighting (trailing-effect correction)

Applied **once**, on the combined grid — not per site.

```
avg = sum(n_ij) / num_nonzero_cells        # across the WHOLE combined grid

W_ij = PSCF_ij * weight(n_ij)
```

`weight()` is the existing `weighting_function` in `pscf.ipynb` (cell `6d764070`) —
no changes needed there. Just make sure `avg_endpoints` is computed from the
post-accumulation `n_ij`, not from any single site's `n_ij_s`.

### openair's actual weighting scheme (verified from source)

The Python `weighting_function` docstring claims its `'standard'` tiers match
openair's `trajLevel()` defaults. Confirmed directly against the installed
`openair` 2.17.0 source (`getAnywhere("trajLevel")`, the `statistic == "pscf"`
branch) — this is openair's real, exact logic, not a paraphrase:

```r
if (statistic == "pscf") {
  Q90 <- quantile(mydata[[pollutant]], probs = percentile/100, na.rm = TRUE)
  mydata <- mydata %>% group_by(across(vars)) %>%
    summarise(N = length(date),
              count = length(which(.data[[pollutant]] > Q90)) / N)
  mydata[[pollutant]] <- mydata$count

  n <- mean(mydata$N)                                    # avg endpoints per OCCUPIED cell
  id <- which(mydata$N >  n     & mydata$N <= 2 * n); mydata[id, pollutant] <- mydata[id, pollutant] * 0.75
  id <- which(mydata$N > (n/2)  & mydata$N <=     n); mydata[id, pollutant] <- mydata[id, pollutant] * 0.5
  id <- which(mydata$N <= (n/2));                     mydata[id, pollutant] <- mydata[id, pollutant] * 0.15
  # N > 2*n is left unmultiplied -> implicit weight of 1.0
}
```

Translated to this project's notation (`n_ij` = openair's `N`, `avg` = openair's `n`):

```
weight(n_ij) = 1.00   if n_ij > 2 * avg
             = 0.75   if avg   < n_ij <= 2 * avg
             = 0.50   if avg/2 < n_ij <= avg
             = 0.15   if n_ij <= avg/2

avg = mean(n_ij) over cells that have at least one endpoint (occupied cells only —
      matches `(denominator > 0).sum()` in the Python code, not total cell count)
```

Two details worth noting:

- **openair's own `Q90` (its version of `C_s`) uses a strict `>`**, not `>=` —
  already called out as a known, intentional divergence from the Python/traPSA
  side in `run_openair_pscf.R`'s header comment. Doesn't affect the *weighting*
  scheme, but affects the `p`/flag step upstream of it.
- **This confirms the current `weighting_function` defaults in `pscf.ipynb`
  (`tier_bounds=(2, 1, 0.5)`, `tier_weights=(1.0, 0.75, 0.5)`, `min_weight=0.15`)
  already match openair exactly** — no change needed there for the multi-site
  refactor, just apply it once on the combined grid per §6 above.

---

## 7. Refactor mapping — old vs. new

| Current (`single_site_pscf`) | Multi-site replacement |
|---|---|
| Computes `numerator`/`denominator` via `histogram2d` | Same, but per site → `n_ij_s`, `m_ij_s` (unchanged logic) |
| Divides immediately: `numerator / denominator` | **Removed from per-site step** — moved to a final step after accumulation |
| Computes `avg_endpoints` from that site's `denominator` | **Removed from per-site step** — computed once from combined `grid.n_ij` |
| Calls `weighting_function` per site | Called **once**, on the combined `PSCF_ij` grid |
| `threshold` passed in as a fixed argument | Computed **inside** the per-site step, from that site's own `combine_df` |
| Returns a finished weighted PSCF grid | Per-site step returns raw `(n_ij_s, m_ij_s)`; a separate final step returns the finished weighted grid |

**New shape, roughly (long-format version):**

1. Build one combined endpoint-level dataframe across all sites (`site`,
   `trajectory_id`, `lat`, `lon`, `ppb`).
2. `C_s` per site via `groupby('site')['ppb'].transform(...)` → flag `p` column.
3. Two `histogram2d` calls over the whole combined table (all rows → `n_ij`;
   `p == 1` rows → `m_ij`) — no per-site loop needed, accumulation is implicit.
4. `finalize_pscf(n_ij, m_ij, weighting=...)` → one division, one
   `avg_endpoints`, one call to `weighting_function`, returns the final
   multi-site `W_ij` grid.

---

## 8. Things to double-check while implementing

- The quantile call (`C_s`) must be grouped **by site** (`groupby('site')`),
  never computed on the whole combined table ungrouped — that's the difference
  between a per-site criterion and an accidental network-wide one.
- Compute `C_s` and the `p` flag on the `hour == 0` subset only (§2), then merge
  `p` back onto every hour offset of that trajectory before binning. Don't
  compute the quantile on the full endpoint-exploded table — filtering to hour
  0 first is guaranteed correct and doesn't depend on trajectory lengths being
  uniform, even though they currently are.
- Make sure the trajectory table actually has a reliable `hour` (or step-offset)
  column marking which row is 0 — don't rely on row order/position to find it.
- All sites' endpoints must be binned against the exact same `grid.lat_edges` /
  `grid.lon_edges` (one shared `PSCFGrid` instance) so cells line up.
- Time base: confirm every site's `combine_df` is paired same-label (trajectory
  hour H ↔ measurement label H) and in UTC before this stage — see `CLAUDE.md`
  §"Time pairing" and §"Per-species hourly resampling convention." Multi-site
  accumulation will silently produce garbage if one site's pairing convention
  drifts from the others.

---

## 9. Limitations — multi-site pooling dilutes real hotspots, and ideas for later

Observed empirically: pooling even two nearby sites (BRZ + LUR) visibly
suppresses hotspot cells relative to what a single-site PSCF map shows for the
same region. This is a real, inherent property of the pooled multi-site
formula (§5) — not a bug — worth documenting explicitly as a limitation in any
writeup, and worth returning to.

### Mechanism

A cell that's a genuine hotspot for site A has an above-baseline flagged
fraction for A's own trajectories there (e.g. 50%, vs. A's overall ~20% base
rate from the 80th-percentile criterion). Once pooled with site B, any of B's
endpoints that land in that same cell — even rarely — carry roughly *B's own*
base rate (~20%), not a rate tied to A's hotspot, since there's no causal
reason B's marginal endpoints there should reflect A's conditions. The pooled
ratio is a volume-weighted average of "A's enhanced rate" and "B's baseline
rate," so it necessarily gets pulled toward the baseline. The more endpoint
volume a distant/unrelated site contributes to a cell relative to the site
it's actually a hotspot for, the more dilution shows up.

**Distance is a proxy for this, not the actual driver** — the real quantity is
each site's *relative endpoint volume* in a given cell. It's also compounded
by a second, physically-grounded effect: endpoints far from their own receptor
are individually less reliable indicators of a true source-receptor link in
the first place, since trajectory positional/dispersion uncertainty grows with
`|traj_age|` (same basis as the age-scaled jitter/kernel work in
`uncertainty_dispersion_notes.md` §3, §5). So at inter-site boundary cells,
dilution and unreliability compound in the same direction.

### The existing trailing-effect weighting doesn't protect against this

`weighting_function` (§6) only ever sees *total pooled* `n_ij` — it can't
distinguish "high count, all from one reliable nearby site" from "high count,
partly diluted by noisy contributions from a distant site." A cell near site B
can easily have healthy-looking total `n_ij` (driven by B's large local
volume) and earn full weight, while still being quietly distorted by a
trickle of unreliable endpoints from site A mixed in. The weighting scheme has
no visibility into *composition*, only total volume.

### Ideas to investigate later

- **Quantify it directly first**: run `PSCF` on each site's `combine_data`
  alone and compare against the pooled multi-site result at the specific
  cells that look suppressed, to measure how much dilution is actually
  happening before deciding whether/how to address it.
- **Tuning `weight_function` tier thresholds is unlikely to fix this** —
  tried as a first, cheap thing to check, but since the weighting function is
  composition-blind regardless of where the thresholds sit, changing them
  shifts *which* cells get penalized for low volume without adding any
  sensitivity to *where* the endpoints came from. Worth confirming empirically
  rather than assuming, but don't expect it to solve the root cause.
- **Dispersion-kernel weighting (SQTBA-style) is the more principled fix** —
  see `uncertainty_dispersion_notes.md` §5–§6. Down-weighting endpoints by age/
  distance from their own receptor would naturally suppress the influence of
  the noisy, far-flung endpoints causing this, since they'd contribute less to
  both numerator and denominator instead of counting as a full, equal vote the
  way raw histogram binning does now.
- **A composition-aware weighting term** (not something traPSA/openair ship a
  precedent for — would be a documented departure from the standard method):
  e.g. a weight that also accounts for how concentrated a cell's endpoint
  count is across contributing sites, down-weighting cells where the count is
  a thin trickle from many distant sites rather than a solid contribution from
  one or two nearby ones.
- **Supplementary endpoint-density map** as a visual diagnostic companion to
  the PSCF map — lets a viewer cross-reference a hotspot against how
  well-supported it is by real trajectory density, without needing the
  weighting function to catch it automatically. `ResultMap` already works for
  this since it's generic over any 2D grid array (§ResultMap in `pscf.py`) —
  just feed it `n_ij` instead of the final weighted result. Two versions worth
  considering: a combined density map (simple, "well-sampled vs. sparse"), or
  a per-site breakdown (more diagnostic — would actually show *which* sites
  contribute to a given cell, directly surfacing the boundary cells where this
  dilution effect lives, at the cost of more build effort).
  - **Implementation note**: `PSCF.pscf()` currently only returns the final
    weighted result — `numerator`/`denominator` are local variables, not
    exposed afterward. Supporting the density-map idea means having `pscf()`
    stash them (e.g. `self.n_ij`, `self.m_ij`) as a side effect so they're
    accessible after calling `.pscf()`.

### A second, distinct limitation: record length acts as an implicit weight

Separate from the dilution issue above — this one is about *volume*, not
*composition*. The formula pools **raw** endpoint counts across sites
(`Σ_s Σ_l τ_ij,s,l`, traPSA guide §3.7), with no normalization for how much
data each site has. A site with a 10-year paired record contributes
proportionally more endpoints to every cell its trajectories visit than a
site with a 6-month record — not because its signal is more meteorologically
meaningful, just because it has more sampled hours.

**The per-site criterion (`C_s`) doesn't fix this** — it normalizes a
different thing. Each site's own 80th percentile means roughly ~20% of *that
site's* hours get flagged regardless of record length, so the *fraction*
flagged is fair across sites. But the *absolute count* of endpoints (both
flagged and unflagged) each site contributes to the pooled `n_ij`/`m_ij` sums
still scales directly with its data volume. So a longer-record site gets more
total "votes" in the pooled ratio at any cell it substantially visits, purely
as an artifact of how long it's been operating.

Same category as the dilution issue: an inherent property of pooling raw
counts as the formula is literally defined, not a bug — worth documenting as
a limitation regardless of whether it gets addressed.

**Possible fix to consider later**: normalize each site's contribution before
pooling — e.g. convert each site's endpoint counts to a fraction of *that
site's own* total endpoints before summing across sites, so a short-record
and a long-record site each contribute proportionally to their own internal
spatial pattern rather than their absolute data volume. No citable precedent
for this specific variant in the PSCF literature (unlike the weighting
scheme, which was verified against openair's actual source) — if pursued,
this would need to be documented as an intentional, reasoned departure from
the standard formula, not presented as "the" method.
