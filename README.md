# pscf_py

A Python implementation of multi-site PSCF (Potential Source Contribution Function) analysis for the Front Range Ozone Study. The goal is to replicate the behavior of traPSA so results can be validated against a known reference.

## What is PSCF?

PSCF is a receptor modeling technique for identifying likely pollution source regions. The core idea:

1. Run back trajectories from a receptor site (e.g., using HYSPLIT) to trace where air parcels came from over some period.
2. Bin all trajectory endpoints onto a spatial grid — each cell accumulates a count `n_ij`.
3. Separately count endpoints that occurred during high-pollution episodes (above a threshold) — this gives `m_ij`.
4. The PSCF value for each cell is `m_ij / n_ij`: the conditional probability that air passing through that cell arrived at the receptor during a high-pollution event.

High PSCF values highlight candidate source regions.

### Multi-site PSCF
Rather than a single receptor, trajectories from multiple monitoring sites are combined. This reduces the influence of site-specific meteorological patterns and gives a more robust regional picture. The traPSA formula:

```
PSCF_ij = Σ_s Σ_l ( p_s,l × τ_ij,s,l )
           / Σ_s Σ_l ( τ_ij,s,l )
```

Where s = site index, l = trajectory index, τ = endpoint count in cell (i,j), and `p_s,l = 1` if the concentration at site s for trajectory l exceeded that site's threshold, else 0. All sites contribute to a single shared grid.

### Threshold (criterion value)
Each site gets its own threshold C_s — traPSA does **not** use a single network-wide cutoff. The default in traPSA is the **80th percentile** of each site's own concentration distribution (criterion = 0.8). The user guide notes a typical range of 50%–90%.

### Weighting function (trailing effect correction)
Cells crossed by very few trajectories can produce spuriously high PSCF values — this is called the "trailing effect." A step-function weight is applied based on each cell's endpoint count relative to the grid average:

```
avg = total_endpoints / number_of_cells
W_ij = PSCF_ij × weight_factor(n_ij / avg)
```

Weight factors decrease as `n_ij` falls below `avg`. Defaults (`tier_bounds`/`tier_weights`/`min_weight` in `config.yaml`) are verified directly against openair's `trajLevel()` source: 1.0 / 0.75 / 0.5 / 0.15 at 2×/1×/0.5× the grid average.

**`ss_scaled`**: an alternative to the discrete tiers above — a continuous, statistically-grounded trailing-effect correction using inverse-variance weighting (`w_ij ∝ 1/CI_width²`, normalized to a max of 1.0), where the CI width comes from a Wilson score interval computed at a **fixed p̂=0.5** (not the cell's own measured proportion) so the weight reflects sample-size adequacy only, not the magnitude of the estimate itself. See `uncertainty_dispersion_notes.md` §8 for the full derivation, including why the fixed-p̂ decoupling matters and a structural pitfall (receptor-location endpoint pileup) that has to be excluded before this normalization is meaningful.

## Key references

- **PSCF method and limitations**: https://www.sciencedirect.com/science/article/pii/S2307187724000373#bib18

  > "However, there are drawbacks as well. First, the PSCF value in a cell with a small number of trajectories becomes very sensitive to a few high-concentration events, resulting in unrealistically high PSCF values. Generally, an arbitrary weighting function is applied to reduce this effect [9], [25], but this is not a completely satisfactory solution. Second, the PSCF values indicate the probability of potential sources located in the cells, but they do not represent the intensity of the sources. Furthermore, the PSCF model cannot provide users with information on the reliability of estimated sources. Direct and quantitative evaluation of both the intensity and reliability of certain areas cannot be achieved using PSCF values. Finally, the estimated values are subject to change with a change in the criteria applied in the calculation. Several studies extended and modified the PSCF model to partially address these issues [15], [14], [20], [31], but fundamental limitations remain."

## Reference implementation
This project targets parity with **traPSA** (Trajectory-based Potential Source Apportionment), developed at Clarkson University / CARES. The user guide is `TraPSA Software User's Guide.pdf` in this directory. traPSA is Windows-only and MATLAB-based; this Python implementation should produce matching results for the same inputs.

## Project structure

```
pscf_py/
├── main.py                          # Entry point: wires everything together and runs the pipeline
├── pscf.py                          # SpatialGrid, PSCF, TrajectoryDataset, ResultMap classes
├── project_specific_functions.py    # CDPHE/Boulder Air/HYSPLIT readers (project-specific — see below)
├── config.yaml                      # All configuration: paths, species, sites, grid, criterion, weighting, map styling
├── documentation/
│   ├── multi_site_pscf_notes.md         # Reference notes: criterion/summation/weighting derivation, known limitations
│   ├── uncertainty_dispersion_notes.md  # Reference notes: uncertainty quantification and dispersion-kernel exploration
│   └── TraPSA Software User's Guide.pdf # Reference implementation's user guide
├── output/                          # Generated maps/overlays from main.py (gitignored)
├── testing/                         # Old notebooks, prior iteration outputs, validation scripts (gitignored)
├── CLAUDE.md                        # Claude Code collaboration notes
└── README.md                        # This file
```

### Architecture
- **`SpatialGrid`**: pure spatial geometry (lat/lon bin edges/centers), no statistical state. Read from `config.yaml`'s `pscf.grid` section.
- **`TrajectoryDataset`**: validates raw pollution/trajectory data against `config.yaml`'s `column_mapping`, renames to the pipeline's canonical schema, merges on `(site_id, timestamp_utc)`, computes each site's own criterion `C_s`, and flags every row — all as part of construction, ending with a `sanity_check()` pass.
- **`PSCF`**: computes the weighted multi-site PSCF grid from an already-assembled `TrajectoryDataset`. Deliberately has no per-site logic — every site's rows are expected to already be in the combined dataframe, and accumulation across sites happens automatically since nothing filters by site.
- **`ResultMap`**: builds the folium map from a finished result grid. Generic over the result's meaning (works for PSCF today, CWT/SQTBA later) — takes a `SpatialGrid` and any 2D array, not something PSCF-specific.
- **`project_specific_functions.py`**: intentionally kept separate from the above — these are the CDPHE/Boulder Air/HYSPLIT file-format readers, which anyone reusing this pipeline with different raw data sources would replace. Everything downstream only depends on the unified schemas documented at the bottom of `config.yaml`.

`config.yaml` is the single source of configuration — species, paths, site lists, grid bounds/resolution, criterion, weighting scheme, and all map/plot styling (title, legend, scale bar, city labels) are read from it rather than hardcoded.

## Time pairing convention

This is critical for reproducibility. traPSA pairs each trajectory with the concentration measurement that shares the **same timestamp** — no hour offset. Conventions used here:

- All timestamps are **UTC**.
- EPA AQS data is in local standard time → convert to UTC before use.
- Boulder Air data is in UTC, except BRZ site (some years in MST) → convert those to UTC.
- AQS labels the **start** of the 1-hour averaging window (label 12:00 UTC = average over 12:00–13:00 UTC).
- Boulder Air minute data is resampled to hourly using the same convention: left-closed, left-labeled bins.
- A trajectory initialized at 12:00 UTC is paired with the measurement labeled 12:00 UTC at that site.

### Why trajectory H pairs with label H and not H-1

A back trajectory initialized at hour H represents an air parcel that **arrived at the receptor at time H**. That air then remained at or flowed through the receptor during H to H+1, contributing to the concentration measured during that period. Since AQS label H covers the averaging window H to H+1, it captures the same air mass the trajectory describes. Pairing with label H-1 (the hour ending at H) would instead describe air that had already passed through the receptor before the trajectory-initialized parcel arrived.

There is an inherent ±30 minute ambiguity: the trajectory is instantaneous while the concentration is a 1-hour average, so measurements near the end of the averaging window are less representative of the air at exactly H. However, label H and label H-1 have symmetric error relative to the instantaneous trajectory time, so neither is strictly more accurate. The H=H convention is chosen for physical consistency and to match traPSA. In practice, HYSPLIT trajectory positional uncertainty (typically 15–20% of travel distance per day) far exceeds any pairing ambiguity from hourly averaging.

A centered average (e.g., 12:30–13:30 for a trajectory at 13:00) would be most physically accurate but is not possible with pre-averaged AQS data, and would create inconsistency between data sources.

### Per-species hourly resampling

Different species use different resampling methods, chosen to maximize internal consistency across data sources:

| Species | Method | Reason |
|---------|--------|--------|
| O3, NOx | Floor to hour (`resample('1h', closed='left', label='left').mean()`) | Both CDPHE and Boulder Air sites are used together. Flooring matches AQS left-label convention exactly, keeping all sites on an identical time axis. |
| VOCs | Round to nearest hour | No CDPHE VOC data exists, so there is no cross-source consistency constraint. Boulder Air VOC data is a ~10-minute grab sample taken near the top of each hour, so rounding to nearest hour is more accurate than flooring. Earlier years had less strict sampling times with some samples taken later in the hour, but rounding remains the best single convention. |

## Implementation roadmap

- [x] Spatial grid (`SpatialGrid`) with lat/lon binning, config-driven bounds/resolution
- [x] Unify EPA/CDPHE and Boulder Air data into a common `(site_id, timestamp_utc, species, ppb)` schema
- [x] Parse HYSPLIT trajectory files into endpoint lat/lon series, keyed by `(site_id, timestamp_utc)`
- [x] Per-site criterion `C_s` (80th percentile by default), computed on each site's own `traj_age == 0` rows
- [x] Multi-site accumulation of `n_ij`/`m_ij` across all sites and trajectories
- [x] Weighted PSCF computation, weighting scheme verified against openair's actual source
- [x] Interactive map output (`ResultMap`): georeferenced overlay, legend, distance scale, site/city labels, config-driven styling
- [x] Trajectory reading scoped to sites relevant to the current species (`active_sites`), instead of reading the full multi-year archive
- [x] Wilson score confidence interval on the raw `m_ij`/`n_ij` ratio, exposed from `pscf()` and shown as a toggleable "sample size uncertainty (CI width)" map overlay — see `uncertainty_dispersion_notes.md` §2
- [x] `ss_scaled` weighting option — CI-based (inverse-variance) alternative to the discrete trailing-effect tiers, decoupled from the cell's own measured proportion — see `uncertainty_dispersion_notes.md` §8
- [ ] Validate against traPSA output on the same inputs
- [ ] CWT / SQTBA (concentration-weighted, dispersion-kernel-aware alternative to PSCF) — see `uncertainty_dispersion_notes.md`
- [ ] Monte Carlo / bootstrap uncertainty estimation (positional + dispersion uncertainty, not just sampling) — see `uncertainty_dispersion_notes.md` §3
- [ ] Per-site (not network-wide) reference point for the multi-site trailing-effect weighting — see `uncertainty_dispersion_notes.md` §8
- [ ] Major roadways / oil & gas well overlays (shapefile-based)

## Dependencies

- Python 3.x
- `numpy`, `pandas`, `pyyaml`
- `folium`, `branca`, `matplotlib` (mapping/plotting)
- `geopandas`, `shapely` (planned: roadway/well shapefile overlays)
- HYSPLIT (external, for trajectory generation)
- Monitoring data from CDPHE / Boulder Airshed network

## Domain notes

- Grid domain covers approximately Colorado: 37–41°N, 109–102°W
- Grid resolution: ~0.027° latitude × 0.035° longitude (roughly 3 km, matching HRRR meteorological resolution)
- traPSA default PSCF criterion: 0.8 (80th percentile, per site)
- traPSA CPF minimum wind speed: 5 km/h
