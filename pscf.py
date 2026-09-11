import pandas as pd
import numpy as np
import os
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.stats.proportion as smp


class SpatialGrid:
    """
    Defines the spatial grid over which PSCF (or CWT) is computed.

    The domain is divided into equal-width lat/lon bins. For each bin (i,j):
        n_ij = total trajectory endpoints that fell in that cell
        m_ij = endpoints where the paired concentration exceeded the site threshold
        PSCF_ij = m_ij / n_ij (computed separately, after accumulation, by PSCF)

    Bounds and resolution are read from config.yaml's pscf.grid section.
    """

    def __init__(self, params):
        self.params = params
        grid_cfg = params['pscf']['grid']
        lat_min = grid_cfg['lat_min']
        lat_max = grid_cfg['lat_max']
        lon_min = grid_cfg['lon_min']
        lon_max = grid_cfg['lon_max']
        dlat = grid_cfg.get('dlat', 0.027)
        dlon = grid_cfg.get('dlon', 0.035)

        # Anchor cell centers to exact multiples of dlat/dlon (from zero), matching
        # openair's round_any(x, inc) grid convention — so a Python grid and an R
        # grid built with the same lat.inc/lon.inc line up exactly, cell for cell,
        # instead of being offset by whatever lat_min/lon_min happen to be.
        first_lat_edge = (np.floor(lat_min / dlat - 0.5) + 0.5) * dlat
        first_lon_edge = (np.floor(lon_min / dlon - 0.5) + 0.5) * dlon
        self.lat_edges = np.arange(first_lat_edge, lat_max + dlat, dlat)
        self.lon_edges = np.arange(first_lon_edge, lon_max + dlon, dlon)

        # Cell centers are the midpoints between adjacent edges
        self.lat_centers = (self.lat_edges[:-1] + self.lat_edges[1:]) / 2
        self.lon_centers = (self.lon_edges[:-1] + self.lon_edges[1:]) / 2

        self.n_lat = len(self.lat_centers)
        self.n_lon = len(self.lon_centers)

    def get_cell(self, lat, lon):
        """
        Return the (row, col) grid indices for a given lat/lon point.

        Returns (None, None) if the point falls outside the grid domain.
        """
        # np.digitize gives the bin index (1-based), so subtract 1 for 0-based.
        i = np.digitize(lat, self.lat_edges) - 1
        j = np.digitize(lon, self.lon_edges) - 1
        if 0 <= i < self.n_lat and 0 <= j < self.n_lon:
            return i, j
        else:
            return None, None


class PSCF:
    """
    Computes the weighted multi-site PSCF grid for an assembled analysis dataset.

    Takes a SpatialGrid instance and a combine_data dataframe (typically
    TrajectoryDataset.data) carrying one row per trajectory endpoint, each with
    that endpoint's lat/lon and a 'criterion_binary' flag (1 = this endpoint's
    trajectory was paired with an hour that met its site's criterion, 0
    otherwise). Sites are not handled individually in this class — combine_data
    is expected to already contain every site's rows, and PSCF_ij accumulates
    across all of them automatically since no per-site filtering happens here.
    """

    def __init__(self, grid, combine_data, params):
        self.grid = grid
        self.combine_data = combine_data
        self.params = params

    def weighting_function(self, denominator, ave_endpoints, method=None, weighting=None, weighting_value=None,
                        comparison=None, tier_bounds=None, tier_weights=None, min_weight=None):
        """
        Compute a per-cell weighting factor that corrects for the "trailing effect" —
        cells crossed by only a few trajectories can produce misleadingly high PSCF
        or CWT values, so cells with low endpoint counts get down-weighted.

        Parameters:
        - denominator: 2D array of trajectory endpoint counts per grid cell (n_ij).
        - ave_endpoints: average endpoint count per cell across the whole grid.
        - method: "pscf" or "cwt" — selects which config.yaml pscf.weight_function
          sub-block to read defaults from. Leave as None to use analysis.method
          (the normal case when called directly); pscf() and cwt() each pass this
          explicitly so they're correct regardless of what analysis.method says.
        - weighting, weighting_value, comparison, tier_bounds, tier_weights,
          min_weight: override the corresponding config.yaml value for this call.
          Leave as None to use what's already in self.params.
        - comparison: "relative" (tiers are multiples of ave_endpoints, traPSA/PSCF's
          convention) or "absolute" (tiers are fixed endpoint-count thresholds,
          openair's CWT convention) — verified directly against openair's
          trajLevel() source, see multi_site_pscf_notes.md §6.

        Returns:
        - w_ij: 2D array of weighting factors, same shape as denominator — meant
          to be multiplied elementwise into the result grid.
        """
        method = method if method is not None else self.params.get('analysis', {}).get('method', 'pscf')
        weight_cfg = self.params['pscf']['weight_function'][method]

        weighting = weighting if weighting is not None else weight_cfg.get('weighting', 'standard')
        weighting_value = weighting_value if weighting_value is not None else weight_cfg.get('weighting_value')
        comparison = comparison if comparison is not None else weight_cfg.get('comparison', 'relative')
        tier_bounds = tier_bounds if tier_bounds is not None else weight_cfg.get('tier_bounds', (2, 1, 0.5))
        tier_weights = tier_weights if tier_weights is not None else weight_cfg.get('tier_weights', (1.0, 0.75, 0.5))
        min_weight = min_weight if min_weight is not None else weight_cfg.get('min_weight', 0.15)

        if weighting == 'standard':
            # "relative" (pscf): tiers are multiples of the grid average, e.g.
            #   > 2x average endpoints -> full weight (1.0), down to <= 0.5x -> min_weight
            # "absolute" (cwt): tiers are fixed endpoint-count thresholds, e.g.
            #   > 80 endpoints -> full weight (1.0), down to <= 10 -> min_weight
            if comparison == 'relative':
                conditions = [denominator > bound * ave_endpoints for bound in tier_bounds]
            else:
                conditions = [denominator > bound for bound in tier_bounds]
            w_ij = np.select(conditions, list(tier_weights), default=min_weight)
        elif weighting == 'binary' and weighting_value is not None:
            # Hard cutoff: full weight if this cell has more than weighting_value
            # endpoints, otherwise zeroed out entirely.
            w_ij = np.where(denominator > weighting_value, 1.0, np.nan)
        elif weighting == 'ss_scaled_pessimistic' and self.upper_scaling_ci is not None and self.lower_scaling_ci is not None:
            # Scale by sample-size uncertainty: full weight where n_ij is large, down to min_weight where n_ij is small
            # Here, we assume denominator is n_ij
            eps = .00000001 # tiny correction factor to ensure denominator is never 0
            ci_width = (self.upper_scaling_ci - self.lower_scaling_ci) + eps
            w_ij = (1/(ci_width ** 2))
            w_ij = w_ij / np.nanmax(w_ij)  # normalize to max weight = 1.0
        elif weighting == 'ss_scaled' and self.upper_ci is not None and self.lower_ci is not None:
            eps = .00000001 # tiny correction factor to ensure denominator is never 0
            ci_width = (self.upper_ci - self.lower_ci) + eps
            w_ij = (1/(ci_width ** 2))
            w_ij = w_ij / np.nanmax(w_ij)  # normalize to max weight = 1.0
        elif weighting == 'ss_scaled_balanced' and weighting_value is not None:
            # Shrinkage compromise between 'ss_scaled' (real p_hat, lets a
            # lucky extreme p_hat from a tiny sample look falsely precise)
            # and 'ss_scaled_pessimistic' (fixed p_hat=0.5, worst case for
            # every cell regardless of how much data it has). weighting_value
            # is k: how many "pretend" observations of p=0.5 to mix in before
            # trusting the real m_ij/n_ij. As n_ij -> infinity, p_shrunk -> the
            # real p_hat; as n_ij -> 0, p_shrunk -> 0.5.
            eps = .00000001 # tiny correction factor to ensure denominator is never 0
            k = weighting_value
            m_ij = self.numerator
            n_ij = denominator
            p_shrunk = (m_ij + k * 0.5) / (n_ij + k)

            mask = n_ij > 0
            lower_balanced_ci = np.full(n_ij.shape, np.nan)
            upper_balanced_ci = np.full(n_ij.shape, np.nan)
            lower_balanced_ci[mask], upper_balanced_ci[mask] = smp.proportion_confint(
                count=(p_shrunk[mask] * n_ij[mask]), nobs=n_ij[mask], alpha=0.05, method='wilson')

            ci_width = (upper_balanced_ci - lower_balanced_ci) + eps
            w_ij = (1/(ci_width ** 2))
            w_ij = w_ij / np.nanmax(w_ij)  # normalize to max weight = 1.0
        elif weighting == 'se_scaled':
            p0 = self.numerator.sum() / self.denominator.sum()
            mask = self.denominator > 0
            self.se = np.full(self.denominator.shape, np.nan)
            self.se[mask] = np.sqrt(p0 * (1 - p0) / self.denominator[mask])
            eps = .0000000001
            w_ij = 1 / (self.se ** 2 + eps)
            w_ij = w_ij / np.nanmax(w_ij)  # normalize to max weight = 1.0
            
            

        else:
            raise ValueError("Invalid weighting type or missing weighting value.")

        return w_ij

    def pscf(self, weighting=None, weighting_value=None):
      """
      Compute the weighted PSCF grid over self.combine_data (set at construction),
      using self.grid for the shared lat/lon bin edges. combine_data must have a
      'criterion_binary' column indicating whether each row's trajectory meets
      the criterion for the numerator.

      Parameters:
      - weighting, weighting_value: override the corresponding config.yaml
        pscf.weight_function value for this call. Leave as None to use what's
        already in self.params (the normal case) — see weighting_function().

      Returns:
      - result: 2D array, the weighted PSCF value per grid cell.
      - lower_ci, upper_ci: Wilson CI bounds on the raw m_ij/n_ij ratio (see
        pscf_cell_sample_size_error_estimation()).
      - z_score: per-cell z-score vs. the network-wide baseline rate (see
        cell_significance_zscore()).
      """
      combine_df = self.combine_data
      combine_df = combine_df.loc[combine_df['traj_age'] != 0]
      site_denom = combine_df.groupby('site_id').count()['traj_age'].to_list()
      grid = self.grid

      # m_ij: endpoints from hours at or above the site's threshold
      self.numerator = np.histogram2d(combine_df.loc[combine_df['criterion_binary'] == 1, 'lat'],
                                  combine_df.loc[combine_df['criterion_binary'] == 1, 'lon'],
                                  bins=[grid.lat_edges, grid.lon_edges])[0]

      # n_ij: every endpoint, regardless of concentration
      self.denominator = np.histogram2d(combine_df['lat'], combine_df['lon'], bins=[grid.lat_edges, grid.lon_edges])[0]

      # Sets self.lower_ci/self.upper_ci as a side effect (Wilson CI on the
      # raw m_ij/n_ij, i.e. sample-size uncertainty) — no return value to
      # capture here, it already assigns directly onto self.
      self.pscf_cell_sample_size_error_estimation()

      # Sets self.z_score as a side effect — how unusual this cell's rate is
      # relative to the network-wide baseline, a different question from the
      # CI above (see cell_significance_zscore()'s docstring).
      self.cell_significance_zscore()

      total_endpoints = np.sum(self.denominator)
      avg_endpoints = total_endpoints / (self.denominator > 0).sum()  # average endpoints per cell — the weighting reference point
      w_ij = self.weighting_function(self.denominator, avg_endpoints, method='pscf', weighting=weighting, weighting_value=weighting_value)

      # m_ij / n_ij per cell; cells with zero trajectories are left as NaN
      # (genuinely undefined — no data) rather than a fake 0, so they can be
      # told apart from cells where n_ij > 0 but m_ij == 0 (well-sampled,
      # legitimately zero — evidence the cell is NOT a source region).
      unscaled_result = np.divide(self.numerator, self.denominator, out=np.full_like(self.numerator, np.nan), where=self.denominator!=0)
      result = np.multiply(unscaled_result, w_ij)  # apply the trailiong-effect weighting

      return result, self.lower_ci, self.upper_ci, self.z_score

    def cwt(self, weighting=None, weighting_value=None, statistic=None):
      """
      Compute the weighted CWT (Concentration Weighted Trajectory) grid over
      self.combine_data — the per-cell pollutant concentration (mean or
      median, see `statistic`), weighted by the same trailing-effect scheme
      as pscf(), but using CWT's own tiers (config.yaml
      pscf.weight_function.cwt — absolute endpoint-count thresholds, not
      multiples of the grid average; see weighting_function()).

      Parameters:
      - weighting, weighting_value: override the corresponding config.yaml
        pscf.weight_function.cwt value for this call. Leave as None to use
        what's already in self.params.
      - statistic: "mean" or "median" — which per-cell statistic to compute.
        Leave as None to use config.yaml's pscf.cwt_statistic (defaults to
        "mean" if unset). Median is the more robust choice for a skewed/
        outlier-prone quantity (e.g. a ratio metric like ethane/NOx, where a
        near-zero denominator can produce an extreme spike) — a handful of
        extreme hours can't drag a per-cell median around the way they can a
        mean, since the weighting below only corrects for sample *size*, not
        for outliers within an already-large sample.

      Returns:
      - result: 2D array, the weighted mean/median concentration per grid cell.
      """
      combine_df = self.combine_data
      grid = self.grid
      statistic = statistic if statistic is not None else self.params['pscf'].get('cwt_statistic', 'mean')

      # denominator: every endpoint, regardless of concentration — same as pscf()
      self.denominator = np.histogram2d(combine_df['lat'], combine_df['lon'],
                                    bins=[grid.lat_edges, grid.lon_edges])[0]

      if statistic == 'median':
        # No vectorized weighted-median in numpy/histogram2d, so bin each
        # row to a cell index directly (same convention as SpatialGrid.get_cell)
        # and take the per-cell median via groupby.
        lat_idx = np.digitize(combine_df['lat'], grid.lat_edges) - 1
        lon_idx = np.digitize(combine_df['lon'], grid.lon_edges) - 1
        valid = (lat_idx >= 0) & (lat_idx < grid.n_lat) & (lon_idx >= 0) & (lon_idx < grid.n_lon)
        cell_df = pd.DataFrame({'i': lat_idx[valid], 'j': lon_idx[valid], 'ppb': combine_df['ppb'].to_numpy()[valid]})
        medians = cell_df.groupby(['i', 'j'])['ppb'].median()

        unscaled_result = np.full(self.denominator.shape, np.nan)
        ij = np.array(medians.index.tolist())
        unscaled_result[ij[:, 0], ij[:, 1]] = medians.to_numpy()
        self.numerator = None  # not a meaningful quantity for median — nothing downstream reads it
      else:
        # numerator: sum of concentration values, weighted by how many endpoints
        # each trajectory contributes to a cell (np.histogram2d's weights= does
        # this sum directly — see multi_site_pscf_notes.md for the derivation).
        self.numerator = np.histogram2d(combine_df['lat'], combine_df['lon'],
                              bins=[grid.lat_edges, grid.lon_edges],
                              weights=combine_df['ppb'])[0]
        # NaN (not 0) where there's genuinely no data — a fake 0 would read on
        # the map as "measured zero concentration" instead of "no data here",
        # and would corrupt np.nanmin/np.nanmax-based color scaling.
        unscaled_result = np.divide(self.numerator, self.denominator, out=np.full_like(self.numerator, np.nan), where=self.denominator!=0)

      total_endpoints = np.sum(self.denominator)
      avg_endpoints = total_endpoints / (self.denominator > 0).sum()
      w_ij = self.weighting_function(self.denominator, avg_endpoints, method='cwt', weighting=weighting, weighting_value=weighting_value)

      result = np.multiply(unscaled_result, w_ij)

      return result
    
    def pscf_cell_sample_size_error_estimation(self):
        m_ij = self.numerator
        n_ij = self.denominator
        mask = n_ij > 0

        self.lower_ci = np.full(n_ij.shape, np.nan)
        self.upper_ci = np.full(n_ij.shape, np.nan)

        self.lower_ci[mask], self.upper_ci[mask] = smp.proportion_confint(
        count=m_ij[mask], nobs=n_ij[mask], alpha=0.05, method='wilson')

        # Separate, fixed-p_hat=0.5 CI used only by weighting_function's
        # 'ss_scaled' branch — a pure sample-size measure, decoupled from
        # the cell's actual measured proportion (see discussion: using the
        # real m_ij here would let extreme p_hat from tiny samples "cheat"
        # their way to a narrow CI, rewarding the exact small-sample noise
        # this weighting is meant to catch). self.lower_ci/self.upper_ci
        # above stay untouched — those still reflect genuine estimation
        # uncertainty for the CI-width map overlay.
        self.lower_scaling_ci = np.full(n_ij.shape, np.nan)
        self.upper_scaling_ci = np.full(n_ij.shape, np.nan)

        self.lower_scaling_ci[mask], self.upper_scaling_ci[mask] = smp.proportion_confint(
        count=(n_ij[mask])/2, nobs=n_ij[mask], alpha=0.05, method='wilson')

    def cell_significance_zscore(self):
        """
        Per-cell z-score testing whether this cell's observed rate (m_ij/n_ij)
        looks statistically different from p0 — the network-wide baseline rate
        (total_m_ij / total_n_ij across the whole grid), i.e. "how would a cell
        with no special relationship to high-pollution hours look, on average,
        across this entire dataset." A different question from
        pscf_cell_sample_size_error_estimation (which asks "how precisely do we
        know THIS cell's own rate") — this asks "does this cell's rate look
        different from an unremarkable, non-source cell at all."

        Normal-approximation one-proportion z-test, not the non-parametric
        bootstrap approach used in the published PSCF literature (Hopke et al.
        1995) — see uncertainty_dispersion_notes.md for that distinction. Rule
        of thumb for validity: n_ij*p0 > 5 and n_ij*(1-p0) > 5 (same caveat as
        Wilson vs. plain SE elsewhere in this file).

        Must be called after pscf() (needs self.numerator/self.denominator).
        Sets self.z_score (2D array, same shape as self.denominator) as a side
        effect — NaN wherever n_ij == 0 (no data). Positive z_score = this
        cell's rate is higher than the network baseline (candidate source
        direction); negative = lower.
        """
        m_ij = self.numerator
        n_ij = self.denominator
        mask = n_ij > 0

        # p0 is computed empirically from the pooled data (total_m_ij/total_n_ij)
        # rather than read as 1 - criterion from config. Each site's criterion
        # is computed on that site's own traj_age==0 rows, so 1 - criterion
        # holds almost exactly at the per-site level by construction — but
        # once multiple sites with different record lengths get pooled onto
        # one shared grid (each contributing a different number of total
        # endpoints), the actual pooled rate can drift slightly away from
        # that theoretical value. Computing p0 directly from this run's own
        # totals measures what the accumulated dataset actually shows,
        # instead of assuming the theoretical per-site figure survives
        # pooling unchanged.
        p0 = m_ij.sum() / n_ij.sum()
        p_hat = np.divide(m_ij, n_ij, out=np.full_like(m_ij, np.nan), where=n_ij != 0)

        self.z_score = np.full(n_ij.shape, np.nan)
        self.z_score[mask] = (p_hat[mask] - p0) / np.sqrt(p0 * (1 - p0) / n_ij[mask])

        return self.z_score



class TrajectoryDataset:
    """
    Assembles and validates the merged pollution + trajectory analysis dataset.

    Given raw pollution and trajectory dataframes (in whatever column names the
    caller's data actually uses) plus the loaded config.yaml params, this
    class validates the raw data, renames it to the pipeline's canonical schema
    (see config.yaml's column_mapping / Output schema / Trajectory schema),
    merges the two on (site_id, timestamp_utc), computes each site's own PSCF
    criterion (C_s) and flags every row with a 'criterion_binary' column, and
    runs a final sanity check — all as part of construction. self.data is the
    finished result, ready to hand to PSCF (or a future CWT class).
    """

    def __init__(self, params, pollution_data, trajectory_data):
        self.params = params
        self.pollution_data = pollution_data
        self.trajectory_data = trajectory_data

        self.data_checker()
        self.build_analysis_dataset()
        self.sanity_check()

    def data_checker(self):
        """
        Validate the RAW data against the raw (pre-rename) column names
        configured in config.yaml's column_mapping — i.e. what the
        incoming user data is expected to look like before prepare_data()
        renames it to the pipeline's canonical schema.

        Raises ValueError if either dataframe is missing/empty, or missing any
        of its expected raw columns.
        """
        if self.pollution_data is None or len(self.pollution_data) == 0:
            raise ValueError("Pollution data is missing or empty.")
        if self.trajectory_data is None or len(self.trajectory_data) == 0:
            raise ValueError("Trajectory data is missing or empty.")

        pollution_mapping = self.params['column_mapping']['pollution']
        trajectory_mapping = self.params['column_mapping']['trajectory']

        missing_pollution_cols = [col for col in pollution_mapping if col not in self.pollution_data.columns]
        if missing_pollution_cols:
            raise ValueError(f"Pollution data is missing expected column(s): {missing_pollution_cols}")

        missing_trajectory_cols = [col for col in trajectory_mapping if col not in self.trajectory_data.columns]
        if missing_trajectory_cols:
            raise ValueError(f"Trajectory data is missing expected column(s): {missing_trajectory_cols}")

    def prepare_data(self):
        """
        Validate the raw data first, then rename the user's raw columns to the
        canonical names the rest of the pipeline (PSCF/CWT) expects, using
        config.yaml's column_mapping.
        """
        self.data_checker()

        pollution_mapping = self.params['column_mapping']['pollution']
        trajectory_mapping = self.params['column_mapping']['trajectory']

        self.pollution_data = self.pollution_data.rename(columns=pollution_mapping)
        self.trajectory_data = self.trajectory_data.rename(columns=trajectory_mapping)

    def get_criterion_value(self, temp_df):
        """
        Compute a single site's criterion value (C_s) from temp_df — that
        site's rows, already filtered to traj_age == 0 (the receptor/hour-0
        point) by create_criterion_dict.

        Reads criterion_type ('percentile' or 'value') and criterion from
        config.yaml's pscf section. For 'percentile', criterion is a 0-1
        fraction (e.g. 0.8 = 80th percentile); np.percentile expects its
        percentile argument on a 0-100 scale, so it's multiplied by 100 here.
        """
        criterion_type = self.params['pscf'].get('criterion_type')
        criterion = self.params['pscf'].get('criterion')

        if criterion_type == 'percentile' and (0 <= criterion <= 1):
            c_val = np.percentile(temp_df['ppb'], criterion * 100)
        elif criterion_type == 'value':
            c_val = criterion
        else:
            raise ValueError(f"Invalid criterion_type: {criterion_type}")
        return c_val

    def create_criterion_dict(self):
        """
        Compute each site's own criterion value (C_s), using only that site's
        traj_age == 0 (receptor) rows — never a network-wide value (see
        multi_site_pscf_notes.md, section 2).

        Returns a dict of {site_id: C_s}.
        """
        criterion_dict = {}
        for site in self.data['site_id'].unique():
            temp_df = self.data.loc[(self.data['site_id'] == site) & (self.data['traj_age'] == 0)].copy()
            c_val = self.get_criterion_value(temp_df)
            criterion_dict[site] = c_val
        return criterion_dict

    def build_analysis_dataset(self):
        """
        Validate + rename the raw data (prepare_data), inner-merge pollution
        and trajectory data on (timestamp_utc, site_id), compute each site's
        criterion, and flag every row with 'criterion_binary' (1 if that row's
        trajectory's paired ppb met its site's criterion, else 0).

        Sets and returns self.data, the finished multi-site analysis dataset.
        """
        self.prepare_data()  # validate raw columns, then rename to canonical names
        self.data = self.pollution_data.merge(self.trajectory_data, on=['timestamp_utc', 'site_id'], how='inner')
        self.data = self.data.loc[self.data['traj_age'] >= self.params['trajectories']['hours_back']]
        self.criterion_dict = self.create_criterion_dict()
        self.data['criterion_binary'] = self.data['site_id'].map(self.criterion_dict)
        self.data['criterion_binary'] = (self.data['ppb'] >= self.data['criterion_binary']).astype(int)


        return self.data

    def sanity_check(self):
        """
        Final check on the fully assembled analysis dataset — catches the kind
        of issue that would otherwise stay silent until PSCF runs on a
        wrong-looking result, e.g. a site's rows vanishing because of a
        timestamp/dtype mismatch in the merge (as happened with LUR earlier).

        Raises ValueError if:
        - self.data is empty,
        - a site present in both raw inputs is entirely absent from self.data
          (almost always a join-key mismatch for that site specifically), or
        - criterion_binary has any NaN or non-0/1 values.
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError(
                "Merged analysis dataset is empty — check that pollution_data "
                "and trajectory_data share overlapping site_id/timestamp_utc values."
            )

        # A site present in both raw inputs but entirely absent from the merged
        # data almost always means a join-key mismatch (dtype, timezone, etc.)
        # for that site specifically, rather than a real lack of overlap.
        raw_sites = set(self.pollution_data['site_id'].unique()) & set(self.trajectory_data['site_id'].unique())
        merged_sites = set(self.data['site_id'].unique())
        dropped_sites = raw_sites - merged_sites
        if dropped_sites:
            raise ValueError(
                f"Site(s) present in both raw inputs but missing entirely from "
                f"the merged data (likely a timestamp/dtype mismatch): {dropped_sites}"
            )

        if self.data['criterion_binary'].isna().any():
            raise ValueError(
                "criterion_binary contains NaN values — check that every site_id "
                "in the merged data has a matching entry in criterion_dict."
            )
        if not self.data['criterion_binary'].isin([0, 1]).all():
            raise ValueError("criterion_binary contains values other than 0/1.")


class ResultMap:
    """
    Builds a folium map for a finished PSCF (or CWT) result grid.

    Takes a SpatialGrid instance, the finished 2D result array (e.g. from
    PSCF.pscf()), a TrajectoryDataset (for site names/locations), and params
    (for criterion/weighting metadata to render as supplemental caption text).
    Deliberately generic to the result array's meaning — nothing here assumes
    PSCF specifically, so the same class works for a future CWT/SQTBA result too.

    Usage:
        rmap = ResultMap(grid, result, dataset, params)
        m = rmap.build_map()  # title/caption/zoom_start default from config.yaml's map section
        m.save("pscf_result.html")
    """

    def __init__(self, grid, result, dataset, params, overlay_path=None, lower_ci=None, upper_ci=None, z_score=None):
        self.grid = grid
        self.result = result
        self.dataset = dataset
        self.params = params
        self.overlay_path = overlay_path or os.path.join(
            params.get('paths', {}).get('output_dir', '.'), 'result_overlay.png'
        )
        # Optional: PSCF.pscf()'s Wilson confidence-interval bounds (same shape
        # as result). If both are given, build_map() adds a toggleable
        # "uncertainty (CI width)" overlay layer alongside the main result —
        # CWT has no equivalent yet (uncertainty_dispersion_notes.md §7/§8).
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        # Optional: PSCF.pscf()'s cell_significance_zscore() output (same shape
        # as result). If given, build_map() adds a toggleable "significance
        # (z-score)" overlay layer — same pattern as the CI-width overlay.
        self.z_score = z_score

    def get_site_locations(self):
        """
        Return {site_id: (lat, lon)} for every site in the analysis dataset,
        taken from each site's traj_age == 0 (receptor) row.
        """
        data = self.dataset.data
        locations = {}
        for site in data['site_id'].unique():
            row = data.loc[(data['site_id'] == site) & (data['traj_age'] == 0)].iloc[0]
            locations[site] = (row['lat'], row['lon'])
        return locations

    def render_overlay(self, data=None, path=None, symmetric=False, vmin=None):
        """
        Render a 2D array as a jet-colormap RGBA image and save it to disk,
        ready to be used as a georeferenced ImageOverlay. Defaults to
        self.result / self.overlay_path (the main result), but also used to
        render the CI-width and z-score overlays onto their own separate
        files — same rendering logic, different array/path.

        Parameters:
        - symmetric: if True, centers the color scale at 0 (vmin=-vmax,
          vmax=max absolute value) instead of the default vmin=0. Needed for
          z-score, which is meaningfully negative as well as positive (below
          vs. above the network baseline) — the default vmin=0 range would be
          wrong for that, silently clipping/misrepresenting negative values.
          Overrides vmin below (symmetric always wins).
        - vmin: explicit lower bound for the color scale. Leave as None for
          the default (0) — the right floor for PSCF, where 0 is a genuine,
          meaningful value (no chance of a source). Pass
          np.nanmin(data)/similar for magnitude data like CWT, where 0 never
          actually occurs among the visible (non-NaN) cells and anchoring at
          0 just compresses all the real variation into a sliver of the
          color scale — see 'binary' weighting, which leaves whole regions
          NaN (transparent) rather than diluting them, so the true floor is
          whatever the lowest surviving cell's value is, not 0.

        Returns (vmin, vmax) — the color scale's bounds, so build_map's
        legend can share the exact same scale as the image.
        """
        data = data if data is not None else self.result
        path = path if path is not None else self.overlay_path

        cmap = plt.get_cmap('jet')
        if symmetric:
            vmax = np.nanmax(np.abs(data))  # nanmax: 'binary' weighting can leave NaN for below-cutoff cells
            vmin = -vmax
        else:
            vmax = np.nanmax(data)
            vmin = vmin if vmin is not None else 0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        rgba = cmap(norm(data))
        rgba[..., 3] = np.where(np.isnan(data), 0.0, 0.85)  # transparent only where there's genuinely no data (NaN); true zeros are plotted like any other value

        # data[0, :] is the southernmost row, but image row 0 renders at the
        # image's top — flip vertically so the overlay lines up with the map.
        rgba = np.flipud(rgba)

        plt.imsave(path, rgba)
        return vmin, vmax

    def default_caption(self, method=None):
        """
        Build a default supplemental caption (criterion, weighting scheme)
        from self.params, for use under the map title when no caption is
        passed explicitly to build_map(). Wording comes from
        config.yaml's map.caption_templates, not hardcoded here.

        Parameters:
        - method: "pscf" or "cwt" — selects which config.yaml
          pscf.weight_function sub-block to read the weighting label from.
          Leave as None to use analysis.method.
        """
        method = method if method is not None else self.params.get('analysis', {}).get('method', 'pscf')
        pscf_cfg = self.params.get('pscf', {})
        templates = self.params.get('map', {}).get('caption_templates', {})
        criterion_type = pscf_cfg.get('criterion_type')
        criterion = pscf_cfg.get('criterion')
        weight_cfg = pscf_cfg.get('weight_function', {}).get(method, {})
        weighting = weight_cfg.get('weighting')
        comparison = weight_cfg.get('comparison')

        parts = []
        if criterion_type == 'percentile' and criterion is not None:
            template = templates.get('criterion_percentile', "Criterion: {value:.0f}th percentile")
            parts.append(template.format(value=criterion * 100))
        elif criterion is not None:
            template = templates.get('criterion_value', "Criterion: {value} ({criterion_type})")
            parts.append(template.format(value=criterion, criterion_type=criterion_type))
        show_weighting = self.params.get('map', {}).get('show_weighting', False)
        if weighting and show_weighting:
            # The weighting scheme is internal/technical detail most viewers
            # don't need — hidden by default (map.show_weighting). Turning it
            # on reveals "standard"/"binary", and show_weighting_detail further
            # reveals whether PSCF's tiers are multiples of the grid average
            # vs CWT's fixed endpoint-count tiers.
            show_detail = self.params.get('map', {}).get('show_weighting_detail', False)
            if show_detail:
                template = templates.get('weighting_detailed', "Weighting: {weighting} ({comparison} thresholds)")
            else:
                template = templates.get('weighting', "Weighting: {weighting}")
            parts.append(template.format(weighting=weighting, comparison=comparison))

        separator = templates.get('separator', " &nbsp;|&nbsp; ")
        return separator.join(parts)

    def build_map(self, title=None, caption=None, zoom_start=None, method=None):
        """
        Build and return the folium Map (call .save(path) on the result
        yourself — this method doesn't write the HTML file, only builds it).

        Parameters:
        - title, caption, zoom_start: override the corresponding
          config.yaml map.* value for this call. Leave as None to use
          what's already in self.params (the normal case). title may contain
          "{species}"/"{method}" placeholders, filled in from analysis.species
          and method — so the title stays correct automatically when you
          change species or plot a different statistic, instead of needing to
          be updated by hand each time.
        - method: "pscf" or "cwt" — which statistic self.result actually is.
          Determines the title's {method} text, which legend_caption to use,
          and (via default_caption()) which weighting label to show. Leave as
          None to use analysis.method — but since self.result is just a plain
          array with no memory of which method produced it, pass this
          explicitly if you're building maps for both statistics in the same
          run (analysis.method would only be correct for one of the two calls).

        title_font_size, caption_font_size, and legend_scale aren't exposed as
        method arguments (config-only) — set them in config.yaml's map
        section if you want the title/caption text or the legend bigger, e.g.
        for a screenshot.
        """
        grid = self.grid
        map_cfg = self.params.get('map', {})
        method = method if method is not None else self.params.get('analysis', {}).get('method', 'pscf')

        title = title if title is not None else map_cfg.get('title', 'Multi-Site {method} — {species}')
        species = self.params.get('analysis', {}).get('species', '')
        title = title.format(method=method.upper(), species=species)

        zoom_start = zoom_start if zoom_start is not None else map_cfg.get('zoom_start', 7)

        # CWT is a magnitude (ppb), not a bounded probability — 0 isn't a
        # meaningful floor among the cells actually shown (especially with
        # 'binary' weighting, which hides low-n cells as NaN rather than
        # diluting them toward 0), so anchor the scale at the visible
        # minimum instead. PSCF keeps the default vmin=0 — 0 there is a
        # real, meaningful value (no chance of a source).
        result_vmin = np.nanmin(self.result) if method == 'cwt' else None
        vmin, vmax = self.render_overlay(vmin=result_vmin)

        m = folium.Map(
            location=[grid.lat_centers.mean(), grid.lon_centers.mean()],
            zoom_start=zoom_start,
            tiles='CartoDB positron',
            control_scale=True,  # adds a metric+imperial distance scale bar (bottom-left)
        )

        grid_bounds = [[grid.lat_edges[0], grid.lon_edges[0]], [grid.lat_edges[-1], grid.lon_edges[-1]]]

        # Georeferenced overlay — bounds match the grid's true edges so the
        # image lines up cell-for-cell with the underlying map's real scale.
        # Named + always visible, so it still shows up in the LayerControl
        # toggle list below (consistent with the CI overlay, not just it).
        main_layer_name = f"{method.upper()} result"
        folium.raster_layers.ImageOverlay(
            image=self.overlay_path,
            bounds=grid_bounds,
            name=main_layer_name,
            show=True,
        ).add_to(m)

        # Optional toggleable overlay: Wilson CI width (upper_ci - lower_ci),
        # a per-cell uncertainty-from-sample-size map (see
        # uncertainty_dispersion_notes.md §2). Only added if PSCF.pscf()'s CI
        # bounds were actually passed in — CWT has no equivalent yet, so this
        # stays absent for CWT maps. Hidden by default (show=False); the
        # LayerControl checkbox lets the viewer reveal it on demand instead of
        # cluttering the default view with a second overlay.
        ci_layer_name = "Sample size uncertainty (CI width)"
        ci_vmin = ci_vmax = None
        if self.lower_ci is not None and self.upper_ci is not None:
            ci_width = self.upper_ci - self.lower_ci
            ci_overlay_path = os.path.join(os.path.dirname(self.overlay_path), 'ci_width_overlay.png')
            ci_vmin, ci_vmax = self.render_overlay(data=ci_width, path=ci_overlay_path)  # own scale — CI width isn't on the same range as the main result
            folium.raster_layers.ImageOverlay(
                image=ci_overlay_path,
                bounds=grid_bounds,
                name=ci_layer_name,
                show=False,
            ).add_to(m)

        # Optional toggleable overlay: per-cell significance z-score vs. the
        # network baseline rate (see PSCF.cell_significance_zscore()) — a
        # different question from the CI-width layer above ("is this cell
        # unusual" vs. "how precisely do we know this cell's own rate").
        # symmetric=True since z-scores are meaningfully negative (below
        # baseline) as well as positive (above it), unlike the other two
        # overlays which are naturally non-negative.
        z_layer_name = "Significance (z-score vs. baseline)"
        z_vmin = z_vmax = None
        if self.z_score is not None:
            z_overlay_path = os.path.join(os.path.dirname(self.overlay_path), 'z_score_overlay.png')
            z_vmin, z_vmax = self.render_overlay(data=self.z_score, path=z_overlay_path, symmetric=True)
            folium.raster_layers.ImageOverlay(
                image=z_overlay_path,
                bounds=grid_bounds,
                name=z_layer_name,
                show=False,
            ).add_to(m)

        # Analysis site markers, labeled by site_id
        for site_id, (lat, lon) in self.get_site_locations().items():
            folium.Marker(
                [lat, lon],
                tooltip=site_id,
                icon=folium.Icon(color='gray', icon='info-sign'),
            ).add_to(m)

        # Reference city labels — plain text, always visible (not a hover
        # tooltip), for geographic context in screenshots. White text-shadow
        # outline keeps them legible over both the base map and the colored
        # PSCF overlay without needing an opaque background box that would
        # cover part of the heatmap. Added after the ImageOverlay above, so
        # it renders on top by Leaflet's normal layer-order stacking.
        city_font_size = map_cfg.get('city_label_font_size', 14)
        for name, coords in self.params.get('reference_cities', {}).items():
            folium.Marker(
                location=[coords['lat'], coords['lon']],
                icon=folium.DivIcon(html=f'''
                    <div style="font-size: {city_font_size}px; font-weight: bold; color: black;
                                font-family: sans-serif; white-space: nowrap;
                                text-shadow: 1px 1px 2px white, -1px -1px 2px white,
                                             1px -1px 2px white, -1px 1px 2px white;">
                        {name}
                    </div>
                '''),
            ).add_to(m)

        # Legend — hand-built HTML/CSS gradient bar instead of branca's
        # LinearColormap. Confirmed (from actual generated output) that
        # branca's per-instance script always does
        # `d3.select(".legend.leaflet-control").append("svg").attr("id", 'legend')`
        # — a CSS-class selector, not a reference scoped to that specific
        # instance. `d3.select()` (singular) only ever grabs the FIRST
        # matching element on the page, and the appended svg's own id
        # collides too, so with two LinearColormaps on one map, both
        # legends' content ends up appended into the SAME first container
        # instead of staying separate — which is why toggling one legend's
        # container previously hid/showed both at once. Building our own
        # divs with unique ids sidesteps the collision entirely, and as a
        # bonus we're no longer stuck with branca's hardcoded 'topright'.
        jet_hex = [mcolors.to_hex(plt.get_cmap('jet')(x)) for x in np.linspace(0, 1, 13)]
        cwt_statistic = self.params.get('pscf', {}).get('cwt_statistic', 'mean')
        default_legend_captions = {'pscf': 'PSCF probability', 'cwt': f'{cwt_statistic.capitalize()} (ppb)'}
        legend_caption = map_cfg.get('legend_caption', {}).get(method) or default_legend_captions.get(method, 'Result')

        legend_scale = map_cfg.get('legend_scale', 1.8)
        legend_font_size = round(14 * legend_scale)
        legend_bar_w = round(200 * legend_scale)
        legend_bar_h = round(20 * legend_scale)

        def _gradient_legend_html(element_id, vmax_value, caption_text, visible, vmin_value=0):
            gradient = ', '.join(jet_hex)
            ticks = [vmin_value, (vmin_value + vmax_value) / 2, vmax_value]
            tick_spans = ''.join(
                f'<span style="flex: 1; text-align: {"left" if i == 0 else "right" if i == len(ticks) - 1 else "center"};">{t:.2f}</span>'
                for i, t in enumerate(ticks)
            )
            display = '' if visible else 'none'
            return f'''
                <div id="{element_id}" style="position: fixed; top: 10px; right: 10px; z-index: 9999;
                            background-color: white; border: 3px solid grey; border-radius: 6px;
                            padding: 10px 14px; font-family: sans-serif; font-size: {legend_font_size}px;
                            display: {display};">
                    <div style="margin-bottom: 6px;">{caption_text}</div>
                    <div style="width: {legend_bar_w}px; height: {legend_bar_h}px;
                                background: linear-gradient(to right, {gradient});
                                border: 1px solid black;"></div>
                    <div style="display: flex; width: {legend_bar_w}px; margin-top: 2px;">{tick_spans}</div>
                </div>
            '''

        main_legend_id = 'main-legend-box'
        m.get_root().html.add_child(folium.Element(
            _gradient_legend_html(main_legend_id, vmax, legend_caption, visible=True, vmin_value=vmin)
        ))

        # Second legend for the CI-width overlay — genuinely different scale
        # from the main result (not just different text), since CI width and
        # PSCF probability aren't the same quantity even though both happen to
        # fall in a similar numeric range.
        ci_legend_id = None
        if ci_vmax is not None:
            ci_legend_id = 'ci-legend-box'
            m.get_root().html.add_child(folium.Element(
                _gradient_legend_html(ci_legend_id, ci_vmax, "Sample size uncertainty (CI width)", visible=False)
            ))

        # Third legend for the z-score overlay — vmin_value passed explicitly
        # since z-score's scale is centered at 0 (symmetric), not the default
        # vmin=0 the other two legends use.
        z_legend_id = None
        if z_vmax is not None:
            z_legend_id = 'z-legend-box'
            m.get_root().html.add_child(folium.Element(
                _gradient_legend_html(z_legend_id, z_vmax, "Significance (z-score vs. baseline)", visible=False, vmin_value=z_vmin)
            ))


        # Leaflet's built-in scale bar is a fixed-size control with no
        # font/size parameters of its own — scaling it via CSS transform
        # would distort its actual distance meaning (see note below), so
        # font-size is used instead. It's pulled out of Leaflet's own
        # bottom-left control container via `position: fixed` (same
        # technique the title banner and legend boxes use) and given a
        # white background + border box to match their look. The legend
        # boxes above already bake `legend_scale` into their own font/bar
        # dimensions directly, so no CSS transform is needed for them either.
        #
        # Note: the scale bar's horizontal centering below uses a fixed
        # margin-left estimate (half its typical rendered width), not a true
        # dynamic center — good enough visually, but worth nudging by hand via
        # map.scale_bar_scale/CSS if it looks off-center at your map's actual
        # width once rendered.
        #
        # Moved to the bottom-right corner instead of trying to fit it into
        # the crowded title/legend row at the top — sidesteps the overlap
        # problem entirely rather than calculating around it. One thing to
        # leave clearance for: Leaflet's default attribution text ("Leaflet |
        # (c) OpenStreetMap contributors") also lives in the bottom-right by
        # default, so this sits above it (bottom: 30px) rather than at the
        # very corner.
        # IMPORTANT: the scale bar must NOT be enlarged with `transform:
        # scale()` the way the legend is. Leaflet computes the bar's pixel
        # width to accurately represent a round real-world distance at the
        # current zoom — visually stretching that box afterward via transform
        # distorts the one thing the element exists to get right (its length
        # no longer matches its own "X km" label). Legibility here comes from
        # font-size instead, which enlarges the text without touching the
        # bar's actual (accurate) length.
        legend_scale = map_cfg.get('legend_scale', 1.8)
        scale_bar_font_size = map_cfg.get('scale_bar_font_size', 16)
        control_style_html = f'''
            <style>
                .leaflet-control-scale {{
                    position: fixed !important;
                    bottom: 30px;
                    right: 10px;
                    top: auto;
                    left: auto;
                    background-color: white;
                    border: 3px solid grey;
                    border-radius: 6px;
                    padding: 6px 12px;
                    z-index: 9999;
                }}
                .leaflet-control-scale-line {{
                    font-size: {scale_bar_font_size}px !important;
                    border-width: 3px !important;
                }}
            </style>
        '''
        m.get_root().html.add_child(folium.Element(control_style_html))

        # Title + supplemental caption — folium has no native title element,
        # so render one as a floating HTML banner instead.
        caption = caption if caption is not None else (map_cfg.get('caption') or self.default_caption(method=method))
        title_font_size = map_cfg.get('title_font_size', 32)
        caption_font_size = map_cfg.get('caption_font_size', 18)
        title_html = f'''
            <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                        background-color: white; padding: 16px 24px; border: 3px solid grey;
                        border-radius: 6px; font-size: {title_font_size}px; font-weight: bold;
                        font-family: sans-serif;">
                {title}
                <div style="font-size: {caption_font_size}px; font-weight: normal; margin-top: 6px;">
                    {caption}
                </div>
            </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Makes the named/toggleable overlays above (main result + optional CI
        # width) actually show up as checkboxes on the map — without this,
        # `name=`/`show=` on the ImageOverlays has no visible UI effect at all.
        folium.LayerControl(position='bottomleft', collapsed=False).add_to(m)

        # Ties each legend's visibility to its own layer's checkbox, instead
        # of both legends sitting stacked/always-visible in the same corner.
        # Each legend box above has its own unique `id`, so this references
        # them directly via getElementById — no ambiguity, unlike branca's
        # shared-selector-based container lookups.
        if ci_legend_id is not None:
            ci_toggle_add = f'if (e.name === "{ci_layer_name}") {{ ciLegendEl.style.display = ""; }}'
            ci_toggle_remove = f'if (e.name === "{ci_layer_name}") {{ ciLegendEl.style.display = "none"; }}'
            ci_js_var = f"var ciLegendEl = document.getElementById('{ci_legend_id}');"
        else:
            ci_js_var = ''
            ci_toggle_add = ''
            ci_toggle_remove = ''

        if z_legend_id is not None:
            z_toggle_add = f'if (e.name === "{z_layer_name}") {{ zLegendEl.style.display = ""; }}'
            z_toggle_remove = f'if (e.name === "{z_layer_name}") {{ zLegendEl.style.display = "none"; }}'
            z_js_var = f"var zLegendEl = document.getElementById('{z_legend_id}');"
        else:
            z_js_var = ''
            z_toggle_add = ''
            z_toggle_remove = ''

        map_var = m.get_name()
        legend_toggle_js = f'''
            <script>
                window.addEventListener('load', function() {{
                    var mainLegendEl = document.getElementById('{main_legend_id}');
                    {ci_js_var}
                    {z_js_var}

                    {map_var}.on('overlayadd', function(e) {{
                        if (e.name === "{main_layer_name}") {{ mainLegendEl.style.display = ""; }}
                        {ci_toggle_add}
                        {z_toggle_add}
                    }});

                    {map_var}.on('overlayremove', function(e) {{
                        if (e.name === "{main_layer_name}") {{ mainLegendEl.style.display = "none"; }}
                        {ci_toggle_remove}
                        {z_toggle_remove}
                    }});
                }});
            </script>
        '''
        m.get_root().html.add_child(folium.Element(legend_toggle_js))

        return m
