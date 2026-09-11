import pandas as pd
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

with open('config.yaml', 'r') as f:
    data_config = yaml.safe_load(f)

def read_cdphe_data(file_path):
    """
    Reads a single CDPHE/EPA AQS CSV and returns it in the unified schema:
        site_id (str), timestamp_utc (datetime64, UTC),
        species (str), ppb (float64)

    Site and species are inferred from the filename:
        "FTC_O3_data.csv"   -> site_id="FTC",  species="O3"
        "CASA_NO2_data.csv" -> site_id="CASA", species="NO2"

    Expects the standard AQS column set including date_gmt, time_gmt,
    and sample_measurement (ppm).
    """
    fname = os.path.basename(file_path)    # e.g. "FTC_O3_data.csv"
    parts = fname.split('_')               # ["FTC", "O3", "data.csv"]
    site_id = parts[0]
    species = parts[1]

    df = pd.read_csv(file_path)

    # UTC timestamp — date_gmt / time_gmt are already UTC
    df['timestamp_utc'] = pd.to_datetime(
        df['date_gmt'] + ' ' + df['time_gmt'], utc=True
    )

    # ppm -> ppb: AQS reports O3 in ppm, but NO/NO2/NOx directly in ppb
    # already (confirmed against units_of_measure in the raw CDPHE files) —
    # only O3 needs the factor applied.
    if species.upper() == 'O3':
        df['sample_measurement'] = df['sample_measurement'] * 1000
    df['ppb'] = df['sample_measurement']

    # drop missing measurements
    df = df[df['ppb'].notna()].copy()

    # average across POCs (handles collocated instruments e.g. ROMO O3)
    df = df.groupby('timestamp_utc', as_index=False)['ppb'].mean()

    df['site_id'] = site_id
    df['species'] = species

    return (
        df[['site_id', 'timestamp_utc', 'species', 'ppb']]
        .sort_values('timestamp_utc')
        .reset_index(drop=True)
    )


def read_ba_data(species_dir, species_col=None):
    """
    Reads all BA CSVs from a single site/species directory and returns the
    unified schema:
        site_id (str), timestamp_utc (datetime64, UTC),
        species (str), ppb (float64)

    species_dir      : path to a site/species subdirectory,
                    e.g. "{ba_data_dir}/BRZ/ozone"
    species_col      : column to extract. If None, inferred from the metadata
                    row — works automatically for ozone. Must be specified
                    for NOx ("no", "nox") and VOC ("ethane", etc.) files
                    since those contain multiple compound columns.

    Handles file format automatically:
    - Standard BA format: metadata row ("o3,tz=0,v=1.1"), Unix epoch timestamp
    """

    site_id = os.path.basename(os.path.dirname(species_dir))

    dfs = []
    inferred_col = None

    for fname in sorted(os.listdir(species_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(species_dir, fname)

        with open(fpath) as f:
            first_line = f.readline().strip()

        # standard BA format: row 0 = metadata, row 1 = column names
        meta_col = first_line.split(',')[0]   # e.g. "o3", "nox", "voc"
        if inferred_col is None:
            inferred_col = meta_col
        df = pd.read_csv(fpath, header=1)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        dfs.append(df)

    col = species_col if species_col is not None else inferred_col

    col_dfs = [df[['time', col]] for df in dfs if col in df.columns]
    data = pd.concat(col_dfs, ignore_index=True)

    data = data.sort_values('time').drop_duplicates(subset='time')

    # resample minute -> hourly mean, drop hours with no data
    if 'voc' not in species_dir: 
        data = (
            data.set_index('time')[col]
            .resample('1h')
            .mean()
            .dropna()
            .reset_index()
            .rename(columns={'time': 'timestamp_utc', col: 'ppb'})
        )
    else:
        data['time'] = data['time'].dt.round('1h')
        data = (
        data.set_index('time')[col]
        .groupby('time')
        .mean()
        .dropna()
        .reset_index()
        .rename(columns={'time': 'timestamp_utc', col: 'ppb'})
        )

    data['site_id'] = site_id
    data['species'] = col.upper()

    return (
        data[['site_id', 'timestamp_utc', 'species', 'ppb']]
        .reset_index(drop=True)
    )

def species_to_cdphe_sites(species):
    # Returns the list of CDPHE site IDs (from data_config) that measure a given species.
    # Used by cdphe_read_in to know which site files to load.
    sites_mapping = {
        'o3':  data_config['cdphe']['o3_sites'],
        'no2': data_config['cdphe']['no2_sites'],
        'no':  data_config['cdphe']['no_sites'],
        'nox': data_config['cdphe']['nox_sites'],
    }
    return sites_mapping.get(species.lower(), [])


def cdphe_read_in(data_config):
    # Top-level loader for all CDPHE data for the configured species.
    # Iterates over all sites that measure that species and concatenates them
    # into one unified DataFrame in the standard schema.
    #
    # NOx is a special case: AQS doesn't report NOx directly, so it is computed
    # by reading NO and NO2 separately for each site and summing them.
    species = data_config['analysis']['species']
    cdphe_dir = data_config['paths']['cdphe_data_dir']
    pattern = data_config['cdphe']['file_pattern']

    if species.upper() == 'NOX':
        # NOx is not a direct AQS output — compute as NO + NO2 per site
        dfs = []
        for site in species_to_cdphe_sites('nox'):
            no_path  = os.path.join(cdphe_dir, pattern.format(site=site, species='NO'))
            no2_path = os.path.join(cdphe_dir, pattern.format(site=site, species='NO2'))
            no_df  = read_cdphe_data(no_path).rename(columns={'ppb': 'no_ppb'}).drop(columns='species')
            no2_df = read_cdphe_data(no2_path).rename(columns={'ppb': 'no2_ppb'}).drop(columns='species')
            # inner join keeps only hours where both NO and NO2 are present
            merged = no_df.merge(no2_df, on=['site_id', 'timestamp_utc'], how='inner')
            merged['ppb'] = merged['no_ppb'] + merged['no2_ppb']
            merged['species'] = 'NOX'
            dfs.append(merged[['site_id', 'timestamp_utc', 'species', 'ppb']])
        return pd.concat(dfs, ignore_index=True)

    else:
        dfs = []
        for site in species_to_cdphe_sites(species):
            file_path = os.path.join(cdphe_dir, pattern.format(site=site, species=species))
            dfs.append(read_cdphe_data(file_path))
        if not dfs:
            # VOC compounds (and anything else with no CDPHE sites configured)
            # have no CDPHE equivalent — BA is the only source, so return an
            # empty frame in the standard schema instead of failing on
            # pd.concat([]).
            return pd.DataFrame(columns=['site_id', 'timestamp_utc', 'species', 'ppb'])
        return pd.concat(dfs, ignore_index=True)
    
def ba_read_in(data_config):
    # Top-level loader for all Boulder Air data for the configured species.
    # Looks up per-site config (subdirectory, column name, UTC offset) from
    # data_config, then calls read_ba_data for each site and concatenates results.
    #
    # VOC compounds (e.g. "ethane", "propane") are not top-level keys in
    # species_config — they fall back to the "voc" entry, using the compound
    # name itself as the column to extract.
    species = data_config['analysis']['species']
    ba_dir = data_config['paths']['ba_data_dir']
    species_cfg = data_config['ba']['species_config']

    # VOC compounds (ethane, propane, etc.) aren't top-level keys — fall back to "voc"
    if species in species_cfg:
        cfg = species_cfg[species]
        col = cfg.get('col')          # None -> auto-infer inside read_ba_data
    else:
        cfg = species_cfg['voc']
        col = species.lower()         # compound name IS the column (e.g. "ethane")

    default_subdir = cfg['subdir']
    sites_cfg = cfg.get('sites', {})

    dfs = []
    for site, overrides in sites_cfg.items():
        # Per-site overrides allow different subdirs, column names, or UTC offsets
        subdir = overrides.get('subdir', default_subdir)
        site_col = overrides.get('col', col)
        #utc_offset = overrides.get('utc_offset_hours', 0)
        species_dir = os.path.join(ba_dir, site, subdir)
        dfs.append(read_ba_data(species_dir, species_col=site_col))

    return pd.concat(dfs, ignore_index=True)

def species_to_cdphe_sites(species):
    # Returns the list of CDPHE site IDs (from data_config) that measure a given species.
    # Used by cdphe_read_in to know which site files to load.
    sites_mapping = {
        'o3':  data_config['cdphe']['o3_sites'],
        'no2': data_config['cdphe']['no2_sites'],
        'no':  data_config['cdphe']['no_sites'],
        'nox': data_config['cdphe']['nox_sites'],
    }
    return sites_mapping.get(species.lower(), [])


def cdphe_read_in(data_config):
    # Top-level loader for all CDPHE data for the configured species.
    # Iterates over all sites that measure that species and concatenates them
    # into one unified DataFrame in the standard schema.
    #
    # NOx is a special case: AQS doesn't report NOx directly, so it is computed
    # by reading NO and NO2 separately for each site and summing them.
    species = data_config['analysis']['species']
    cdphe_dir = data_config['paths']['cdphe_data_dir']
    pattern = data_config['cdphe']['file_pattern']

    if species.upper() == 'NOX':
        # NOx is not a direct AQS output — compute as NO + NO2 per site
        dfs = []
        for site in species_to_cdphe_sites('nox'):
            no_path  = os.path.join(cdphe_dir, pattern.format(site=site, species='NO'))
            no2_path = os.path.join(cdphe_dir, pattern.format(site=site, species='NO2'))
            no_df  = read_cdphe_data(no_path).rename(columns={'ppb': 'no_ppb'}).drop(columns='species')
            no2_df = read_cdphe_data(no2_path).rename(columns={'ppb': 'no2_ppb'}).drop(columns='species')
            # inner join keeps only hours where both NO and NO2 are present
            merged = no_df.merge(no2_df, on=['site_id', 'timestamp_utc'], how='inner')
            merged['ppb'] = merged['no_ppb'] + merged['no2_ppb']
            merged['species'] = 'NOX'
            dfs.append(merged[['site_id', 'timestamp_utc', 'species', 'ppb']])
        return pd.concat(dfs, ignore_index=True)

    else:
        dfs = []
        for site in species_to_cdphe_sites(species):
            file_path = os.path.join(cdphe_dir, pattern.format(site=site, species=species))
            dfs.append(read_cdphe_data(file_path))
        if not dfs:
            # VOC compounds (and anything else with no CDPHE sites configured)
            # have no CDPHE equivalent — BA is the only source, so return an
            # empty frame in the standard schema instead of failing on
            # pd.concat([]).
            return pd.DataFrame(columns=['site_id', 'timestamp_utc', 'species', 'ppb'])
        return pd.concat(dfs, ignore_index=True)
    
def _read_ba_site_df(ba_dir, cfg, site, overrides):
    # Resolves one site's subdir/column against its species_config entry
    # (with per-site overrides applied) and reads its data. Factored out so
    # ba_read_in's normal path and its NO2-derivation path (which reads two
    # species per site instead of one) share the exact same resolution logic.
    subdir = overrides.get('subdir', cfg['subdir'])
    col = overrides.get('col', cfg.get('col'))
    species_dir = os.path.join(ba_dir, site, subdir)
    return read_ba_data(species_dir, species_col=col)


def ba_read_in(data_config):
    # Top-level loader for all Boulder Air data for the configured species.
    # Looks up per-site config (subdirectory, column name, UTC offset) from
    # data_config, then calls read_ba_data for each site and concatenates results.
    #
    # VOC compounds (e.g. "ethane", "propane") are not top-level keys in
    # species_config — they fall back to the "voc" entry, using the compound
    # name itself as the column to extract.
    species = data_config['analysis']['species']
    ba_dir = data_config['paths']['ba_data_dir']
    species_cfg = data_config['ba']['species_config']

    if species == 'NO2':
        # Only CCF has a direct NO2 file — every other BA site measures NOx
        # and NO but not NO2 directly, so NO2 is derived there as NOx - NO
        # instead of silently dropping those sites from NO2 analysis.
        no2_cfg = species_cfg['NO2']
        nox_cfg = species_cfg['NOx']
        no_cfg = species_cfg['NO']
        direct_sites = set(no2_cfg.get('sites', {}).keys())
        no_sites_cfg = no_cfg.get('sites', {})

        dfs = []
        for site, overrides in no2_cfg.get('sites', {}).items():
            dfs.append(_read_ba_site_df(ba_dir, no2_cfg, site, overrides))

        for site, nox_overrides in nox_cfg.get('sites', {}).items():
            if site in direct_sites or site not in no_sites_cfg:
                continue  # already have direct NO2, or no NO to subtract here
            nox_df = _read_ba_site_df(ba_dir, nox_cfg, site, nox_overrides)
            no_df = _read_ba_site_df(ba_dir, no_cfg, site, no_sites_cfg[site])
            merged = nox_df.merge(no_df, on=['site_id', 'timestamp_utc'], how='inner', suffixes=('_nox', '_no'))
            merged['ppb'] = merged['ppb_nox'] - merged['ppb_no']
            merged['species'] = 'NO2'
            dfs.append(merged[['site_id', 'timestamp_utc', 'species', 'ppb']])

        return pd.concat(dfs, ignore_index=True)

    if species == 'ethane_nox_ratio':
        # Ratio metric for use with CWT (mean ratio per cell) instead of
        # either raw concentration alone — only defined at sites that
        # measure both ethane and NOx.
        voc_cfg = dict(species_cfg['voc'])
        voc_cfg['col'] = 'ethane'
        nox_cfg = species_cfg['NOx']
        voc_sites_cfg = voc_cfg.get('sites', {})
        nox_sites_cfg = nox_cfg.get('sites', {})
        shared_sites = set(voc_sites_cfg) & set(nox_sites_cfg)

        dfs = []
        for site in sorted(shared_sites):
            ethane_df = _read_ba_site_df(ba_dir, voc_cfg, site, voc_sites_cfg[site])
            nox_df = _read_ba_site_df(ba_dir, nox_cfg, site, nox_sites_cfg[site])
            merged = ethane_df.merge(nox_df, on=['site_id', 'timestamp_utc'], how='inner', suffixes=('_ethane', '_nox'))
            merged = merged[merged['ppb_nox'] != 0]  # avoid divide-by-zero
            merged['ppb'] = merged['ppb_ethane'] / merged['ppb_nox']
            merged['species'] = 'ETHANE_NOX_RATIO'
            dfs.append(merged[['site_id', 'timestamp_utc', 'species', 'ppb']])

        return pd.concat(dfs, ignore_index=True)

    # VOC compounds (ethane, propane, etc.) aren't top-level keys — fall back to "voc"
    if species in species_cfg:
        cfg = species_cfg[species]
    else:
        cfg = dict(species_cfg['voc'])
        cfg['col'] = species.lower()  # compound name IS the column (e.g. "ethane")

    dfs = []
    for site, overrides in cfg.get('sites', {}).items():
        dfs.append(_read_ba_site_df(ba_dir, cfg, site, overrides))

    return pd.concat(dfs, ignore_index=True)

def cdphe_ba_combine(data_config):
    cdphe_df = cdphe_read_in(data_config)
    ba_df = ba_read_in(data_config)
    return pd.concat([cdphe_df, ba_df], ignore_index=True)


def active_sites(data_config):
    # Returns the set of site_ids relevant to data_config['analysis']['species']
    # — the union of CDPHE and BA sites configured for that species. Same
    # site-selection logic cdphe_read_in/ba_read_in already use internally,
    # exposed here so trajectory reading can be scoped to just the sites that
    # will actually survive the pollution+trajectory merge for this species,
    # instead of reading every site's trajectory archive regardless of
    # whether it's relevant to what's currently configured.
    species = data_config['analysis']['species']

    cdphe_sites = set(species_to_cdphe_sites(species))

    species_cfg = data_config['ba']['species_config']
    if species == 'NO2':
        # Keep in sync with ba_read_in's NO2 branch: direct CCF data plus
        # any site with both NOx and NO configured (NO2 derived as NOx - NO).
        direct_sites = set(species_cfg['NO2'].get('sites', {}).keys())
        nox_sites = set(species_cfg['NOx'].get('sites', {}).keys())
        no_sites = set(species_cfg['NO'].get('sites', {}).keys())
        ba_sites = direct_sites | (nox_sites & no_sites)
    elif species == 'ethane_nox_ratio':
        # Keep in sync with ba_read_in's ratio branch: only sites with both.
        voc_sites = set(species_cfg['voc'].get('sites', {}).keys())
        nox_sites = set(species_cfg['NOx'].get('sites', {}).keys())
        ba_sites = voc_sites & nox_sites
    elif species in species_cfg:
        ba_sites = set(species_cfg[species].get('sites', {}).keys())
    else:
        ba_sites = set(species_cfg['voc'].get('sites', {}).keys())

    return cdphe_sites | ba_sites


def read_in_traj_for_active_sites(data_config):
    # Like read_in_traj, but only walks the trajectory directories for sites
    # relevant to the currently configured species (active_sites), across
    # trajectories.available_years — instead of walking the entire archive
    # regardless of which sites will actually be used. Skips (site, year)
    # folders that don't exist on disk (e.g. a site with no trajectory
    # archive of its own, like BOUR sharing BRZ's location) rather than
    # erroring.
    traj_dir = data_config['paths']['trajectory_dir']
    year_dir_prefix = data_config['trajectories']['year_dir_prefix']
    years = data_config['trajectories']['available_years']

    traj_paths = []
    for site in sorted(active_sites(data_config)):
        for year in years:
            site_dir = os.path.join(traj_dir, f'{year_dir_prefix}{year}', site)
            if not os.path.isdir(site_dir):
                continue
            for fname in os.listdir(site_dir):
                if not fname.startswith('.'):
                    traj_paths.append(os.path.join(site_dir, fname))

    return traj_paths

def data_check_plots(polution_data):
    for site, group in polution_data.groupby('site_id'):
        print(f"\n=== {site} ===")
        print(group['ppb'].describe())

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(group['timestamp_utc'], group['ppb'], linewidth=0.5)
        ax.set_title(site)
        ax.set_xlabel('time (UTC)')
        ax.set_ylabel('ppb')
        plt.tight_layout()
        plt.show()


def read_in_traj(data_config):

    traj_paths = []
    traj_dir = data_config['paths']['trajectory_dir']
    for dirpath, subdirs, files in os.walk(traj_dir):
        for fname in files:
            if not fname.startswith('.'):
                full_path = os.path.join(dirpath, fname)
                traj_paths.append(full_path)

    return traj_paths

traj_paths = read_in_traj(data_config)


# Fixed layout of a HYSPLIT single-trajectory endpoint file: this part of the
# format is invariant across runs, unlike the header line counts below (which
# depend on how many met grids/trajectories a given run used) and the trailing
# diagnostic variables (which depend on what outputs were requested).
def read_trajectory_file(traj_paths):
    CORE_ENDPOINT_COLUMNS = [
        'traj_num', 'met_grid', 'year', 'month', 'day', 'start_hour',
        'minute', 'fhour', 'traj_age', 'lat', 'lon', 'height',
    ]

    rows = []
    var_names = None

    for path in traj_paths:
        site_id = os.path.basename(path).split('_')[0]

        with open(path) as f:
            n_grids = int(f.readline().split()[0])
            for _ in range(n_grids):
                f.readline()

            n_traj = int(f.readline().split()[0])
            for _ in range(n_traj):
                f.readline()

            var_line = f.readline().split()
            if var_names is None:
                var_names = [v.lower() for v in var_line[1:]]

            # Buffer this file's rows separately so we can backfill the
            # timestamp (known only once we hit each trajectory's age==0 row)
            # without re-reading the file.
            file_rows = []
            t0_by_traj = {}
            for line in f:
                fields = line.split()
                file_rows.append(fields)
                if fields[8] == '0.0':
                    t0_by_traj[fields[0]] = fields[2:6]  # year, month, day, start_hour

        # One Timestamp per trajectory in this file (usually just one), not
        # one per row.
        timestamp_by_traj = {
            traj_num: pd.Timestamp(
                year=2000 + int(y), month=int(mo), day=int(d), hour=int(h), tz='UTC'
            )
            for traj_num, (y, mo, d, h) in t0_by_traj.items()
        }

        for fields in file_rows:
            rows.append([site_id, timestamp_by_traj[fields[0]]] + fields)

    names = ['site_id', 'timestamp_utc'] + CORE_ENDPOINT_COLUMNS + var_names
    traj_df = pd.DataFrame(rows, columns=names)

    numeric_cols = CORE_ENDPOINT_COLUMNS + var_names
    traj_df[numeric_cols] = traj_df[numeric_cols].astype(float)

    return traj_df