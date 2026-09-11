if __name__ == "__main__":

    import os
    import glob
    import time
    import yaml
    import numpy as np
    import project_specific_functions as psf
    from pscf import SpatialGrid, PSCF, TrajectoryDataset, ResultMap

    with open('config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    timings = {}
    total_start = time.perf_counter()

    # --- Pollution data (both networks, whatever species is set in config.yaml) ---
    t0 = time.perf_counter()
    pollution_data = psf.cdphe_ba_combine(data_config)
    timings['data_readin'] = time.perf_counter() - t0
    print(f"Pollution data: {len(pollution_data)} rows, sites: {sorted(pollution_data['site_id'].unique())}")
    print(f"  ({timings['data_readin']:.2f}s)")

    # --- Trajectory data — scoped to sites relevant to the current species,
    # not the full multi-year archive ---
    t0 = time.perf_counter()
    traj_paths = psf.read_in_traj_for_active_sites(data_config)
    print(f"Trajectory subset: {len(traj_paths)} files")
    trajectory_data = psf.read_trajectory_file(traj_paths)
    timings['trajectory_readin'] = time.perf_counter() - t0
    print(f"Trajectory data: {len(trajectory_data)} rows, sites: {sorted(trajectory_data['site_id'].unique())}")
    print(f"  ({timings['trajectory_readin']:.2f}s)")

    # --- Assemble, validate, and flag (TrajectoryDataset runs its full
    # pipeline as part of construction: data_checker -> build_analysis_dataset
    # -> sanity_check), then Grid + PSCF ---
    t0 = time.perf_counter()
    dataset = TrajectoryDataset(data_config, pollution_data, trajectory_data)
    grid = SpatialGrid(data_config)
    pscf = PSCF(grid, dataset.data, data_config)

    method = data_config['analysis']['method']
    if method == 'cwt':
        result = pscf.cwt()
        ci_low = ci_high = z_score = None   # .cwt() has no CI/z-score equivalent
    else:
        result, ci_low, ci_high, z_score = pscf.pscf()
    timings['pscf_processing'] = time.perf_counter() - t0
    print(f"Assembled analysis dataset: {len(dataset.data)} rows")
    print(f"Criterion values: {dataset.criterion_dict}")
    print(f"Result grid shape: {result.shape}")
    print(f"Non-zero cells: {(result > 0).sum()} / {result.size}")
    print(f"Max {method.upper()} value: {np.nanmax(result):.3f}")
    print(f"  ({timings['pscf_processing']:.2f}s)")

    # --- Map ---
    t0 = time.perf_counter()
    rmap = ResultMap(grid, result, dataset, data_config,  lower_ci=ci_low, upper_ci=ci_high, z_score=z_score)
    m = rmap.build_map()

    # Filename encodes the parameters that actually change the result, so
    # runs with different settings land in separate files instead of
    # overwriting a single test_pscf_result.html — lets multiple parameter
    # sweeps be compared/stacked side by side afterward.
    species = data_config['analysis']['species']
    pscf_cfg = data_config['pscf']
    weight_cfg = pscf_cfg['weight_function'][method]
    criterion_str = f"{pscf_cfg['criterion_type']}-{pscf_cfg['criterion']}".replace('.', 'p')
    name_parts = [species, method, criterion_str, weight_cfg['weighting']]
    if weight_cfg['weighting'] in ('binary', 'ss_scaled_balanced'):
        name_parts.append(f"k{weight_cfg.get('weighting_value')}")
    output_filename = "_".join(str(p) for p in name_parts) + ".html"

    output_path = os.path.join(data_config['paths']['output_dir'], output_filename)
    m.save(output_path)
    timings['plotting'] = time.perf_counter() - t0
    print(f"Saved map to {output_path}")
    print(f"  ({timings['plotting']:.2f}s)")

    timings['total'] = time.perf_counter() - total_start

    print("\n--- Timing summary ---")
    for stage, seconds in timings.items():
        print(f"{stage:20s}: {seconds:8.2f}s")
