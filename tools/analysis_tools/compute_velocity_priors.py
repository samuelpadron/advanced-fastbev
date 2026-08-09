"""
Compute per-class ground-truth speed percentiles from the nuScenes training
annotations, to derive a data-driven class_max_vel prior for
VelocityPlausibilityLoss, instead of hand-picked or grid-searched values.

Usage:
    python compute_velocity_priors.py \
        --ann-file data/nuscenes/bevdetv3-nuscenes_infos_train.pkl \
        --percentile 99.5

Notes:
- Reads the same train info pkl at 'data_root + 'bevdetv3-nuscenes_infos_train.pkl'.
- gt_velocity in mmdet3d/BEVDet-style nuScenes infos is (N, 2) [vx, vy] in
  m/s, in the global/ego frame. Some entries are NaN when the adjacent
  sample needed to estimate velocity wasn't available (e.g. first/last
  frame in a scene) -- these are dropped before computing percentiles.
- Speed = sqrt(vx^2 + vy^2). Reports several percentiles per class so you
  can see the shape of the distribution.
- Prints a class_max_vel dict pre-formatted to paste directly into
  config, using --percentile as the cutoff, keyed by class index in the
  same order as `class_names` in configs.
"""

import argparse
import pickle

import numpy as np

# Must match `class_names` in the FastBEV configs exactly, in order,
# since class_max_vel in the loss config is keyed by class index.
CLASS_NAMES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

PERCENTILES_TO_REPORT = [50, 90, 95, 99, 99.5, 99.9, 100]


def load_infos(ann_file):
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    # mmdet3d nuscenes pkl format: top-level dict with an 'infos' list,
    # or (older format) a bare list of infos. Handle both.
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise ValueError(
        f"Unrecognized pkl structure in {ann_file}: expected a dict with "
        f"an 'infos' key or a bare list, got {type(data)}"
    )


def collect_speeds_by_class(infos):
    """Returns dict: class_name -> 1D np.array of speeds (m/s), NaNs dropped."""
    speeds_by_class = {name: [] for name in CLASS_NAMES}

    missing_fields_warned = False
    for info in infos:
        gt_names = info.get('gt_names')
        gt_velocity = info.get('gt_velocity')

        if gt_names is None or gt_velocity is None:
            if not missing_fields_warned:
                print("WARNING: some info entries are missing 'gt_names' "
                      "or 'gt_velocity'; skipping those. If this applies "
                      "to ALL entries, check your pkl's actual key names "
                      "(they can differ slightly across mmdet3d/BEVDet "
                      "fork versions -- try info.keys() to inspect).")
                missing_fields_warned = True
            continue

        gt_velocity = np.asarray(gt_velocity, dtype=np.float64)
        speed = np.linalg.norm(gt_velocity, axis=1)  # (N,)

        for name, spd in zip(gt_names, speed):
            if name not in speeds_by_class:
                continue  # e.g. 'ignore' or DontCare-type labels, skip
            if np.isnan(spd):
                continue
            speeds_by_class[name].append(spd)

    return {name: np.array(v) for name, v in speeds_by_class.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ann-file', required=True,
        help="Path to the train info pkl, e.g. "
             "data/nuscenes/bevdetv3-nuscenes_infos_train.pkl")
    parser.add_argument(
        '--percentile', type=float, default=99.5,
        help="Percentile to use as the class_max_vel cutoff (default: 99.5)")
    args = parser.parse_args()

    print(f"Loading {args.ann_file} ...")
    infos = load_infos(args.ann_file)
    print(f"Loaded {len(infos)} samples.\n")

    speeds_by_class = collect_speeds_by_class(infos)

    print("Per-class speed distribution (m/s), NaN entries dropped:\n")
    header = "class".ljust(22) + "n".rjust(8) + "".join(
        f"p{p}".rjust(9) for p in PERCENTILES_TO_REPORT)
    print(header)
    print("-" * len(header))

    class_max_vel = {}
    for idx, name in enumerate(CLASS_NAMES):
        speeds = speeds_by_class[name]
        if len(speeds) == 0:
            print(f"{name.ljust(22)}{'0'.rjust(8)}  (no valid samples, "
                  f"check class name spelling / dataset)")
            class_max_vel[idx] = None
            continue

        pct_values = np.percentile(speeds, PERCENTILES_TO_REPORT)
        row = name.ljust(22) + str(len(speeds)).rjust(8) + "".join(
            f"{v:.2f}".rjust(9) for v in pct_values)
        print(row)

        cutoff = float(np.percentile(speeds, args.percentile))
        class_max_vel[idx] = round(cutoff, 1)

    print(f"\nclass_max_vel at p{args.percentile}, ready to paste into your "
          f"config's loss_vel_plausibility:\n")
    print("class_max_vel={")
    for idx, name in enumerate(CLASS_NAMES):
        val = class_max_vel[idx]
        val_str = f"{val}" if val is not None else "None  # FIX: no data"
        print(f"    {idx}: {val_str},  # {name}")
    print("},")


if __name__ == '__main__':
    main()