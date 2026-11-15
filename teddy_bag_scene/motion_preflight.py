"""CPU-only geometric validation; never claims a successful soft-body grasp.

Run with .venv/bin/python teddy_bag_scene/motion_preflight.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motion import DOWN, SIDES, Planner, default_config, nominal_waypoints


def preflight(config=None, output=None, metadata=None, *, rotations=None,
              initial_finger_opening=.04):
    config = (json.loads(Path(__file__).with_name("config.json").read_text())
              if config is None and Path(__file__).with_name("config.json").exists()
              else default_config() if config is None else config)
    planner = Planner(config)
    waypoints = nominal_waypoints(config, metadata)
    rotations = rotations or {}
    report = {
        "kind": "offline_robot_geometry_only",
        "config": config,
        "explicit_metadata_override": metadata is not None,
        "nominal_waypoints_m": waypoints,
        "rotation_overrides_xyzw": {stage: {side: r.as_quat().tolist()
            for side, r in arms.items()} for stage, arms in rotations.items()},
        "initial_finger_q_m": initial_finger_opening,
        "tcp_hand_local_m": [0., 0., .1029],
        "success": False,
        "limitations": [
            "MuJoCo convex collision geometry only; no Genesis dynamic validation in this report.",
            "Soft bag and teddy are not collision obstacles in this preflight: intended frictional contacts require runtime checking.",
            "The teddy must hang beneath the grasp before insertion; its fit and the unsupported bag mouth must be measured at runtime.",
            "Nominal grasp locations must be replaced with observed mesh points after settling.",
            "The local panda_tactile.xml extends each real rubber pad inward by 1.5 mm; thin cloth gripping still requires actual contact/friction verification.",
        ],
        "initial_q": {},
        "stages": [],
    }
    qs = {}
    for side in SIDES:
        rotation = rotations.get("initial", {}).get(side, DOWN)
        q, error, angle = planner.solve(side, waypoints["initial"][side], rotation)
        if error > .0005 or angle > .002:
            raise RuntimeError(f"Unreachable initial {side}: {error=}, {angle=}")
        qs[side] = q
        report["initial_q"][side] = q.tolist()
    initial_clearance = planner.clearance(qs, initial_finger_opening)
    report["initial_clearance"] = initial_clearance
    if min(v["distance"] for v in initial_clearance.values()) < .008:
        raise RuntimeError(f"Unsafe initial pose: {initial_clearance}")

    total_duration = 0.
    min_clearance = min(v["distance"] for v in initial_clearance.values())
    for name, targets in waypoints.items():
        if name == "initial":
            continue
        # Finger closure/opening is a separate zero-velocity arm hold stage.
        finger = {"left": .001, "right": .04 if name in
                  ("right_clear_handle", "teddy_pregrasp", "teddy_grasp", "right_retreat") else .001}
        trajectory = planner.trajectory(qs, targets, duration=3., samples=51,
            finger_start=finger, finger_end=finger, validate_samples=121,
            rotations=rotations.get(name))
        extrema = {"peak_joint_velocity_rad_s": 0., "peak_joint_acceleration_rad_s2": 0.,
                   "min_rigid_clearance_m": .2, "max_tcp_path_deviation_m": 0.}
        starts = {side: planner.fk(side, qs[side])[0] for side in targets}
        for t in np.linspace(0, trajectory.duration, 201):
            point = trajectory.sample(float(t))
            extrema["peak_joint_velocity_rad_s"] = max(extrema["peak_joint_velocity_rad_s"],
                max(float(np.abs(point["qd"][s]).max()) for s in SIDES))
            extrema["peak_joint_acceleration_rad_s2"] = max(extrema["peak_joint_acceleration_rad_s2"],
                max(float(np.abs(point["qdd"][s]).max()) for s in SIDES))
            checks = planner.clearance(point["q"], finger)
            closest = min(v["distance"] for v in checks.values())
            extrema["min_rigid_clearance_m"] = min(extrema["min_rigid_clearance_m"], closest)
            for side, end in targets.items():
                a, b = starts[side], np.asarray(end)
                actual, _ = planner.fk(side, point["q"][side])
                direction = b-a
                fraction = float(np.clip(np.dot(actual-a, direction) / max(np.dot(direction,direction), 1e-12), 0, 1))
                distance = float(np.linalg.norm(actual-(a+fraction*direction)))
                extrema["max_tcp_path_deviation_m"] = max(extrema["max_tcp_path_deviation_m"], distance)
        for endpoint in (0., trajectory.duration):
            sample = trajectory.sample(endpoint)
            for side in SIDES:
                if np.max(np.abs(sample["qd"][side])) > 1e-8 or np.max(np.abs(sample["qdd"][side])) > 1e-8:
                    raise RuntimeError("Trajectory endpoint is not C2 stationary")
        if extrema["peak_joint_velocity_rad_s"] > .551 or extrema["peak_joint_acceleration_rad_s2"] > 1.001:
            raise RuntimeError(f"Joint rate limit violation: {extrema}")
        report["stages"].append({"stage": name, "targets_m": targets,
                                  "duration_s": trajectory.duration, **extrema})
        total_duration += trajectory.duration
        min_clearance = min(min_clearance, extrema["min_rigid_clearance_m"])
        qs = trajectory.end_q()
    report["success"] = True
    report["total_motion_duration_s"] = total_duration
    report["minimum_rigid_clearance_m"] = min_clearance
    report["dynamic_task_success_verified"] = False
    if output is not None:
        Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--metadata", type=Path, help="Explicit candidate asset metadata for preflight before activating it.")
    parser.add_argument("--cloth-tangent", action="store_true", help="Use the runtime cloth-normal initial grasp frames and measured minimum pad gaps.")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("report.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    metadata = json.loads(args.metadata.read_text()) if args.metadata else None
    options={}
    frames={}
    if args.cloth_tangent:
        from contact_grasp import cloth_grasp_frame
        config=config or json.loads(Path(__file__).with_name("config.json").read_text())
        path=Path(config["bag"]["npz_file"])
        if not path.is_absolute(): path=Path(__file__).parent/"assets"/path
        with np.load(path,allow_pickle=False) as bag:
            for side,group in (("left","grasp_front"),("right","grasp_back")):
                frames[side]=cloth_grasp_frame(bag["points"],bag["faces"],bag[group],
                    shell_radius=config["bag"]["thickness"])
        rotations={side:frame.rotation for side,frame in frames.items()}
        options={"rotations":{"initial":rotations,"open_bag":rotations},
                 "initial_finger_opening":{side:max(.001,frame.minimum_half_gap) for side,frame in frames.items()}}
    result = preflight(config, None, metadata, **options)
    if frames:
        result["cloth_grasp_frames"]={side:frame.as_dict() for side,frame in frames.items()}
        result["orientation_source"]="contact_grasp.cloth_grasp_frame on configured rest bag asset"
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"success": result["success"], "report": str(args.output),
                      "min_rigid_clearance_m": result["minimum_rigid_clearance_m"],
                      "motion_duration_s": result["total_motion_duration_s"]}, indent=2))
