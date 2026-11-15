"""CPU-only hypothetical left handle tilt study; no Genesis or soft dynamics."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from contact_grasp import cloth_grasp_frame
from motion import Planner, TCP


def main():
    source = HERE / "config_guarded_contact_trial.json"
    source_bytes = source.read_bytes()
    config = json.loads(source_bytes)
    bag = np.load(HERE / "assets" / config["bag"]["npz_file"])
    origin = np.asarray(config["bag"]["position"])
    output = HERE / "runtime_ipc" / "left_tilt_candidates"
    rows = []
    candidates = {}
    baseline_palm = None
    for degrees in (35, 45, 50, 55):
        candidate = copy.deepcopy(config)
        angle = np.radians(degrees)
        candidate["task"]["handle_grasps"]["left"]["preferred_approach"] = [
            float(np.sin(angle)), 0., -float(np.cos(angle))]
        planner = Planner(candidate)
        frames, qs, apertures, initial_errors = {}, {}, {}, {}
        for side in ("left", "right"):
            spec = candidate["task"]["handle_grasps"][side]
            frame = cloth_grasp_frame(bag["points"], bag["faces"], bag[spec["group"]],
                shell_radius=candidate["bag"]["thickness"],
                preferred_approach=spec["preferred_approach"])
            frames[side] = frame
            q, error, rotation_error = planner.solve(side, origin + frame.position, frame.rotation)
            qs[side] = q
            apertures[side] = max(.001, frame.minimum_half_gap)
            initial_errors[side] = {"position_m": error, "rotation_rad": rotation_error}
        palm = origin + frames["left"].position - frames["left"].rotation.apply(TCP)
        if baseline_palm is None:
            baseline_palm = palm.copy()
        row = {
            "preferred_left_tilt_degrees": degrees,
            "left_frame": frames["left"].as_dict(),
            "right_frame": frames["right"].as_dict(),
            "initial_joint_positions": {s: q.tolist() for s, q in qs.items()},
            "initial_ik_errors": initial_errors,
            "initial_tcp_world_m": {s: (origin + f.position).tolist() for s, f in frames.items()},
            "initial_half_openings_m": apertures,
            "left_palm_origin_world_m": palm.tolist(),
            "left_palm_displacement_from_35deg_m": (palm - baseline_palm).tolist(),
            "initial_clearance": planner.clearance(qs, apertures),
        }
        try:
            if any(e["position_m"] > .0005 or e["rotation_rad"] > .002 for e in initial_errors.values()):
                raise ValueError("Initial IK exceeds 0.5mm / 0.002rad")
            if any(v["distance"] < .008 for v in row["initial_clearance"].values()):
                raise ValueError("Initial clearance below 8mm")
            # Actual fingers holding 0.6mm cloth are slightly open. Validate
            # both command-zero geometry and a 1.5mm-per-finger envelope.
            # This does not prove the intervening rigid distance minima.
            summaries = []
            for half_gap in (0., .0015):
                openings = dict.fromkeys(("left", "right"), half_gap)
                speed = float(candidate["task"].get("motion_speed_scale", 1.))
                plan = planner.trajectory(qs, candidate["task"]["open_targets"],
                    rotations={s: f.rotation for s, f in frames.items()},
                    duration=2.5/speed, finger_start=openings, finger_end=openings,
                    clearance_margin=.008, max_joint_velocity=.55*speed,
                    max_joint_acceleration=speed**2, validate_samples=101)
                minima = {}
                for time in np.linspace(0., plan.duration, 101):
                    state = plan.sample(float(time))
                    clearances = planner.clearance(state["q"], state["finger"])
                    for name, value in clearances.items():
                        if name not in minima or value["distance"] < minima[name]["distance"]:
                            minima[name] = {**value, "time_s": float(time)}
                summaries.append({"half_opening_m": half_gap, "duration_s": plan.duration,
                    "sample_count": 101, "minimum_clearance_by_category": minima,
                    "spread_joint_positions": {s: q.tolist() for s, q in plan.end_q().items()},
                    "spread_clearance": planner.clearance(plan.end_q(), openings)})
            row["spread_paths"] = summaries
            row["status"] = "hypothetical_robot_preflight_passed"
            if degrees > 35:
                candidates[degrees] = candidate
        except ValueError as error:
            row["status"] = "rejected"
            row["reason"] = str(error)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("preferred_left_tilt_degrees", "status",
            "left_palm_displacement_from_35deg_m")} | {"reason": row.get("reason")}), flush=True)
    if candidates:
        output.mkdir(exist_ok=True)
        chosen = min(candidates)
        report = {
            "status": "hypothetical_robot_preflight_only",
            "source_config": str(source),
            "source_config_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "source_config_at_test": config,
            "left_tilts_degrees_tested": [35, 45, 50, 55],
            "right_preferred_tilt_degrees": 35,
            "minimum_robot_clearance_m": .008,
            "smallest_tested_feasible_alternative_degrees": chosen,
            "rows": rows,
            "limits": ["No Genesis initialization or dynamics were run.",
                "Same initial left TCP and same spread targets, rest cloth and named material groups.",
                "101 time samples cover each path; joint extrema and rate limits are checked analytically.",
                "Finger-to-finger pairs within one hand are omitted by Planner; two alternative equal openings are evaluated separately.",
                "No inserted bear or deformed bag geometry was evaluated.",
                "A changed left tilt changes cloth loading and shape; these configurations are not dynamic insertion-route proof.",
                "Minimal tested feasible angle does not establish sufficient insertion clearance."]}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        for degrees, candidate in candidates.items():
            (output / f"config_left_tilt_{degrees}.json").write_text(json.dumps(candidate, indent=2))
        print(f"REPORT {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
