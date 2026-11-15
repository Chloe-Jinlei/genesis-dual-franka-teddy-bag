"""Bounded CPU hypotheses from full17's measured entry endpoint; never simulates."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_17"
sys.path.insert(0, str(RUN / "source"))
from motion import Planner, MotionPlan, CubicSpline, minimum_jerk
from contact_robot_geometry import check_translation_sweep
from staged_contact_insertion import _spatial_loop_geometry, left_hand_minimum_z
from validate import mouth_structure
from physical_monitor import tetra_volumes
from insertion_candidates import projected_inside

cfg = json.loads((RUN / "config.json").read_text())
rest = dict(np.load(RUN / "rest.npz"))
state = dict(np.load(RUN / "lower_teddy_to_rim.npz"))
planner = Planner(cfg)
original = Rotation.from_quat(json.loads((RUN / "guarded_contact_candidate.json").read_text())["route"]["rotation_xyzw"])
start = state["tcp_right"]
fingers = {s: float(np.mean(state["q_" + s][7:])) for s in ("left", "right")}
initial_q = {s: state["q_" + s][:7].copy() for s in ("left", "right")}
_, rim, _ = mouth_structure(rest["bag"], rest["bag_faces"], cfg)
polygon, _, spatial = _spatial_loop_geometry(state["bag"][rim], rest["bag"][rim], cfg["bag"]["thickness"])
weights = np.abs(tetra_volumes(rest["bear"], rest["bear_tets"]))
com = np.average(state["bear"][rest["bear_tets"]].mean(axis=1), axis=0, weights=weights)
left_hand_z = left_hand_minimum_z(state["q_left"], cfg, planner=planner)


def checked_path(qs, target, half_openings, duration):
    paths = planner.plan_cartesian(qs, {"right": target}, rotations={"right": original}, samples=45)
    fractions = np.asarray([e["fraction"] for e in paths])
    curves = {s: CubicSpline(fractions, np.asarray([e["q"].get(s, qs[s]) for e in paths]), axis=0)
              for s in ("left", "right")}
    plan = MotionPlan(curves, duration, half_openings, half_openings)
    extrema = plan.validate_joint_limits(planner.limits)
    plan.limit_rate(.35, .5)
    origin = planner.fk("right", qs["right"])[0]
    minimum, worst, first = float("inf"), None, None
    errors = [0., 0.]
    peaks = [0., 0.]
    for time in np.linspace(0., plan.duration, 401):
        sample = plan.sample(float(time))
        checks = planner.clearance(sample["q"], sample["finger"])
        category, value = min(checks.items(), key=lambda item: item[1]["distance"])
        tcp, rotation = planner.fk("right", sample["q"]["right"])
        fraction = float(minimum_jerk(time / plan.duration)[0])
        point = dict(time_s=float(time), path_fraction=fraction, tcp_m=tcp.tolist(), category=category, **value)
        if value["distance"] < minimum:
            minimum, worst = value["distance"], point
        if first is None and value["distance"] < .008:
            first = point
        errors[0] = max(errors[0], float(np.linalg.norm(tcp - (origin + fraction * (target - origin)))))
        errors[1] = max(errors[1], float((rotation * original.inv()).magnitude()))
    for time in np.linspace(0., plan.duration, 501):
        sample = plan.sample(float(time))
        for j, key in enumerate(("qd", "qdd")):
            peaks[j] = max(peaks[j], *[float(np.abs(sample[key][s]).max()) for s in ("left", "right")])
    okay = minimum >= .008 and errors[0] <= .0005 and errors[1] <= .002
    return dict(status="geometric_candidate" if okay else "rejected", duration_s=plan.duration,
        target_tcp_m=target.tolist(), original_rotation_xyzw=original.as_quat().tolist(),
        minimum_sampled_rigid_clearance_m=minimum, worst_rigid_sample=worst,
        first_8mm_violation=first, rigid_sample_count=401,
        maximum_sampled_tcp_straight_path_error_m=errors[0], maximum_sampled_original_rotation_error_rad=errors[1],
        command_velocity_limit_rad_s=.35, command_acceleration_limit_rad_s2=.5,
        sampled_peak_velocity_rad_s=peaks[0], sampled_peak_acceleration_rad_s2=peaks[1], rate_sample_count=501,
        analytic_joint_limits_pass=True, analytic_joint_extrema=extrema, physical_success=False), plan


def sweep(translation, displacement):
    return check_translation_sweep(state["bear"], rest["bear_tets"], state["q_left"], cfg,
        translation_m=translation, displacement_m=displacement, required_gap_m=.005, planner=planner)


def envelope_margin(point):
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(np.sum((point - polygon) * edges, axis=1) / np.sum(edges * edges, axis=1), 0., 1.)
    distance = float(np.linalg.norm(point - (polygon + t[:, None] * edges), axis=1).min())
    inside = bool(projected_inside(np.asarray([point]), polygon)[0])
    return dict(inside_contact_envelope=inside, boundary_distance_m=distance,
        at_least_2mm_inside=inside and distance >= .002, is_actual_opening=False)


results = []
for extra_lower in (.01, .02):
    low_delta = np.array([0., 0., -extra_lower])
    low_target = start + low_delta
    lower_row, lower_plan = checked_path(initial_q, low_target, fingers, 6.)
    lower_sweep = sweep(np.zeros(3), low_delta)
    for dx in (-.02, -.025):
        center_delta = np.array([dx, -.06, 0.])
        center_target = low_target + center_delta
        row = dict(extra_lowering_from_measured_entry_m=extra_lower,
            corresponding_requested_hand_clearance_policy_m=.01 + extra_lower,
            center_delta_m=center_delta.tolist(), physical_success=False, runtime_execution_performed=False,
            extra_lower=lower_row, extra_lower_bear_left_5mm_sweep=lower_sweep,
            frozen_whole_bear_to_measured_left_hand_gap_after_lower_m=left_hand_z - float(state["bear"][:, 2].max()) + extra_lower,
            centered_com_contact_envelope=envelope_margin(com[:2] + center_delta[:2]))
        try:
            row["center"], center_plan = checked_path(lower_plan.end_q(), center_target, fingers, 4.)
            row["center_bear_left_5mm_sweep"] = sweep(low_delta, center_delta)
            final_total_dz = cfg["table"]["height"] + .01 - float(state["bear"][:, 2].min())
            insert_delta = np.array([0., 0., final_total_dz + extra_lower])
            final_target = center_target + insert_delta
            row["insert"], insert_plan = checked_path(center_plan.end_q(), final_target, fingers, 6.)
            row["insert_bear_left_5mm_sweep"] = sweep(low_delta + center_delta, insert_delta)
            final_bear = state["bear"] + center_delta + [0., 0., final_total_dz]
            row["frozen_final_bear_bounds_m"] = [final_bear.min(axis=0).tolist(), final_bear.max(axis=0).tolist()]
            row["whole_frozen_bear_below_lowest_rim_by_2mm"] = bool(final_bear[:, 2].max() <= state["bag"][rim, 2].min() - .002)
            row["frozen_bear_final_table_gap_m"] = float(final_bear[:, 2].min() - cfg["table"]["height"])
            opening = []
            for fraction in np.linspace(0., 1., 21):
                ff = {**fingers, "right": float((1-fraction) * fingers["right"] + fraction * .04)}
                checks = planner.clearance(insert_plan.end_q(), ff)
                category, value = min(checks.items(), key=lambda item: item[1]["distance"])
                opening.append(dict(fraction=float(fraction), right_half_opening_m=ff["right"], category=category, **value))
            worst = min(opening, key=lambda v: v["distance"])
            row["release"] = dict(status="geometric_candidate" if worst["distance"] >= .008 else "rejected",
                minimum_sampled_rigid_clearance_m=worst["distance"], worst_sample=worst, sample_count=21,
                final_right_half_opening_m=.04, physical_release_not_executed=True)
            row["withdraw"], _ = checked_path(insert_plan.end_q(), center_target, {**fingers, "right": .04}, 4.)
            row["all_frozen_geometric_gates_pass"] = (
                all(row[k]["status"] == "geometric_candidate" for k in ("extra_lower", "center", "insert", "release", "withdraw"))
                and all(row[k]["status"] == "clear" for k in ("extra_lower_bear_left_5mm_sweep", "center_bear_left_5mm_sweep", "insert_bear_left_5mm_sweep"))
                and row["centered_com_contact_envelope"]["at_least_2mm_inside"]
                and row["whole_frozen_bear_below_lowest_rim_by_2mm"])
        except (ValueError, KeyError) as exc:
            row.update(error=str(exc), all_frozen_geometric_gates_pass=False)
        results.append(row)
        print(json.dumps({"extra_lower_m": extra_lower, "dx_m": dx,
            "all_frozen_geometric_gates_pass": row["all_frozen_geometric_gates_pass"],
            "rigid_minima_m": {k: row.get(k, {}).get("minimum_sampled_rigid_clearance_m") for k in ("extra_lower", "center", "insert", "release", "withdraw")},
            "error": row.get("error")}), flush=True)
        output = dict(kind="full17_extra_entry_lowering_frozen_remainder_review",
            observed_time_s=float(state["time"]), physical_success=False, runtime_execution_performed=False,
            original_rotation_xyzw=original.as_quat().tolist(), actual_rim_geometry=spatial,
            results=results, input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (RUN / "config.json", RUN / "rest.npz", RUN / "lower_teddy_to_rim.npz", RUN / "guarded_contact_candidate.json")},
            limits=["All extra lowering and subsequent poses/shapes are CPU kinematic or frozen-shape hypotheses; no physical state is resumed or fabricated as observed.",
                    "A future 20/30 mm entry hand-clearance policy starts from pre-entry actual geometry; after deformation it may not equal this simple extra 10/20 mm endpoint translation.",
                    "All original commanded quaternion components are retained. Rigid clearance and tracking are sampled at 401 points; joint positions use continuous cubic extrema and rate retiming.",
                    "Bear-left gap is a continuous frozen tetrahedron translation certificate for all actual left collision solids. It does not certify evolving cloth/teddy contact or grip.",
                    "Diagnostic paths may continue beyond an earlier rejected segment to identify other obstacles; such continuation does not authorize execution.",
                    "Force, strain, grasp retention, final physical containment, contact during finger release/withdrawal and settled success require a fresh native full run."])
        (RUN / "independent-extra-entry-lower-remainder.json").write_text(json.dumps(output, indent=2,
            default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + "\n")
