"""Four bounded CPU wrist-yaw contingencies; no simulator, source/config mutation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_17"
sys.path.insert(0, str(RUN / "source"))
from motion import Planner, MotionPlan, CubicSpline, minimum_jerk
from contact_robot_geometry import (_convex, convex_clearance_certificate,
    check_translation_sweep, left_collision_solids)
from staged_contact_insertion import _spatial_loop_geometry
from validate import mouth_structure
from physical_monitor import tetra_volumes
from insertion_candidates import projected_inside

cfg = json.loads((RUN / "config.json").read_text())
rest = dict(np.load(RUN / "rest.npz"))
state = dict(np.load(RUN / "lower_teddy_to_rim.npz"))
planner = Planner(cfg)
original = Rotation.from_quat(json.loads((RUN / "guarded_contact_candidate.json").read_text())["route"]["rotation_xyzw"])
tcp0 = state["tcp_right"].copy()
fingers = {s: float(np.mean(state["q_" + s][7:])) for s in ("left", "right")}
initial_q = {s: state["q_" + s][:7].copy() for s in ("left", "right")}
left = left_collision_solids(planner, state["q_left"])
_, rim, _ = mouth_structure(rest["bag"], rest["bag_faces"], cfg)
polygon, _, spatial = _spatial_loop_geometry(state["bag"][rim], rest["bag"][rim], cfg["bag"]["thickness"])
weights = np.abs(tetra_volumes(rest["bear"], rest["bear_tets"]))


def checked_path(qs, target, rotation, half_openings, duration):
    origin, initial_rotation = planner.fk("right", qs["right"])
    expected_rotation = Slerp([0., 1.], Rotation.from_quat([initial_rotation.as_quat(), rotation.as_quat()]))
    paths = planner.plan_cartesian(qs, {"right": target}, rotations={"right": rotation}, samples=45)
    fractions = np.asarray([e["fraction"] for e in paths])
    curves = {s: CubicSpline(fractions, np.asarray([e["q"].get(s, qs[s]) for e in paths]), axis=0)
              for s in ("left", "right")}
    plan = MotionPlan(curves, duration, half_openings, half_openings)
    extrema = plan.validate_joint_limits(planner.limits)
    plan.limit_rate(.35, .5)
    minimum, worst, first = float("inf"), None, None
    errors, peaks = [0., 0.], [0., 0.]
    for t in np.linspace(0., plan.duration, 401):
        sample = plan.sample(float(t))
        checks = planner.clearance(sample["q"], sample["finger"])
        category, value = min(checks.items(), key=lambda item: item[1]["distance"])
        tcp, orient = planner.fk("right", sample["q"]["right"])
        fraction = float(minimum_jerk(t / plan.duration)[0])
        point = dict(time_s=float(t), path_fraction=fraction, tcp_m=tcp.tolist(), category=category, **value)
        if value["distance"] < minimum:
            minimum, worst = value["distance"], point
        if first is None and value["distance"] < .008:
            first = point
        errors[0] = max(errors[0], float(np.linalg.norm(tcp - (origin + fraction * (target - origin)))))
        errors[1] = max(errors[1], float((orient * expected_rotation(fraction).inv()).magnitude()))
    for t in np.linspace(0., plan.duration, 501):
        sample = plan.sample(float(t))
        for i, key in enumerate(("qd", "qdd")):
            peaks[i] = max(peaks[i], *[float(np.abs(sample[key][s]).max()) for s in ("left", "right")])
    okay = minimum >= .008 and errors[0] <= .0005 and errors[1] <= .002 and peaks[0] <= .35 and peaks[1] <= .5
    return dict(status="geometric_candidate" if okay else "rejected", duration_s=plan.duration,
        target_tcp_m=target.tolist(), target_rotation_xyzw=rotation.as_quat().tolist(),
        minimum_sampled_rigid_clearance_m=minimum, worst_rigid_sample=worst,
        first_8mm_violation=first, rigid_sample_count=401,
        maximum_sampled_tcp_straight_path_error_m=errors[0], maximum_sampled_slerp_rotation_error_rad=errors[1],
        command_velocity_limit_rad_s=.35, command_acceleration_limit_rad_s2=.5,
        sampled_peak_velocity_rad_s=peaks[0], sampled_peak_acceleration_rad_s2=peaks[1], rate_sample_count=501,
        analytic_joint_limits_pass=True, analytic_joint_extrema=extrema, physical_success=False), plan


def yawed_bear(angle):
    return Rotation.from_euler("z", angle).apply(state["bear"] - tcp0) + tcp0


def rotation_sweep(angle):
    count = max(1, int(np.ceil(abs(angle) / np.deg2rad(5.))))
    knots = np.linspace(0., angle, count + 1)
    radius = float(np.linalg.norm((state["bear"] - tcp0)[:, :2], axis=1).max())
    sagitta = radius * (1. - np.cos(abs(angle) / count / 2.))
    minimum, rows = float("inf"), []
    for i, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        hull = _convex(np.vstack([yawed_bear(a), yawed_bear(b)]))
        segment_min, worst = float("inf"), None
        for solid in left:
            certificate = convex_clearance_certificate(hull, solid, required_gap_m=.005 + sagitta)
            lower = certificate["conservative_gap_lower_bound_m"] - sagitta
            if lower < segment_min:
                segment_min, worst = lower, dict(body=solid["body"], geom_id=solid["geom_id"], **certificate)
            if certificate["status"] != "clear":
                return dict(status="not_certified", segment_index=i, segment_angles_rad=[float(a), float(b)],
                    sagitta_outer_bound_m=sagitta, maximum_xy_radius_m=radius,
                    conservative_gap_lower_bound_m=max(0., lower), worst=worst,
                    segment_count=count, checked_prefix_segments=rows,
                    insufficient_frozen_convex_envelope_certificate=True)
        minimum = min(minimum, segment_min)
        rows.append(dict(segment_index=i, minimum_continuous_gap_lower_bound_m=segment_min,
                         worst_body=worst["body"], worst_geom=worst["geom_id"]))
    return dict(status="clear", conservative_gap_lower_bound_m=minimum,
        sagitta_outer_bound_m=sagitta, maximum_xy_radius_m=radius,
        segment_count=count, all_left_geometry_count=len(left), segments=rows,
        method="Each <=5-degree frozen yaw arc lies within radius*(1-cos(delta/2)) of the convex hull of its endpoint solids; add that bound to the 5mm separation requirement.",
        certificate_scope="Pure frozen yaw about the fixed measured TCP, all tetrahedral points included; no dynamic deformation or continuous FK tracking bound.")


def envelope(bear, delta):
    com = np.average(bear[rest["bear_tets"]].mean(axis=1), axis=0, weights=weights) + delta
    p = com[:2]
    edges = np.roll(polygon, -1, axis=0) - polygon
    t = np.clip(np.sum((p - polygon) * edges, axis=1) / np.sum(edges * edges, axis=1), 0., 1.)
    distance = float(np.linalg.norm(p - (polygon + t[:, None] * edges), axis=1).min())
    inside = bool(projected_inside(np.asarray([p]), polygon)[0])
    return dict(planned_frozen_com_m=com.tolist(), inside_contact_envelope=inside,
        boundary_distance_m=distance, at_least_2mm_inside=inside and distance >= .002,
        is_actual_opening=False)


def sweep(bear, translation, displacement):
    return check_translation_sweep(bear, rest["bear_tets"], state["q_left"], cfg,
        translation_m=translation, displacement_m=displacement, required_gap_m=.005, planner=planner)


results = []
for degrees in (-90., -45., 45., 90.):
    angle = np.deg2rad(degrees)
    target_rotation = Rotation.from_euler("z", angle) * original
    row = dict(world_z_yaw_degrees=degrees, physical_success=False, runtime_execution_performed=False,
               center_remainders=[])
    try:
        row["rotation"], rotation_plan = checked_path(initial_q, tcp0, target_rotation, fingers, 6.)
        print("rotation", degrees, row["rotation"]["status"], row["rotation"]["minimum_sampled_rigid_clearance_m"], flush=True)
        if row["rotation"]["status"] == "geometric_candidate":
            row["frozen_bear_rotation_5mm_sweep"] = rotation_sweep(angle)
            print("bear_yaw_sweep", degrees, row["frozen_bear_rotation_5mm_sweep"]["status"], flush=True)
            if row["frozen_bear_rotation_5mm_sweep"]["status"] == "clear":
                bear = yawed_bear(angle)
                row["rotated_frozen_bear_bounds_m"] = [bear.min(axis=0).tolist(), bear.max(axis=0).tolist()]
                for dx, dy in ((-.02, -.05), (-.02, -.06), (-.025, -.05), (-.025, -.06)):
                    delta = np.array([dx, dy, 0.])
                    item = dict(center_delta_m=delta.tolist(), com_contact_envelope=envelope(bear, delta))
                    try:
                        target = tcp0 + delta
                        item["center"], cp = checked_path(rotation_plan.end_q(), target, target_rotation, fingers, 4.)
                        item["center_bear_left_5mm_sweep"] = sweep(bear, np.zeros(3), delta)
                        dz = cfg["table"]["height"] + .01 - float(bear[:, 2].min())
                        item["insert"], ip = checked_path(cp.end_q(), target + [0., 0., dz], target_rotation, fingers, 6.)
                        item["insert_bear_left_5mm_sweep"] = sweep(bear, delta, np.array([0., 0., dz]))
                        final_bear = bear + delta + [0., 0., dz]
                        item["whole_frozen_bear_below_lowest_rim_by_2mm"] = bool(final_bear[:, 2].max() <= state["bag"][rim, 2].min() - .002)
                        item["final_frozen_bear_bounds_m"] = [final_bear.min(axis=0).tolist(), final_bear.max(axis=0).tolist()]
                        opening = []
                        for fraction in np.linspace(0., 1., 21):
                            ff = {**fingers, "right": float((1-fraction) * fingers["right"] + fraction * .04)}
                            checks = planner.clearance(ip.end_q(), ff)
                            cat, value = min(checks.items(), key=lambda e: e[1]["distance"])
                            opening.append(dict(fraction=float(fraction), category=cat, **value))
                        worst = min(opening, key=lambda e: e["distance"])
                        item["release"] = dict(status="geometric_candidate" if worst["distance"] >= .008 else "rejected",
                            minimum_sampled_rigid_clearance_m=worst["distance"], worst_sample=worst, sample_count=21)
                        item["withdraw"], _ = checked_path(ip.end_q(), target, target_rotation, {**fingers, "right": .04}, 4.)
                        item["all_frozen_geometric_gates_pass"] = (all(item[k]["status"] == "geometric_candidate"
                            for k in ("center", "insert", "release", "withdraw")) and all(item[k]["status"] == "clear"
                            for k in ("center_bear_left_5mm_sweep", "insert_bear_left_5mm_sweep"))
                            and item["com_contact_envelope"]["at_least_2mm_inside"]
                            and item["whole_frozen_bear_below_lowest_rim_by_2mm"])
                    except (ValueError, KeyError) as exc:
                        item.update(error=str(exc), all_frozen_geometric_gates_pass=False)
                    row["center_remainders"].append(item)
                    print("remainder", degrees, dx, dy, item["all_frozen_geometric_gates_pass"], flush=True)
    except (ValueError, KeyError) as exc:
        row.update(error=str(exc), rotation_feasibility_rejected=True)
        print("rotation_error", degrees, str(exc), flush=True)
    results.append(row)
    output = dict(kind="full17_bounded_world_z_yaw_future_contingency", observed_time_s=float(state["time"]),
        physical_success=False, runtime_execution_performed=False,
        baseline_commanded_rotation_xyzw=original.as_quat().tolist(), fixed_rotation_tcp_m=tcp0.tolist(),
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (RUN / "config.json", RUN / "rest.npz", RUN / "lower_teddy_to_rim.npz", RUN / "guarded_contact_candidate.json")},
        results=results,
        limitations=["Future contingency only. The running full18 route, source and config are untouched.",
            "No soft toy is assumed to follow a rigid wrist yaw under real friction; force/torque, material retention and deformation may reject this motion.",
            "The frozen rotation certificate covers the intended pure WORLD-Z arc with a conservative sagitta bound, not evolving material or continuous joint-FK errors.",
            "Rigid clearance and Cartesian/Slerp tracking are sampled at 401 points; joint-position extrema are analytic, rates use the established 501-point retiming procedure.",
            "No native cloth contact, self-contact, plasticity, force, release, settled containment or full-task success is demonstrated.",
            "Diagnostic remainder paths may continue after a rejected segment solely to identify other obstacles; rejected motion is never authorized."])
    (RUN / "independent-yaw-contingency-review.json").write_text(json.dumps(output, indent=2,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + "\n")
