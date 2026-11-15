"""Explicit partial-insertion release experiment, with no soft-state mutation.

The original full-depth geometric candidate remains a required preflight.
This controller instead stops at a higher, smoothly reached waypoint, verifies
actual partial capture, then deliberately opens the fingers. Partial capture
is an engineering release condition, never a full-containment success claim.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from contact_insertion import _policy, _state_context
from contact_robot_geometry import check_translation_sweep
from motion import MotionPlan, CubicSpline, minimum_jerk
from postlift_yaw import _finite_tree, _clearance_rows, continuous_rate_bounds
from validate import containment, mouth_structure, signed_volumes


def release_policy(config):
    spec = config.get("task", {}).get("contact_trial", {}).get("gravity_release")
    if spec is None:
        return None
    if not isinstance(spec, dict) or set(spec) != {
        "mode", "tcp_height_above_table_m", "minimum_inside_vertex_fraction", "settle_seconds"
    } or spec["mode"] != "gravity_seating":
        raise ValueError("Invalid explicit gravity-release policy")
    for key, lo, hi in (("tcp_height_above_table_m", .25, .45),
                        ("minimum_inside_vertex_fraction", .5, 1.),
                        ("settle_seconds", 2., 10.)):
        value = spec[key]
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Invalid gravity-release value: " + key)
    trial = config["task"]["contact_trial"]
    if trial.get("insertion_strategy", "vertical") != "vertical" or trial.get("recenter_enabled") is not False:
        raise ValueError("Gravity release requires fixed-XY vertical insertion without recentering")
    return dict(spec)


def _checked_motion(context, commanded, target, rotation, duration, fingers):
    planner = context["planner"]
    commanded = {s: np.asarray(commanded[s], float).copy() for s in ("left", "right")}
    for side, q in commanded.items():
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Invalid commanded arm configuration")
    paths = planner.plan_cartesian(commanded, {"right": target}, rotations={"right": rotation}, samples=45)
    fractions = np.array([p["fraction"] for p in paths])
    curves = {}
    for side in commanded:
        values = np.array([p["q"].get(side, commanded[side]) for p in paths])
        values[0] = commanded[side]
        curves[side] = CubicSpline(fractions, values, axis=0)
    plan = MotionPlan(curves, duration, fingers, fingers)
    plan.validate_joint_limits(planner.limits)
    plan.limit_rate(.35, .5)
    rates = continuous_rate_bounds(plan)
    v = max(float(np.max(r["velocity_upper_bound_rad_s"])) for r in rates.values())
    a = max(float(np.max(r["acceleration_upper_bound_rad_s2"])) for r in rates.values())
    plan.duration *= max(1., v/.35, np.sqrt(a/.5))*1.002
    rates = continuous_rate_bounds(plan)
    if not _finite_tree(rates) or any(np.max(r["velocity_upper_bound_rad_s"]) > .35
            or np.max(r["acceleration_upper_bound_rad_s2"]) > .5 for r in rates.values()):
        raise ValueError("Invalid continuous gravity-release rate bounds")
    start = planner.fk("right", commanded["right"])[0]
    minimum = float("inf"); position_error = 0.; angle_error = 0.
    for t in np.linspace(0., plan.duration, 401):
        sample = plan.sample(float(t))
        if not _finite_tree(sample) or not np.array_equal(sample["q"]["left"], commanded["left"]):
            raise ValueError("Nonfinite motion or changed left command")
        for left in (context["qs"]["left"], commanded["left"]):
            _, row = _clearance_rows(planner.clearance(dict(sample["q"], left=left), fingers), "gravity release")
            minimum = min(minimum, row["distance"])
        p, r = planner.fk("right", sample["q"]["right"])
        expected = start+float(minimum_jerk(t/plan.duration)[0])*(np.asarray(target)-start)
        pe = float(np.linalg.norm(p-expected)); re = float((r*rotation.inv()).magnitude())
        if not np.isfinite([pe,re]).all() or pe > .0005 or re > .002:
            raise ValueError("Gravity-release FK curve exceeds 0.5 mm / 2 mrad straight-path tolerances")
        position_error = max(position_error, pe); angle_error = max(angle_error, re)
    return plan, {"minimum_sampled_rigid_clearance_m": minimum, "rigid_sample_count": 401,
        "maximum_sampled_position_error_m": position_error, "maximum_sampled_rotation_error_rad": angle_error,
        "continuous_rate_bounds": rates, "exact_command_start_preserved": True,
        "actual_and_commanded_left_checked": True, "continuous_rigid_clearance_certified": False}


def _opening_and_withdrawal(context, qs, outside, rotation):
    planner = context["planner"]
    minimum = float("inf")
    # Cover the full physical finger range, including possible further closing.
    for gap in np.linspace(0., .04, 21):
        fingers = dict(context["fingers"], right=float(gap))
        _, row = _clearance_rows(planner.clearance(qs, fingers), "gravity-release opening")
        clearance = row["distance"]
        minimum = min(minimum, clearance)
        if clearance < .008:
            raise ValueError("High release pose cannot open right fingers with 8 mm rigid clearance")
    opened = dict(context["fingers"], right=.04)
    plan, proof = _checked_motion(context, qs, outside, rotation, 4., opened)
    return {"minimum_opening_rigid_clearance_m": minimum, "opening_samples": 21,
        "withdrawal_duration_s": plan.duration, "withdrawal_rigid_samples": 401,
        "withdrawal_joint_extrema": plan.joint_extrema(),
        "withdrawal_curve_review": proof,
        "scope": "Robot geometry only. Released cloth and teddy remain native dynamic contact."}, plan


def plan_gravity_release(rest, state, config, full_depth, command_qs=None, *, return_plan=False):
    policy = release_policy(config)
    if policy is None:
        raise ValueError("Gravity-release policy is absent")
    if full_depth.get("status") != "candidate" or full_depth.get("report", {}).get("rejections") \
            or full_depth.get("observed_time_s") != float(state["time"]) \
            or full_depth.get("report", {}).get("policy") != _policy(config):
        raise ValueError("Gravity release requires the accepted fresh full-depth preflight")
    context = _state_context(rest, state, config, _policy(config))
    if context["reasons"]:
        raise ValueError("Invalid actual state before gravity-release insertion")
    commanded = context["qs"] if command_qs is None else command_qs
    for side in ("left", "right"):
        pc, rc = context["planner"].fk(side, np.asarray(commanded[side], float))
        pa, ra = context["planner"].fk(side, context["qs"][side])
        if not np.isfinite([*pc,*rc.as_quat()]).all() or np.linalg.norm(pc-pa) > .0005 or (rc*ra.inv()).magnitude() > .002:
            raise ValueError("Actual and commanded poses disagree by over 0.5 mm / 2 mrad")
    start = np.asarray(state["tcp_right"], float)
    low = np.asarray(full_depth["insert_stage"]["right_tcp_m"], float)
    target = start.copy()
    target[2] = float(config["table"]["height"]) + policy["tcp_height_above_table_m"]
    if np.linalg.norm(low[:2]-start[:2]) > 1e-9 or not low[2]+.02 < target[2] < start[2]-.04:
        raise ValueError("Gravity-release waypoint must be an interior point of the accepted vertical descent")
    sweeps = {}
    for label, left in (("actual", context["qs"]["left"]), ("commanded", commanded["left"])):
        sweep = check_translation_sweep(np.asarray(state["bear"]), np.asarray(rest["bear_tets"], int),
            np.r_[left, state["q_left"][7:]], config, translation_m=np.zeros(3),
            displacement_m=target-start, required_gap_m=_policy(config)["left_robot_soft_sweep_clearance_m"]+.001,
            planner=context["planner"])
        if not _finite_tree(sweep) or sweep["status"] != "clear":
            raise ValueError("Partial descent fails complete frozen-teddy/left-robot sweep plus 1 mm path allowance")
        sweeps[label] = sweep
    plan, proof = _checked_motion(context, commanded, target, context["rotation"], 6., context["fingers"])
    opening, _ = _opening_and_withdrawal(context, plan.end_q(), full_depth["outside_tcp_m"], context["rotation"])
    stage = {"name": "insert_to_gravity_release", "right_tcp_m": target.tolist(),
        "rotation_xyzw": context["rotation"].as_quat().tolist(), "duration_s": plan.duration}
    result = {"kind": "explicit_gravity_seating_release", "status": "candidate", "policy": policy,
        "observed_time_s": float(state["time"]), "insert_stage": stage,
        "command_q_start": {s: np.asarray(commanded[s]).tolist() for s in ("left", "right")},
        "command_q_end": {s: q.tolist() for s, q in plan.end_q().items()},
        "actual_source_arms": {s: {"q": np.asarray(state["q_"+s]).tolist(),
            "tcp_m": np.asarray(state["tcp_"+s]).tolist()} for s in ("left", "right")},
        "original_full_depth_tcp_m": low.tolist(), "outside_tcp_m": full_depth["outside_tcp_m"],
        "continuous_reference_zero_endpoint_velocity_acceleration": all(
            np.max(np.abs(plan.sample(t)[key][side])) < 1e-10
            for t in (0., plan.duration) for key in ("qd", "qdd") for side in ("left", "right")),
        "joint_extrema": plan.joint_extrema(), "left_robot_soft_sweeps": sweeps, "insert_curve_review": proof,
        "sweep_approximation_scope": "The extra 1 mm is an engineering allowance; sampled FK errors do not certify continuous Cartesian tracking or changing teddy orientation/shape.",
        "release_opening_and_withdrawal_preflight": opening,
        "physical_success": False, "requires_actual_partial_capture_before_release": True,
        "requires_unchanged_final_full_containment_and_retention": True}
    if not _finite_tree(result):
        raise ValueError("Nonfinite gravity-release planning evidence")
    return (result, plan) if return_plan else result


def review_actual_release(rest, state, config, candidate, contact_row, joint_trace, command_qs=None, *, return_plan=False):
    policy = release_policy(config)
    if policy is None or candidate.get("policy") != policy:
        raise ValueError("Gravity-release policy changed after planning")
    context = _state_context(rest, state, config, _policy(config))
    body, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    if not len(rim) or not topology["two_panels_directly_sewn_along_u"]:
        raise ValueError("No valid two-panel bag body for release review")
    weights = np.abs(signed_volumes(rest["bear"], rest["bear_tets"]))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid rest mass weights")
    captured = containment(state["bag"], state["bear"], body, rim, weights, rest["bear_tets"])
    target = np.asarray(candidate["insert_stage"]["right_tcp_m"], float)
    target_rotation = Rotation.from_quat(candidate["insert_stage"]["rotation_xyzw"])
    position_error = float(np.linalg.norm(np.asarray(state["tcp_right"])-target))
    angle_error = float((context["rotation"]*target_rotation.inv()).magnitude())
    joints = np.asarray(joint_trace, float)
    if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] != 19 \
            or not np.isfinite(joints[-2:]).all() or not 0 < joints[-1,0]-joints[-2,0] <= .02:
        raise ValueError("Missing actual endpoint joint velocity evidence")
    velocity = (joints[-1,10:17]-joints[-2,10:17])/(joints[-1,0]-joints[-2,0])
    reactions = np.asarray(contact_row.get("right_finger_opening_projection_n"), float)
    checks = {"actual_state_has_no_original_preflight_rejection": not context["reasons"],
        "actual_time_after_candidate": float(state["time"]) > candidate["observed_time_s"],
        "actual_endpoint_within_2mm": position_error <= .002,
        "actual_full_orientation_retained": angle_error <= .015,
        "actual_joint_trace_matches_endpoint_time": abs(joints[-1,0]-float(state["time"])) < 1e-9,
        "actual_joint_trace_matches_saved_endpoint_joints": np.array_equal(joints[-1,1:10], state["q_left"])
            and np.array_equal(joints[-1,10:19], state["q_right"]),
        "actual_right_joint_velocity_below_0_05_rad_s": float(np.abs(velocity).max()) <= .05,
        "contact_row_is_current_and_still_holding": abs(float(contact_row["time_s"])-float(state["time"])) < 1e-9
            and contact_row.get("right_two_sided_contact_required") is True and not contact_row.get("stop_reason"),
        "both_right_opening_reactions_above_original_0_1N": reactions.shape == (2,)
            and np.isfinite(reactions).all() and float(reactions.min()) > .1,
        "actual_analysis_body_valid": captured["virtual_closed_body_valid"],
        "actual_mass_COM_inside": captured["bear_center_of_mass_inside"],
        "actual_all_node_fraction_at_least_declared_minimum": captured["bear_vertex_inside_fraction"] >= policy["minimum_inside_vertex_fraction"]}
    checks = {key: bool(value) for key, value in checks.items()}
    actual_opening, _ = _opening_and_withdrawal(context, context["qs"], candidate["outside_tcp_m"], context["rotation"])
    commanded = context["qs"] if command_qs is None else command_qs
    opening, withdraw = _opening_and_withdrawal(context, commanded, candidate["outside_tcp_m"], target_rotation)
    result = {"status": "admissible_release" if all(checks.values()) else "rejected", "checks": checks,
        "actual_time_s": float(state["time"]), "actual_containment": captured,
        "actual_endpoint_position_error_m": position_error, "actual_orientation_error_rad": angle_error,
        "actual_right_endpoint_joint_velocity_rad_s": velocity.tolist(),
        "actual_opening_and_withdrawal_preflight": actual_opening,
        "commanded_opening_and_withdrawal_preflight": opening, "physical_success": False,
        "withdraw_command_q_start": {s: np.asarray(commanded[s]).tolist() for s in ("left", "right")},
        "withdraw_command_q_end": {s: q.tolist() for s, q in withdraw.end_q().items()},
        "limits": ["Partial node inclusion and COM inclusion are a release policy, not stable support proof.",
            "The nonplanar mouth analysis cap is not simulated and its valid flag is not a general body-self-intersection certificate.",
            "All original runtime contact, deformation and holding gates apply until deliberate release.",
            "Final success still requires the original whole-teddy containment, retention and physical audit."]}
    if not _finite_tree(result):
        raise ValueError("Nonfinite actual-release evidence")
    return (result, withdraw) if return_plan else result
