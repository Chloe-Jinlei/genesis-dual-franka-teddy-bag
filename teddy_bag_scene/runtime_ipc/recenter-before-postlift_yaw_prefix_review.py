"""Audit available yaw inputs/curve/endpoints without inventing final traces.

This prefix-only CPU review never runs the helper or physics and cannot return
a complete yaw or task pass. Every-step native evidence remains pending.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import sys

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation, Slerp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_guarded_run as audit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run.resolve()
root = HERE.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = ["config.json", "rest.npz", "lift_teddy.npz", "orient_teddy_postlift.npz",
         "postlift_yaw_candidate.json", "postlift_yaw_execution.json", "postlift_yaw_plan.npz"]
hashes = {name: sha(run / name) for name in names}
cfg = json.loads((run / "config.json").read_text())
plan = json.loads((run / "postlift_yaw_candidate.json").read_text())
execution = json.loads((run / "postlift_yaw_execution.json").read_text())
events = json.loads((run / "events.json").read_text())
rest = dict(np.load(run / "rest.npz", allow_pickle=False))
source = dict(np.load(run / "lift_teddy.npz", allow_pickle=False))
end = dict(np.load(run / "orient_teddy_postlift.npz", allow_pickle=False))
manifest = json.loads((run / "input_sha256.json").read_text())
checks = {}
def check(name, value):
    checks[name] = bool(value)

option = cfg["task"]["contact_trial"]["postlift_yaw"]
angle = audit._yaw_number(option["world_z_degrees"])
minimum_duration = audit._yaw_number(option["duration_s"])
frozen_policy = audit.frozen_postlift_yaw_policy(run)
dt = audit._yaw_number(cfg["simulation"]["dt"])
observed = audit._yaw_number(plan["observed_time_s"])
duration = audit._yaw_number(plan["duration_s"])
steps = int(np.ceil(duration / dt))
expected_end = observed + steps * dt
check("original_optional_config_bounds", set(option) == {"world_z_degrees", "duration_s"} and 0 < abs(angle) <= frozen_policy["maximum_absolute_world_z_degrees"] and minimum_duration >= frozen_policy["minimum_duration_s"] and dt > 0)
required = ["source/postlift_yaw.py", "source/motion.py", "source/run.py", "config.json",
            "source/assets/panda_tactile.xml", "source/assets/" + Path(cfg["bag"]["npz_file"]).name]
check("frozen_sources_assets_and_config_match_manifest", all(name in manifest and sha(run / name) == manifest[name] for name in required))
check("candidate_and_execution_records_finite", audit._yaw_finite_tree(plan) and audit._yaw_finite_tree(execution))
required_checks = {"actual_original_physical_gates", "lift_above_table_40mm", "command_actual_pose_consistent", "analytic_joint_limits", "continuous_reference_rate_bounds", "sampled_all_raw_rigid_checks_finite_and_above_8mm", "sampled_actual_reference_fk_tolerances", "actual_and_commanded_left_frozen_rotation_above_5mm", "returns_exact_checked_plan"}
check("candidate_complete_check_contract", plan["kind"] == "guarded_postlift_yaw" and plan["status"] == "candidate" and set(plan["checks"]) == required_checks and all(v is True for v in plan["checks"].values()) and not plan["rejections"] and plan["parameters"] == option and duration >= option["duration_s"])
check("planning_keeps_native_execution_and_task_success_separate", plan["physical_success"] is False and plan["runtime_execution_performed"] is False and plan["modifies_soft_points"] is False and plan["requires_fresh_actual_preparation_after_execution"] is True and execution["physical_success"] is False)

motion = audit.load_module(run / "source/motion.py", "prefix_frozen_motion")
pc = copy.deepcopy(cfg)
pc["robots"]["mjcf_path"] = str(run / "source/assets" / Path(cfg["robots"].get("mjcf_path", "panda_tactile.xml")).name)
planner = motion.Planner(pc)
actual_rot, command = {}, {}
for side in ("left", "right"):
    q, tcp = np.asarray(source["q_" + side]), np.asarray(source["tcp_" + side])
    assert q.shape == (9,) and tcp.shape == (3,) and np.isfinite(q).all() and np.isfinite(tcp).all()
    actual_p, actual_rot[side] = planner.fk(side, q[:7])
    record, cmd = plan["actual_source_poses"][side], plan["command_source_poses"][side]
    command[side] = np.asarray(cmd["q"], float)
    assert command[side].shape == (7,) and np.isfinite(command[side]).all()
    cp, cr = planner.fk(side, command[side])
    pe, re = float(np.linalg.norm(cp - actual_p)), float((cr * actual_rot[side].inv()).magnitude())
    check(side + "_source_matches_actual_fk", np.allclose(record["q"], q[:7], rtol=0, atol=1e-12) and np.linalg.norm(actual_p - tcp) <= .002 and np.linalg.norm(np.asarray(record["tcp_m"]) - actual_p) <= 1e-9 and (Rotation.from_quat(record["rotation_xyzw"]) * actual_rot[side].inv()).magnitude() <= 1e-9)
    check(side + "_command_fk_record_and_actual_consistency", pe <= .002 and re <= .015 and np.linalg.norm(cp - cmd["tcp_m"]) <= 1e-9 and (cr * Rotation.from_quat(cmd["rotation_xyzw"]).inv()).magnitude() <= 1e-9 and abs(pe - cmd["actual_position_error_m"]) <= 1e-9 and abs(re - cmd["actual_rotation_error_rad"]) <= 1e-9)
target_tcp = np.asarray(source["tcp_right"])
target_rotation = Rotation.from_rotvec([0, 0, np.deg2rad(angle)]) * actual_rot["right"]
check("target_is_world_z_yaw_of_actual_full_rotation_and_fixed_tcp", np.linalg.norm(np.asarray(plan["target_tcp_m"]) - target_tcp) <= 1e-9 and (Rotation.from_quat(plan["target_rotation_xyzw"]) * target_rotation.inv()).magnitude() <= 1e-9)
measurements = {}
for label, state, record in (("source", source, plan["actual_physical_state"]), ("endpoint", end, execution["actual_physical_state"])):
    local, measured = audit._yaw_actual_soft_state(run, cfg, rest, state, record, planner)
    checks.update({label + "_" + k: bool(v) for k, v in local.items()})
    measurements[label] = measured
    check(label + "_original_physical_rejections_empty_and_lifted", not record["physical_state_rejections"] and np.asarray(state["bear"])[:, 2].min() > cfg["table"]["height"] + .04)
check("source_time_is_actual_saved_lift_end", abs(float(source["time"]) - observed) <= 1e-9 and abs(float(np.asarray(source["bear"])[:, 2].min()) - plan["actual_lifted_bear_minimum_z_m"]) <= 1e-9)

raw = dict(np.load(run / "postlift_yaw_plan.npz", allow_pickle=False))
assert set(raw) == {"duration", "observed_time", "left_x", "left_c", "right_x", "right_c", "finger_left", "finger_right"}
assert all(np.isfinite(v).all() for v in raw.values())
check("archive_exact_duration_and_observation", abs(float(raw["duration"]) - duration) <= 1e-12 and abs(float(raw["observed_time"]) - observed) <= 1e-12)
curves, bounds = {}, {}
for side in ("left", "right"):
    x, c = raw[side + "_x"], raw[side + "_c"]
    assert x.ndim == 1 and len(x) >= 2 and x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0) and c.shape == (4, len(x) - 1, 7)
    assert raw["finger_" + side].shape == ()
    expected_finger = 0. if side == "left" else cfg["task"].get("bear_finger_target", .027)
    check(side + "_closed_finger_motor_target_preserved", float(raw["finger_" + side]) == expected_finger)
    joins = []
    for k, width in enumerate(np.diff(x)[:-1]):
        a, b, cc, d = c[:, k]
        joins.extend([np.max(np.abs(((a * width + b) * width + cc) * width + d - c[3, k + 1])),
                      np.max(np.abs((3 * a * width + 2 * b) * width + cc - c[2, k + 1])),
                      np.max(np.abs(6 * a * width + 2 * b - 2 * c[1, k + 1]))])
    check(side + "_cubic_c0_c1_c2_continuity", not joins or np.isfinite(joins).all() and max(joins) <= 1e-10)
    curves[side] = PPoly(c, x)
    bounds[side] = audit._yaw_curve_bounds(curves[side], duration)
    b, limits = bounds[side], planner.limits[side]
    check(side + "_archive_exact_reported_command_endpoints", np.array_equal(curves[side](0), command[side]) and np.array_equal(curves[side](1), plan[side + "_q_end"]))
    check(side + "_analytic_joint_and_continuous_rate_bounds", np.all(b["minimum"] >= limits[:, 0]) and np.all(b["maximum"] <= limits[:, 1]) and np.all(b["velocity"] <= .35) and np.all(b["acceleration"] <= .5))
    recorded = plan["robot_preflight"]["continuous_rate_bounds"][side]
    check(side + "_reported_bounds_cover_independent_bounds", np.all(np.asarray(recorded["velocity_upper_bound_rad_s"]) >= b["velocity"] - 1e-12) and np.all(np.asarray(recorded["acceleration_upper_bound_rad_s2"]) >= b["acceleration"] - 1e-12) and np.all(np.asarray(recorded["velocity_upper_bound_rad_s"]) <= .35) and np.all(np.asarray(recorded["acceleration_upper_bound_rad_s2"]) <= .5))
check("left_command_constant_for_entire_cubic", np.all(curves["left"].c[:3] == 0) and np.all(curves["left"].c[3] == command["left"]))
fingers = {side: float(np.clip(np.asarray(source["q_" + side])[7:].mean(), 0, .04)) for side in curves}
ideal = Slerp([0., 1.], Rotation.from_quat([actual_rot["right"].as_quat(), target_rotation.as_quat()]))
rigid_min, pmax, rmax, rigid_ok = np.inf, 0., 0., True
for u in np.linspace(0, 1, 401):
    fraction = 10 * u**3 - 15 * u**4 + 6 * u**5
    qr = curves["right"](fraction)
    p, r = planner.fk("right", qr)
    pmax = max(pmax, float(np.linalg.norm(p - target_tcp)))
    rmax = max(rmax, float((r * ideal(fraction).inv()).magnitude()))
    for ql in (command["left"], np.asarray(source["q_left"])[:7]):
        rows = planner.clearance({"left": ql, "right": qr}, fingers)
        rigid_ok = rigid_ok and audit._yaw_rigid_rows(rows)
        distances = [audit._yaw_number(v["distance"]) for v in rows.values()]
        rigid_min = min(rigid_min, *distances)
check("independent_401_samples_original_8mm_and_fk_limits", rigid_ok and pmax <= .0005 and rmax <= .002)
robot = plan["robot_preflight"]
check("archive_sampled_metrics_match_report", robot["rigid_sample_count"] == 401 and robot["left_poses_checked"] == ["actual", "commanded"] and robot["continuous_rigid_clearance_certified"] is False and robot["command_start_exactly_preserved"] is True and robot["left_command_constant"] is True and abs(robot["minimum_sampled_rigid_clearance_m"] - rigid_min) <= 1e-9 and abs(robot["maximum_sampled_tcp_error_m"] - pmax) <= 1e-9 and abs(robot["maximum_sampled_slerp_error_rad"] - rmax) <= 1e-9)
check("independent_frozen_bear_left_arc_axes_and_sagitta_5mm", audit._yaw_verify_frozen_arc(plan["frozen_bear_left_rotation_sweep"], source, command, angle, planner, motion.mujoco))

end_p, end_r = planner.fk("right", np.asarray(end["q_right"])[:7])
pe = float(np.linalg.norm(np.asarray(end["tcp_right"]) - target_tcp))
re = float((end_r * target_rotation.inv()).magnitude())
check("native_saved_endpoint_matches_execution_pose", np.linalg.norm(end_p - end["tcp_right"]) <= .002 and np.linalg.norm(np.asarray(end["tcp_right"]) - execution["endpoint_tcp_m"]) <= 1e-9 and (end_r * Rotation.from_quat(execution["endpoint_rotation_xyzw"]).inv()).magnitude() <= 1e-9 and abs(pe - execution["endpoint_position_error_m"]) <= 1e-9 and abs(re - execution["endpoint_rotation_error_rad"]) <= 1e-9)
check("endpoint_original_pose_and_rejection_gates", pe <= .002 and re <= .015 and not execution["endpoint_rejections"])
check("execution_declares_same_plan_without_baseline_reset", execution["status"] == "executed_endpoint_passed" and execution["exact_checked_plan_executed"] is True and execution["contact_guard_baseline_reset"] is False)
check("actual_integer_steps_and_endpoint_times", type(execution["actual_step_count"]) is int and execution["actual_step_count"] == steps and abs(execution["start_time_s"] - observed) <= 1e-9 and abs(execution["planned_duration_s"] - duration) <= 1e-12 and max(abs(execution[k] - expected_end) for k in ("end_time_s", "endpoint_time_s")) <= 1e-8 and abs(float(end["time"]) - expected_end) <= 1e-8)
lift = [(i, e) for i, e in enumerate(events) if e["phase"] == "lift_teddy"]
yaw = [(i, e) for i, e in enumerate(events) if e["phase"] == "orient_teddy_postlift"]
check("unique_yaw_immediately_follows_lift", len(lift) == len(yaw) == 1 and yaw[0][0] == lift[0][0] + 1 and abs(lift[0][1]["start"] + np.ceil(lift[0][1]["duration"] / dt) * dt - observed) <= 1e-8 and abs(yaw[0][1]["start"] - observed) <= 1e-8 and abs(yaw[0][1]["duration"] - duration) <= 1e-12)

# Static contract evidence is explicitly separate from actual trace coverage.
def methods(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
current_methods = methods(run / "source/run.py")
audited_methods = methods(root / "runs/two_panel_full_20/source/run.py")
contract_names = ["begin_guarded_contact_monitor", "run_postlift_yaw", "run_guarded_contact_trial", "observe_physics", "execute"]
contract = {name: current_methods.get(name) is not None and current_methods[name] == audited_methods.get(name) for name in contract_names}
check("frozen_runner_yaw_baseline_and_sampling_contract_matches_audited_full20", all(contract.values()))
check("immutable_prefix_inputs_unchanged_during_review", all(sha(run / name) == digest for name, digest in hashes.items()))
result = dict(kind="independent_actual_postlift_yaw_prefix_review", status="incomplete_pending_native_trace", passed=False,
    frozen_angle_policy=frozen_policy,
    physical_success=False, complete_run_audited=False, complete_native_guard_trace_audited=False,
    all_available_prefix_checks_passed=all(checks.values()), checks=checks, failed_available_checks=[k for k,v in checks.items() if not v],
    actual_measurements=measurements, actual_endpoint_position_error_m=pe, actual_endpoint_rotation_error_rad=re,
    observed_time_s=observed, planned_duration_s=duration, actual_step_count=steps, actual_endpoint_time_s=float(end["time"]),
    independent_reference_metrics=dict(minimum_sampled_rigid_clearance_m=rigid_min, maximum_sampled_tcp_error_m=pmax, maximum_sampled_slerp_error_rad=rmax),
    archived_reference_bounds={side:{key:value.tolist() for key,value in b.items()} for side,b in bounds.items()},
    continuous_single_baseline=dict(frozen_source_contract_checks=contract, execution_declares_no_reset=True,
        actual_baseline_record_verified=False, every_step_coverage_verified=False, native_force_torque_grip_audit_passed=None,
        status="pending_saved_native_baseline_and_complete_guard_trace"),
    pending=["Actual baseline before yaw, equality with final baseline, and every yaw integration step need final report and contact_guard_trace.json.",
        "Every-step native reaction forces/torques/grip and joint trace are not available in this prefix review.",
        "Fresh preparation and the remaining task require their own saved evidence and final audit."],
    events_observed=events, input_sha256=hashes, auditor_sha256=sha(HERE / "audit_guarded_run.py"),
    limitations=["Analytic rate bounds concern the archived commanded reference, not actual dynamic velocities or continuous native clearance.",
        "401-sample robot/FK checks remain sampled; frozen arc axes and sagitta concern ideal rigid grasp and two static left poses.",
        "No final yaw pass or physical task-success conclusion is made while complete native evidence is pending."])
if not all(checks.values()): result["status"] = "failed_available_evidence"
out = run / "independent-postlift-yaw-prefix-review.json"
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(report=str(out), status=result["status"], available_checks_passed=result["all_available_prefix_checks_passed"], checks=len(checks), failed=result["failed_available_checks"], endpoint_position_error_m=pe, endpoint_rotation_error_rad=re, reference=result["independent_reference_metrics"]), indent=2))
assert result["all_available_prefix_checks_passed"]
