"""Read-only full21 saved-evidence summary. No simulation or input mutation."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_21"
sys.path.insert(0, str(RUN / "source"))
from validate import signed_volumes, mesh_edges, mouth_structure, containment
from physics_config import _map_shell_material_regions

def read(name):
    return json.loads((RUN / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)

report = read("report.json")
cfg = read("config.json")
events = read("events.json")
rest = dict(np.load(RUN / "rest.npz", allow_pickle=False))
states = dict(np.load(RUN / "states.npz", allow_pickle=False))
joint = np.load(RUN / "joint_trace.npy", allow_pickle=False)
soft = np.load(RUN / "soft_metrics.npy", allow_pickle=False)
audit = read("independent-audit.json")
final = read("independent-final-review.json")
guard = read("independent-guard-reactions.json")
yaw = read("independent-postlift-yaw-sequence.json")
low = read("independent-final-low-region-winding.json")
manifest = read("input_sha256.json")
launch = json.loads((ROOT / "runtime_ipc/full21-launch.json").read_text())
dt = float(cfg["simulation"]["dt"])
checks = {}
checks["frozen_manifest_matches"] = final["input_manifest_matches"]
checks["live_source_assets_still_match_run_snapshot"] = all(
    sha(ROOT / key.removeprefix("source/")) == digest
    for key, digest in manifest.items() if key.startswith("source/"))
checks["launch_config_hash_and_frozen_semantics_match"] = (
    sha(Path(launch["config"])) == launch["config_sha256"]
    and json.loads(Path(launch["config"]).read_text()) == cfg)
checks["launch_source_hashes_match_live_and_frozen"] = all(
    sha(ROOT / key) == sha(RUN / "source" / key) == digest
    for key, digest in launch["source_frozen"].items())

tets = rest["bear_tets"]
volumes = signed_volumes(rest["bear"], tets)
weights = np.abs(volumes)
edges = mesh_edges(rest["bag_faces"])[0]
rest_lengths = np.linalg.norm(rest["bag"][edges[:, 1]] - rest["bag"][edges[:, 0]], axis=1)
body, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], cfg)
rates = {}
for side, column in (("left", 1), ("right", 10)):
    velocity = np.gradient(joint[:, column:column+7], joint[:, 0], axis=0)
    acceleration = np.gradient(velocity, joint[:, 0], axis=0)
    rates[side] = (velocity, acceleration)

phase_rows = []
for n, event in enumerate(events):
    path = RUN / (event["phase"] + ".npz")
    state = dict(np.load(path, allow_pickle=False))
    time = float(state["time"])
    next_start = float(events[n+1]["start"]) if n+1 < len(events) else float(report["simulated_seconds"])
    step_count = int(np.ceil(float(event["duration"]) / dt))
    expected = float(event["start"]) + step_count*dt
    matches = np.flatnonzero(np.isclose(states["time"], time, atol=1e-9, rtol=0))
    index = int(matches[-1]) if len(matches) else None
    exact = index is not None and all(np.array_equal(state[k], states[k][index]) for k in state)
    jt = np.flatnonzero(np.isclose(joint[:, 0], time, atol=1e-9, rtol=0))
    joint_exact = bool(len(jt) and np.array_equal(joint[jt[-1], 1:10], state["q_left"])
                       and np.array_equal(joint[jt[-1], 10:19], state["q_right"]))
    sm = (soft[:, 0] >= event["start"] - 1e-9) & (soft[:, 0] <= time + 1e-9)
    jm = (joint[:, 0] >= event["start"] - 1e-9) & (joint[:, 0] <= time + 1e-9)
    com = np.sum(state["bear"][tets].mean(1)*weights[:, None], axis=0)/weights.sum()
    row = dict(phase=event["phase"], actual_start_s=event["start"], actual_end_s=time,
        nominal_duration_s=event["duration"], actual_integration_steps=step_count,
        expected_end_by_integer_steps_s=expected,
        timestamp_matches_integer_steps=abs(time-expected)<1e-8,
        endpoint_matches_next_start_or_final=abs(time-next_start)<1e-8,
        endpoint_exactly_matches_saved_full_frame=exact,
        endpoint_actual_joints_exactly_match_every_step_trace=joint_exact,
        all_endpoint_values_finite=all(np.isfinite(x).all() for x in state.values()),
        endpoint_minimum_J=float(np.min(signed_volumes(state["bear"], tets)/volumes)),
        endpoint_maximum_cloth_edge_stretch=float(np.max(np.linalg.norm(state["bag"][edges[:,1]]-state["bag"][edges[:,0]], axis=1)/rest_lengths)),
        endpoint_bear_COM_m=com, endpoint_bear_min_m=state["bear"].min(0),
        endpoint_bear_max_m=state["bear"].max(0), endpoint_left_TCP_m=state["tcp_left"],
        endpoint_right_TCP_m=state["tcp_right"], endpoint_right_finger_positions_m=state["q_right"][7:],
        per_step_soft_extrema=dict(rows=int(sm.sum()), minimum_J=float(soft[sm,1].min()),
            maximum_bag_table_penetration_m=float(soft[sm,2].max()),
            maximum_bear_table_penetration_m=float(soft[sm,3].max()),
            maximum_cloth_edge_stretch=float(soft[sm,4].max())),
        actual_sampled_joint_motion={side:dict(maximum_velocity_rad_s=float(np.abs(v[jm]).max()),
            maximum_acceleration_rad_s2=float(np.abs(a[jm]).max())) for side,(v,a) in rates.items()})
    if event["phase"] in ("insert_teddy", "release_teddy", "withdraw_right_arm", "final_settle"):
        row["endpoint_original_containment"] = containment(state["bag"],state["bear"],body,rim,weights,tets)
    phase_rows.append(row)
checks["all_19_phase_endpoints_match_integer_steps_saved_geometry_and_actual_joints"] = (
    len(phase_rows)==19 and all(all(row[key] for key in (
        "timestamp_matches_integer_steps", "endpoint_matches_next_start_or_final",
        "endpoint_exactly_matches_saved_full_frame", "endpoint_actual_joints_exactly_match_every_step_trace",
        "all_endpoint_values_finite")) for row in phase_rows))

shell = report["shell_conversion"]
entity = shell["entities"][0]
source_asset = RUN / "source/assets" / Path(cfg["bag"]["npz_file"]).name
regions, mapping = _map_shell_material_regions(source_asset, cfg["bag"]["position"], rest["bag"], rest["bag_faces"])
region_checks = {}
for name, expected in (("body",50.),("handle_front",250.),("handle_back",250.)):
    c = entity["region_coefficients"][name]
    region_checks[name] = bool(c["shear_stiffness_n_per_m"] == expected and c["all_faces_equal"]
        and c["face_count"] == len(regions[name])
        and np.array_equal(regions[name], entity["material_region_mapping"]["regions"][name]["native_face_indices"]))
checks["native_initialization_material_readback_matches_50_250_and_independent_face_mapping"] = all(region_checks.values())
checks["native_material_source_hash_matches_frozen_asset"] = sha(source_asset) == entity["material_region_mapping"]["source_sha256"]
checks["native_material_protected_attributes_reported_unchanged"] = all(entity["protected_attributes_unchanged"].values())
checks["single_guard_baseline_record_unchanged"] = report["contact_guard_baseline"] == report["contact_guard"]["baseline"]
expected_limits = dict(max_net_force_n=8., max_tcp_torque_nm=.15, max_finger_force_n=8., minimum_left_opening_reaction_n=.1)
checks["native_raw_force_limits_equal_original_explicit_limits"] = guard["raw_limits"] == expected_limits
checks["all_guarded_steps_and_yaw_contract_pass"] = guard["passed"] and yaw["passed"]

w = low["cap_winding"]
wd = np.asarray(w["mean_apex_raw_winding"])
distances = low["physical_cloth_query_distance"]
low_guards = (low["winding_computed"] and low["height_precondition"]["all_bear_vertices_and_com_at_least_required_margin_below_rim"]
    and low["actual_topology_orientation"]["passed"] and low["actual_spatial_rim"]["passed"]
    and w["caps_agree"] and w["integer_residual_passed"] and not w["multiple_winding_query_indices"]
    and all(w[key]["is_watertight"] and w[key]["is_winding_consistent"] for key in ("mean_apex_surface","high_apex_surface"))
    and distances["queries_separated_from_body_at_numerical_guard"])
low_summary = dict(actual_time_s=low["time_s"], diagnostic_guards_passed=low_guards,
    eligible_bear_nodes=1450, eligible_COM_queries=1, inside_unit_degree_nodes=int(np.sum(np.abs(np.rint(wd[:-1]))==1)),
    outside_zero_degree_nodes=int(np.sum(np.rint(wd[:-1])==0)), COM_zero_degree=bool(np.rint(wd[-1])==0),
    height=low["height_precondition"], maximum_cap_difference=w["maximum_absolute_cap_difference"],
    maximum_integer_residual=w["maximum_integer_residual"], minimum_body_distance_m=distances["minimum_body_distance_m"],
    COM_distance_to_actual_body_m=distances["body_distance_m"][-1],
    maximum_outside_node_distance_to_actual_body_m=max(distances["body_distance_m"][:-1]),
    all_points_above_cap_or_empty_query_false_negative=False,
    scope="Actual pointwise degree relative to two coherently closed physical-body triangle chains. Does not certify embedded/nonintersecting body or whole tetrahedron volume.")

insert_review = read("independent-actual-insert-diagnosis/review.json")
insert_low = insert_review["phases"]["insert_teddy"]["low_region"]
output = dict(kind="independent_full21_complete_failure_evidence_summary", utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    physical_task_success=False, action_sequence_completed=report["sequence_completed"], native_status=report["status"],
    native_error=report["error"], final_simulated_seconds=report["simulated_seconds"],
    all_independent_evidence_consistency_checks_passed=all(checks.values()), checks=checks,
    failed_consistency_checks=[k for k,v in checks.items() if not v],
    task_failures=final["failed_geometry_or_soft_checks"],
    conclusion="All 19 native action phases completed including release, withdrawal and settling. The task failed: actual final bear nodes and mass COM are outside the bag; physical limits passing does not establish containment.",
    original_physical_limits=dict(minimum_J=.05, maximum_cloth_edge_stretch=1.15, maximum_table_penetration_m=.002, **expected_limits),
    actual_every_step_soft=audit["soft_metric_sampling"], actual_guard=guard,
    actual_jointrates=audit["motion"], actual_jointrate_thresholds=audit["motion_thresholds"],
    rate_scope="Every-step actual joint positions differentiated by numpy.gradient; phase masks use the same full-run derivatives. Separate from planned reference 0.35/0.5 gates and yaw continuous reference bounds.",
    source_input_manifest=read("independent-input-manifest-review.json"),
    final_material_readback=dict(scope=shell["scope"], written_in_final_report=True,
        final_time_material_resampling_performed=False, calibrated=shell["calibrated"],
        regions=entity["region_coefficients"], independent_region_checks=region_checks,
        all_protected_attributes_unchanged=entity["protected_attributes_unchanged"],
        rest_area_density_derived_and_native_reported_mass_kg=dict(independent=audit["mass"]["bag_kg"], native=entity["native_rest_mass_kg"]),
        note="Native attribute readback was taken during pre-initialization and persisted in the final report; this is not a second end-time material-buffer readback."),
    final_original_containment=audit["final_containment"], original_retention_samples=audit["hold_containment_samples"],
    final_contact_sampling=audit["final_contact"], final_low_region_pointwise_diagnostic=low_summary,
    actual_insert_low_region_existing_independent_diagnostic=insert_low,
    actual_phase_endpoints=phase_rows,
    relevant_subaudit_status={k:v for k,v in final.items() if k.endswith("status")},
    limitations=["Integration-step and saved-frame observations do not certify continuous collision freedom.",
        "Positive opening reactions and named material proximity are holding evidence, not a force-closure proof.",
        "Three final 0.5-second retention samples are the original criterion; no exhaustive continuous retention proof is claimed.",
        "Effective cloth and bear parameters remain uncalibrated.",
        "Zero eligible low-region queries means not applicable, never zero containment; final queries here are all 1451 eligible."],
    input_sha256={str(RUN/name):sha(RUN/name) for name in ["report.json","validation.json","config.json","input_sha256.json","events.json","states.npz","rest.npz","joint_trace.npy","soft_metrics.npy","contact_guard_trace.json","coupling_trace.json","handle_grip_trace.json","independent-final-review.json","independent-final-low-region-winding.json"]},
    audit_source_sha256=sha(ROOT/"runtime_ipc/audit_guarded_run.py"), summary_source_sha256=sha(Path(__file__).resolve()))
out = RUN / "independent-failure-summary.json"
out.write_text(json.dumps(output, indent=2, default=serial)+"\n")
print(json.dumps(dict(report=str(out), sha256=sha(out), consistency_passed=all(checks.values()), failed_checks=output["failed_consistency_checks"], final_low_region=low_summary), indent=2))
