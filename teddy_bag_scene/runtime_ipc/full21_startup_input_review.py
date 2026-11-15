"""One read-only source/config/native-initial-state review; no simulation."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_21"
PREV = ROOT / "runs/two_panel_full_20"
sys.path[:0] = [str(RUN / "source"), str(ROOT / "runtime_ipc")]
from audit_input_manifest import audit
from physics_config import _map_shell_material_regions
from validate import mouth_structure, signed_volumes

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a) | set(b)):
            p = path + "." + key if path else key
            if key not in a or key not in b:
                rows.append(dict(path=p, before=a.get(key, "__missing__"), after=b.get(key, "__missing__")))
            else:
                rows.extend(diff(a[key], b[key], p))
        return rows
    return [] if a == b else [dict(path=path, before=a, after=b)]

cfg = json.loads((RUN / "config.json").read_text())
old = json.loads((PREV / "config.json").read_text())
launch = json.loads((ROOT / "runtime_ipc/full21-launch.json").read_text())
manifest = audit(RUN)
checks = {}
def check(name, value):
    checks[name] = bool(value)

check("all_30_frozen_input_hashes_match", manifest["manifest_entry_count"] == 30 and manifest["all_declared_hashes_match"])
check("all_snapshot_files_manifested", manifest["all_snapshot_files_in_manifest"])
check("external_robot_meshes_available", not manifest["missing_external_model_files"])
check("supplied_config_matches_declared_launch_hash", sha(Path(launch["config"])) == launch["config_sha256"] == "d712fe0e04ee6283c58b75a0fd1c1b50e8f125a3a6dc2b30d2df7c87219654ac")
check("frozen_config_semantically_matches_supplied", cfg == json.loads(Path(launch["config"]).read_text()))
check("all_launch_source_hashes_match_frozen_and_live", all(sha(ROOT / name) == sha(RUN / "source" / name) == digest for name, digest in launch["source_frozen"].items()))
current = json.loads((RUN / "input_sha256.json").read_text())
prior = json.loads((PREV / "input_sha256.json").read_text())
changed = sorted(k for k in set(current) & set(prior) if current[k] != prior[k])
added, removed = sorted(set(current) - set(prior)), sorted(set(prior) - set(current))
check("only_expected_source_changes_since_full20", changed == ["config.json", "source/run.py", "source/staged_contact_insertion.py"])
check("only_new_scaled_teddy_obj_and_mtl_replace_old_assets", added == ["source/assets/bear_uniform_coarse_s85.mtl", "source/assets/bear_uniform_coarse_s85.obj"] and removed == ["source/assets/bear_uniform_coarse.mtl", "source/assets/bear_uniform_coarse.obj"])
check("all_live_source_assets_match_frozen_snapshot", all(sha(ROOT / name.removeprefix("source/")) == digest for name, digest in current.items() if name.startswith("source/")))
differences = diff(old, cfg)
expected_changed = {"bear.maxvolume", "bear.mesh", "task.bear_finger_force_limit", "task.bear_grasp_local", "task.bear_grasp_tracking_radius_m", "task.contact_trial.center_offsets_xy_m", "task.contact_trial.insertion_strategy", "task.contact_trial.recenter_offsets_xy_m", "task.contact_trial.staged"}
check("config_change_paths_are_explicit_small_teddy_vertical_trial", {d["path"] for d in differences} == expected_changed)
check("bag_robot_table_and_integration_configs_unchanged", all(cfg[key] == old[key] for key in ("bag", "robots", "table", "simulation")))
check("bear_material_density_unchanged", all(cfg["bear"][key] == old["bear"][key] for key in ("E", "nu", "rho", "friction_mu")))
check("scaled_initial_grasp_and_meshing_settings", np.allclose(cfg["task"]["bear_grasp_local"], np.asarray(old["task"]["bear_grasp_local"]) * .85, atol=1e-15, rtol=0) and cfg["task"]["bear_grasp_tracking_radius_m"] == .032 * .85 and np.isclose(cfg["bear"]["maxvolume"], 4e-6 * .85**3, atol=1e-20, rtol=0) and cfg["task"]["bear_finger_force_limit"] == 3.)
check("vertical_route_with_original_yaw_and_base_gates", cfg["task"]["contact_trial"]["insertion_strategy"] == "vertical" and "staged" not in cfg["task"]["contact_trial"] and cfg["task"]["contact_trial"]["postlift_yaw"] == old["task"]["contact_trial"]["postlift_yaw"] and all(cfg["task"]["contact_trial"][key] == old["task"]["contact_trial"][key] for key in ("bottom_clearance_m", "left_robot_soft_sweep_clearance_m", "minimum_section_envelope_overlap_fraction", "max_lateral_excursion_fraction_of_mouth_width", "maximum_initial_com_offset_m")))

rest = dict(np.load(RUN / "rest.npz"))
initial = dict(np.load(RUN / "initial.npz"))
old_rest = dict(np.load(PREV / "rest.npz"))
old_initial = dict(np.load(PREV / "initial.npz"))
check("native_initial_states_are_finite_at_zero_time", float(initial["time"]) == 0 and all(np.isfinite(v).all() for v in initial.values()) and all(np.isfinite(v).all() for v in rest.values()))
check("native_initial_equals_native_rest_for_both_soft_objects", all(np.array_equal(initial[k], rest[k]) for k in ("bag", "bear")))
check("native_bag_rest_topology_and_points_identical_to_full20", all(np.array_equal(rest[k], old_rest[k]) for k in ("bag", "bag_faces")))
check("native_robot_initial_q9_tcp_and_bag_identical_to_full20", all(np.array_equal(initial[k], old_initial[k]) for k in ("q_left", "q_right", "tcp_left", "tcp_right", "bag")))
topology_same = np.array_equal(rest["bear_tets"], old_rest["bear_tets"])
p = np.asarray(cfg["bear"]["position"])
scale_errors = np.linalg.norm(rest["bear"] - (old_rest["bear"] - p) * .85 - p, axis=1) if rest["bear"].shape == old_rest["bear"].shape else None
check("native_bear_matches_uniform_085_within_0point2um", topology_same and scale_errors is not None and float(scale_errors.max()) < 2e-7)
volumes = signed_volumes(rest["bear"], rest["bear_tets"])
old_volumes = signed_volumes(old_rest["bear"], old_rest["bear_tets"])
check("native_rest_tetra_volumes_finite_and_positive", np.isfinite(volumes).all() and np.all(volumes > 0))
check("native_rest_volume_ratio_matches_085cubed_with_import_roundoff", abs(volumes.sum() / old_volumes.sum() - .85**3) < 2e-6)

asset = RUN / "source/assets" / cfg["bag"]["npz_file"]
a = dict(np.load(asset, allow_pickle=False))
meta = json.loads((RUN / "source/assets" / cfg["bag"]["metadata"]).read_text())["bag"]
front, back, seam = (set(map(int, a[k])) for k in ("panel_front", "panel_back", "u_seam"))
pf, pb, body_faces = (set(map(int, a[k])) for k in ("panel_front_face_indices", "panel_back_face_indices", "body_face_indices"))
fg, bg = a["panel_front_grid"], a["panel_back_grid"]
check("two_cloth_panels_share_exactly_u_seam", front & back == seam and np.array_equal(fg[0], bg[0]) and np.array_equal(fg[:, 0], bg[:, 0]) and np.array_equal(fg[:, -1], bg[:, -1]))
check("exactly_two_panel_face_sets_cover_body", not pf & pb and pf | pb == body_faces and len(pf) == len(pb) == 1480)
zero_width = float(np.ptp(a["points"][a["bottom_seam"], 1]))
check("zero_width_bottom_no_hidden_support_or_fixed_vertices", zero_width == 0 and meta["depth_m"] == 0 and meta["bottom_seam_transverse_depth_m"] == 0 and not meta["bottom_seam_has_area"] and not meta["hidden_rigid_support"] and meta["fixed_particles"] == 0)
_, _, topology = mouth_structure(rest["bag"], rest["bag_faces"], cfg)
check("native_import_is_open_two_panel_u_seam_body", topology["actual_open_body_boundary"] and topology["two_panels_directly_sewn_along_u"] and topology["body_triangle_count"] == 2960 and topology["independent_flat_bottom_triangles"] == 0 and topology["independent_flat_side_triangles"] == 0)
regions, mapping = _map_shell_material_regions(asset, cfg["bag"]["position"], rest["bag"], rest["bag_faces"])
check("native_face_partition_body2960_and_handles352each", {k: len(v) for k, v in regions.items()} == {"body": 2960, "handle_front": 352, "handle_back": 352} and mapping["body_and_handles_disjoint_complete_face_partition"])
check("same_native_material_implementation_and_50_250_configuration", all(current["source/" + k] == prior["source/" + k] for k in ("scene.py", "physics_config.py")) and cfg["bag"]["shear_stiffness"] == 50 and cfg["bag"]["handle_shear_stiffness"] == 250)
log = Path(launch["log"]).read_text()
check("native_initialization_completed_and_physics_began", "BUILD_DONE" in log and "PHASE close_both_handles" in log)
logged_points = [int(x) for x in re.findall(r"Mesh points:\s*(\d+)", log)]
logged_tets = [int(x) for x in re.findall(r"Mesh tetrahedra:\s*(\d+)", log)]
report_path = RUN / "report.json"
native_material = json.loads(report_path.read_text()).get("shell_conversion") if report_path.exists() else None
default_path = ROOT / "config.json"
split_path = RUN / "independent-rest-tet-split-review.json"
scale_path = RUN / "independent-initial-teddy-scale-review.json"
split_review = json.loads(split_path.read_text())
scale_review = json.loads(scale_path.read_text())
split = split_review["verified_split"]
check("independent_split_review_uses_identical_frozen_inputs", all(sha(ROOT / name) == digest for name, digest in split_review["inputs"].items()))
check("logged_raw_counts_exactly_explained_by_native_surface_tet_split", logged_points == [1362] and logged_tets == [5222] and split["parent_tetrahedra"] == 88 and split["all_child_patterns_match_source_exactly"] and split["all_and_only_reconstructed_parents_have_four_boundary_vertices"] and split["all_added_vertices_equal_source_float32_centroids_exactly"] and split["parent_to_children_total_volume_difference_m3"] == 0 and len(rest["bear"]) == 1362 + 88 and len(rest["bear_tets"]) == 5222 + 3 * 88)
report = dict(kind="independent_full21_startup_input_review", created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    physical_success=False, runtime_execution_performed_by_review=False, startup_input_checks_passed=all(checks.values()), checks=checks, failed_checks=[k for k, v in checks.items() if not v],
    manifest=manifest, launch=launch, config_changes_from_full20=differences, changed_common_input_files=changed, added_input_files=added, removed_input_files=removed,
    native_saved_initial_comparison=dict(full21_points=len(rest["bear"]), full21_tetrahedra=len(rest["bear_tets"]), full20_points=len(old_rest["bear"]), full20_tetrahedra=len(old_rest["bear_tets"]), tetra_connectivity_identical=topology_same,
        uniform_scale=.85, maximum_pointwise_scale_error_m=float(scale_errors.max()) if scale_errors is not None else None, initial_bounding_box_dimensions_m=np.ptp(rest["bear"], axis=0).tolist(),
        minimum_signed_tetra_volume_m3=float(volumes.min()), total_rest_volume_m3=float(volumes.sum()), volume_ratio_to_full20=float(volumes.sum() / old_volumes.sum()), density_kg_m3=cfg["bear"]["rho"], rest_volume_times_declared_density_mass_kg=float(volumes.sum() * cfg["bear"]["rho"]),
        old_rest_volume_times_density_mass_kg=float(old_volumes.sum() * old["bear"]["rho"]), native_mass_attribute_readback_available=False,
        minimum_mean_ratio_shape_quality=split_review["native_saved_rest"]["minimum_mean_ratio_shape_quality"], median_mean_ratio_shape_quality=split_review["native_saved_rest"]["shape_quality_quantiles_0_1_10_50_90_100_percent"][3],
        initial_grasp_roi_node_count=split_review["roi"]["native_point_count"], initial_grasp_roi_radius_m=split_review["roi"]["radius_m"],
        maximum_scaled_obj_surface_to_native_rest_node_error_m=scale_review["every_scaled_asset_point_is_runtime_rest_node_max_error_m"]),
    mesher_log_observation=dict(logged_point_counts=logged_points, logged_tetrahedron_counts=logged_tets, differs_from_saved_entity_rest=logged_points != [len(rest["bear"])] or logged_tets != [len(rest["bear_tets"])], discrepancy_resolved=True,
        interpretation=split_review["interpretation"], verified_split=split, source_algorithm=split_review["source_algorithm"]),
    independent_geometry_reviews=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in (split_path,scale_path)],
    native_rest_topology=topology, native_rest_face_partition={k: len(v) for k, v in regions.items()}, maximum_material_mapping_error_m=mapping["maximum_vertex_mapping_error_m"],
    actual_native_material_readback=dict(status="available_in_final_report" if native_material is not None else "pending_final_report", verified_now=False, evidence=native_material,
        scope="50/250 N/m is confirmed in frozen configuration and unchanged initialization code. latest_contact_diagnostics.json contains rigid coupling data, not cloth material attributes. The shell_conversion readback is written in report.json at finish."),
    default_config_observation=dict(path=str(default_path), sha256=sha(default_path), same_as_trial=json.loads(default_path.read_text()) == cfg),
    saved_input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (RUN / "rest.npz", RUN / "initial.npz", RUN / "config.json", RUN / "input_sha256.json")},
    limitations=["All native task validation remains pending. Startup consistency does not prove grip, insertion, release or retention.", "The effective cloth material has not been calibrated to measured fabric.", "rho times saved rest volume is an independent mass calculation, not a direct native mass-attribute readback.", "Mean-ratio shape quality and positive rest volumes do not guarantee resistance to dynamic inversion; post-split dihedral limits are not inferred from TetGen input requests."])
out = RUN / "independent-startup-input-review.json"
out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(report=str(out), passed=report["startup_input_checks_passed"], checks=len(checks), failed_checks=report["failed_checks"], native_comparison=report["native_saved_initial_comparison"], material_readback=report["actual_native_material_readback"]["status"]), indent=2))
assert report["startup_input_checks_passed"]
