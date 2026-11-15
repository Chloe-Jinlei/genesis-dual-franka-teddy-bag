"""Bounded geometry-first initial offset search on actual full18 lift, CPU only."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_18"
sys.path.insert(0, str(RUN / "source"))
from contact_insertion import (_policy, _state_context, _mouth, _section_envelopes,
    plan_guarded_contact_trial)
from insertion_candidates import projected_inside, polygon_centroid
from contact_slice_geometry import convex_polygon_overlap_area, convex_hull_metrics

cfg = json.loads((RUN / "config.json").read_text())
rest = dict(np.load(RUN / "rest.npz"))
state = dict(np.load(RUN / "lift_teddy.npz"))
old = json.loads((RUN / "guarded_contact_candidate.json").read_text())
policy = _policy(cfg)
context = _state_context(rest, state, cfg, policy)
polygon, triangles, mouth = _mouth(rest, state, cfg)
anchor = polygon_centroid(polygon)
sections, envelopes = _section_envelopes(state["bear"], rest["bear_tets"])
old_rows = old["report"]["selected_sections"]["sections"]
critical = {min(range(len(old_rows)), key=lambda i: old_rows[i]["opening_overlap_fraction"]),
            max(range(len(old_rows)), key=lambda i: old_rows[i]["maximum_outside_excursion_bound_m"]),
            max(range(len(old_rows)), key=lambda i: old_rows[i]["mouth_and_section_envelope_perimeter_m"]),
            max(range(len(envelopes)), key=lambda i: envelopes[i]["area"])}
critical.update(np.linspace(0, len(envelopes)-1, 9).astype(int).tolist())
padding = policy["rim_perimeter_clearance_m"] * np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
offsets = [np.array([x, y], dtype=float) / 1000. for x in range(-35, 36, 5)
           for y in range(-35, 36, 5) if np.hypot(x, y) <= 35.]
offsets.append(np.array([.024, .020]))
screened = []
for index, offset in enumerate(offsets):
    target = anchor + offset
    inside, distances = projected_inside(np.array([target]), polygon)
    row = dict(offset_xy_m=offset.tolist(), source="original" if index == len(offsets)-1 else "5mm_grid",
        center_of_mass_target_xy_m=target.tolist(), com_inside=bool(inside[0]),
        com_boundary_distance_m=float(distances[0]), exact_necessary_condition_rejections=[])
    if not inside[0] or distances[0] < policy["com_boundary_clearance_m"]:
        row["exact_necessary_condition_rejections"].append("COM lacks 2mm interior margin")
    delta = target - context["com"][:2]
    vi, vd = projected_inside(state["bear"][:, :2] + delta, polygon)
    outside = float(vd[~vi].max()) if np.any(~vi) else 0.
    row["vertex_outside_distance_lower_bound_m"] = outside
    if outside > policy["lateral_excursion_fraction_of_mouth_width"] * mouth["projected_minimum_caliper_width_m"]:
        row["exact_necessary_condition_rejections"].append("Actual vertex outside distance exceeds unchanged lateral limit")
    minimum_overlap, maximum_perimeter = 1., 0.
    if not row["exact_necessary_condition_rejections"]:
        for i in sorted(critical):
            envelope = envelopes[i]
            hull = envelope["hull_xy"] + delta
            overlap = convex_polygon_overlap_area(hull, pretriangles=triangles) / envelope["area"]
            minimum_overlap = min(minimum_overlap, overlap)
            buffered = (hull[:, None, :] + padding[None, :, :]).reshape(-1, 2)
            perimeter = convex_hull_metrics(np.vstack([polygon, buffered]))["perimeter"]
            maximum_perimeter = max(maximum_perimeter, perimeter)
        if minimum_overlap < policy["minimum_section_envelope_overlap_fraction"]:
            row["exact_necessary_condition_rejections"].append("At least one actual slab fails unchanged overlap fraction")
        if maximum_perimeter > mouth["available_nonstretch_rim_perimeter_m"]:
            row["exact_necessary_condition_rejections"].append("At least one actual slab fails unchanged rim perimeter")
        row.update(critical_slab_minimum_overlap_fraction=float(minimum_overlap),
            critical_slab_maximum_required_perimeter_m=float(maximum_perimeter))
    row["passes_necessary_screen_only"] = not row["exact_necessary_condition_rejections"]
    screened.append(row)
survivors = [r for r in screened if r["passes_necessary_screen_only"]]
survivors.sort(key=lambda r: (r["vertex_outside_distance_lower_bound_m"],
    -r["critical_slab_minimum_overlap_fraction"], np.linalg.norm(np.asarray(r["offset_xy_m"]) - [.024, .020])))
print("grid", len(offsets), "necessary_screen_survivors", len(survivors), flush=True)
batches = []
accepted = None
for start in range(0, len(survivors), 25):
    batch = survivors[start:start+25]
    trial = copy.deepcopy(cfg)
    trial["task"]["contact_trial"]["center_offsets_xy_m"] = [r["offset_xy_m"] for r in batch]
    result = plan_guarded_contact_trial(rest, state, trial)
    batch_row = dict(batch_index=len(batches), offset_count=len(batch), offsets_xy_m=trial["task"]["contact_trial"]["center_offsets_xy_m"],
        status=result["status"], rejections=result["report"].get("rejections", []),
        centering_candidates=result["report"].get("centering_candidates", []),
        robot_centering_attempts=result["report"].get("robot_centering_attempts", []))
    batches.append(batch_row)
    print("batch", len(batches), "status", result["status"], "robot_attempts", len(batch_row["robot_centering_attempts"]), flush=True)
    if result["status"] == "candidate":
        accepted = result
        (RUN / "independent-alternate-initial-offset-candidate.json").write_text(json.dumps(result, indent=2) + "\n")
        break
output = dict(kind="full18_bounded_alternate_initial_COM_offset_search",
    physical_success=False, runtime_execution_performed=False, observed_time_s=float(state["time"]),
    original_policy=policy, original_rotation_xyzw=context["rotation"].as_quat().tolist(),
    actual_state_global_rejections=context["reasons"], mouth_area_centroid_xy_m=anchor.tolist(),
    grid_spacing_m=.005, search_radius_m=.035, grid_plus_original_count=len(offsets),
    critical_slab_indices=sorted(critical), complete_slab_count=len(envelopes),
    necessary_screen_survivor_count=len(survivors), screened=screened, full_helper_batches=batches,
    accepted_full_preparation_candidate=accepted is not None,
    accepted_offset_xy_m=(np.asarray(accepted["report"]["selected_sections"]["center_of_mass_xy_target_m"]) - anchor).tolist() if accepted else None,
    candidate_report="independent-alternate-initial-offset-candidate.json" if accepted else None,
    input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (RUN / "config.json", RUN / "rest.npz", RUN / "lift_teddy.npz", RUN / "source/contact_insertion.py")},
    limits=["Only in-memory offset lists changed, in batches of at most25 as required by the unchanged preparation helper. Every other policy and physical threshold remains unchanged.",
        "Geometry screen rejects only existing necessary conditions; survivors require the full helper's all-slab checks and complete frozen robot/soft sweep route.",
        "The first accepted full helper candidate stops this bounded feasibility search; remaining survivors are not presumed to pass.",
        "A candidate is a CPU route from the saved actual lift. No force, deformation, grasp retention, actual entry/centering, final containment or physical task success is inferred."])
(RUN / "independent-alternate-initial-offset-search.json").write_text(json.dumps(output, indent=2,
    default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + "\n")
print(json.dumps({k: output[k] for k in ("grid_plus_original_count", "necessary_screen_survivor_count", "accepted_full_preparation_candidate", "accepted_offset_xy_m")}), flush=True)
