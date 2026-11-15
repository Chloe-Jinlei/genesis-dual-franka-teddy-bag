"""Saved completed-run diagnostic only: no native, Genesis, MuJoCo or planner imports.

Usage: python completed_run_native_axis_strain_review.py two_panel_full_38
The source-grounded normal-to-Z chart is the native BW chart, not texture UV,
construction directions, a first-edge frame, principal directions or measured yarns.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ac89e1009c7d8efb72d260790bb5ab31d7460542"
BASE_URL = f"https://github.com/spiriMirror/libuipc/blob/{COMMIT}/"
METRIC_NAMES = (
    "native_a_ratio", "native_b_ratio", "maximum_native_axis_ratio",
    "native_angle_change_deg", "absolute_native_angle_change_deg",
    "native_shear_invariant", "absolute_native_shear_invariant",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reject_live_pid(launch):
    pid = launch.get("pid")
    require(isinstance(pid, int) and not isinstance(pid, bool) and pid > 0,
            "A valid target launch PID is required for the completion check")
    # Fail conservatively even on PID reuse; never inspect another process's
    # command line or scan all processes (in particular no active sibling run).
    require(not (Path("/proc") / str(pid)).exists(),
            f"Refusing run with existing launch PID {pid}; active or reused PID")


def completion_contract(run, output):
    require(not output.exists(), f"Refusing to overwrite {output}")
    name = run.name
    number = name.removeprefix("two_panel_full_")
    launch_path = ROOT / "runtime_ipc" / f"full{number}-launch.json"
    required = [run / n for n in (
        "report.json", "config.json", "rest.npz", "states.npz", "events.json",
        "validation.json", "input_sha256.json",
    )] + [launch_path]
    require(all(p.is_file() for p in required), "Incomplete run input files")
    report, events, launch = read(run / "report.json"), read(run / "events.json"), read(launch_path)
    require(launch.get("output") == str(run), "Launch is bound to a different run")
    reject_live_pid(launch)
    require(report.get("status") == "completed" and not report.get("error"),
            "Refusing failed, interrupted or incomplete run")
    require(report.get("requested_task") == "full", "Only a completed full task is supported")
    require(isinstance(events, list) and events and events == report.get("events"),
            "Final report/event sequence disagreement")
    starts = np.array([e["start"] for e in events], float)
    durations = np.array([e["duration"] for e in events], float)
    require(np.isfinite(starts).all() and np.isfinite(durations).all()
            and np.all(np.diff(starts) > 0) and np.all(durations > 0),
            "Invalid event times/durations")
    require(len({e["phase"] for e in events}) == len(events), "Repeated phase names unsupported")
    require(events[-1]["phase"] == "final_settle", "Missing final settle phase")
    final_time = float(report["simulated_seconds"])
    dt = float(read(run / "config.json")["simulation"]["dt"])
    require(np.isfinite(dt) and dt > 0 and np.isfinite(final_time), "Invalid time configuration")
    require(abs(final_time - (starts[-1] + durations[-1])) <= dt + 1e-7,
            "Final settle not completed according to report")
    validation = read(run / "validation.json")
    require(abs(float(validation["duration_s"]) - final_time) < 1e-7,
            "Validation and report final times disagree")
    return report, events, launch, launch_path, required


def triangle_edges(points, faces):
    tri = points[faces]
    return np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=-1)


def native_inverse(reference_edges):
    """Literal pinned Dm2x2 construction, including the native cos==-1 branch."""
    normal = np.cross(reference_edges[:, :, 0], reference_edges[:, :, 1])
    double_area = np.linalg.norm(normal, axis=1)
    require(np.isfinite(double_area).all() and np.all(double_area > 0), "Degenerate reference triangle")
    normal /= double_area[:, None]
    vec = np.cross(normal, np.array([0., 0., 1.]))
    cosine = normal[:, 2]
    k = np.zeros((len(normal), 3, 3), float)
    k[:, 0, 1], k[:, 0, 2] = -vec[:, 2], vec[:, 1]
    k[:, 1, 0], k[:, 1, 2] = vec[:, 2], -vec[:, 0]
    k[:, 2, 0], k[:, 2, 1] = -vec[:, 1], vec[:, 0]
    rotation = np.broadcast_to(np.eye(3), k.shape).copy()
    special = cosine + 1 == 0
    regular = ~special
    rotation[regular] += k[regular] + (k[regular] @ k[regular]) / (1 + cosine[regular, None, None])
    rotation[special] = np.diag([-1., -1., 1.])
    dm = (rotation @ reference_edges)[:, :2, :]
    require(np.isfinite(dm).all(), "Nonfinite native rest chart")
    inv = np.linalg.inv(dm)
    require(np.isfinite(inv).all(), "Nonfinite native inverse rest chart")
    f0 = reference_edges @ inv
    errors = np.linalg.norm(np.swapaxes(f0, 1, 2) @ f0 - np.eye(2), axis=(1, 2))
    require(float(errors.max()) < 1e-9, "Native rest chart not numerically orthonormal")
    return inv, double_area / 2, {
        "maximum_rest_orthonormal_error": float(errors.max()),
        "cos_minus_one_branch_face_count": int(special.sum()),
        "maximum_Dm_condition_number": float(np.linalg.cond(dm).max()),
    }


def frame_metrics(edges, inverse):
    f = edges @ inverse
    a, b = f[:, :, 0], f[:, :, 1]
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    require(np.all(la > 0) and np.all(lb > 0), "Collapsed native material direction")
    shear = np.einsum("ni,ni->n", a, b)
    cosine = shear / (la * lb)
    require(np.isfinite(cosine).all() and np.all(np.abs(cosine) <= 1 + 1e-10),
            "Invalid native material angle")
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.))) - 90.
    metrics = np.stack((la, lb, np.maximum(la, lb), angle, np.abs(angle), shear, np.abs(shear)))
    require(np.isfinite(metrics).all(), "Nonfinite native strain metric")
    return metrics


def small_analytic_checks():
    edges = np.array([[[1., 0.], [0., 1.], [0., 0.]]])
    inverse, _, _ = native_inverse(edges)
    identity = frame_metrics(edges, inverse)[:, 0]
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    rigid = frame_metrics(rotation[None] @ edges, inverse)[:, 0]
    diagonal = frame_metrics(edges @ np.diag([1.02, .98]), inverse)[:, 0]
    shear = frame_metrics(edges @ np.array([[1., .1], [0., 1.]]), inverse)[:, 0]
    checks = {
        "identity_axes_and_angle": bool(np.allclose(identity, [1, 1, 1, 0, 0, 0, 0], atol=1e-12, rtol=0)),
        "current_rigid_rotation_invariance": bool(np.allclose(identity, rigid, atol=1e-12, rtol=0)),
        "known_native_diagonal_stretch": bool(np.allclose(diagonal[:3], [1.02, .98, 1.02], atol=1e-12, rtol=0)
                                                 and abs(diagonal[3]) < 1e-12),
        "known_simple_shear": bool(np.allclose(shear[:3], [1, np.sqrt(1.01), np.sqrt(1.01)], atol=1e-12, rtol=0)
                                   and abs(shear[5] - .1) < 1e-12
                                   and abs(shear[3] + np.degrees(np.arctan(.1))) < 1e-12),
    }
    require(all(checks.values()), "Analytic diagnostic regression failed")
    return checks


def weighted_quantiles(values, weights):
    order = np.argsort(values, kind="stable")
    total = float(weights.sum())
    require(total > 0, "Empty/zero-area region")
    cdf = np.cumsum(weights[order]) / total
    return {name: float(values[order[min(np.searchsorted(cdf, q), len(order) - 1)]])
            for name, q in (("minimum", 0.), ("p50", .5), ("p95", .95), ("p99", .99), ("maximum", 1.))}


def new_period():
    return {"sample_count": 0, "first_saved_time_s": None, "last_saved_time_s": None,
            "extrema": {}, "maximum_over_frames_spatial_area_weighted_p95": {},
            "last_saved_sample_in_period": None,
            "maximum_reference_area_fraction_native_axis_ratio_over": {}}


def witness(metrics, metric_index, face, time, frame_index, phase, faces, areas):
    return {"value": float(metrics[metric_index, face]), "time_s": float(time),
            "saved_unique_frame_index": int(frame_index), "phase_at_time": phase,
            "native_and_source_face_index": int(face), "vertex_indices": faces[face].tolist(),
            "reference_triangle_area_m2": float(areas[face]),
            "all_native_metrics_at_witness": {n: float(metrics[i, face]) for i, n in enumerate(METRIC_NAMES)}}


def update_period(period, stats, metrics, ids, time, frame_index, phase, faces, areas):
    period["sample_count"] += 1
    if period["first_saved_time_s"] is None:
        period["first_saved_time_s"] = float(time)
    period["last_saved_time_s"] = float(time)
    for k, name in enumerate(METRIC_NAMES):
        for sense, select in (("minimum", np.argmin), ("maximum", np.argmax)):
            face = int(ids[select(metrics[k, ids])])
            value = float(metrics[k, face])
            key = name + "_" + sense
            previous = period["extrema"].get(key)
            if previous is None or (value < previous["value"] if sense == "minimum" else value > previous["value"]):
                period["extrema"][key] = witness(metrics, k, face, time, frame_index, phase, faces, areas)
        p95 = stats["area_weighted_quantiles"][name]["p95"]
        previous = period["maximum_over_frames_spatial_area_weighted_p95"].get(name)
        if previous is None or p95 > previous["value"]:
            period["maximum_over_frames_spatial_area_weighted_p95"][name] = {
                "value": p95, "time_s": float(time), "phase_at_time": phase,
                "saved_unique_frame_index": int(frame_index),
            }
    for threshold, fraction in stats["reference_area_fraction_native_axis_ratio_over"].items():
        previous = period["maximum_reference_area_fraction_native_axis_ratio_over"].get(threshold)
        if previous is None or fraction > previous["fraction"]:
            period["maximum_reference_area_fraction_native_axis_ratio_over"][threshold] = {
                "fraction": fraction, "time_s": float(time), "phase_at_time": phase,
            }
    period["last_saved_sample_in_period"] = {"time_s": float(time), "phase_at_time": phase, **stats}


def main(run_name):
    require(re.fullmatch(r"two_panel_full_[1-9][0-9]*", run_name) is not None, "Invalid run_name")
    run = ROOT / "runs" / run_name
    output = ROOT / "runtime_ipc" / f"{run_name}-native-axis-strain-review.json"
    report, events, launch, launch_path, required = completion_contract(run, output)
    cfg = read(run / "config.json")
    asset_path = run / "source/assets" / cfg["bag"]["npz_file"]
    manifest = read(run / "input_sha256.json")
    require(all(sha(run / k) == v for k, v in manifest.items()), "Frozen input manifest mismatch")
    protected = required + [asset_path, Path(__file__)] + [run / k for k in manifest]
    baseline_path = ROOT / "runtime_ipc" / (run_name.removeprefix("two_panel_") + "-cloth-material-baseline.json")
    comparator = None
    if baseline_path.exists():
        old = read(baseline_path)
        require(old.get("source_run") == run_name, "Existing comparator belongs to another run")
        protected.append(baseline_path)
        comparator = {"path": str(baseline_path), "sha256": sha(baseline_path),
                      "scope": "Existing principal-SVD/construction-axis diagnostic is referenced unchanged; not recomputed."}
    checks = small_analytic_checks()
    hashes = {str(p): sha(p) for p in protected}
    with np.load(run / "rest.npz") as z:
        vertices, faces = z["bag"], z["bag_faces"]
    with np.load(asset_path) as z:
        asset = dict(z)
    require(np.array_equal(asset["faces"], faces), "Source/native face order differs; no implicit remapping")
    mapping_error = float(np.linalg.norm(vertices - (asset["points"] + cfg["bag"]["position"]), axis=1).max())
    require(mapping_error < 1e-7, "Source/native vertices differ; no implicit remapping")
    groups = {n: np.asarray(asset[n + "_face_indices"], int) for n in ("panel_front", "panel_back")}
    body = set(map(int, asset["body_face_indices"]))
    for side in ("front", "back"):
        members = set(map(int, asset["handle_" + side]))
        groups["handle_" + side] = np.array([i for i, f in enumerate(faces)
                                             if i not in body and set(map(int, f)) <= members], int)
    require(np.array_equal(np.sort(np.concatenate(list(groups.values()))), np.arange(len(faces))),
            "Four-region partition is not disjoint and complete")
    with np.load(run / "states.npz") as z:
        raw_times, positions = z["time"], z["bag"]
    require(np.isfinite(raw_times).all() and np.isfinite(positions).all(), "Nonfinite saved input")
    require(len(raw_times) == len(positions) and len(raw_times) > 1
            and np.all(np.diff(raw_times) >= -1e-9), "Invalid saved frame clock")
    keep = np.r_[np.diff(raw_times) > 1e-9, True]
    times, positions = raw_times[keep], positions[keep]
    require(np.all(np.diff(times) > 0) and abs(times[0]) < 1e-9,
            "Saved series lacks ordered time-zero reference")
    require(abs(times[-1] - report["simulated_seconds"]) < 1e-7,
            "Saved series does not include completed final state")
    require(np.max(np.abs(positions[0] - vertices)) < 1e-9, "Initial cloth is not saved rest geometry")
    inverse, area, chart_check = native_inverse(triangle_edges(vertices, faces))
    result_groups = {name: {"face_count": len(ids), "reference_area_m2": float(area[ids].sum()),
                            "periods": {"entire_run": new_period(), **{e["phase"]: new_period() for e in events}}}
                     for name, ids in groups.items()}
    starts = np.array([e["start"] for e in events])
    # Independent equality against all three previously checked native witnesses.
    prior_path = ROOT / "runtime_ipc/full38-three-peak-native-axis-review.json"
    witnesses = []
    if run_name == "two_panel_full_38":
        require(prior_path.is_file(), "Full38 three-witness regression reference missing")
        hashes[str(prior_path)] = sha(prior_path)
        witnesses = read(prior_path)["rows"]
    witness_results = []
    for frame_index, (time, points) in enumerate(zip(times, positions)):
        metrics = frame_metrics(triangle_edges(points, faces), inverse)
        phase = events[max(0, int(np.searchsorted(starts, time + 1e-8, side="right") - 1))]["phase"]
        if frame_index == 0:
            require(np.max(np.abs(metrics[:3] - 1)) < 1e-9 and np.max(np.abs(metrics[3:])) < 1e-7,
                    "Time-zero metrics are not identity")
        for w in witnesses:
            if abs(time - w["time_s"]) < 1e-8:
                f = w["face_index"]
                ok = (np.allclose(metrics[:2, f], w["native_chart_a_b_lengths"], atol=1e-10, rtol=0)
                      and abs(metrics[3, f] - w["native_chart_angle_change_deg"]) < 1e-8
                      and abs(metrics[5, f] - w["native_C"][0][1]) < 1e-10)
                require(ok, "Previously verified native witness disagrees")
                witness_results.append({"label": w["label"], "time_s": float(time), "face": f, "passed": bool(ok)})
        for name, ids in groups.items():
            weights = area[ids]
            stats = {"area_weighted_quantiles": {n: weighted_quantiles(metrics[k, ids], weights)
                                                for k, n in enumerate(METRIC_NAMES)},
                     "reference_area_fraction_native_axis_ratio_over": {
                         str(threshold): float(weights[metrics[2, ids] > threshold].sum() / weights.sum())
                         for threshold in (1.01, 1.02, 1.05)}}
            for period in ("entire_run", phase):
                update_period(result_groups[name]["periods"][period], stats, metrics, ids,
                              time, frame_index, phase, faces, area)
        if frame_index and frame_index % 500 == 0:
            print(json.dumps({"diagnostic_saved_frames_processed": frame_index, "total": len(times)}), flush=True)
    require(len(witness_results) == len(witnesses), "A prior witness was not covered")
    require(all(p["sample_count"] > 0 for g in result_groups.values() for p in g["periods"].values()),
            "An executed phase has no saved samples")
    reject_live_pid(launch)
    checks.update({"input_manifest_verified": True, "source_native_mapping_identity": True,
                   "four_regions_complete_disjoint": True, "saved_time_zero_identity": True,
                   "all_saved_metrics_finite": True, "completed_final_saved_frame_present": True,
                   "prior_three_witnesses_match_when_applicable": len(witness_results) == len(witnesses),
                   "protected_inputs_unchanged": all(sha(Path(p)) == h for p, h in hashes.items())})
    require(all(checks.values()), "Input changed or diagnostic check failed; report not written")
    result = {
        "kind": "completed_run_native_axis_strain_diagnostic", "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run), "run_name": run_name, "all_diagnostic_checks_passed": all(checks.values()), "checks": checks,
        "physical_task_success_claim_from_this_diagnostic": False, "native_simulation_performed": False,
        "material_calibration_performed": False,
        "completion_contract": {"recorded_report_status": report["status"], "recorded_task_passed": report.get("task_passed"),
                                "recorded_validation_status": read(run / "validation.json").get("status"),
                                "launch_pid_absent_before_and_after": True, "launch": str(launch_path),
                                "scope": "Read-only completion prerequisites, not a repeat task-success audit. Existing PID, incomplete records or existing output are refused."},
        "sampling": {"raw_saved_frames": len(raw_times), "unique_saved_frames": len(times),
                     "duplicates_removed_keep_last": int(len(raw_times) - len(times)),
                     "maximum_saved_interval_s": float(np.max(np.diff(times))),
                     "start_s": float(times[0]), "end_s": float(times[-1]), "native_face_count": len(faces),
                     "source_mapping_maximum_error_m": mapping_error,
                     "phase_rule": "Unique nonoverlapping attribution: next phase owns a shared boundary using start+1e-8 tolerance; a period endpoint is its last saved sample, not an interpolated phase endpoint."},
        "definitions": {
            "native_chart": "Literal source Dm2x2: rotate each initial/rest triangle normal toward world+Z, retain rotated XY edges, invert Dm; F=current_edges*inverse(Dm). Native a=(1,0), b=(0,1). Special cos==-1 branch is preserved exactly.",
            "native_a_b_ratio": "Norms of Fa and Fb in a numerically orthonormal rest chart. Value-1 is directional line strain in the native chart, not principal or measured yarn strain.",
            "angle": "acos((Fa dot Fb)/(|Fa||Fb|)) in degrees minus90. Signed angle and its absolute value are separately reported.",
            "shear_invariant": "I6=Fa dot Fb, dimensionless; native energy density is mu*I6^2. I6 also depends on the two column lengths; the normalized angle separates that length factor.",
            "p95": "Per-frame spatial triangle-rest-area weighted quantile; maximum_over_frames_spatial_area_weighted_p95 is the largest such p95 in the period. It is not a time-pooled percentile or a largest-5%-of-nodes metric.",
            "final": "The entire_run last_saved_sample_in_period is the actual completed final saved frame. Phase endpoints are explicitly the last saved samples in each phase.",
            "principal_construction_distinction": "No SVD/principal or construction-direction metrics are recomputed. Native axes vary with initial triangle orientation; neither they nor construction-grid axes identify real warp/weft without fabric measurements.",
            "rest_area": "Constant physical reference triangle area, not current area or vertex count.",
        },
        "chart_numerical_check": chart_check, "three_witness_regression": witness_results,
        "groups": result_groups, "existing_principal_construction_comparator": comparator,
        "bag_material_configuration": cfg["bag"],
        "source_formula_provenance": [
            {"url": BASE_URL + "src/backends/cuda/finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h",
             "location": "lines29–68: Dm2x2;101–129: energy", "formula": "W=lambda*sum[(norm(Fa_i)-1)^2+q*max(norm(Fa_i)-1,0)^3]+mu*(Fa dot Fb)^2"},
            {"url": BASE_URL + "src/backends/cuda/finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.cu",
             "location": "lines17–28: inverse chart;57–69: fixed a/b and rest area weighting"},
            {"path": str(ROOT / "runtime_ipc/full38_native_axis_bounded_review.py"),
             "scope": "Prior three-hotspot implementation; full38 numerical values are checked independently above."},
        ],
        "limitations": [
            "Saved frames are at most about50 ms apart; native-axis or shear peaks between them may be missed. This is not an integration-step strain log.",
            "The native chart is not a calibrated warp/weft chart. Low native-axis extension does not prove low extension in every material direction or inextensibility.",
            "Angle change can produce appreciable principal stretch even if both native-axis norms remain near1; do not interpret all principal stretch as yarn lengthening.",
            "An individual coarse triangle has no internal wrinkle degrees of freedom; continuum compression and shear are not direct fibre-scale measurements.",
            "No contact-force attribution, effective damping, rebound, hysteresis, textile calibration or future candidate success follows from these statistics.",
            "No native initialization/run, planner search, production/config/assets/default/docs/manifest edit, or observation of an active sibling run was performed.",
        ],
        "input_sha256": hashes, "script_sha256": sha(__file__),
    }
    with output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    compact = {}
    for name, group in result_groups.items():
        allp = group["periods"]["entire_run"]
        compact[name] = {
            "native_axis_max_percent": 100 * (allp["extrema"]["maximum_native_axis_ratio_maximum"]["value"] - 1),
            "max_spatial_native_axis_p95_percent": 100 * (allp["maximum_over_frames_spatial_area_weighted_p95"]["maximum_native_axis_ratio"]["value"] - 1),
            "maximum_abs_angle_change_deg": allp["extrema"]["absolute_native_angle_change_deg_maximum"]["value"],
            "final_native_axis_max_percent": 100 * (allp["last_saved_sample_in_period"]["area_weighted_quantiles"]["maximum_native_axis_ratio"]["maximum"] - 1),
            "final_native_axis_p95_percent": 100 * (allp["last_saved_sample_in_period"]["area_weighted_quantiles"]["maximum_native_axis_ratio"]["p95"] - 1),
        }
    print(json.dumps({"output": str(output), "sha256": sha(output), "checks_passed": len(checks),
                      "all_checks_passed": all(checks.values()), "regions": compact}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name")
    args = parser.parse_args()
    try:
        main(args.run_name)
    except Exception as exc:
        print(json.dumps({"status": "diagnostic_refused_or_failed", "error_type": type(exc).__name__,
                          "error": str(exc), "native_simulation_performed": False}), file=sys.stderr)
        raise SystemExit(2)
