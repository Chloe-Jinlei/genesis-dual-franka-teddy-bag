"""Bounded saved-state-only Full38/39 COM-preflight comparison; no native imports."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime_ipc/full39-independent-preflight-material-review.json"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def need(x, message):
    if not x: raise ValueError(message)

def point_review(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    mask = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    crossings = a[mask, 0] + (point[1] - a[mask, 1]) * (b[mask, 0] - a[mask, 0]) / (b[mask, 1] - a[mask, 1])
    delta = b - a
    u = np.clip(np.sum((point - a) * delta, axis=1) / np.sum(delta * delta, axis=1), 0, 1)
    nearest = a + u[:, None] * delta
    distances = np.linalg.norm(point - nearest, axis=1)
    edge = int(np.argmin(distances))
    distance = float(distances[edge])
    inside = bool(np.count_nonzero(point[0] < crossings) % 2) or distance < 1e-8
    return {"inside": inside, "boundary_distance_m": distance,
            "signed_inside_distance_m": distance if inside else -distance,
            "nearest_ordered_edge": [edge, (edge + 1) % len(polygon)],
            "nearest_boundary_xy_m": nearest[edge].tolist(),
            "horizontal_line_x_intersections_m": sorted(crossings.tolist())}

def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [r for k in sorted(set(a) | set(b))
                for r in differences(a.get(k), b.get(k), prefix + "/" + k)]
    return [] if a == b else [{"path": prefix, "full38": a, "full39": b}]

def main():
    need(not OUT.exists(), "Refusing existing output")
    protected, rows, configs, manifests = {}, {}, {}, {}
    for number in (38, 39):
        run = ROOT / "runs" / f"two_panel_full_{number}"
        launch_path = ROOT / "runtime_ipc" / f"full{number}-launch.json"
        launch = read(launch_path)
        need(launch["output"] == str(run), "Launch binding differs")
        need(not (Path("/proc") / str(launch["pid"])).exists(), "Target launch PID active or reused")
        manifest = read(run / "input_sha256.json")
        need(all(sha(run / k) == v for k, v in manifest.items()), "Frozen manifest mismatch")
        manifests[number] = manifest
        paths = [run / n for n in ("config.json", "report.json", "guarded_contact_candidate.json", "orient_teddy_postlift.npz", "rest.npz", "input_sha256.json")] + [launch_path]
        protected.update({str(p): sha(p) for p in paths})
        cfg = configs[number] = read(run / "config.json")
        report = read(run / "report.json")
        candidate = read(run / "guarded_contact_candidate.json")
        q, mouth = candidate["report"], candidate["report"]["mouth"]
        with np.load(run / "orient_teddy_postlift.npz") as z: state = dict(z)
        with np.load(run / "rest.npz") as z: rest = dict(z)
        time = float(np.asarray(state["time"]).reshape(-1)[0])
        need(abs(time - q["observed_time_s"]) < 1e-9, "Source time differs")
        rim_ids = np.asarray(mouth["ordered_rim_indices"], int)
        rim = state["bag"][rim_ids]
        need(np.array_equal(rim, np.asarray(mouth["ordered_rim_world_m"])), "Recorded rim differs from actual saved state")
        polygon = rim[:, :2]
        nxt = np.roll(polygon, -1, axis=0)
        cross = polygon[:, 0] * nxt[:, 1] - nxt[:, 0] * polygon[:, 1]
        signed_area = float(cross.sum() / 2)
        centroid = np.sum((polygon + nxt) * cross[:, None], axis=0) / (6 * signed_area)
        need(abs(abs(signed_area) - mouth["projected_area_m2"]) < 1e-12, "Area differs")
        need(np.allclose(centroid, q["observed_mouth_area_centroid_xy_m"], atol=1e-12, rtol=0), "Area centroid differs")
        offsets = np.asarray(q["declared_com_offsets_from_mouth_area_centroid_m"])
        need(np.array_equal(offsets, [[-.01, .03075]]), "Unexpected declared candidates")
        target = centroid + offsets[0]
        check = point_review(target, polygon)
        recorded = q["centering_com_checks"][0]
        need(np.allclose(target, recorded["center_of_mass_xy_target_m"], atol=1e-12, rtol=0), "COM target differs")
        need(check["inside"] == recorded["projected_inside"] and abs(check["boundary_distance_m"] - recorded["boundary_distance_m"]) < 1e-12, "Independent COM gate differs")
        need(recorded["required_boundary_clearance_m"] == .002, "Boundary gate changed")
        need(recorded["admissible"] == (check["inside"] and check["boundary_distance_m"] >= .002), "Admissibility differs")
        hull = ConvexHull(polygon)
        intervals = np.asarray(check["horizontal_line_x_intersections_m"])
        need(len(intervals) == 2, "Unexpected diagnostic horizontal crossing count")
        heights = float(np.ptp(state["bear"][:, 2]))
        need(abs(heights - q["vertical_capacity"]["observed_bear_height_m"]) < 1e-12, "Bear height differs")
        check.update({"target_xy_m": target.tolist(), "signed_margin_after_required_2mm_m": check["signed_inside_distance_m"] - .002,
                      "nearest_native_rim_nodes": rim_ids[check["nearest_ordered_edge"]].tolist(),
                      "inside_convex_hull_diagnostic_only": point_review(target, polygon[hull.vertices])["inside"],
                      "horizontal_open_interval_width_at_target_y_m": float(intervals[1] - intervals[0])})
        rows[str(number)] = {"run_path": str(run), "saved_state": str(run / "orient_teddy_postlift.npz"),
            "actual_source_time_s": time, "phase": "orient_teddy_postlift endpoint, initial guarded-contact preflight",
            "run_status": report["status"], "candidate_status": candidate["status"], "rejections": q["rejections"],
            "declared_com_offsets_m": offsets.tolist(), "mouth_area_centroid_xy_m": centroid.tolist(),
            "mouth_projected_area_m2": abs(signed_area), "mouth_convex_hull_area_m2_diagnostic": float(hull.volume),
            "actual_polygon_to_hull_area_ratio": abs(signed_area) / float(hull.volume),
            "recorded_projected_crossings": mouth["projected_crossings"],
            "recorded_minimum_caliper_width_m": mouth["projected_minimum_caliper_width_m"],
            "rim_z_range_m": [float(rim[:, 2].min()), float(rim[:, 2].max())],
            "actual_rim_point_count": len(rim), "vertical_capacity": q["vertical_capacity"],
            "independent_declared_COM_test": check,
            "number_section_candidates_evaluated": len(q["centering_candidates"]),
            "number_robot_attempts_evaluated": len(q["robot_centering_attempts"]),
            "observed_physical_state": {k: q["physical_state"][k] for k in ("minimum_tetra_volume_ratio", "maximum_bag_edge_stretch_ratio", "bag_table_penetration_m", "bear_table_penetration_m", "physical_state_rejections")},
            "policy": q["policy"]}
    diff = differences(configs[38], configs[39])
    need(diff == [{"path": "/bag/handle_shear_stiffness", "full38": 250.0, "full39": 500.0}], "Unexpected config difference")
    source_diff = [k for k in sorted(set(manifests[38]) | set(manifests[39])) if k != "config.json" and manifests[38].get(k) != manifests[39].get(k)]
    need(not source_diff, "Frozen source/asset differences")
    need(rows["39"]["rejections"] == ["No planned COM lies inside actual projected opening with 2 mm margin"], "Unexpected failure causes")
    need(rows["39"]["number_section_candidates_evaluated"] == 0 and rows["39"]["number_robot_attempts_evaluated"] == 0, "Unexpected downstream planning")
    need(rows["38"]["policy"] == rows["39"]["policy"], "Policy differs")
    protected[str(Path(__file__).resolve())] = sha(__file__)
    result = {"kind": "independent_same_phase_preflight_COM_geometry_review", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Only immutable completed Full38/failed Full39 inputs; no new candidate planning, native simulation or production changes.",
        "observations": rows, "config_differences": diff, "nonconfig_frozen_manifest_differences": source_diff,
        "independent_checks_passed": True,
        "interpretation_zh": [
            "Full39的唯一预声明COM候选在实际有序口缘的世界XY投影外8.636毫米；它在凸包内不构成通过依据。Full38相同相对偏移在实际投影内42.986毫米。",
            "两次同为23.315秒的实际转腕末态；绝对COM目标各自依赖本轮实际袋口面积质心，不能把相同偏移误称相同世界坐标。",
            "Full39高度余量为正53.099毫米，不是最低口缘高度拒绝。未执行运输或插入，也未进入section候选或机器人路径尝试。",
            "该拒绝限定于唯一声明偏移[-10,+30.75]毫米；35毫米只约束偏移半径，程序没有搜索整片区域。它不证明所有抓姿/位置物理无解，亦不授权绕过原门。",
            "区域面积与凹陷是实际观测差异，单次38/39比较不能给出提带剪切刚度到袋口形状的单调或独立因果关系。"
        ],
        "bounded_next_material_hypothesis": {"proposed_handle_shear_N_per_m": 350., "relative_to_completed_full38_factor": 1.4,
            "unchanged": "Full38 other material values, bag/teddy geometry, control, solver and all acceptance gates",
            "basis_zh": "在已通过的250与被拒绝的500之间，仅作一次较小提带抗剪刚度增量；此前Full38全帧显示提带夹角变形明显，故此区域参数方向有机制依据。",
            "not_established_zh": "350未经织物标定，不能保证袋口更宽、COM门通过、主伸长下降或全流程成功；必须重新从原生初态观察及保留原最终验收。"},
        "source_contract": {"COM_guard": "source/contact_insertion.py:434-447, actual ordered rim XY polygon, point inside AND >=2mm boundary distance",
            "target": "source/contact_insertion.py:380-396, actual polygon area centroid + declared offsets",
            "point_predicate": "source/insertion_candidates.py:21-36, ray crossing + nearest segment distance",
            "body_boundary": "source/validate.py:101-157, body opening boundary excluding handle loops",
            "independent_review": "Separate child read-only source inspection confirmed unchanged contracts; no production import or native initialization used."},
        "input_sha256": protected, "all_protected_inputs_unchanged": all(sha(p) == h for p, h in protected.items()),
        "future_physical_success_claim": False, "new_simulation_performed": False}
    need(result["all_protected_inputs_unchanged"], "Input changed")
    with OUT.open("x") as f: json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"path": str(OUT), "sha256": sha(OUT), "config_diff": diff, "checks_passed": True}))

if __name__ == "__main__": main()
