"""Bounded CPU regression for staged spatial-rim engineering geometry.

Synthetic loops are mathematical fixtures, not physical simulations. The actual
case uses only full13's saved body-rim vertices; no scene, engine, GPU, forces,
whole-teddy contact scan, or insertion-success inference is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
sys.path.insert(0, str(SCENE))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def bow_tie(height):
    return np.array([[-.05, -.05, 0.], [.05, .05, 0.],
                     [-.05, .05, height], [.05, -.05, height]])


def segment_distance(a, b, c, d):
    """Independent constrained quadratic: four boundaries plus interior."""
    u, v, w = b-a, d-c, a-c
    aa, bb, ee, cc, ff = (float(u@u), float(u@v), float(v@v),
                         float(u@w), float(v@w))
    candidates = []
    for s in (0., 1.):
        candidates.append((s, float(np.clip((ff+bb*s)/ee, 0., 1.)) if ee else 0.))
    for t in (0., 1.):
        candidates.append((float(np.clip((bb*t-cc)/aa, 0., 1.)) if aa else 0., t))
    det = aa*ee-bb*bb
    if det > 16*np.finfo(float).eps*aa*ee:
        s, t = (bb*ff-cc*ee)/det, (aa*ff-bb*cc)/det
        if 0. <= s <= 1. and 0. <= t <= 1.:
            candidates.append((s, t))
    return min(float(np.linalg.norm(a+s*u-c-t*v)) for s, t in candidates)


def independent_minimum(points):
    n = len(points)
    return min(segment_distance(points[i], points[(i+1)%n], points[j], points[(j+1)%n])
               for i in range(n) for j in range(i+2, n)
               if not (i == 0 and j == n-1))


def cases():
    square = np.array([[-.05, -.05, 0.], [.05, -.05, 0.],
                       [.05, .05, 0.], [-.05, .05, 0.]])
    nan_points = square.copy()
    nan_points[1, 2] = np.nan
    duplicate = square.copy()
    duplicate[1] = duplicate[0]
    collinear = np.column_stack((np.arange(4)*.02, np.zeros((4, 2))))
    yield dict(name="simple_square", points=square, reference=square.copy(), radius=.0003,
               accept=True, expected_minimum=.1, expected_crossings=0)
    yield dict(name="xy_crossing_with_20mm_height_separation", points=bow_tie(.02),
               reference=square.copy(), radius=.0003, accept=True,
               expected_minimum=.02, expected_crossings=1)
    yield dict(name="genuine_3d_interior_crossing", points=bow_tie(0.),
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="gap_0_5mm_below_0_6mm", points=bow_tie(.0005),
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="gap_5nm_above_threshold_guard_rejects", points=bow_tie(.000600005),
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="shell_diameter_larger_than_gap", points=bow_tie(.0015),
               reference=square.copy(), radius=.001, accept=False)
    yield dict(name="degenerate_collinear_loop", points=collinear,
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="duplicate_adjacent_vertex", points=duplicate,
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="nonfinite_current_vertex", points=nan_points,
               reference=square.copy(), radius=.0003, accept=False)
    yield dict(name="mismatched_reference_shape", points=square,
               reference=square[:3].copy(), radius=.0003, accept=False)

    from validate import mouth_structure
    run = SCENE / "runs/two_panel_full_13"
    config = json.loads((run / "config.json").read_text())
    with np.load(run / "rest.npz", allow_pickle=False) as data:
        rest = {key: data[key].copy() for key in data.files}
    with np.load(run / "lower_teddy_to_rim.npz", allow_pickle=False) as data:
        state = {key: data[key].copy() for key in data.files}
    _, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    assert topology["two_panels_directly_sewn_along_u"] and len(rim) == 74
    yield dict(name="actual_full13_lower_teddy_to_rim", points=state["bag"][rim],
               reference=rest["bag"][rim], radius=float(config["bag"]["thickness"]), accept=True,
               expected_minimum=.0022814178295033423, expected_crossings=2,
               actual_time_s=float(state["time"]), topology=topology,
               ordered_rim_source="mouth_structure -> boundary_loops(body_faces); asset rim is membership only",
               source_hashes={str(path.relative_to(SCENE)): sha256(path)
                              for path in (run / "rest.npz", run / "config.json", run / "lower_teddy_to_rim.npz")})


def check_success(result, fixture):
    """Check hull/triangulation without treating hull as a physical opening."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise AssertionError("Expected (polygon, triangles, evidence)")
    polygon, triangles, evidence = result
    polygon, triangles = np.asarray(polygon, float), np.asarray(triangles, float)
    assert polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3
    assert triangles.ndim == 3 and triangles.shape[1:] == (3, 2)
    assert np.isfinite(polygon).all() and np.isfinite(triangles).all()
    assert isinstance(evidence, dict)
    hull = ConvexHull(fixture["points"][:, :2])
    expected_area = float(hull.volume)
    polygon_area = abs(float(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                                   -polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2)
    u, v = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    triangle_area = float(np.abs(u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]).sum()/2)
    assert np.isclose(polygon_area, expected_area, atol=1e-12, rtol=1e-10)
    assert np.isclose(triangle_area, expected_area, atol=1e-12, rtol=1e-10)
    for vertices in (polygon, triangles.reshape(-1, 2)):
        assert np.max(vertices @ hull.equations[:, :2].T + hull.equations[:, 2]) <= 1e-10
    # Compare evidence fields after the helper contract is available.
    minimum = independent_minimum(fixture["points"])
    assert abs(minimum-fixture["expected_minimum"]) < 1e-10
    from validate import proper_segment_crossings
    crossings = proper_segment_crossings(fixture["points"][:, :2])
    assert crossings == fixture["expected_crossings"]
    assert evidence["representation"] == "spatial_loop_contact_envelope"
    assert evidence["is_actual_opening"] is False
    assert evidence["does_not_certify_open_aperture_or_passage"] is True
    assert evidence["actual_rim_xy_projection_crossings"] == crossings
    assert abs(evidence["minimum_nonadjacent_3d_edge_distance_m"]-minimum) < 1e-10
    n = len(fixture["points"])
    assert evidence["nonadjacent_3d_pair_count"] == n*(n-3)//2
    guard = float(evidence["edge_distance_numerical_guard_m"])
    assert guard == 1e-8
    assert abs(evidence["minimum_nonadjacent_3d_edge_distance_lower_bound_m"]-(minimum-guard)) < 1e-12
    assert evidence["required_nonadjacent_3d_edge_separation_m"] == max(.0006, 2*fixture["radius"])
    assert minimum-guard > evidence["required_nonadjacent_3d_edge_separation_m"]
    return dict(evidence=evidence, convex_hull_area_m2=expected_area,
                triangulated_area_m2=triangle_area,
                independent_nonadjacent_segment_minimum_m=minimum,
                independent_projected_proper_crossings=crossings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "staged_spatial_loop_geometry_regression.json")
    args = parser.parse_args()
    import staged_contact_insertion as module
    source = SCENE / "staged_contact_insertion.py"
    initial_hash = sha256(source)
    helper = getattr(module, "_spatial_loop_geometry", None)
    if helper is None:
        print("Pending: _spatial_loop_geometry has not been installed; no test result claimed.")
        return 2
    rows = []
    for fixture in cases():
        points = np.asarray(fixture["points"], float).copy()
        reference = np.asarray(fixture["reference"], float).copy()
        original_points, original_reference = points.copy(), reference.copy()
        points.setflags(write=False)
        reference.setflags(write=False)
        row = {key: value for key, value in fixture.items() if key not in ("points", "reference")}
        try:
            result = helper(points, reference, fixture["radius"])
        except Exception as exc:
            row.update(actual_accept=False, exception_type=type(exc).__name__, exception=str(exc))
            row["passed"] = not fixture["accept"] and isinstance(exc, ValueError)
        else:
            row["actual_accept"] = True
            row["passed"] = fixture["accept"]
            if fixture["accept"]:
                try:
                    row.update(check_success(result, fixture))
                except Exception as exc:
                    row.update(passed=False, validation_error=f"{type(exc).__name__}: {exc}")
        unchanged = (np.array_equal(points, original_points, equal_nan=True)
                     and np.array_equal(reference, original_reference, equal_nan=True))
        row["inputs_unchanged"] = unchanged
        row["passed"] = bool(row["passed"] and unchanged)
        rows.append(row)
    final_hash = sha256(source)
    report = dict(kind="staged_spatial_loop_geometry_cpu_regression", source=str(source),
                  source_sha256=initial_hash, source_sha256_after=final_hash,
                  source_unchanged_during_run=initial_hash == final_hash,
                  script_sha256=sha256(Path(__file__)),
                  passed=all(row["passed"] for row in rows) and initial_hash == final_hash,
                  total_cases=len(rows), passed_cases=sum(row["passed"] for row in rows), cases=rows,
                  physical_execution_performed=False, physical_success_claim=False,
                  scope="Pure rim function only; topology wrapper, full section planning, robot route and native dynamics not executed",
                  limitations=["Convex XY hull is an engineering envelope, not the actual spatial opening.",
                               "Rim segment separation does not establish body-surface separation or future insertion success."])
    args.output.write_text(json.dumps(report, default=serialize, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": report["passed"], "cases": len(rows),
                      "passed_cases": report["passed_cases"], "source_sha256": initial_hash,
                      "failed": [row for row in rows if not row["passed"]],
                      "output": str(args.output)}, default=serialize, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
