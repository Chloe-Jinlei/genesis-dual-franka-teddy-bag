"""Small CPU-only synthetic regression for audit_staged_hand.

The fixture is not a simulation, physical result, or proof of grasp success.
Only this new regression and its JSON output are written; audit/runtime source
and existing run data remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import numpy as np


HERE = Path(__file__).resolve().parent
NAMES = ["link7", "left_finger", "right_finger"]


def fixture():
    """Three native surfaces, each with four noncoplanar reference vertices."""
    base = np.array([[0., 0., 0.], [.02, 0., .001],
                     [0., .015, .002], [0., 0., .008]])
    vertices = {"left_" + name: base * (1. + .1*k)
                for k, name in enumerate(NAMES)}
    geometry = []
    for k, name in enumerate(NAMES):
        points = vertices["left_" + name]
        selection = {"collision_body": name}
        if name == "link7":
            selection = {"collision_body": "hand", "merge_target": "link7",
                         "native_subset_max_error_m": 0.,
                         "native_subset_vertex_indices": [0, 1, 2, 3]}
        geometry.append({"link": name, "coupling_index": k,
                         "vertices": len(points), "selection": selection,
                         "local_bounds_m": [points.min(0).tolist(), points.max(0).tolist()],
                         "local_vertices_sha256": hashlib.sha256(
                             points.astype("<f8").tobytes()).hexdigest()})
    report = {
        "events": [{"phase": "lower_teddy_to_rim", "start": 0., "duration": .01},
                   {"phase": "center_teddy_at_rim", "start": .01, "duration": .01},
                   {"phase": "insert_teddy", "start": .02, "duration": .01},
                   {"phase": "release_teddy", "start": .03, "duration": .005}],
        "simulated_seconds": .04,
        "staged_native_hand_monitor": {"minimum_gap_m": .005,
            "bear_vertex_count": 4, "cached_reference_geometry": geometry},
    }
    rows, frames, frame_times, coupling = [], [], [], []
    for i, time in enumerate(np.arange(2, 7) * .005):
        matrices = []
        for k in range(3):
            matrix = np.eye(4)
            # Positive determinant, deliberately non-rigid affine transforms.
            matrix[:3, :3] = [[1.01 + .001*i, .012, 0.],
                              [0., .99 + .001*k, .008],
                              [-.04, .02, 1.002 + .001*i]]
            matrix[:3, 3] = [.1*k, -.02*k, 1.10 + .01*k + .0001*i]
            matrices.append(matrix)
        matrices = np.asarray(matrices)
        minima = np.array([np.min(vertices["left_" + name] @ matrix[2, :3]
                                  + matrix[2, 3])
                           for name, matrix in zip(NAMES, matrices)])
        bear = np.array([[0., 0., 1.01], [.01, 0., 1.02],
                         [0., .01, 1.03 + .0001*i], [0., 0., 1.00]])
        top = float(bear[:, 2].max())
        bottom = float(minima.min())
        rows.append({"time_s": float(time), "physics_step": i + 2,
                     "left_link_names": NAMES.copy(),
                     "left_distal_ipc_transforms": matrices.tolist(),
                     "left_link_minimum_z_m": minima.tolist(),
                     "left_distal_minimum_z_m": bottom,
                     "bear_vertex_count": len(bear), "bear_maximum_z_m": top,
                     "highest_bear_vertex_index": int(bear[:, 2].argmax()),
                     "vertical_gap_m": bottom - top, "minimum_gap_m": .005,
                     "stop_reason": None})
        if i in (0, 2, 4):
            frames.append(bear)
            frame_times.append(time)
            coupling.append({"time": float(time), "links": [
                {"entity": "left_franka", "link": "link7",
                 "translation_error_m": 0.},
                *[{"entity": "left_franka", "link": NAMES[k],
                   "ipc_transform": matrices[k].tolist()} for k in (1, 2)]]})
    trace = np.zeros((8, 19))
    trace[:, 0] = np.arange(1, 9) * .005
    return {"report": report, "rows": rows, "vertices": vertices,
            "states": {"time": np.asarray(frame_times), "bear": np.asarray(frames)},
            "trace": trace, "coupling": coupling, "dt": .005}


def mutate(case, data):
    if case == "delete_first_row":
        data["rows"].pop(0)
    elif case == "delete_last_row":
        data["rows"].pop()
    elif case == "repeat_physics_step":
        data["rows"][2]["physics_step"] = data["rows"][1]["physics_step"]
    elif case == "lower_threshold_in_rows_and_metadata":
        for row in data["rows"]:
            row["minimum_gap_m"] = .001
        data["report"]["staged_native_hand_monitor"]["minimum_gap_m"] = .001
    elif case == "nonfinite_gap":
        data["rows"][2]["vertical_gap_m"] = float("nan")
    elif case == "tamper_native_affine_matrix":
        # Non-Z shear leaves height summaries unchanged; independent saved
        # finger transform comparison must still detect this alteration.
        data["rows"][2]["left_distal_ipc_transforms"][1][0][1] += .03
    elif case == "argmax_index_out_of_range":
        data["rows"][2]["highest_bear_vertex_index"] = 4
    elif case == "full_bear_frame_disagrees":
        data["states"]["bear"][1, 2, 2] += .02
    elif case != "valid_affine_fixture":
        raise ValueError(case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "staged_height_audit_regression.json")
    args = parser.parse_args()
    audit_path = HERE / "audit_guarded_run.py"
    before = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("synthetic_staged_height_audit", audit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(audit_path.read_bytes()).hexdigest() != before:
        raise RuntimeError("Audit source changed during import; retry with a stable source")
    cases = ["valid_affine_fixture", "delete_first_row", "delete_last_row",
             "repeat_physics_step", "lower_threshold_in_rows_and_metadata",
             "nonfinite_gap", "tamper_native_affine_matrix",
             "argmax_index_out_of_range", "full_bear_frame_disagrees"]
    results = []
    initial = fixture()
    with tempfile.TemporaryDirectory(prefix="staged_height_audit_regression_") as temp:
        for case in cases:
            data = copy.deepcopy(initial)
            mutate(case, data)
            run = Path(temp) / case
            run.mkdir()
            np.savez_compressed(run / "staged_native_hand_reference.npz", **data["vertices"])
            (run / "staged_hand_trace.json").write_text(json.dumps(data["rows"]))
            (run / "coupling_trace.json").write_text(json.dumps(data["coupling"]))
            expected = case == "valid_affine_fixture"
            try:
                actual = module.audit_staged_hand(run, data["report"], data["states"],
                                                 data["trace"], data["dt"])
                passed = bool(actual.get("passed")) == expected
                if not expected:
                    passed = passed and actual.get("status") == "failed"
                results.append({"case": case, "expected_audit_passed": expected,
                                "regression_passed": passed, "audit_result": actual})
            except Exception as error:
                results.append({"case": case, "expected_audit_passed": expected,
                                "regression_passed": False,
                                "unexpected_exception": type(error).__name__ + ": " + str(error)})
    output = {"kind": "synthetic_native_height_postrun_audit_regression",
              "physical_success": False, "runtime_execution_performed": False,
              "fixture_is_synthetic": True, "audit_source_modified": False,
              "fixture": {"native_reference_bodies": 3, "vertices_per_body": 4,
                          "trace_rows": 5, "full_bear_frames": 3,
                          "native_finger_transform_crosschecks": 6,
                          "center_start_s": .01, "insert_endpoint_release_start_s": .03,
                          "integration_step_s": .005, "native_affine_is_nonrigid": True},
              "source_sha256_at_import": before,
              "source_sha256_at_finish": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
              "regression_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "passed_count": sum(r["regression_passed"] for r in results),
              "test_count": len(results), "tests": results,
              "limitations": ["Tests only audit_staged_hand on a small synthetic fixture.",
                              "Does not validate physical runs, native monitor acquisition, staged sequence audit, or dynamics."]}
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({"report": str(args.output), "passed_count": output["passed_count"],
                      "test_count": output["test_count"], "source_sha256": before,
                      "failed_cases": [r["case"] for r in results if not r["regression_passed"]]}, indent=2))
    raise SystemExit(0 if output["passed_count"] == output["test_count"] else 1)


if __name__ == "__main__":
    main()
