"""Apply the immutable full30 diagnostic method to a completed subsequent run.

Example (only after the run finishes):
    .venv/bin/python runtime_ipc/compare_cloth_material_run.py two_panel_full_31

No simulation, planning, material change, or new deformation metrics. The full30
method/report stay byte-identical. Refuse active/incomplete runs and overwrites.
"""
from pathlib import Path
import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SCRIPT = ROOT / "runtime_ipc/full30_cloth_material_baseline.py"
REFERENCE_REPORT = ROOT / "runtime_ipc/full30-cloth-material-baseline.json"
REFERENCE_RUN = ROOT / "runs/two_panel_full_30"
SCRIPT_SHA256 = "2bc32f735837f90f7cb7f44f21616f8194db9985317293cfbe541307ebb44b47"
REPORT_SHA256 = "ec66543e57231e8b544ec110b7fcc9c7f8470ccc049dcb5612dd9452c383d177"
GROUPS = ("panel_front", "panel_back", "handle_front", "handle_back")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_method(run, output):
    """Change only two reporting labels in memory; override two input/output paths.

    Mathematical functions, loops, sampling, metrics and reference directions are
    loaded from the pinned original source without edits. Its guarded main is
    never triggered by module loading.
    """
    if sha(REFERENCE_SCRIPT) != SCRIPT_SHA256 or sha(REFERENCE_REPORT) != REPORT_SHA256:
        raise ValueError("Immutable full30 diagnostic source/report hash mismatch")
    labels = {
        "independent_full30_cloth_material_baseline":
            "independent_"+run.name.removeprefix("two_panel_")+"_cloth_material_baseline",
        "two_panel_full_30": run.name,
    }
    replacements = {key: 0 for key in labels}

    class Labels(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in labels:
                replacements[node.value] += 1
                return ast.copy_location(ast.Constant(labels[node.value]), node)
            return node

    tree = Labels().visit(ast.parse(REFERENCE_SCRIPT.read_text()))
    if replacements != {key: 1 for key in labels}:
        raise ValueError("Unexpected reference reporting-label structure")
    namespace = {"__name__": "immutable_cloth_baseline_method", "__file__": str(REFERENCE_SCRIPT)}
    exec(compile(ast.fix_missing_locations(tree), str(REFERENCE_SCRIPT), "exec"), namespace)
    namespace.update(ROOT=ROOT, RUN=run, OUTPUT=output)
    return namespace, replacements


def ready_inputs(run):
    """Require finalized complete output before loading any deformation arrays."""
    required = ("report.json", "config.json", "rest.npz", "states.npz", "soft_metrics.npy",
                "events.json", "input_sha256.json")
    absent = [name for name in required if not (run/name).is_file()]
    if absent:
        raise ValueError("Finalized run evidence missing: "+", ".join(absent))
    report = json.loads((run/"report.json").read_text())
    if report.get("status") != "completed" or report.get("sequence_completed") is not True or report.get("error"):
        raise ValueError("This wrapper requires a completed, error-free native sequence; partial output is not analyzed")
    launch_path = ROOT/"runtime_ipc"/(run.name.removeprefix("two_panel_")+"-launch.json")
    launch = json.loads(launch_path.read_text())
    try:
        command = [v.decode() for v in (Path("/proc")/str(int(launch["pid"]))/"cmdline").read_bytes().split(b"\0") if v]
    except FileNotFoundError:
        command = []
    if command and "--output" in command and command[command.index("--output")+1] == run.name:
        raise ValueError("Native process for this run is still active")
    events = json.loads((run/"events.json").read_text())
    if not events or events[-1].get("phase") != "final_settle" or events != report["events"]:
        raise ValueError("Final settle/event output is incomplete or inconsistent")
    cfg = json.loads((run/"config.json").read_text())
    frozen = json.loads((run/"input_sha256.json").read_text())
    for key in ("config.json", "source/assets/"+cfg["bag"]["npz_file"], "source/two_panel_builder.py"):
        if sha(run/key) != frozen[key]:
            raise ValueError("Frozen method input hash mismatch: "+key)
    import numpy as np
    states = np.load(run/"states.npz")
    soft = np.load(run/"soft_metrics.npy")
    end = float(report["simulated_seconds"])
    expected_end = events[-1]["start"]+np.ceil(events[-1]["duration"]/cfg["simulation"]["dt"])*cfg["simulation"]["dt"]
    if not (np.isfinite(end) and abs(float(states["time"][-1])-end)<1e-8
            and abs(float(soft[-1,0])-end)<1e-8 and abs(expected_end-end)<1e-8):
        raise ValueError("Final states/soft summaries do not cover the completed native endpoint")
    own_rest, ref_rest = np.load(run/"rest.npz"), np.load(REFERENCE_RUN/"rest.npz")
    if not all(np.array_equal(own_rest[key],ref_rest[key]) for key in ("bag", "bag_faces")):
        raise ValueError("This direct comparison requires identical native cloth rest geometry/topology")
    hashes = {str(run/name):sha(run/name) for name in required}
    hashes[str(launch_path)] = sha(launch_path)
    return report, cfg, hashes, {"native_process_for_run_active":False,
        "completed_sequence_endpoint_time_s":end,"cloth_rest_exact_full30":True,
        "native_report_task_passed":report.get("task_passed"),
        "scope":"Finalization readiness only; reported task status is not independently recertified here."}


def peak_comparison(old, new):
    if set(old) != set(new):
        raise ValueError("Metric inventory differs from immutable baseline")
    result = {}
    for name, a in old.items():
        b = new[name]
        row = {"full30_value":a["value"],"new_value":b["value"],
               "difference_new_minus_full30":b["value"]-a["value"],
               "full30_peak_time_s":a["time_s"],"new_peak_time_s":b["time_s"]}
        if name == "shear_angle_change_deg":
            row.update(full30_absolute_degrees=a["absolute_value"],
                       new_absolute_degrees=b["absolute_value"],
                       absolute_magnitude_difference_degrees=b["absolute_value"]-a["absolute_value"])
        else:
            row["ratio_difference_percentage_points"] = 100*(b["value"]-a["value"])
        result[name] = row
    return result


def final_comparison(old, new):
    result = {"full30_time_s":old["time_s"],"new_time_s":new["time_s"],"same_area_weighted_quantiles":{}}
    for name, a in old["area_weighted_triangle_quantiles"].items():
        b = new["area_weighted_triangle_quantiles"][name]
        result["same_area_weighted_quantiles"][name] = {
            q:{"full30":value,"new":b[q],"difference_new_minus_full30":b[q]-value}
            for q,value in a.items()}
    return result


def comparisons(old, new):
    if tuple(old["groups"]) != GROUPS or tuple(new["groups"]) != GROUPS:
        raise ValueError("Group inventory differs from baseline")
    result = {}
    for name in GROUPS:
        a,b = old["groups"][name],new["groups"][name]
        if a["face_count"] != b["face_count"] or set(a["each_recorded_phase"]) != set(b["each_recorded_phase"]):
            raise ValueError("Group faces or phase inventory differ from baseline")
        result[name] = {
            "entire_run_peak_metrics":peak_comparison(a["periods"]["entire_run"]["extrema"],b["periods"]["entire_run"]["extrema"]),
            "opening_period_peak_metrics":peak_comparison(a["periods"]["bag_opening_close_drape_spread"]["extrema"],b["periods"]["bag_opening_close_drape_spread"]["extrema"]),
            "opening_endpoint":final_comparison(a["periods"]["bag_opening_close_drape_spread"]["endpoint"],b["periods"]["bag_opening_close_drape_spread"]["endpoint"]),
            "final":final_comparison(a["final"],b["final"]),
            "each_recorded_phase":{phase:{"peak_metrics":peak_comparison(a["each_recorded_phase"][phase]["extrema"],b["each_recorded_phase"][phase]["extrema"]),
                "endpoint":final_comparison(a["each_recorded_phase"][phase]["endpoint"],b["each_recorded_phase"][phase]["endpoint"])} for phase in a["each_recorded_phase"]}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_name", help="Completed run, e.g. two_panel_full_31")
    args = parser.parse_args()
    if not re.fullmatch(r"two_panel_full_\d+",args.run_name) or args.run_name == REFERENCE_RUN.name:
        raise ValueError("Provide a subsequent full-run directory name; never overwrite full30")
    run = ROOT/"runs"/args.run_name
    output = ROOT/"runtime_ipc"/(args.run_name.removeprefix("two_panel_")+"-cloth-material-baseline.json")
    if output.exists():
        raise FileExistsError(output)
    report,cfg,inputs,readiness = ready_inputs(run)
    with tempfile.TemporaryDirectory(prefix="cloth-baseline-staging-",dir=ROOT/"runtime_ipc") as directory:
        temporary = Path(directory)/"report.json"
        method, labels = load_method(run,temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            method["main"]()
        measured = json.loads(temporary.read_text())
    if measured["source_run"] != args.run_name or not measured["all_diagnostic_checks_passed"]:
        raise ValueError("Method output label/check mismatch")
    reference = json.loads(REFERENCE_REPORT.read_text())
    measured["material_configuration"] = measured.pop("old_material_configuration")
    measured["method_reuse"] = {"wrapper":str(Path(__file__).resolve()),"wrapper_sha256":sha(__file__),
        "immutable_reference_script":str(REFERENCE_SCRIPT),"reference_script_sha256":SCRIPT_SHA256,
        "immutable_reference_report":str(REFERENCE_REPORT),"reference_report_sha256":REPORT_SHA256,
        "in_memory_reporting_label_replacements":labels,"material_field_relabel":"old_material_configuration -> material_configuration","numeric_method_changes":False,
        "source_config":str(run/"config.json"),"source_config_sha256":sha(run/"config.json"),
        "source_material_configuration":cfg["bag"],"native_completion_readiness":readiness}
    measured["comparison_to_full30"] = comparisons(reference, measured)
    measured["comparison_scope"] = "Same rest, faces, sampling, formulas and four groups. Compare phase labels and their own actual endpoints, not equal wall time. Identical commands do not imply identical dynamic loads; material effects are not isolated from changed contact/drape response. This is not a new physical-task audit."
    if not all(sha(path)==digest for path,digest in inputs.items()):
        raise ValueError("Run output changed during analysis")
    if sha(REFERENCE_SCRIPT)!=SCRIPT_SHA256 or sha(REFERENCE_REPORT)!=REPORT_SHA256:
        raise ValueError("Immutable reference changed during analysis")
    measured["completed_run_input_sha256"] = inputs
    measured["wrapper_finished_utc"] = datetime.now(timezone.utc).isoformat()
    with output.open("x") as file:
        json.dump(measured,file,indent=2,allow_nan=False)
        file.write("\n")
    print(json.dumps({"path":str(output),"sha256":sha(output),"source_run":args.run_name,
        "all_diagnostic_checks_passed":measured["all_diagnostic_checks_passed"],
        "maximum_principal_stretch_by_group":{name:{"full30":reference["groups"][name]["periods"]["entire_run"]["extrema"]["principal_max"]["value"],
            "new":measured["groups"][name]["periods"]["entire_run"]["extrema"]["principal_max"]["value"]} for name in GROUPS}},indent=2))


if __name__ == "__main__":
    main()
