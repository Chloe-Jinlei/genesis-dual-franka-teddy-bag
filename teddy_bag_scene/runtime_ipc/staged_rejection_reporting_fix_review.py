"""Bounded reporting-only regression and full20 saved-state CPU replay.

Never starts physics. Dependencies and geometry come from full20's snapshot;
only the two list-copy changes in the current staged helper are exercised.
"""
from pathlib import Path
import copy
import difflib
import hashlib
import importlib.util
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_20"
sys.path.insert(0, str(RUN / "source"))
spec = importlib.util.spec_from_file_location("staged_reporting_fix_under_review", ROOT / "staged_contact_insertion.py")
staged = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staged)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def changes(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        rows = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                rows.append(dict(path=path + "/" + key, before=a.get(key), after=b.get(key)))
            else:
                rows.extend(changes(a[key], b[key], path + "/" + key))
        return rows
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [row for i, (x, y) in enumerate(zip(a, b)) for row in changes(x, y, path + "/" + str(i))]
    return [] if a == b else [dict(path=path, before=a, after=b)]

cfg = json.loads((RUN / "config.json").read_text())
entry = json.loads((RUN / "staged_contact_entry.json").read_text())
rest = dict(np.load(RUN / "rest.npz"))
state = dict(np.load(RUN / "lower_teddy_to_rim.npz"))
original = json.loads((RUN / "staged_contact_center.json").read_text())
paths = [RUN / name for name in ("config.json", "rest.npz", "lower_teddy_to_rim.npz", "staged_contact_entry.json", "staged_contact_center.json")]
paths += sorted(p for p in (RUN / "source").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
paths += [ROOT / "staged_contact_insertion.py"]
before_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
before_inputs = copy.deepcopy((rest, state, cfg, entry))

old_source = (RUN / "source/staged_contact_insertion.py").read_text()
new_source = (ROOT / "staged_contact_insertion.py").read_text()
expected_source = old_source.replace('center_rejections=center["report"]["rejections"]', 'center_rejections=list(center["report"]["rejections"])').replace('frozen_remainder_rejections=remainder["report"]["rejections"]', 'frozen_remainder_rejections=list(remainder["report"]["rejections"])')
assert new_source == expected_source, "Source differs beyond the two approved list copies"

# A controlled, intentionally rejected remainder verifies that final aggregate
# reasons do not mutate per-center reasons. A nonfinite score case exercises
# the separate remainder-copy rationale without running any robot planning.
saved_core = staged._plan_staged_contact_single
synthetic = []
for mode in ("remainder_rejected", "nonfinite_score"):
    centers, remainders = [], []
    def fake_core(_rest, current, _cfg, *, phase, previous=None, output=None, center_delta_override=None):
        if phase == "center":
            target = np.asarray(current["tcp_right"]) + np.r_[center_delta_override, 0.]
            result = dict(status="candidate", stage=dict(right_tcp_m=target.tolist(), right_q_end=[0.] * 7, duration_s=4.),
                report=dict(rejections=[], robot_trajectory={"minimum_sampled_rigid_clearance_m": .02},
                    planned_com_contact_envelope_boundary_distance_m=.01))
            centers.append(result)
            return result
        result = dict(status="rejected" if mode == "remainder_rejected" else "candidate", report=dict(
            rejections=["individual frozen remainder rejection"] if mode == "remainder_rejected" else [],
            robot_trajectory={"minimum_sampled_rigid_clearance_m": float("nan")},
            release_withdraw_preflight={"minimum_release_opening_clearance_m": .02}))
        remainders.append(result)
        return result
    staged._plan_staged_contact_single = fake_core
    out = staged.plan_staged_contact(rest, state, cfg, phase="center", previous=entry)
    attempts = out["center_selection"]["attempts"]
    checks = dict(no_selected_index=out["center_selection"]["selected_index"] is None,
        overall_rejected=out["status"] == "rejected",
        all_individual_centers_have_empty_rejections=all(a["center_rejections"] == [] for a in attempts),
        aggregate_reason_only_in_overall=out["report"]["rejections"] == ["No predeclared center passes center and frozen remainder preflight"],
        remainder_source_reasons_unchanged=all(r["report"]["rejections"] == (["individual frozen remainder rejection"] if mode == "remainder_rejected" else []) for r in remainders),
        score_rejections_preserved_in_attempt=all(a["frozen_remainder_rejections"] == (["individual frozen remainder rejection"] if mode == "remainder_rejected" else ["Nonfinite raw center/insert/release clearance"]) for a in attempts))
    synthetic.append(dict(case=mode, checks=checks, passed=all(checks.values())))
staged._plan_staged_contact_single = saved_core
assert all(row["passed"] for row in synthetic)

print("Two controlled alias regressions passed; starting full20 actual-entry CPU replay", flush=True)
started = time.monotonic()
replay_path = RUN / "independent-center-reporting-fix-replay.json"
replayed = staged.plan_staged_contact(rest, state, cfg, phase="center", previous=entry, output=replay_path)
elapsed = time.monotonic() - started
diff = changes(original, replayed)
expected_diff = [dict(path="/center_selection/attempts/0/center_rejections", before=["No predeclared center passes center and frozen remainder preflight"], after=[])]
checks = dict(source_exactly_two_list_copies=new_source == expected_source,
    source_and_frozen_inputs_unchanged_during_replay=all(sha(p) == before_hashes[str(p.relative_to(ROOT))] for p in paths),
    input_arrays_unchanged=all(np.array_equal(v, before_inputs[i][k]) for i, arrays in enumerate((rest, state)) for k, v in arrays.items()),
    input_config_and_entry_unchanged=(cfg == before_inputs[2] and entry == before_inputs[3]),
    exact_only_serialized_difference_is_first_attempt_reason=diff == expected_diff,
    rejected_with_no_selection=replayed["status"] == "rejected" and replayed["center_selection"]["selected_index"] is None,
    first_individual_center_candidate_has_no_rejections=replayed["center_selection"]["attempts"][0]["center_status"] == "candidate" and replayed["center_selection"]["attempts"][0]["center_rejections"] == [],
    original_historical_record_unmodified=sha(RUN / "staged_contact_center.json") == before_hashes[str((RUN / "staged_contact_center.json").relative_to(ROOT))])
regression = json.loads((ROOT / "runtime_ipc/staged-center-selection-audit-regression.json").read_text())
report = dict(kind="staged_rejection_reporting_only_fix_review", physical_success=False, runtime_execution_performed=False,
    source_sha256=sha(ROOT / "staged_contact_insertion.py"), prior_source_sha256=sha(RUN / "source/staged_contact_insertion.py"),
    source_diff="".join(difflib.unified_diff(old_source.splitlines(True), new_source.splitlines(True), fromfile="full20 frozen source", tofile="live reporting fix")),
    checks=checks, all_checks_passed=all(checks.values()), controlled_alias_regressions=synthetic,
    existing_center_selection_audit_regression=dict(all_checks_passed=regression["all_checks_passed"], case_count=regression["case_count"],
        legacy_sequence_and_budget_regressions_passed=regression["legacy_sequence_and_budget_regressions_passed"]),
    full20_replay=dict(actual_observation_time_s=float(state["time"]), elapsed_wall_s=elapsed,
        output=str(replay_path.relative_to(ROOT)), sha256=sha(replay_path), status=replayed["status"],
        selected_index=replayed["center_selection"]["selected_index"], exact_json_differences=diff,
        attempts=replayed["center_selection"]["attempts"]), input_sha256=before_hashes,
    scope="Reporting list isolation only. Native simulation was not run; all planning gates and full20 rejection outcomes remain unchanged.")
outpath = ROOT / "runtime_ipc/staged-rejection-reporting-fix-review.json"
outpath.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(report=str(outpath), all_checks_passed=report["all_checks_passed"], source_sha256=report["source_sha256"], exact_json_differences=diff, elapsed_wall_s=elapsed), indent=2), flush=True)
assert report["all_checks_passed"] and regression["all_checks_passed"]
