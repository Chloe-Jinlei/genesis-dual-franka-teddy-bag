"""Offline frozen-body alignment ranking; never executes or authorizes a motion.

Run with --self-test for the bounded ranking/validity contract tests. Otherwise
provide one actual saved state, its adaptive center record, and the config used
to produce that record. Only an explicitly named JSON report is written.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SCENE))
import audit_guarded_run as selection_audit
import diagnose_low_region_winding as low_region

DISTANCE_TOLERANCE_M = 1e-6
ORIGINAL_SCORE_LOSS_BAND_M = .001
METRIC_KEYS = ("inside_node_count", "node_count", "node_inside_fraction",
               "signed_com_body_distance_m", "maximum_outside_node_body_distance_m")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unknown_metrics(reason, guards=None):
    return dict(status="unknown", physical_success=False, reasons=[str(reason)],
                guards=guards or {}, **{key: None for key in METRIC_KEYS})


def metrics_from_diagnostic(diagnostic, node_count):
    """Outside points are valid measurements; ambiguity is unknown, never zero.

    Reconstruct the already-reviewed low-region diagnostic's validity checks.
    Its status can be failed_point_diagnostic simply because a point is outside.
    Only physical body distances are used; cap and handle distances are excluded.
    """
    try:
        if type(node_count) is not int or node_count <= 0:
            raise ValueError("Invalid common node count")
        h = diagnostic["height_precondition"]
        caps = diagnostic["cap_winding"]
        distances = diagnostic["physical_cloth_query_distance"]
        wa = np.asarray(caps["mean_apex_raw_winding"], float)
        wb = np.asarray(caps["high_apex_raw_winding"], float)
        dd = np.asarray(distances["body_distance_m"], float)
        if any(x.shape != (node_count + 1,) for x in (wa, wb, dd)):
            raise ValueError("All nodes and COM must have one common set of query values")
        finite = all(np.isfinite(x).all() for x in (wa, wb, dd))
        guards = {
            "all_queries_at_least_2mm_below_actual_lowest_rim": bool(
                np.isfinite(float(h["minimum_query_height_gap_m"])) and
                float(h["minimum_query_height_gap_m"]) >= low_region.HEIGHT_MARGIN_M),
            "actual_body_topology_orientation_nondegeneracy":
                diagnostic["actual_topology_orientation"].get("passed") is True,
            "actual_spatial_rim": diagnostic["actual_spatial_rim"].get("passed") is True,
            "both_analysis_caps_coherently_closed": all(
                caps[k].get("is_watertight") is True and caps[k].get("is_winding_consistent") is True
                for k in ("mean_apex_surface", "high_apex_surface")),
            "raw_winding_and_physical_body_distances_finite": finite,
            "caps_agree_at_every_query": bool(finite and np.max(np.abs(wa-wb)) <= low_region.CAP_AGREEMENT_TOLERANCE),
            "integer_winding_residual_bounded": bool(finite and max(
                np.max(np.abs(wa-np.rint(wa))), np.max(np.abs(wb-np.rint(wb)))) <= low_region.INTEGER_RESIDUAL_TOLERANCE),
            "no_multiple_winding": bool(finite and np.all(np.abs(np.rint(wa)) <= 1) and np.all(np.abs(np.rint(wb)) <= 1)),
            "all_queries_farther_than_1um_from_physical_body": bool(finite and np.min(dd) > low_region.BODY_AMBIGUITY_DISTANCE_M),
        }
        if not all(guards.values()):
            result = unknown_metrics("Low-region metric validity is not established", guards)
            result["reasons"] = [name for name, passed in guards.items() if not passed]
            return result
        inside = (np.abs(np.abs(wa)-1) <= low_region.INTEGER_RESIDUAL_TOLERANCE) & (
            np.abs(np.abs(wb)-1) <= low_region.INTEGER_RESIDUAL_TOLERANCE)
        exterior = ~inside[:-1]
        count = int(inside[:-1].sum())
        return dict(status="valid", physical_success=False, reasons=[], guards=guards,
            inside_node_count=count, node_count=node_count, node_inside_fraction=count/node_count,
            com_inside=bool(inside[-1]), signed_com_body_distance_m=float(dd[-1] if inside[-1] else -dd[-1]),
            maximum_outside_node_body_distance_m=float(dd[:-1][exterior].max()) if exterior.any() else 0.,
            outside_node_count=int(exterior.sum()), minimum_query_body_distance_m=float(dd.min()),
            maximum_cap_difference=float(np.max(np.abs(wa-wb))),
            maximum_integer_residual=float(max(np.max(np.abs(wa-np.rint(wa))), np.max(np.abs(wb-np.rint(wb))))),
            whole_tetrahedral_volume_containment_certified=False,
            measure="Unweighted count of this same frozen mesh's nodes; not a volume, mass, or success fraction")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as error:
        return unknown_metrics(error)


def frozen_query_metrics(run, query, node_count, diagnostic_function=None):
    """A failed candidate geometry measurement is unknown; source checks stay fatal."""
    try:
        diagnostic = (diagnostic_function or low_region.diagnose)(run, query)
    except Exception as error:
        details = dict(exception_type=type(error).__name__, exception_message=str(error))
        return unknown_metrics("Candidate geometry diagnostic failed: "+str(error)), {
            "diagnostic_error": details, "physical_success": False}
    metrics = metrics_from_diagnostic(diagnostic, node_count)
    # The wrapped diagnostic's temporary 'actual state' pathname is deliberately
    # omitted; every point here is an explicitly frozen hypothetical query.
    evidence = {k: diagnostic.get(k) for k in (
        "height_precondition", "actual_topology_orientation", "actual_spatial_rim",
        "cap_winding", "physical_cloth_query_distance", "reasons")} if isinstance(diagnostic, Mapping) else {
            "diagnostic_error": "Diagnostic returned a non-mapping result"}
    return metrics, evidence


def dominates(a, b, distance_tolerance_m=DISTANCE_TOLERANCE_M):
    """Transitive, one-sided tolerance: never accept a worse component.

    All three components must be exactly no worse, and at least one improvement
    must exceed its declared tolerance. This is NOT pairwise +/- epsilon
    dominance, which can cycle. Node counts are exact integers on one common mesh.
    """
    aa = np.array([a["inside_node_count"], a["signed_com_body_distance_m"],
                   -a["maximum_outside_node_body_distance_m"]], float)
    bb = np.array([b["inside_node_count"], b["signed_com_body_distance_m"],
                   -b["maximum_outside_node_body_distance_m"]], float)
    tolerances = np.array([0., distance_tolerance_m, distance_tolerance_m])
    return bool(np.all(aa >= bb) and np.any(aa-bb > tolerances))


def rank_candidates(candidates, distance_tolerance_m=DISTANCE_TOLERANCE_M):
    if not candidates or not np.isfinite(distance_tolerance_m) or distance_tolerance_m < 0:
        raise ValueError("Finite nonnegative ranking tolerance and nonempty candidates required")
    indices = [c["index"] for c in candidates]
    if any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Original candidate indices must be unique nonnegative integers")
    try:
        scores_valid = all(np.isfinite(float(c["original_score_m"])) for c in candidates)
    except (KeyError, TypeError, ValueError):
        scores_valid = False
    if not scores_valid:
        raise ValueError("Original finite score required for every accepted candidate")
    original = sorted(candidates, key=lambda c: (-float(c["original_score_m"]), c["index"]))
    unknown = []
    counts = set()
    for c in candidates:
        m = c.get("body_metrics")
        if not isinstance(m, Mapping) or m.get("status") != "valid":
            unknown.append(c["index"]); continue
        try:
            if type(m["node_count"]) is not int or type(m["inside_node_count"]) is not int or not (
                    0 <= m["inside_node_count"] <= m["node_count"] and m["node_count"] > 0):
                raise ValueError("Malformed node counts")
            if not np.isfinite([m[k] for k in METRIC_KEYS]).all() or m["maximum_outside_node_body_distance_m"] < 0:
                raise ValueError("Malformed finite body metrics")
            if abs(m["node_inside_fraction"] - m["inside_node_count"]/m["node_count"]) > 1e-12:
                raise ValueError("Node fraction and count disagree")
            counts.add(m["node_count"])
        except (KeyError, TypeError, ValueError):
            unknown.append(c["index"])
    if len(counts) > 1:
        unknown = sorted(set(unknown + indices))
    result = dict(physical_success=False, creates_new_feasibility_gate=False,
        distance_ranking_tolerance_m=distance_tolerance_m, node_count_tolerance=0,
        dominance_definition="All components exactly no worse; one improves beyond its tolerance. Count up, signed COM-body distance up, maximum exterior-body distance down.",
        within_front_order="Original score descending, original candidate index ascending",
        original_score_order=[c["index"] for c in original], unknown_candidate_indices=sorted(unknown))
    if unknown:
        result.update(method="whole_group_original_score_fallback", fallback_reason="At least one body metric is unknown, malformed, or uses a different node denominator",
            pareto_fronts=None, ranked_indices=[c["index"] for c in original], selected_index=original[0]["index"])
        return result
    remaining = {c["index"]: c for c in candidates}; fronts = []
    while remaining:
        front = [c for c in remaining.values() if not any(
            other["index"] != c["index"] and dominates(other["body_metrics"], c["body_metrics"], distance_tolerance_m)
            for other in remaining.values())]
        if not front:
            raise RuntimeError("No Pareto front: transitive dominance contract violated")
        front.sort(key=lambda c: (-float(c["original_score_m"]), c["index"]))
        fronts.append([c["index"] for c in front])
        for c in front: del remaining[c["index"]]
    ranked = [i for front in fronts for i in front]
    result.update(method="body_pareto_fronts_then_original_score", fallback_reason=None,
                  pareto_fronts=fronts, ranked_indices=ranked, selected_index=ranked[0])
    return result


def rank_with_original_score_band(candidates, maximum_score_loss_m=ORIGINAL_SCORE_LOSS_BAND_M,
                                 distance_tolerance_m=DISTANCE_TOLERANCE_M):
    """Optional offline ranking heuristic, never an additional feasibility gate.

    Check metric validity on the entire original accepted set before filtering.
    Thus even an unknown candidate outside the band triggers whole-set fallback.
    """
    if not np.isfinite(maximum_score_loss_m) or maximum_score_loss_m < 0:
        raise ValueError("Original-score loss band must be finite and nonnegative, in metres")
    pure = rank_candidates(candidates, distance_tolerance_m)
    best = max(float(c["original_score_m"]) for c in candidates)
    cutoff = best - maximum_score_loss_m
    decisions = [dict(index=c["index"], original_score_m=float(c["original_score_m"]),
        score_loss_from_best_m=best-float(c["original_score_m"]),
        meets_score_band=float(c["original_score_m"]) >= cutoff,
        reason="Original score >= best score - declared band" if float(c["original_score_m"]) >= cutoff
            else "Original score < best score - declared band; excluded from this optional ranking subset only")
        for c in sorted(candidates, key=lambda c: c["index"])]
    result = dict(physical_success=False, creates_new_feasibility_gate=False,
        heuristic="Original-score loss band, then body Pareto fronts, then original score and index",
        maximum_original_score_loss_m=float(maximum_score_loss_m), parameter_unit="metre",
        best_original_score_m=best, inclusive_original_score_cutoff_m=cutoff,
        original_candidate_indices=sorted(c["index"] for c in candidates),
        candidate_band_decisions=decisions,
        original_score_order=pure["original_score_order"],
        unknown_candidate_indices=pure["unknown_candidate_indices"],
        limitations="Preserves only a bound on the recorded original score. Does not certify continuous motion, withdraw surplus, future cloth shape, grasp, forces, or physical success.")
    if pure["unknown_candidate_indices"]:
        result.update(method="whole_group_original_score_fallback", band_applied=False,
            retained_indices=None, excluded_indices=None, pareto_fronts=None,
            ranked_indices=pure["ranked_indices"], selected_index=pure["selected_index"],
            fallback_reason="At least one metric in the entire original set is unknown; no band or body ranking applied")
        return result
    kept = [c for c in candidates if float(c["original_score_m"]) >= cutoff]
    ranked = rank_candidates(kept, distance_tolerance_m)
    result.update(method="score_band_then_body_pareto_then_original_score", band_applied=True,
        retained_indices=sorted(c["index"] for c in kept),
        excluded_indices=sorted(d["index"] for d in decisions if not d["meets_score_band"]),
        pareto_fronts=ranked["pareto_fronts"], ranked_indices=ranked["ranked_indices"],
        selected_index=ranked["selected_index"], fallback_reason=None,
        distance_ranking_tolerance_m=distance_tolerance_m, node_count_tolerance=0)
    return result


def evaluate(run, state_path, selection_path, config_path, maximum_score_loss_m=ORIGINAL_SCORE_LOSS_BAND_M):
    run = Path(run).resolve(); state_path = Path(state_path).resolve()
    selection_path = Path(selection_path).resolve(); config_path = Path(config_path).resolve()
    inputs = [state_path, selection_path, config_path, run/"config.json", run/"rest.npz",
              run/"source/validate.py", SCENE/"validate.py", HERE/"diagnose_low_region_winding.py",
              HERE/"audit_guarded_run.py", Path(__file__).resolve()]
    before = {str(p): sha(p) for p in inputs}
    cfg = json.loads(config_path.read_text()); actual_cfg = json.loads((run/"config.json").read_text())
    # The shared diagnostic loads the run's actual geometry/material definition.
    # A retrospective candidate-config override may only differ in planning policy.
    if any(cfg[k] != actual_cfg[k] for k in ("bag", "bear", "table", "simulation")):
        raise ValueError("Candidate config changes the actual geometry/material/simulation definition")
    if sha(run/"source/validate.py") != sha(SCENE/"validate.py"):
        raise ValueError("Reviewed diagnostic's validator differs from the frozen source")
    asset = actual_cfg["bag"]["npz_file"]
    if sha(run/"source/assets"/asset) != sha(SCENE/"assets"/asset):
        raise ValueError("Reviewed diagnostic's body asset differs from frozen source")
    inputs += [run/"source/assets"/asset, SCENE/"assets"/asset]
    before.update({str(p): sha(p) for p in inputs})
    with np.load(state_path, allow_pickle=False) as archive:
        state = {k: archive[k].copy() for k in archive.files}
    original_state = {k: v.copy() for k, v in state.items()}
    if state["bag"].ndim != 2 or state["bear"].ndim != 2:
        raise ValueError("Provide a single actual observed state, not a state history")
    record = json.loads(selection_path.read_text()); contract = selection_audit.staged_center_contract(cfg)
    if not contract["adaptive"] or record["status"] != "candidate" or record["phase"] != "center":
        raise ValueError("An accepted adaptive center record is required")
    source = float(state["time"]); start = state["tcp_right"]
    selection_checks = selection_audit.audit_adaptive_center_selection(record, contract, start,
        np.asarray(record["stage"]["right_tcp_m"], float), source, float(cfg["simulation"]["dt"]))
    selection_checks["checkpoint_policy_matches_config"] = selection_audit._same_staged_policy(
        record["checkpoint"]["staged_policy"], contract["expected_staged_policy"])
    selection_checks["record_time_matches_actual_state"] = abs(float(record["observed_time_s"])-source) <= 1e-9
    if not all(selection_checks.values()):
        raise ValueError("Candidate source/selection record failed independent validation: " + str(selection_checks))
    clearance = float(cfg["task"]["contact_trial"].get("bottom_clearance_m", .03))
    dz = float(cfg["table"]["height"]) + clearance - float(state["bear"][:, 2].min())
    if not np.isfinite(dz): raise ValueError("Nonfinite shared final vertical translation")
    node_count = len(state["bear"]); candidates = []; excluded = []
    # Each transient file is an explicitly hypothetical query, not a runtime state.
    # Calling the reviewed diagnostic directly keeps every validity definition intact.
    with tempfile.TemporaryDirectory(prefix="frozen-body-ranking-queries-") as temp:
        for attempt in record["center_selection"]["attempts"]:
            accepted = attempt.get("center_status") == "candidate" and attempt.get("frozen_remainder_status") == "candidate" and not attempt.get("frozen_remainder_rejections")
            if not accepted:
                excluded.append(dict(index=attempt["index"], reason="Not accepted by existing center/remainder checks")); continue
            offset = np.asarray(attempt["center_offset_xy_m"], float)
            score = float(attempt["score_m"])
            if offset.shape != (2,) or not np.isfinite(offset).all() or not np.isfinite(score):
                raise ValueError("Accepted candidates must have finite offsets and original scores")
            translated = state["bear"] + np.r_[offset, dz]
            expected_bounds = np.asarray(attempt["frozen_remainder_report"]["target_bear_bounds_world_m"], float)
            if expected_bounds.shape != (2, 3) or not np.isfinite(expected_bounds).all() or not np.allclose(expected_bounds, [translated.min(0), translated.max(0)], rtol=0, atol=1e-9):
                raise ValueError("Accepted remainder does not share the actual frozen shape, final Z and translation")
            query = Path(temp)/f"hypothetical_candidate_{attempt['index']}.npz"
            np.savez(query, time=state["time"], bag=state["bag"], bear=translated)
            metrics, evidence = frozen_query_metrics(run, query, node_count)
            candidates.append(dict(index=int(attempt["index"]), offset_xy_m=offset.tolist(),
                original_score_m=score, frozen_final_translation_m=[*offset.tolist(), dz],
                orientation="Unrotated actual observed teddy shape for every candidate",
                body_metrics=metrics, frozen_query_diagnostic=evidence, physical_success=False))
            print(f"Frozen candidate {attempt['index']}: {metrics['status']}", flush=True)
    if len(candidates) != sum(a.get("center_status") == "candidate" and a.get("frozen_remainder_status") == "candidate" and not a.get("frozen_remainder_rejections") for a in record["center_selection"]["attempts"]):
        raise ValueError("An accepted candidate was omitted")
    unchanged = all(np.array_equal(state[k], original_state[k], equal_nan=True) for k in state)
    hashes_unchanged = before == {str(p): sha(p) for p in inputs}
    if not unchanged or not hashes_unchanged: raise ValueError("Input source/state changed during offline evaluation")
    return dict(kind="offline_frozen_body_alignment_ranking", physical_success=False,
        runtime_execution_performed=False, runtime_or_config_modified=False,
        actual_state_path=str(state_path), actual_observed_time_s=source,
        common_frozen_final_z_translation_m=dz, common_node_count=node_count,
        candidate_selection_checks={k: bool(v) for k, v in selection_checks.items()},
        all_actual_state_arrays_unchanged=unchanged, all_input_source_hashes_unchanged=hashes_unchanged,
        input_sha256=before, candidates=candidates, excluded_nonaccepted_candidates=excluded,
        ranking=rank_candidates(candidates),
        optional_original_score_band_ranking=rank_with_original_score_band(candidates, maximum_score_loss_m),
        limitations=["This ranks already accepted offline hypotheses and adds no motion, attachment, feasibility gate, or runtime policy.",
            "The actual cloth and teddy are frozen; contact deformation, forces, release, and settling are not predicted.",
            "Node inclusion is not mass/volume inclusion; even all nodes and COM inside cannot certify whole tetrahedra or passage.",
            "Closed-chain and winding guards do not prove the physical body has no self-intersection everywhere.",
            "Only physical body triangles define distances; the analysis cap and handles never contribute to the distance score.",
            "Any unknown body metric causes the entire accepted set to use its original score ordering.",
            "No decision here changes active full19 or replaces its final physical validation."])


def self_test():
    checks = []
    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
    def diagnostic(w, distances):
        return dict(status="failed_point_diagnostic", height_precondition={"minimum_query_height_gap_m": .01},
            actual_topology_orientation={"passed": True}, actual_spatial_rim={"passed": True},
            cap_winding=dict(mean_apex_raw_winding=w, high_apex_raw_winding=w,
                mean_apex_surface={"is_watertight": True, "is_winding_consistent": True},
                high_apex_surface={"is_watertight": True, "is_winding_consistent": True}),
            physical_cloth_query_distance={"body_distance_m": distances, "full_cloth_distance_m": [999.]*len(w)})
    valid = diagnostic([1.,0.,1.], [.01,.04,.02])
    mm = metrics_from_diagnostic(valid, 2)
    check("normal_exterior_point_is_valid_not_unknown", mm["status"] == "valid" and mm["inside_node_count"] == 1)
    check("distance_uses_physical_body_not_full_cloth", mm["maximum_outside_node_body_distance_m"] == .04 and mm["signed_com_body_distance_m"] == .02)
    other = metrics_from_diagnostic(diagnostic([1.,1.,0.], [.01,.02,.03]), 2)
    check("outside_com_gets_negative_physical_body_distance", other["signed_com_body_distance_m"] == -.03)
    check("no_exterior_nodes_has_zero_maximum_outside_distance", other["maximum_outside_node_body_distance_m"] == 0.)
    mutations = {
        "height_not_low_region": lambda d: d["height_precondition"].update(minimum_query_height_gap_m=.001),
        "invalid_body_topology": lambda d: d["actual_topology_orientation"].update(passed=False),
        "invalid_spatial_rim": lambda d: d["actual_spatial_rim"].update(passed=False),
        "open_cap": lambda d: d["cap_winding"]["high_apex_surface"].update(is_watertight=False),
        "cap_disagreement": lambda d: d["cap_winding"].update(high_apex_raw_winding=[0.,0.,1.]),
        "multiple_winding": lambda d: d["cap_winding"].update(mean_apex_raw_winding=[2.,0.,1.], high_apex_raw_winding=[2.,0.,1.]),
        "noninteger_winding": lambda d: d["cap_winding"].update(mean_apex_raw_winding=[.8,0.,1.], high_apex_raw_winding=[.8,0.,1.]),
        "near_body_at_exact_1um_boundary": lambda d: d["physical_cloth_query_distance"].update(body_distance_m=[1e-6,.04,.02]),
        "nonfinite_distance": lambda d: d["physical_cloth_query_distance"].update(body_distance_m=[float('nan'),.04,.02]),
        "query_count_mismatch": lambda d: d["physical_cloth_query_distance"].update(body_distance_m=[.01,.02]),
        "malformed_topology_mapping": lambda d: d.update(actual_topology_orientation=None),
        "malformed_cap_surface_mapping": lambda d: d["cap_winding"].update(mean_apex_surface=[]),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(valid); mutate(bad); result = metrics_from_diagnostic(bad, 2)
        check(name+"_is_unknown_not_zero", result["status"] == "unknown" and all(result[k] is None for k in METRIC_KEYS))
    def diagnostic_error(*args): raise RuntimeError("Synthetic geometry failure")
    failed_geometry, failed_evidence = frozen_query_metrics(None,None,2,diagnostic_error)
    check("geometry_exception_becomes_unknown_with_error_evidence", failed_geometry["status"] == "unknown" and
          all(failed_geometry[k] is None for k in METRIC_KEYS) and failed_evidence["diagnostic_error"]["exception_type"] == "RuntimeError")
    nonmapping, _ = frozen_query_metrics(None,None,2,lambda *args: [])
    check("nonmapping_diagnostic_becomes_unknown", nonmapping["status"] == "unknown")
    def metric(count, com, outside, n=100):
        return dict(status="valid", inside_node_count=count, node_count=n, node_inside_fraction=count/n,
                    signed_com_body_distance_m=com, maximum_outside_node_body_distance_m=outside)
    def item(index, score, m): return dict(index=index, original_score_m=score, body_metrics=m)
    a = item(0, .1, metric(80,.05,.05)); b = item(1,.01,metric(82,.06,.04))
    rr = rank_candidates([a,b]); check("body_dominance_precedes_original_score", rr["pareto_fronts"] == [[1],[0]])
    trade = item(2,.03,metric(84,.04,.06)); rr = rank_candidates([a,b,trade])
    check("tradeoffs_share_front_original_score_breaks_tie", rr["pareto_fronts"] == [[2,1],[0]])
    same = [item(5,.02,metric(80,.05,.05)), item(2,.02,metric(80,.05,.05))]
    check("equal_score_and_body_choose_original_index", rank_candidates(same)["ranked_indices"] == [2,5])
    check("input_permutation_does_not_change_ranking", rank_candidates(same[::-1])["ranked_indices"] == [2,5])
    micro = item(1,.01,metric(80,.05+.5e-6,.05))
    check("sub_tolerance_improvement_shares_front", rank_candidates([a,micro])["pareto_fronts"] == [[0,1]])
    major = item(1,.01,metric(80,.05+2e-6,.05))
    check("beyond_tolerance_improvement_dominates", rank_candidates([a,major])["pareto_fronts"] == [[1],[0]])
    # Counterexample to symmetric epsilon dominance: exact no-worse comparison
    # makes all three tradeoffs incomparable, so front extraction cannot cycle.
    cycle = [item(i,.01,metric(50+n,.01+x*1e-6,.01-y*1e-6))
             for i,(n,x,y) in enumerate([(0,0,0),(2,-1,-1),(1,1,-2)])]
    check("epsilon_tradeoff_example_has_deterministic_fronts", rank_candidates(cycle)["pareto_fronts"] == [[0,1,2]])
    uu = copy.deepcopy(b); uu["body_metrics"] = unknown_metrics("ambiguous")
    rr = rank_candidates([a,uu]); check("one_unknown_falls_back_entire_group", rr["method"] == "whole_group_original_score_fallback" and rr["ranked_indices"] == [0,1] and rr["pareto_fronts"] is None)
    rr = rank_candidates([item(0,.1,unknown_metrics("x")),item(1,.2,unknown_metrics("y"))])
    check("all_unknown_still_uses_original_finite_scores", rr["ranked_indices"] == [1,0])
    mismatch = copy.deepcopy(b); mismatch["body_metrics"] = metric(82,.06,.04,101)
    check("different_denominators_force_whole_group_fallback", rank_candidates([a,mismatch])["method"] == "whole_group_original_score_fallback")
    for malformed in (None, [], "malformed", {}):
        changed = copy.deepcopy(b); changed["body_metrics"] = malformed
        check("malformed_metric_"+type(malformed).__name__+"_whole_group_fallback",
              rank_candidates([a,changed])["ranked_indices"] == [0,1] and rank_candidates([a,changed])["pareto_fronts"] is None)
    missing = copy.deepcopy(b); del missing["body_metrics"]
    check("missing_body_metrics_whole_group_fallback", rank_candidates([a,missing])["method"] == "whole_group_original_score_fallback")
    check("geometry_exception_falls_back_entire_group", rank_candidates([a,item(1,.2,failed_geometry)])["ranked_indices"] == [1,0])
    for name, values in [("duplicate_indices",[a,copy.deepcopy(a)]),
                         ("nonfinite_original_score",[item(0,float('nan'),mm)]),
                         ("missing_original_score",[dict(index=0,body_metrics=mm)])]:
        try: rank_candidates(values); rejected=False
        except ValueError: rejected=True
        check(name+"_rejected", rejected)
    # A lower-scoring better-body candidate is considered only when in the band.
    band_items = [item(5,.0045,metric(70,.04,.06)), item(0,.0043,metric(80,.05,.05)),
                  item(2,.0015,metric(90,.06,.04))]
    bounded = rank_with_original_score_band(band_items)
    check("pure_pareto_and_score_band_can_differ", rank_candidates(band_items)["selected_index"] == 2 and
          bounded["selected_index"] == 0 and bounded["retained_indices"] == [0,5] and bounded["excluded_indices"] == [2])
    check("score_band_records_complete_set_and_reasons", len(bounded["candidate_band_decisions"]) == 3 and
          bounded["maximum_original_score_loss_m"] == .001 and bounded["original_candidate_indices"] == [0,2,5])
    boundary = .0045 - .001
    edge = [band_items[0], item(1,boundary,metric(81,.05,.05)),
            item(2,float(np.nextafter(boundary,-np.inf)),metric(82,.06,.04))]
    check("score_band_inclusive_boundary_and_adjacent_float", rank_with_original_score_band(edge)["retained_indices"] == [1,5])
    check("zero_band_keeps_only_original_best", rank_with_original_score_band(band_items,0)["selected_index"] == 5)
    check("wide_band_matches_pure_ranking", rank_with_original_score_band(band_items,.01)["ranked_indices"] == rank_candidates(band_items)["ranked_indices"])
    band_unknown = copy.deepcopy(band_items); band_unknown[2]["body_metrics"] = unknown_metrics("1um boundary outside score band")
    fallback = rank_with_original_score_band(band_unknown)
    check("unknown_outside_score_band_still_falls_back_whole_group", fallback["method"] == "whole_group_original_score_fallback" and
          fallback["ranked_indices"] == [5,0,2] and fallback["retained_indices"] is None and not fallback["band_applied"])
    check("score_band_permutation_deterministic", rank_with_original_score_band(band_items[::-1]) == bounded)
    for width in (-.001, float('nan'), float('inf')):
        try: rank_with_original_score_band(band_items,width); rejected=False
        except ValueError: rejected=True
        check("invalid_score_band_"+str(width)+"_rejected",rejected)
    return dict(kind="offline_body_alignment_ranking_boundary_tests", physical_success=False,
        all_checks_passed=all(c["passed"] for c in checks), check_count=len(checks), checks=checks,
        source_sha256=sha(__file__), scope="Tests metric validity and ranking contracts only; no simulated task success")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--maximum-original-score-loss-m", type=float, default=ORIGINAL_SCORE_LOSS_BAND_M,
        help="Optional offline score-loss band in metres; default 0.001. Pure Pareto is always reported too.")
    args = parser.parse_args()
    if args.self_test:
        report = self_test(); protected = {Path(__file__).resolve()}
    else:
        if args.run is None or args.state is None or args.selection is None:
            parser.error("run, --state and --selection are required")
        cfg = args.config or args.run/"config.json"
        report = evaluate(args.run, args.state, args.selection, cfg, args.maximum_original_score_loss_m)
        protected = {Path(p).resolve() for p in report["input_sha256"]}
    output = args.output.resolve()
    if output in protected or output.suffix.lower() != ".json":
        raise ValueError("Output must be a separate JSON report, never an input")
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    summary = {k: report[k] for k in ("ranking","optional_original_score_band_ranking")} if "ranking" in report else {
        k: report[k] for k in ("check_count","all_checks_passed")}
    print(json.dumps(summary, indent=2))
    if args.self_test and not report["all_checks_passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
