"""CPU-only post-run audit of an explicitly guarded contact insertion trial.

Uses the run's frozen geometry validator, then independently checks every-step
reaction records, raw limits, coverage, material proximity and release flags.
No Genesis imports; this file does not affect simulation source or callbacks.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _json_scalar(value):
    """Preserve NumPy scalar values when serializing measured audit results."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit_reactions(run, report, trace):
    path = run / "contact_guard_trace.json"
    if not path.exists():
        return {"status": "not_entered", "passed": False,
                "reason": "No guarded contact-trial reaction trace was saved."}
    rows = json.loads(path.read_text())
    if not rows:
        return {"status": "missing", "passed": False, "reason": "Empty guard trace."}
    limits = report["contact_guard"]["limits"]
    t = np.asarray([r["time_s"] for r in rows])
    steps = np.asarray([r["physics_step"] for r in rows])
    dt = float(json.loads((run / "config.json").read_text())["simulation"]["dt"])
    def array(key):
        return np.asarray([r[key] for r in rows], float)
    force = array("force_world_n")
    torque = array("torque_at_aim_tcp_world_nm")
    links = array("right_link_force_world_n")
    ropen = array("right_finger_opening_projection_n")
    lopen = array("left_finger_opening_projection_n")
    rightnorm = np.linalg.norm(links[:, 1:], axis=2)
    netnorm = np.linalg.norm(force, axis=1)
    taunorm = np.linalg.norm(torque, axis=1)
    left_distance = array("left_designated_material_distance_m")
    required = np.asarray([r["right_two_sided_contact_required"] for r in rows], bool)
    right_distance = array("right_nearest_teddy_distance_m")
    release = next((e["start"] for e in report["events"] if e["phase"] == "release_teddy"), None)
    # A row at the insertion endpoint is measured before toggling the flag.
    expected_required = np.ones(len(t), bool) if release is None else t <= release + 1e-9
    joint_times = trace[:, 0]
    expected = joint_times[joint_times >= t[0] - 1e-9]
    if len(expected) == 0 or abs(expected[0] - t[0]) > 1e-9:
        expected = np.r_[t[0], expected]
    baseline = report["contact_guard"]["baseline"]
    delta_keys = ("force_world_n", "torque_at_aim_tcp_world_nm",
                  "right_finger_opening_projection_n", "right_finger_force_norm_n",
                  "left_finger_opening_projection_n", "left_finger_force_norm_n")
    delta_error = max(float(np.max(np.abs(array("delta_" + k) -
        (array(k) - np.asarray(baseline[k]))))) for k in delta_keys)
    raw_finite = all(np.isfinite(a).all() for a in
        (t, force, torque, links, ropen, lopen, left_distance, right_distance[required]))
    checks = {
        "all_required_raw_values_finite": bool(raw_finite),
        "strictly_increasing_consecutive_steps": bool(np.all(np.diff(t) > 0) and np.all(np.diff(steps) == 1)),
        "timestamps_match_fixed_integration_step": bool(np.allclose(np.diff(t), dt, rtol=0, atol=1e-9)),
        "covers_every_step_from_guard_entry_to_run_end": bool(len(t) == len(expected) and np.allclose(t, expected, rtol=0, atol=1e-9)),
        "last_time_matches_report": bool(abs(t[-1] - report["simulated_seconds"]) < 1e-9),
        "right_hold_flag_agrees_with_release_event": bool(np.array_equal(required, expected_required)),
        "summed_raw_link_force_matches_net_force": bool(np.allclose(links.sum(axis=1), force, rtol=0, atol=1e-9)),
        "stored_force_norms_match_raw_vectors": bool(np.allclose(netnorm, array("net_force_norm_n"), rtol=0, atol=1e-9)),
        "stored_torque_norms_match_raw_vectors": bool(np.allclose(taunorm, array("tcp_torque_norm_nm"), rtol=0, atol=1e-9)),
        "stored_finger_norms_match_raw_vectors": bool(np.allclose(rightnorm, array("right_finger_force_norm_n"), rtol=0, atol=1e-9)),
        "baseline_differences_are_only_recorded_subtractions": bool(delta_error < 1e-9),
        "raw_net_force_below_limit": bool(np.all(netnorm <= limits["max_net_force_n"])),
        "raw_tcp_torque_below_limit": bool(np.all(taunorm <= limits["max_tcp_torque_nm"])),
        "raw_each_right_finger_force_below_limit": bool(np.all(rightnorm <= limits["max_finger_force_n"])),
        "left_both_soft_opening_reactions_above_minimum": bool(np.all(lopen > limits["minimum_left_opening_reaction_n"])),
        "right_both_soft_opening_reactions_above_minimum_until_release": bool(np.all(ropen[required] > .1)),
        "designated_left_material_within_18mm": bool(np.all(left_distance <= .018)),
        "right_teddy_vertex_within_45mm_until_release": bool(np.all(right_distance[required] <= .045)),
        "no_recorded_stop_reason": not any(r.get("stop_reason") for r in rows),
    }
    violations = []
    for i, row in enumerate(rows):
        reasons = []
        if netnorm[i] > limits["max_net_force_n"]: reasons.append("raw net force")
        if taunorm[i] > limits["max_tcp_torque_nm"]: reasons.append("raw TCP torque")
        if rightnorm[i].max() > limits["max_finger_force_n"]: reasons.append("raw right finger force")
        if lopen[i].min() <= limits["minimum_left_opening_reaction_n"]: reasons.append("left soft opening reaction")
        if required[i] and ropen[i].min() <= .1: reasons.append("right soft opening reaction before release")
        if left_distance[i] > .018: reasons.append("left designated material distance")
        if required[i] and right_distance[i] > .045: reasons.append("right material proximity")
        if reasons: violations.append({"time_s": float(t[i]), "reasons": reasons})
    return {"status": "passed" if all(checks.values()) else "failed", "passed": all(checks.values()),
            "rows": len(rows), "start_time_s": float(t[0]), "end_time_s": float(t[-1]),
            "checks": checks, "first_physical_violation": violations[0] if violations else None,
            "raw_limits": limits,
            "maximum_net_force_n": float(netnorm.max()), "maximum_tcp_torque_nm": float(taunorm.max()),
            "maximum_right_finger_force_n": rightnorm.max(axis=0).tolist(),
            "minimum_right_opening_reaction_until_release_n": ropen[required].min(axis=0).tolist() if required.any() else None,
            "minimum_left_opening_reaction_n": lopen.min(axis=0).tolist(),
            "maximum_designated_left_material_distance_m": float(left_distance.max()),
            "maximum_right_nearest_material_distance_until_release_m": float(right_distance[required].max()) if required.any() else None,
            "maximum_saved_baseline_subtraction_error": delta_error,
            "scope": "IPC soft-coupling STC feedback; excludes native Genesis rigid-contact force",
            "limitations": ["Positive finger-axis projections plus material proximity are grip evidence, not force-closure proof.",
                ("Saved per-link origins and aim axes are independently checked by the richer contact trace audit."
                 if "right_link_aim_origin_world_m" in rows[0] else
                 "Link origins and aim axes are absent, so reference shifting and opening projections cannot be independently reconstructed from these rows."),
                "Raw engineering force and moment limits are uncalibrated; baseline subtraction never waives them."]}


def audit_corrections(report, states, dt):
    """Recompute bounded correction endpoint displacements from saved TCPs."""
    if "guarded_contact_corrections" not in report:
        return {"status":"not_entered","passed":False,
                "reason":"No correction/reobservation loop record was saved."}
    corrections=report["guarded_contact_corrections"]
    observations=report.get("guarded_contact_reobservations",[])
    events=report["events"]
    checks={"at_most_two_corrections":len(corrections)<=2,
            "at_most_three_fresh_observations":len(observations)<=3}
    summaries=[]; actual_total=0.; incomplete=[]
    def check(name,value):checks[name]=bool(value)
    def frame_at(t):
        times=np.asarray(states["time"])
        indices=np.flatnonzero(np.abs(times-float(t))<1e-8)
        if not len(indices):raise ValueError(f"No full-state frame at correction endpoint {t}")
        return np.asarray(states["tcp_right"])[indices[-1]]
    for index,correction in enumerate(corrections,1):
        prefix=f"correction_{index}_"
        phase_names=[f"recenter_height_{index}",f"recenter_teddy_xy_{index}"]
        phase_events=[e for e in events if e["phase"] in phase_names]
        if correction["status"]!="candidate":
            check(prefix+"rejected_proposal_not_executed",not phase_events)
            incomplete.append(f"Correction {index} was rejected before execution")
            continue
        check(prefix+"attempt_index",correction.get("attempt_index")==index)
        check(prefix+"preceded_by_rejected_fresh_observation",len(observations)>=index and observations[index-1]["status"]=="rejected")
        stages=correction["stages_to_execute"]
        check(prefix+"only_height_and_translation",[s["name"] for s in stages]==["height_at_current_xy","translate_outside_mouth"])
        if len(stages)!=2:
            incomplete.append(f"Correction {index} has malformed stage list");continue
        source_time=float(correction["observed_time_s"])
        try:start=frame_at(source_time)
        except ValueError as error:
            check(prefix+"start_frame_exists",False);incomplete.append(str(error));continue
        targets=np.asarray([s["right_tcp_m"] for s in stages],float)
        delta=targets[1,:2]-start[:2]
        planned=float(np.linalg.norm(delta))
        before=float(correction["cumulative_xy_m_before"])
        after=float(correction["cumulative_xy_m_after"])
        check(prefix+"finite_targets",np.isfinite(targets).all())
        check(prefix+"height_preserves_observed_xy",np.linalg.norm(targets[0,:2]-start[:2])<=1e-9)
        check(prefix+"declared_delta_matches_target",np.linalg.norm(delta-np.asarray(correction["correction_xy_m"]))<=1e-9)
        check(prefix+"declared_norm_matches_target",abs(planned-float(correction["correction_distance_m"]))<=1e-9)
        check(prefix+"planned_move_within_30mm",planned<=.030000001)
        check(prefix+"before_matches_prior_actual_endpoints",abs(before-actual_total)<=1e-9)
        check(prefix+"declared_after_matches_observed_before_plus_planned_move",abs(after-before-planned)<=1e-9)
        check(prefix+"planned_total_within_50mm",after<=.050000001)
        check(prefix+"observed_source_matches_saved_tcp",np.linalg.norm(start[:2]-np.asarray(correction["observed_tcp_xy_m"]))<=1e-9)
        if len(phase_events)!=2 or [e["phase"] for e in phase_events]!=phase_names:
            incomplete.append(f"Correction {index} did not record both physical preparation stages");continue
        check(prefix+"physical_motion_starts_at_observed_time",abs(phase_events[0]["start"]-source_time)<=1e-8)
        end_event=phase_events[-1]
        end_time=float(end_event["start"]+np.ceil(end_event["duration"]/dt)*dt)
        try:end=frame_at(end_time)
        except ValueError as error:
            incomplete.append(str(error));continue
        actual=float(np.linalg.norm(end[:2]-start[:2]));actual_total+=actual
        check(prefix+"actual_move_within_30mm",actual<=.030000001)
        check(prefix+"actual_total_within_50mm",actual_total<=.050000001)
        check(prefix+"actual_target_error_within_2mm",np.linalg.norm(end[:2]-targets[1,:2])<=.002)
        check(prefix+"runtime_observed_distance_matches_saved_tcp",abs(actual-float(correction.get("runtime_observed_endpoint_displacement_m",np.nan)))<=1e-9)
        check(prefix+"runtime_observed_total_matches_saved_tcp",abs(actual_total-float(correction.get("runtime_observed_cumulative_endpoint_displacement_m",np.nan)))<=1e-9)
        fresh=observations[index] if len(observations)>index else None
        check(prefix+"strictly_new_fresh_observation_after_execution",fresh is not None and
              float(fresh.get("observed_time_s",-1))>source_time and
              abs(float(fresh.get("observed_time_s",-1))-end_time)<=1e-8)
        summaries.append({"attempt_index":index,"source_time_s":source_time,"actual_endpoint_time_s":end_time,
            "planned_xy_endpoint_displacement_m":planned,"actual_xy_endpoint_displacement_m":actual,
            "actual_cumulative_endpoint_displacement_m":actual_total,
            "following_fresh_status":fresh["status"] if fresh else None})
    insert_events=[e for e in events if e["phase"]=="insert_teddy"]
    if insert_events:
        accepted=bool(observations and observations[-1]["status"]=="candidate")
        check("insertion_requires_last_fresh_acceptance",accepted)
        check("insertion_starts_at_last_observation_time",accepted and len(insert_events)==1 and
              abs(insert_events[0]["start"]-float(observations[-1].get("observed_time_s",-1)))<=1e-8)
    else:
        check("no_unrecorded_insertion_claim",not report.get("sequence_completed",False))
    passed=all(checks.values()) and not incomplete
    return {"status":"passed" if passed else "incomplete" if incomplete else "failed",
        "passed":passed,"checks":checks,"correction_count":len(corrections),
        "fresh_observation_count":len(observations),"actual_cumulative_endpoint_displacement_m":actual_total,
        "corrections":summaries,"incomplete_evidence":incomplete,
        "distance_scope":"Sum of observed correction endpoint XY displacements, not oscillatory path arc length",
        "limits":"CPU saved-state sequencing/budget audit; physical guards and final containment must independently pass."}


def audit_staged_hand(run,report,states,trace,dt):
    """Rebuild native affine hand heights; cross-check bear maxima where saved."""
    result={"status":"not_entered","passed":False,"checks":{},"limitations":[
        "Hand minima are reconstructed for every trace row from saved reference vertices and native affine transforms.",
        "Complete bear geometry exists only at saved frames/endpoints; intervening bear maxima are monitor summaries, not independently reconstructed shapes.",
        "Recorded integration-state separation does not establish continuous-time clearance or final bag containment."]}
    rows_path=run/"staged_hand_trace.json";reference_path=run/"staged_native_hand_reference.npz"
    events=report.get("events",[])
    center=next((e for e in events if e["phase"]=="center_teddy_at_rim"),None)
    if center is None:
        result["reason"]="No physical low-height centering phase was entered."
        return result
    checks=result["checks"]
    try:
        if not rows_path.exists() or not reference_path.exists():raise ValueError("Missing native hand trace or reference vertices")
        rows=json.loads(rows_path.read_text())
        if not rows:raise ValueError("Empty native hand trace after low-height centering started")
        names=["link7","left_finger","right_finger"]
        with np.load(reference_path,allow_pickle=False) as source:
            if set(source.files)!={"left_"+name for name in names}:raise ValueError("Unexpected native hand reference mapping")
            vertices=[np.asarray(source["left_"+name],float) for name in names]
        if any(v.ndim!=2 or v.shape[1]!=3 or len(v)<4 or not np.isfinite(v).all() for v in vertices):
            raise ValueError("Malformed native reference vertices")
        metadata=report["staged_native_hand_monitor"]
        geometry=metadata["cached_reference_geometry"]
        by_name={g["link"]:g for g in geometry}
        checks["exact_three_native_reference_links"]=len(geometry)==3 and set(by_name)==set(names)
        checks["reference_vertices_match_recorded_hashes"]=all(
            hashlib.sha256(v.astype('<f8').tobytes()).hexdigest()==by_name[name]["local_vertices_sha256"]
            and len(v)==int(by_name[name]["vertices"]) for name,v in zip(names,vertices))
        checks["reference_bounds_match_recorded_geometry"]=all(np.allclose(
            [v.min(0),v.max(0)],by_name[name]["local_bounds_m"],rtol=0,atol=1e-10)
            for name,v in zip(names,vertices))
        selection=by_name["link7"]["selection"]
        checks["link7_reference_is_true_palm_subset_not_arm_link7"]=bool(selection.get("collision_body")=="hand" and
            selection.get("merge_target")=="link7" and float(selection.get("native_subset_max_error_m",np.inf))<=1e-7)
        time=np.asarray([r["time_s"] for r in rows],float)
        steps=np.asarray([r["physics_step"] for r in rows],float)
        matrices=np.asarray([r["left_distal_ipc_transforms"] for r in rows],float)
        minima=np.asarray([r["left_link_minimum_z_m"] for r in rows],float)
        bottom=np.asarray([r["left_distal_minimum_z_m"] for r in rows],float)
        top=np.asarray([r["bear_maximum_z_m"] for r in rows],float)
        gaps=np.asarray([r["vertical_gap_m"] for r in rows],float)
        threshold=np.asarray([r["minimum_gap_m"] for r in rows],float)
        indices=np.asarray([r["highest_bear_vertex_index"] for r in rows],float)
        counts=np.asarray([r["bear_vertex_count"] for r in rows],float)
        finite=all(np.isfinite(a).all() for a in (time,steps,matrices,minima,bottom,top,gaps,threshold,indices,counts))
        checks["all_native_rows_finite_and_shapes_valid"]=bool(finite and matrices.shape==(len(rows),3,4,4) and minima.shape==(len(rows),3))
        if not checks["all_native_rows_finite_and_shapes_valid"]:raise ValueError("Nonfinite or malformed native height rows")
        checks["stable_native_link_order"]=all(r["left_link_names"]==names for r in rows)
        checks["native_affine_homogeneous_rows_valid"]=bool(np.allclose(matrices[:,:,3,:],[0.,0.,0.,1.],rtol=0,atol=1e-10))
        checks["native_affine_bodies_noninverted"]=bool(np.all(np.linalg.det(matrices[:,:,:3,:3])>0.))
        reconstructed=np.column_stack([np.min(np.einsum('vi,ri->rv',v,matrices[:,i,2,:3])+matrices[:,i,2,3,None],axis=1)
                                        for i,v in enumerate(vertices)])
        checks["all_hand_minima_reconstruct_from_native_affine_vertices"]=bool(np.allclose(reconstructed,minima,rtol=0,atol=1e-9))
        checks["whole_hand_minimum_matches_three_components"]=bool(np.allclose(reconstructed.min(1),bottom,rtol=0,atol=1e-9))
        checks["reported_vertical_gaps_reconstruct"]=bool(np.allclose(bottom-top,gaps,rtol=0,atol=1e-9))
        checks["minimum_threshold_is_5mm"]=bool(np.allclose(threshold,.005,rtol=0,atol=1e-12) and abs(float(metadata["minimum_gap_m"])-.005)<=1e-12)
        checks["all_recorded_vertical_gaps_at_least_5mm"]=bool(np.all(gaps>=.005))
        n_bear=int(np.asarray(states["bear"]).shape[1])
        checks["whole_bear_counts_and_max_vertex_indices_valid"]=bool(np.all(counts==n_bear) and np.all(indices==np.floor(indices)) and np.all((indices>=0)&(indices<n_bear)))
        release=next((e for e in events if e["phase"]=="release_teddy"),None)
        stop=float(release["start"]) if release else float(report["simulated_seconds"])
        start=float(center["start"])
        expected=np.asarray(trace)[:,0];expected=expected[(expected>=start-1e-9)&(expected<=stop+1e-9)]
        checks["covers_every_step_from_center_entry_through_insert_endpoint"]=bool(len(time)==len(expected) and np.allclose(time,expected,rtol=0,atol=1e-9))
        checks["strictly_increasing_consecutive_steps"]=bool(np.all(np.diff(time)>0) and np.all(np.diff(steps)==1) and np.all(steps==np.floor(steps)))
        checks["timestamps_match_fixed_step"]=bool(np.allclose(np.diff(time),dt,rtol=0,atol=1e-9))
        checks["no_native_height_stop_reason"]=not any(r.get("stop_reason") for r in rows)
        frame_times=np.asarray(states["time"],float);frame_errors=[];argmax_ok=True;matched=0
        for i,t in enumerate(frame_times):
            if t<start-1e-9 or t>stop+1e-9:continue
            matches=np.flatnonzero(np.abs(time-t)<1e-8)
            if not len(matches):continue
            j=int(matches[-1]);points=np.asarray(states["bear"])[i]
            frame_errors.append(abs(float(points[:,2].max())-top[j]));argmax_ok&=abs(float(points[int(indices[j]),2])-top[j])<=1e-9;matched+=1
        expected_frames=int(np.sum((frame_times>=start-1e-9)&(frame_times<=stop+1e-9)))
        checks["all_full_bear_frames_in_monitored_window_matched"]=matched==expected_frames and matched>0
        checks["full_frame_bear_maxima_and_argmax_reconstruct"]=bool(frame_errors and max(frame_errors)<=1e-9 and argmax_ok)
        # Failure states may be more recent than the last regular capture.
        failure_checked=False
        if (run/"failed_soft_state.npz").exists():
            with np.load(run/"failed_soft_state.npz",allow_pickle=False) as failed:
                t=float(failed["time"]);where=np.flatnonzero(np.abs(time-t)<1e-8)
                if len(where):
                    j=int(where[-1]);points=np.asarray(failed["bear"]);failure_checked=True
                    checks["failed_full_bear_frame_matches_height_row"]=bool(abs(float(points[:,2].max())-top[j])<=1e-9)
        coupling=json.loads((run/"coupling_trace.json").read_text());coupling_errors=[];matched_fingers=0
        for sample in coupling:
            where=np.flatnonzero(np.abs(time-float(sample["time"]))<1e-8)
            if not len(where):continue
            j=int(where[-1]);links={x["link"]:x for x in sample["links"] if x["entity"]=="left_franka"}
            for k,name in enumerate(names[1:],1):
                if name in links and "ipc_transform" in links[name]:
                    coupling_errors.append(float(np.max(np.abs(matrices[j,k]-np.asarray(links[name]["ipc_transform"])))));matched_fingers+=1
        checks["saved_coupling_finger_transforms_match_height_trace"]=bool(coupling_errors and max(coupling_errors)<=1e-9)
        result.update(status="passed" if all(checks.values()) else "failed",passed=all(checks.values()),
            rows=len(rows),start_time_s=float(time[0]),end_time_s=float(time[-1]),minimum_gap_m=float(gaps.min()),
            maximum_hand_reconstruction_error_m=float(np.max(np.abs(reconstructed-minima))),
            full_bear_frame_crosschecks=matched,failed_full_bear_frame_crosschecked=failure_checked,
            bear_summary_only_rows_without_matching_full_frame=int(sum(
                not np.any(np.abs(frame_times-t)<1e-8) for t in time)),
            finger_transform_crosschecks=matched_fingers)
    except (ValueError,KeyError,IndexError,TypeError,OSError) as error:
        result.update(status="failed",passed=False,error=str(error))
    return result


def staged_xy_budget_limits(config):
    """Independently bound the explicit frozen-config XY planning allowance."""
    staged=config.get("task",{}).get("contact_trial",{}).get("staged",{})
    individual=float(staged.get("maximum_individual_xy_m",.03))
    cumulative=float(staged.get("maximum_cumulative_xy_m",.05))
    if not np.isfinite([individual,cumulative]).all() or not .03<=individual<=.10 \
            or not .05<=cumulative<=.12 or cumulative<individual:
        raise ValueError("Invalid declared staged XY limits: individual [0.03,0.10] m, cumulative [0.05,0.12] m and cumulative >= individual required")
    return dict(maximum_individual_xy_m=individual,maximum_cumulative_xy_m=cumulative)


def staged_center_contract(config):
    """Derive allowed centers from frozen config without importing the planner."""
    settings=config.get("task",{}).get("contact_trial",{})
    staged=settings.get("staged",{});budgets=staged_xy_budget_limits(config)
    fallback=np.asarray(staged.get("center_offset_xy_m",[-.027,-.010]),float)
    if fallback.shape!=(2,) or not np.isfinite(fallback).all() or not \
            1e-6<=np.linalg.norm(fallback)<=budgets["maximum_individual_xy_m"]:
        raise ValueError("Invalid frozen-config single center offset")
    if "center_offsets_xy_m" not in staged:
        return dict(adaptive=False,allowed_offsets_xy_m=[fallback.tolist()])
    offsets=np.asarray(staged["center_offsets_xy_m"],float)
    if offsets.ndim!=2 or offsets.shape[1]!=2 or not 1<=len(offsets)<=8 \
            or not np.isfinite(offsets).all() or np.any(np.linalg.norm(offsets,axis=1)<1e-6) \
            or np.any(np.linalg.norm(offsets,axis=1)>budgets["maximum_individual_xy_m"]) \
            or len(np.unique(offsets,axis=0))!=len(offsets):
        raise ValueError("Frozen config must declare 1–8 unique finite nonzero center offsets within its XY budget")
    gap=float(staged.get("hand_clearance_m",.01));ratio=float(staged.get("maximum_rim_stretch_ratio",1.01))
    lateral=float(staged.get("entry_max_lateral_excursion_fraction_of_mouth_width",1.))
    overlap=float(staged.get("entry_minimum_section_envelope_overlap_fraction",0.))
    low_geometry=staged.get("low_rim_geometry","ordered_vertical_projection")
    low_gate=staged.get("low_phase_section_gate","strict")
    if settings.get("insertion_strategy")!="lower_then_center" or not np.isfinite([gap,ratio,lateral,overlap]).all() \
            or not .01<=gap<=.03 or not 1.<=ratio<=1.01 or not 0.<=lateral<=1. or not 0.<=overlap<=1. \
            or low_geometry not in ("ordered_vertical_projection","spatial_loop_contact_envelope") \
            or low_gate not in ("strict","diagnostic_only"):
        raise ValueError("Invalid frozen-config adaptive staged policy")
    expected=dict(planned_hand_clearance_m=gap,observed_required_hand_clearance_m=.005,
        low_rim_geometry=low_geometry,low_phase_section_gate=low_gate,center_offset_xy_m=fallback.tolist(),
        maximum_rim_stretch_ratio=ratio,entry_max_lateral_excursion_fraction_of_mouth_width=lateral,
        entry_minimum_section_envelope_overlap_fraction=overlap,**budgets,
        maximum_original_rotation_error_rad=.015,maximum_previous_endpoint_error_m=.002,
        rigid_clearance_m=.008,left_robot_sweep_clearance_m=.005,maximum_joint_velocity_rad_s=.35,
        maximum_joint_acceleration_rad_s2=.5,minimum_downward_duration_s=6.,minimum_center_duration_s=4.,
        each_stage_requires_strictly_new_actual_observation=True,wall_contact_is_risk_not_success=True,
        physical_limits_unchanged=True,center_offsets_xy_m=offsets.tolist(),
        center_selection_policy="fresh_actual_center_then_frozen_remainder_preflight",
        center_selection_score="minimum center/insert rigid surplus above 8mm and COM envelope surplus above 2mm",
        frozen_remainder_does_not_authorize_later_execution=True)
    return dict(adaptive=True,allowed_offsets_xy_m=offsets.tolist(),expected_staged_policy=expected)


def _same_staged_policy(actual,expected):
    return isinstance(actual,dict) and actual==expected and all(
        actual.get(k) is v for k,v in expected.items() if isinstance(v,bool))


def audit_adaptive_center_selection(plan,contract,start,target,source,dt):
    """Check recorded finite selection; frozen remainder is never physical proof."""
    checks={};allowed=np.asarray(contract["allowed_offsets_xy_m"],float)
    selected=np.asarray(plan.get("selected_center_offset_xy_m",[]),float)
    def same_xy(value,expected):
        value=np.asarray(value,float)
        return bool(value.shape==(2,) and np.isfinite(value).all() and np.linalg.norm(value-expected)<=1e-12)
    selection=plan["center_selection"];attempts=selection["attempts"]
    checks["selection_is_planning_only_and_requires_fresh_insert"]=bool(
        selection.get("kind")=="predeclared_finite_center_search" and selection.get("physical_success") is False
        and selection.get("later_insert_requires_fresh_actual_observation") is True)
    checks["selection_observation_matches_actual_center_source"]=bool(
        np.isfinite(float(selection["actual_observed_time_s"])) and abs(float(selection["actual_observed_time_s"])-source)<=1e-9)
    index=selection.get("selected_index")
    index_ok=type(index) is int and 0<=index<len(allowed)
    checks["selection_index_and_offsets_match_frozen_list"]=bool(index_ok and
        same_xy(selected,allowed[index]) and same_xy(selection.get("selected_offset_xy_m",[]),allowed[index]))
    checks["selected_offset_matches_actual_source_to_target"]=bool(selected.shape==(2,) and
        np.isfinite(selected).all() and np.linalg.norm((target-start)[:2]-selected)<=1e-9 and abs((target-start)[2])<=1e-9)
    order_ok=isinstance(attempts,list) and len(attempts)==len(allowed)
    feasible=[];attempts_ok=order_ok;scores_ok=order_ok;chosen_hypothesis_ok=False
    if order_ok:
        for i,attempt in enumerate(attempts):
            if not isinstance(attempt,dict):
                attempts_ok=False;scores_ok=False
                continue
            attempts_ok=bool(attempts_ok and type(attempt.get("index")) is int and attempt["index"]==i
                and same_xy(attempt.get("center_offset_xy_m",[]),allowed[i])
                and attempt.get("physical_success") is False and attempt.get("center_status") in ("candidate","rejected")
                and "stage" not in attempt)
            if attempt.get("center_status")!="candidate":
                attempts_ok=bool(attempts_ok and attempt.get("frozen_remainder_status")!="candidate")
                continue
            if attempt.get("frozen_remainder_status")!="candidate":continue
            # A core candidate may still be rejected by the wrapper's finite
            # margin check. Such an unselected hypothesis is not an executed
            # physics failure and must not enter the ranking.
            if attempt.get("frozen_remainder_rejections"):
                attempts_ok=bool(attempts_ok and i!=index and not any(k in attempt for k in
                    ("score_m","minimum_center_insert_release_rigid_clearance_m",
                     "center_com_boundary_margin_m","frozen_remainder_report")))
                continue
            remainder=attempt["frozen_remainder_report"]
            rigid=float(attempt["minimum_center_insert_release_rigid_clearance_m"])
            com=float(attempt["center_com_boundary_margin_m"]);score=float(attempt["score_m"])
            insert_rigid=float(remainder["robot_trajectory"]["minimum_sampled_rigid_clearance_m"])
            release=float(remainder["release_withdraw_preflight"]["minimum_release_opening_clearance_m"])
            valid=bool(np.isfinite([rigid,com,score,insert_rigid,release]).all() and rigid>=.008 and com>=.002
                and insert_rigid>=.008 and release>=.008 and rigid<=min(insert_rigid,release)+1e-12
                and abs(score-min(rigid-.008,com-.002))<=1e-12
                and attempt.get("frozen_remainder_is_hypothesis_not_actual_observation") is True
                and not attempt.get("center_rejections") and not attempt.get("frozen_remainder_rejections")
                and remainder.get("physical_success") is False and not remainder.get("rejections") and "stage" not in remainder
                and _same_staged_policy(remainder.get("staged_policy"),contract["expected_staged_policy"]))
            scores_ok=bool(scores_ok and valid);feasible.append((score,-i))
            if index_ok and i==index:
                actual_center=plan["report"]
                center_rigid=float(actual_center["robot_trajectory"]["minimum_sampled_rigid_clearance_m"])
                center_com=float(actual_center.get("planned_com_contact_envelope_boundary_distance_m",
                    actual_center.get("planned_com_boundary_distance_m",np.nan)))
                expected_time=source+np.ceil(float(plan["stage"]["duration_s"])/dt)*dt
                chosen_hypothesis_ok=bool(valid and np.isfinite([center_rigid,center_com]).all()
                    and center_rigid>=.008 and center_com>=.002 and abs(rigid-min(center_rigid,insert_rigid,release))<=1e-12
                    and abs(com-center_com)<=1e-12 and abs(float(attempt["hypothetical_time_s"])-expected_time)<=1e-9)
    checks["all_predeclared_attempts_recorded_once_in_order"]=bool(attempts_ok)
    checks["feasible_scores_reconstruct_from_recorded_margins"]=bool(scores_ok and feasible)
    checks["chosen_attempt_has_consistent_frozen_remainder"]=chosen_hypothesis_ok
    checks["selected_index_maximizes_recorded_score_with_first_index_tie_break"]=bool(
        index_ok and feasible and -max(feasible)[1]==index)
    return checks


def audit_staged_sequence(run,report,states,trace,config,reactions):
    """Audit actual staged observations separately from the old high-entry loop."""
    from scipy.spatial.transform import Rotation
    result={"status":"not_entered","passed":False,"checks":{},"stages":[],"incomplete_evidence":[],
        "kind":"independent_lower_then_center_sequence",
        "distance_scope":"Sum of actual staged endpoint XY displacements, not oscillatory path arc length."}
    checks=result["checks"];incomplete=result["incomplete_evidence"]
    plans=report.get("staged_contact_plans",[])
    if not plans:
        result["reason"]="No staged contact plans were saved."
        return result
    try:
        budgets=staged_xy_budget_limits(config)
        individual_limit=budgets["maximum_individual_xy_m"]
        cumulative_limit=budgets["maximum_cumulative_xy_m"]
        result["independently_validated_frozen_config_xy_limits_m"]=budgets
        checks["frozen_config_xy_limits_within_allowed_bounds"]=True
        centers=staged_center_contract(config)
        result["independently_validated_frozen_config_center_contract"]=centers
        checks["frozen_config_center_offsets_valid"]=True
        names=["lower_teddy_to_rim","center_teddy_at_rim","insert_teddy"]
        phases=["entry","center","insert"];events=report["events"]
        dt=float(config["simulation"]["dt"])
        if not np.isfinite(dt) or dt<=0.:raise ValueError("Invalid fixed integration step")
        trace=np.asarray(trace,float)
        if trace.ndim!=2 or trace.shape[1]<19 or not np.isfinite(trace).all():
            raise ValueError("Malformed or nonfinite actual joint trace")
        relevant=[e for e in events if e["phase"] in names]
        checks["at_most_one_plan_for_each_ordered_phase"]=[p.get("phase") for p in plans]==phases[:len(plans)] and len(plans)<=3
        checks["physical_phase_order_unique_and_complete"]=[e["phase"] for e in relevant]==names
        checks["three_accepted_staged_plans"]=len(plans)==3 and all(p.get("status")=="candidate" for p in plans)
        checks["old_high_entry_correction_loop_not_fabricated"]=not report.get("guarded_contact_corrections") and not report.get("guarded_contact_reobservations")
        choice=report["guarded_contact_candidate"]
        original=Rotation.from_quat(choice["route"]["rotation_xyzw"])
        motion=load_module(run/"source/motion.py","frozen_staged_audit_motion")
        planner=motion.Planner(config)
        times=np.asarray(states["time"],float);actual_total=0.;prior_end=None
        def frame_at(t):
            ids=np.flatnonzero(np.abs(times-float(t))<1e-8)
            if not len(ids):raise ValueError(f"No actual full-state frame at staged endpoint {t}")
            k=int(ids[-1]);return np.asarray(states["tcp_right"])[k],np.asarray(states["q_right"])[k]
        transport=[e for e in events if e["phase"]=="transport_teddy"]
        checks["one_completed_initial_transport"]=len(transport)==1
        initial_end=float(transport[0]["start"]+np.ceil(float(transport[0]["duration"])/dt)*dt) if len(transport)==1 else np.nan
        for index,p in enumerate(plans):
            prefix=phases[index]+"_" if index<3 else "unexpected_"
            found=[e for e in relevant if e["phase"]==names[index]] if index<3 else []
            if centers["adaptive"]:
                expected_policy=centers["expected_staged_policy"]
                checks[prefix+"reported_policy_matches_frozen_config"]=_same_staged_policy(
                    p.get("report",{}).get("staged_policy"),expected_policy)
                if p.get("status")=="candidate":
                    checks[prefix+"checkpoint_policy_matches_frozen_config"]=_same_staged_policy(
                        p.get("checkpoint",{}).get("staged_policy"),expected_policy)
            if p.get("status")!="candidate":
                checks[prefix+"rejected_plan_not_executed"]=not found;incomplete.append(f"{p.get('phase')} rejected before execution");continue
            if len(found)!=1:
                incomplete.append(f"{p.get('phase')} lacks exactly one physical event");continue
            event=found[0];source=float(p["observed_time_s"]);target=np.asarray(p["stage"]["right_tcp_m"],float)
            quat=np.asarray(p["stage"]["rotation_xyzw"],float);reference=np.asarray(p["original_rotation_xyzw"],float)
            checks[prefix+"finite_observation_target_orientation"]=bool(np.isfinite(source) and target.shape==(3,) and quat.shape==(4,) and reference.shape==(4,) and all(np.isfinite(x).all() for x in (target,quat,reference)))
            if not checks[prefix+"finite_observation_target_orientation"]:raise ValueError("Malformed staged plan")
            checks[prefix+"stage_name_matches_phase"]=p["stage"]["name"]==names[index]
            checks[prefix+"original_complete_rotation_target_preserved"]=bool((Rotation.from_quat(quat).inv()*original).magnitude()<=1e-8 and (Rotation.from_quat(reference).inv()*original).magnitude()<=1e-8)
            checks[prefix+"event_starts_at_actual_plan_observation"]=abs(float(event["start"])-source)<=1e-8
            checks[prefix+"strict_new_observation_after_previous_completed_stage"]=abs(source-(initial_end if prior_end is None else prior_end))<=1e-8 and source>float(choice["observed_time_s"])
            try:start,q_start=frame_at(source)
            except ValueError as error:incomplete.append(str(error));continue
            end_time=float(event["start"]+np.ceil(float(event["duration"])/dt)*dt)
            delta=target-start;planned=float(np.linalg.norm(delta[:2]));before=float(p["cumulative_xy_m_before"]);after=float(p["cumulative_xy_m_after"])
            checks[prefix+"planned_budget_finite_and_within_declared_xy_limits"]=bool(np.isfinite([planned,before,after]).all() and 0<=planned<=individual_limit+1e-9 and 0<=before<=after<=cumulative_limit+1e-9)
            recorded_budget=p["checkpoint"]["staged_policy"]
            checks[prefix+"checkpoint_budget_matches_frozen_config"]=all(
                np.isfinite(float(recorded_budget[key])) and abs(float(recorded_budget[key])-value)<=1e-12
                for key,value in budgets.items())
            checks[prefix+"planned_budget_matches_actual_prior_and_target"]=abs(before-actual_total)<=1e-9 and abs(after-before-planned)<=1e-9
            checks[prefix+"checkpoint_source_matches_actual_state"]=np.linalg.norm(np.asarray(p["checkpoint"]["observed_tcp_world_m"])-start)<=1e-9
            checks[prefix+"duration_is_slow_guarded_motion"]=float(event["duration"])>= (4. if index==1 else 6.)
            if index==1:
                allowed=np.asarray(centers["allowed_offsets_xy_m"],float)
                checks[prefix+"bounded_declared_horizontal_target"]=abs(delta[2])<=1e-9 and bool(np.any(np.linalg.norm(allowed-delta[:2],axis=1)<=1e-9))
                if centers["adaptive"]:
                    checks.update({prefix+k:v for k,v in audit_adaptive_center_selection(
                        p,centers,start,target,source,dt).items()})
                    checks[prefix+"checkpoint_target_matches_selected_stage"]=bool(np.linalg.norm(
                        np.asarray(p["checkpoint"]["stage_target_tcp_world_m"],float)-target)<=1e-9)
            else:checks[prefix+"strictly_down_at_actual_xy"]=np.linalg.norm(delta[:2])<=1e-9 and delta[2]<0.
            try:end,q_end=frame_at(end_time)
            except ValueError as error:incomplete.append(str(error));continue
            actual=float(np.linalg.norm(end[:2]-start[:2]));actual_total+=actual
            checks[prefix+"actual_endpoint_within_2mm"]=np.linalg.norm(end-target)<=.002
            if index==1 and centers["adaptive"]:
                selected=np.asarray(p["selected_center_offset_xy_m"],float)
                checks[prefix+"actual_executed_offset_tracks_selected_candidate_within_2mm"]=bool(
                    np.linalg.norm((end-start)-np.r_[selected,0.])<=.002)
            checks[prefix+"actual_xy_within_declared_xy_limits"]=actual<=individual_limit+1e-9 and actual_total<=cumulative_limit+1e-9
            checks[prefix+"runtime_distance_records_match_actual_tcp"]=abs(actual-float(p.get("runtime_observed_endpoint_displacement_m",np.nan)))<=1e-9 and abs(actual_total-float(p.get("runtime_observed_cumulative_endpoint_displacement_m",np.nan)))<=1e-9
            checks[prefix+"runtime_endpoint_time_matches_actual_completion"]=abs(float(p.get("runtime_endpoint_time_s",np.nan))-end_time)<=1e-8
            mask=(trace[:,0]>=source-1e-8)&(trace[:,0]<=end_time+1e-8)
            angles=[float((planner.fk("right",q)[1].inv()*original).magnitude()) for q in trace[mask,10:17]]
            checks[prefix+"actual_full_rotation_within_015rad_every_joint_step"]=bool(angles and max(angles)<=.015)
            fk_errors=[float(np.linalg.norm(planner.fk("right",q[:7])[0]-pnt)) for pnt,q in ((start,q_start),(end,q_end))]
            checks[prefix+"actual_tcp_matches_joint_fk"]=max(fk_errors)<=.002
            result["stages"].append(dict(phase=phases[index],observed_time_s=source,actual_end_time_s=end_time,
                planned_xy_distance_m=planned,actual_xy_distance_m=actual,actual_cumulative_xy_m=actual_total,
                actual_max_rotation_error_rad=max(angles) if angles else None,maximum_endpoint_fk_error_m=max(fk_errors)))
            prior_end=end_time
        checks["continuous_original_contact_guard_passed"]=bool(reactions.get("passed"))
        baseline=report.get("contact_guard_baseline");final_baseline=report.get("contact_guard",{}).get("baseline")
        checks["single_original_guard_baseline_preserved"]=bool(baseline is not None and baseline==final_baseline)
        rows=json.loads((run/"contact_guard_trace.json").read_text())
        if "postlift_yaw" in config.get("task",{}).get("contact_trial",{}):
            guard_start=float(report.get("postlift_yaw",{}).get("observed_time_s",np.nan))
            checks["guard_started_before_optional_yaw_not_reset_at_stages"]=bool(rows and baseline and np.isfinite(guard_start) and abs(float(rows[0]["time_s"])-guard_start)<=1e-8 and abs(float(baseline["time_s"])-guard_start)<=1e-8 and guard_start<float(choice["observed_time_s"]))
        else:
            checks["guard_started_with_initial_contact_plan_not_reset_at_stages"]=bool(rows and baseline and abs(float(rows[0]["time_s"])-float(choice["observed_time_s"]))<=1e-8 and abs(float(baseline["time_s"])-float(rows[0]["time_s"]))<=1e-8)
        hand=audit_staged_hand(run,report,states,trace,dt);result["native_hand_height_audit"]=hand
        checks["native_hand_height_trace_valid"]=bool(hand["passed"])
        release=[e for e in events if e["phase"]=="release_teddy"]
        if report.get("sequence_completed"):
            checks["release_only_after_third_actual_staged_endpoint"]=len(release)==1 and prior_end is not None and abs(float(release[0]["start"])-prior_end)<=1e-8
        result.update(status="passed" if all(checks.values()) and not incomplete else "incomplete" if incomplete else "failed",
            passed=all(checks.values()) and not incomplete,actual_cumulative_endpoint_displacement_m=actual_total)
    except (ValueError,KeyError,IndexError,TypeError,OSError) as error:
        result.update(status="failed",passed=False,error=str(error))
    return result


def _yaw_finite_tree(value):
    if isinstance(value,dict):return all(_yaw_finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(_yaw_finite_tree(v) for v in value)
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,(int,float,np.number)):return bool(np.isfinite(value))
    return value is None or isinstance(value,str)


def _yaw_number(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or not np.isfinite(value):
        raise ValueError('Expected finite numeric yaw scalar')
    return float(value)


def _yaw_rigid_rows(rows):
    required={'left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms'}
    return bool(isinstance(rows,dict) and set(rows)==required and _yaw_finite_tree(rows) and
        all(isinstance(v,dict) and _yaw_number(v.get('distance'))>=.008 for v in rows.values()))


def _yaw_curve_bounds(curve,duration):
    """Independent analytic cubic extrema and conservative min-jerk rate bounds."""
    maxima1=np.zeros(7);maxima2=np.zeros(7);minimum=np.full(7,np.inf);maximum=-minimum.copy()
    for i,width in enumerate(np.diff(curve.x)):
        for j in range(7):
            a,b,c,d=curve.c[:,i,j];points=[0.,float(width)]
            roots=np.roots(np.trim_zeros([3*a,2*b,c],'f')) if a!=0 or b!=0 else []
            points += [float(v.real) for v in roots if abs(v.imag)<1e-12 and 0<v.real<width]
            values=np.polyval([a,b,c,d],points)
            minimum[j]=min(minimum[j],float(min(values)));maximum[j]=max(maximum[j],float(max(values)))
            derivative_points=[0.,float(width)]
            if a!=0 and 0< -b/(3*a)<width:derivative_points.append(float(-b/(3*a)))
            maxima1[j]=max(maxima1[j],float(np.abs(np.polyval([3*a,2*b,c],derivative_points)).max()))
            maxima2[j]=max(maxima2[j],abs(2*b),abs(6*a*width+2*b))
    return dict(minimum=minimum,maximum=maximum,
        velocity=maxima1*1.875/duration,
        acceleration=(maxima2*1.875**2+maxima1*10/np.sqrt(3))/duration**2)


def _yaw_actual_soft_state(run,config,rest,state,record,planner):
    """Recompute endpoint material/geometry limits without the yaw helper."""
    from scipy.spatial import cKDTree
    checks={};bag=np.asarray(state['bag'],float);bear=np.asarray(state['bear'],float)
    if bag.shape!=np.asarray(rest['bag']).shape or bear.shape!=np.asarray(rest['bear']).shape or not np.isfinite(bag).all() or not np.isfinite(bear).all():raise ValueError('Nonfinite/mismatched actual soft state')
    faces=np.asarray(rest['bag_faces'],int);tets=np.asarray(rest['bear_tets'],int)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    def determinants(points):
        tetra=points[tets];return np.linalg.det(np.stack([tetra[:,i]-tetra[:,0] for i in (1,2,3)],axis=-1))
    references=determinants(np.asarray(rest['bear'],float));length0=np.linalg.norm(np.diff(np.asarray(rest['bag'])[edges],axis=1)[:,0],axis=1)
    if np.any(np.abs(references)<6e-14) or np.any(length0<1e-10):raise ValueError('Degenerate rest tetra/edge')
    ratio=float((determinants(bear)/references).min());stretch=float((np.linalg.norm(np.diff(bag[edges],axis=1)[:,0],axis=1)/length0).max())
    table=float(config['table']['height']);half=np.asarray(config['table']['size'][:2],float)/2
    penetration=[]
    for points in (bag,bear):
        inside=np.all(np.abs(points[:,:2])<=half,axis=1)
        penetration.append(max(0.,table-float(points[inside,2].min())) if inside.any() else 0.)
    asset=Path(config['bag']['npz_file']);asset=run/'source/assets'/asset.name
    with np.load(asset,allow_pickle=False) as source:
        group=config['task']['handle_grasps']['left']['group']
        material=source['points'][source[group]]+np.asarray(config['bag']['position'])
    errors,ids=cKDTree(rest['bag']).query(material)
    if errors.max()>1e-6:raise ValueError('Frozen designated handle material cannot be mapped')
    left=float(np.linalg.norm(bag[np.unique(ids)]-state['tcp_left'],axis=1).min())
    right=float(np.linalg.norm(bear-state['tcp_right'],axis=1).min())
    measured={'time_s':float(state['time']),'minimum_tetra_volume_ratio':ratio,
        'maximum_bag_edge_stretch_ratio':stretch,'bag_table_penetration_m':penetration[0],
        'bear_table_penetration_m':penetration[1],'left_designated_material_distance_m':left,'right_nearest_teddy_distance_m':right,'all_soft_states_finite':1.}
    checks['independent_soft_and_grasp_limits']=bool(ratio>=.05 and stretch<=1.15 and max(penetration)<=.002 and left<=.018 and right<=.045)
    checks['physical_record_matches_saved_geometry']=all(np.isfinite(float(record.get(k,np.nan))) and abs(float(record[k])-v)<=1e-8 for k,v in measured.items())
    qs={s:np.asarray(state['q_'+s])[:7] for s in ('left','right')}
    fingers={s:float(np.clip(np.asarray(state['q_'+s])[7:].mean(),0,.04)) for s in qs}
    rigid=planner.clearance(qs,fingers)
    checks['independent_actual_rigid_8mm']=_yaw_rigid_rows(rigid)
    stored=record['actual_rigid_clearances']
    checks['actual_rigid_record_matches_recomputed_distances']=bool(_yaw_rigid_rows(stored) and all(abs(rigid[k]['distance']-stored[k]['distance'])<=1e-8 for k in rigid))
    checks['actual_joints_fingers_and_tcp_valid']=all(np.all(qs[s]>=planner.limits[s][:,0]) and np.all(qs[s]<=planner.limits[s][:,1]) and
        np.all(np.asarray(state['q_'+s])[7:]>=-.0005) and np.all(np.asarray(state['q_'+s])[7:]<=.0405) and
        np.linalg.norm(planner.fk(s,qs[s])[0]-state['tcp_'+s])<=.002 for s in qs)
    return checks,measured


def _yaw_verify_frozen_arc(sweep,source,command,angle_degrees,planner,mujoco):
    """Check saved separating axes against robot vertices and bear arc endpoints.

    This repeats no collision search: each recorded axis must itself prove the
    bound after the exact sagitta allowance and a 1e-7 m numeric guard.
    """
    import itertools
    from scipy.spatial.transform import Rotation
    angle=float(np.deg2rad(angle_degrees));count=max(1,int(np.ceil(abs(angle)/np.deg2rad(5))))
    rel=np.asarray(source['bear'])-np.asarray(source['tcp_right']);radius=float(np.linalg.norm(rel[:,:2],axis=1).max())
    sagitta=float(radius*(1-np.cos(abs(angle)/count/2)))
    if not isinstance(sweep,dict) or not _yaw_finite_tree(sweep) or sweep.get('status')!='clear' or sweep.get('physical_success') is not False or sweep.get('required_gap_m')!=.005 or sweep.get('segment_count')!=count or sweep.get('left_poses_checked')!=['actual','commanded']:return False
    if abs(float(sweep['maximum_xy_vertex_radius_m'])-radius)>1e-12 or abs(float(sweep['sagitta_outer_bound_m'])-sagitta)>1e-12:return False
    if len(sweep['per_pose'])!=2:return False
    m,d=planner.m,planner.d;original=d.qpos.copy();all_gaps=[]
    try:
        for pose,label in zip(sweep['per_pose'],('actual','commanded')):
            q=np.asarray(source['q_left']).copy()
            if label=='commanded':q[:7]=command['left']
            if pose.get('pose')!=label or not np.array_equal(pose['q_left'],q) or len(pose['segments'])!=count:return False
            d.qpos[planner.qindices['left']]=q[:7]
            for finger in (1,2):d.qpos[m.joint(f'left_finger_joint{finger}').qposadr[0]]=q[6+finger]
            mujoco.mj_forward(m,d);solids={}
            for geom in range(m.ngeom):
                body=planner.body_name(geom)
                if not body.startswith('left_') or body.endswith('link0') or not (m.geom_contype[geom] or m.geom_conaffinity[geom]):continue
                kind=int(m.geom_type[geom])
                if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh=int(m.geom_dataid[geom]);offset=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh]);vertices=m.mesh_vert[offset:offset+n].astype(float)
                elif kind==int(mujoco.mjtGeom.mjGEOM_BOX):vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))*m.geom_size[geom]
                else:raise ValueError('Unsupported frozen left collision shape')
                solids[geom]=(body,vertices@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom])
            for k,segment in enumerate(pose['segments']):
                rows=segment['per_geom']
                if segment['index']!=k or len(rows)!=len(solids) or {r['geom_id'] for r in rows}!=set(solids):return False
                ends=np.vstack([Rotation.from_rotvec([0.,0.,angle*f]).apply(rel)+source['tcp_right'] for f in (k/count,(k+1)/count)])
                gaps=[]
                for row in rows:
                    body,vertices=solids[row['geom_id']];cert=row['certificate'];axis=np.asarray(cert['separating_axis_world'],float)
                    if row['left_body']!=body or axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-8:return False
                    axis=axis/np.linalg.norm(axis);a=ends@axis;b=vertices@axis;gap=max(0.,float(b.min()-a.max()),float(a.min()-b.max()))
                    guard=_yaw_number(cert['numerical_guard_m']);lower=max(0.,gap-guard);bound=lower-sagitta
                    if cert['status']!='clear' or guard<1e-7 or abs(cert['raw_separating_projection_gap_m']-gap)>1e-9 or abs(cert['conservative_gap_lower_bound_m']-lower)>1e-9 or abs(row['continuous_arc_lower_bound_m']-bound)>1e-9 or bound<.005:return False
                    gaps.append(bound)
                if abs(segment['minimum_continuous_gap_lower_bound_m']-min(gaps))>1e-9:return False
                all_gaps+=gaps
    finally:d.qpos[:]=original;mujoco.mj_forward(m,d)
    return bool(all_gaps and abs(sweep['minimum_continuous_gap_lower_bound_m']-min(all_gaps))<=1e-9)


def audit_postlift_yaw(run,report,states,trace,config,reactions):
    """Read saved native evidence; never plan/execute or import the yaw helper.

    Report acceptance concerns this optional phase only. Even complete evidence
    is not insertion success, continuous native clearance, or force closure.
    """
    from scipy.interpolate import PPoly
    from scipy.spatial.transform import Rotation,Slerp
    enabled='postlift_yaw' in config.get('task',{}).get('contact_trial',{})
    if not enabled:
        unexpected=any(k in report for k in ('postlift_yaw','postlift_yaw_execution')) or any(
            e.get('phase')=='orient_teddy_postlift' for e in report.get('events',[]))
        return dict(status='failed' if unexpected else 'not_configured',passed=not unexpected,
            required=False,physical_success=False,checks={'no_unconfigured_yaw_evidence':not unexpected})
    result=dict(kind='independent_optional_postlift_yaw_audit',status='incomplete',passed=False,
        required=True,physical_success=False,checks={},incomplete_evidence=[],
        limitations=['This is an audit of saved native states and the archived reference, not task success.',
            'Robot clearance/FK reference checks are sampled; force and grip evidence remains the every-step native guard.',
            'Frozen bear/left arc certificates concern ideal rigid grasp and two static left poses only.',
            'A saved spline is execution evidence, not an independently observed command stream. Actual joints are checked separately.'])
    checks=result['checks'];incomplete=result['incomplete_evidence']
    try:
        option=config['task']['contact_trial']['postlift_yaw']
        if not isinstance(option,dict) or set(option)!={'world_z_degrees','duration_s'}:raise ValueError('Invalid frozen yaw config structure')
        angle=_yaw_number(option['world_z_degrees']);minimum_duration=_yaw_number(option['duration_s'])
        if not 0<abs(angle)<=15 or minimum_duration<6:raise ValueError('Frozen yaw config outside original bounds')
        dt=_yaw_number(config['simulation']['dt'])
        if dt<=0:raise ValueError('Invalid integration step')
        checks['frozen_optional_config_valid']=True
        manifest=json.loads((run/'input_sha256.json').read_text())
        required_sources=['source/postlift_yaw.py','source/motion.py','source/run.py','config.json',
            'source/assets/'+Path(config['robots'].get('mjcf_path','panda_tactile.xml')).name,
            'source/assets/'+Path(config['bag']['npz_file']).name]
        checks['yaw_sources_and_config_match_frozen_manifest']=all(
            key in manifest and (run/key).is_file() and hashlib.sha256((run/key).read_bytes()).hexdigest()==manifest[key]
            for key in required_sources)
        checks['in_memory_config_matches_frozen_config']=config==json.loads((run/'config.json').read_text())
        events=report.get('events',[])
        checks['events_file_matches_report']=events==json.loads((run/'events.json').read_text())
        yaw_events=[e for e in events if e.get('phase')=='orient_teddy_postlift']
        plan=report.get('postlift_yaw')
        if plan is None:
            checks['yaw_not_executed_without_candidate']=not yaw_events and 'postlift_yaw_execution' not in report
            incomplete.append('Configured yaw was not entered or candidate evidence was not saved')
            result['status']='not_entered' if all(checks.values()) else 'failed';return result
        checks['candidate_json_matches_report']=plan==json.loads((run/'postlift_yaw_candidate.json').read_text())
        checks['planning_record_does_not_claim_physical_execution']=plan.get('physical_success') is False and plan.get('runtime_execution_performed') is False and plan.get('modifies_soft_points') is False
        if plan.get('status')!='candidate':
            checks['rejected_candidate_not_executed']=not yaw_events and 'postlift_yaw_execution' not in report
            incomplete.append('Yaw preflight rejected before execution')
            result['status']='preflight_rejected' if all(checks.values()) else 'failed';return result
        required_checks={'actual_original_physical_gates','lift_above_table_40mm','command_actual_pose_consistent',
            'analytic_joint_limits','continuous_reference_rate_bounds','sampled_all_raw_rigid_checks_finite_and_above_8mm',
            'sampled_actual_reference_fk_tolerances','actual_and_commanded_left_frozen_rotation_above_5mm','returns_exact_checked_plan'}
        checks['accepted_candidate_has_full_finite_check_set']=bool(_yaw_finite_tree(plan) and plan.get('kind')=='guarded_postlift_yaw' and
            isinstance(plan.get('checks'),dict) and set(plan['checks'])==required_checks and all(v is True for v in plan['checks'].values()) and not plan.get('rejections'))
        checks['parameters_equal_frozen_config']=plan.get('parameters')==option
        checks['fresh_native_preparation_required']=plan.get('requires_fresh_actual_preparation_after_execution') is True
        observed=_yaw_number(plan['observed_time_s']);duration=_yaw_number(plan['duration_s'])
        checks['duration_respects_config']=duration>=minimum_duration
        steps=int(np.ceil(duration/dt));expected_end=observed+steps*dt
        result.update(observed_time_s=observed,planned_duration_s=duration,expected_step_count=steps,expected_end_time_s=expected_end)
        times=np.asarray(states['time'],float)
        if times.ndim!=1 or not np.isfinite(times).all():raise ValueError('Malformed saved state times')
        def frame_at(t):
            ids=np.flatnonzero(np.abs(times-t)<=1e-8)
            if not len(ids):raise ValueError('Missing complete native endpoint at '+str(t))
            k=int(ids[-1]);return {key:np.asarray(value)[k] for key,value in states.items()}
        source=frame_at(observed)
        motion=load_module(run/'source/motion.py','frozen_postlift_audit_motion')
        planner_config=json.loads(json.dumps(config));planner_config['robots']['mjcf_path']=str(run/'source/assets'/Path(config['robots'].get('mjcf_path','panda_tactile.xml')).name)
        planner=motion.Planner(planner_config)
        with np.load(run/'rest.npz',allow_pickle=False) as archive:rest={k:archive[k].copy() for k in archive.files}
        actual_rot={};command={}
        for side in ('left','right'):
            q=np.asarray(source['q_'+side],float);tcp=np.asarray(source['tcp_'+side],float)
            if q.shape!=(9,) or tcp.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(tcp).all():raise ValueError('Malformed actual yaw source '+side)
            actual_p,actual_rot[side]=planner.fk(side,q[:7])
            record=plan['actual_source_poses'][side];cmd=plan['command_source_poses'][side]
            command[side]=np.asarray(cmd['q'],float)
            if command[side].shape!=(7,) or not np.isfinite(command[side]).all():raise ValueError('Malformed archived command '+side)
            cp,cr=planner.fk(side,command[side]);pe=float(np.linalg.norm(cp-actual_p));re=float((cr*actual_rot[side].inv()).magnitude())
            checks[side+'_source_matches_saved_actual_fk']=bool(np.allclose(record['q'],q[:7],rtol=0,atol=1e-12) and
                np.linalg.norm(actual_p-tcp)<=.002 and np.linalg.norm(np.asarray(record['tcp_m'])-actual_p)<=1e-9 and
                (Rotation.from_quat(record['rotation_xyzw'])*actual_rot[side].inv()).magnitude()<=1e-9)
            checks[side+'_command_source_fk_consistent']=bool(np.linalg.norm(cp-np.asarray(cmd['tcp_m']))<=1e-9 and
                (cr*Rotation.from_quat(cmd['rotation_xyzw']).inv()).magnitude()<=1e-9 and pe<=.002 and re<=.015 and
                abs(pe-float(cmd['actual_position_error_m']))<=1e-9 and abs(re-float(cmd['actual_rotation_error_rad']))<=1e-9)
        target_rotation=Rotation.from_rotvec([0.,0.,np.deg2rad(angle)])*actual_rot['right']
        target_tcp=np.asarray(source['tcp_right'],float)
        checks['target_is_world_yaw_of_actual_source_with_fixed_tcp']=bool(np.linalg.norm(np.asarray(plan['target_tcp_m'])-target_tcp)<=1e-9 and
            (Rotation.from_quat(plan['target_rotation_xyzw'])*target_rotation.inv()).magnitude()<=1e-9)
        lift=[e for e in events if e.get('phase')=='lift_teddy']
        checks['observed_source_is_actual_lift_endpoint']=bool(len(lift)==1 and abs(float(lift[0]['start'])+np.ceil(float(lift[0]['duration'])/dt)*dt-observed)<=1e-8)
        table=float(config['table']['height'])
        checks['actual_source_bear_above_table_40mm']=bool(np.asarray(source['bear'])[:,2].min()>table+.04 and
            abs(float(np.asarray(source['bear'])[:,2].min())-float(plan['actual_lifted_bear_minimum_z_m']))<=1e-9)
        checks['actual_source_raw_rigid_record_complete_and_valid']=_yaw_rigid_rows(plan['actual_physical_state']['actual_rigid_clearances'])
        source_checks,source_measurements=_yaw_actual_soft_state(run,config,rest,source,plan['actual_physical_state'],planner)
        checks.update({'source_'+k:v for k,v in source_checks.items()});result['actual_source_measurements']=source_measurements
        execution=report.get('postlift_yaw_execution')
        if execution is None:
            incomplete.append('Accepted yaw lacks native execution record');return result
        checks['execution_file_matches_report']=execution==json.loads((run/'postlift_yaw_execution.json').read_text())
        checks['execution_evidence_finite_and_same_plan']=bool(_yaw_finite_tree(execution) and execution.get('physical_success') is False and
            execution.get('exact_checked_plan_executed') is True and execution.get('contact_guard_baseline_reset') is False)
        checks['execution_times_and_step_count_match_plan']=bool(abs(_yaw_number(execution['start_time_s'])-observed)<=1e-9 and
            abs(_yaw_number(execution['planned_duration_s'])-duration)<=1e-9 and
            type(execution['actual_step_count']) is int and execution['actual_step_count']==steps)
        checks['one_yaw_event_at_actual_source']=bool(len(yaw_events)==1 and abs(float(yaw_events[0]['start'])-observed)<=1e-8 and abs(float(yaw_events[0]['duration'])-duration)<=1e-9)
        if len(yaw_events)!=1:raise ValueError('Missing or duplicate native yaw event')
        yaw_index=events.index(yaw_events[0]);lift_index=events.index(lift[0]) if len(lift)==1 else -2
        checks['yaw_follows_lift_before_any_contact_transport']=yaw_index==lift_index+1 and not any(
            e.get('phase') in ('adjust_teddy_height_contact_trial','transport_teddy_along_y','transport_teddy','lower_teddy_to_rim','center_teddy_at_rim','insert_teddy','release_teddy') for e in events[:yaw_index])
        baseline=report.get('contact_guard_baseline');final_baseline=report.get('contact_guard',{}).get('baseline')
        rows=json.loads((run/'contact_guard_trace.json').read_text())
        checks['single_guard_baseline_starts_before_yaw_and_is_never_reset']=bool(rows and baseline and baseline==final_baseline and
            abs(float(baseline['time_s'])-observed)<=1e-8 and abs(float(rows[0]['time_s'])-observed)<=1e-8)
        guard_times=np.asarray([r['time_s'] for r in rows],float)
        run_end=_yaw_number(report['simulated_seconds'])
        checks['reported_run_reaches_claimed_successful_yaw_endpoint']=bool(run_end>=observed-1e-8 and (execution.get('status')!='executed_endpoint_passed' or run_end>=expected_end-1e-8))
        stop=min(run_end,expected_end)
        expected_times=observed+np.arange(int(round((stop-observed)/dt))+1)*dt
        got=guard_times[(guard_times>=observed-1e-8)&(guard_times<=stop+1e-8)]
        checks['native_guard_covers_every_yaw_step']=bool(np.isfinite(guard_times).all() and len(got)==len(expected_times) and np.allclose(got,expected_times,rtol=0,atol=1e-8))
        checks['native_guard_keeps_right_grasp_required_through_yaw']=all(r.get('right_two_sided_contact_required') is True for r in rows if observed-1e-8<=r['time_s']<=expected_end+1e-8)
        checks['complete_run_native_reaction_audit_passed']=reactions.get('passed') is True
        # Reconstruct the saved cubics directly; never call the production helper.
        with np.load(run/'postlift_yaw_plan.npz',allow_pickle=False) as archive:
            raw={k:archive[k].copy() for k in archive.files}
        expected_keys={'duration','observed_time','left_x','left_c','right_x','right_c','finger_left','finger_right'}
        if set(raw)!=expected_keys or not all(np.isfinite(v).all() for v in raw.values()):raise ValueError('Malformed/nonfinite exact curve archive')
        checks['archive_clock_matches_candidate']=abs(float(raw['duration'])-duration)<=1e-12 and abs(float(raw['observed_time'])-observed)<=1e-12
        curves={};bounds={}
        for side in ('left','right'):
            x=raw[side+'_x'];c=raw[side+'_c']
            if x.ndim!=1 or len(x)<2 or x[0]!=0 or x[-1]!=1 or np.any(np.diff(x)<=0) or c.shape!=(4,len(x)-1,7):raise ValueError('Curve must cover [0,1] exactly with 7 joints')
            if raw['finger_'+side].shape!=() or not -.04<=float(raw['finger_'+side])<=.04:raise ValueError('Malformed finite-force finger target')
            expected_finger=0. if side=='left' else _yaw_number(config['task'].get('bear_finger_target',.027))
            checks[side+'_archived_motor_target_preserves_prior_grasp']=abs(float(raw['finger_'+side])-expected_finger)<=1e-12
            # Production CubicSpline is C2. Independent piecewise bounds are
            # insufficient if a corrupted archive jumps between intervals.
            joins=[]
            for k,width in enumerate(np.diff(x)[:-1]):
                a,b,cc,d=c[:,k];next_a,next_b,next_c,next_d=c[:,k+1]
                joins.extend([np.max(np.abs(((a*width+b)*width+cc)*width+d-next_d)),
                    np.max(np.abs((3*a*width+2*b)*width+cc-next_c)),
                    np.max(np.abs(6*a*width+2*b-2*next_b))])
            checks[side+'_cubic_c0_c1_c2_continuity']=bool(not joins or np.isfinite(joins).all() and max(joins)<=1e-10)
            curves[side]=PPoly(c,x);bounds[side]=_yaw_curve_bounds(curves[side],duration)
            b=bounds[side];limits=planner.limits[side]
            checks[side+'_archive_exact_command_endpoints']=bool(np.array_equal(curves[side](0.),command[side]) and np.array_equal(curves[side](1.),np.asarray(plan[side+'_q_end'])))
            checks[side+'_analytic_joint_limits_and_continuous_rates']=bool(np.all(b['minimum']>=limits[:,0]) and np.all(b['maximum']<=limits[:,1]) and np.all(b['velocity']<=.35) and np.all(b['acceleration']<=.5))
            recorded=plan['robot_preflight']['continuous_rate_bounds'][side]
            checks[side+'_reported_rate_bounds_cover_independent_bounds']=bool(np.all(np.asarray(recorded['velocity_upper_bound_rad_s'])>=b['velocity']-1e-12) and
                np.all(np.asarray(recorded['acceleration_upper_bound_rad_s2'])>=b['acceleration']-1e-12) and
                np.all(np.asarray(recorded['velocity_upper_bound_rad_s'])<=.35) and np.all(np.asarray(recorded['acceleration_upper_bound_rad_s2'])<=.5))
        checks['left_command_entire_cubic_is_constant']=bool(np.all(curves['left'].c[:3]==0) and np.all(curves['left'].c[3]==command['left']))
        if not all(checks.values()):
            result.update(status='failed',passed=False);return result
        # Actual and commanded left positions are checked separately. Measured
        # finger geometry matches the preflight; motor targets are not geometry.
        measured_fingers={s:float(np.clip(np.asarray(source['q_'+s])[7:].mean(),0,.04)) for s in curves}
        ideal=Slerp([0.,1.],Rotation.from_quat([actual_rot['right'].as_quat(),target_rotation.as_quat()]))
        rigid_min=np.inf;pmax=0.;rmax=0.;all_rigid=True
        for u in np.linspace(0,1,401):
            fraction=10*u**3-15*u**4+6*u**5;qright=curves['right'](fraction)
            p,r=planner.fk('right',qright);pmax=max(pmax,float(np.linalg.norm(p-target_tcp)))
            rmax=max(rmax,float((r*ideal(fraction).inv()).magnitude()))
            for qleft in (command['left'],np.asarray(source['q_left'])[:7]):
                cr=planner.clearance({'left':qleft,'right':qright},measured_fingers)
                all_rigid=all_rigid and _yaw_rigid_rows(cr)
                rigid_min=min(rigid_min,*[_yaw_number(v['distance']) for v in cr.values()])
        checks['independent_401_reference_samples_preserve_8mm_and_fk_limits']=bool(all_rigid and pmax<=.0005 and rmax<=.002)
        robot=plan['robot_preflight']
        checks['reference_summary_matches_archive_recomputation']=bool(robot['rigid_sample_count']==401 and robot['left_poses_checked']==['actual','commanded'] and
            robot['continuous_rigid_clearance_certified'] is False and robot['command_start_exactly_preserved'] is True and robot['left_command_constant'] is True and
            abs(float(robot['minimum_sampled_rigid_clearance_m'])-rigid_min)<=1e-9 and abs(float(robot['maximum_sampled_tcp_error_m'])-pmax)<=1e-9 and
            abs(float(robot['maximum_sampled_slerp_error_rad'])-rmax)<=1e-9)
        sweep=plan['frozen_bear_left_rotation_sweep']
        checks['independent_frozen_left_arc_axes_certify_complete_5mm_scope']=_yaw_verify_frozen_arc(sweep,source,command,angle,planner,motion.mujoco)
        if execution.get('status')!='executed_endpoint_passed':
            incomplete.append('Native yaw did not pass its fresh endpoint; status='+str(execution.get('status')))
            checks['failed_yaw_does_not_prepare_contact']='guarded_contact_candidate' not in report
            result['status']='execution_rejected' if all(checks.values()) else 'failed'
            return result
        end=frame_at(expected_end);qp=np.asarray(end['q_right'],float);end_p,end_r=planner.fk('right',qp[:7])
        ep=np.asarray(end['tcp_right'],float);pe=float(np.linalg.norm(ep-target_tcp));re=float((end_r*target_rotation.inv()).magnitude())
        checks['fresh_native_endpoint_clock_matches_integer_steps']=bool(abs(float(execution['end_time_s'])-expected_end)<=1e-8 and abs(float(execution['endpoint_time_s'])-expected_end)<=1e-8)
        checks['native_endpoint_record_matches_saved_fk_and_tcp']=bool(np.linalg.norm(ep-np.asarray(execution['endpoint_tcp_m']))<=1e-9 and np.linalg.norm(end_p-ep)<=.002 and
            (end_r*Rotation.from_quat(execution['endpoint_rotation_xyzw']).inv()).magnitude()<=1e-9 and
            abs(float(execution['endpoint_position_error_m'])-pe)<=1e-9 and abs(float(execution['endpoint_rotation_error_rad'])-re)<=1e-9)
        checks['actual_endpoint_within_2mm_15mrad_and_bear_lifted']=bool(pe<=.002 and re<=.015 and np.asarray(end['bear'])[:,2].min()>table+.04)
        checks['endpoint_has_no_physical_rejections_and_valid_raw_rigid']=bool(not execution['endpoint_rejections'] and not execution['actual_physical_state']['physical_state_rejections'] and _yaw_rigid_rows(execution['actual_physical_state']['actual_rigid_clearances']))
        endpoint_checks,endpoint_measurements=_yaw_actual_soft_state(run,config,rest,end,execution['actual_physical_state'],planner)
        checks.update({'endpoint_'+k:v for k,v in endpoint_checks.items()});result['actual_endpoint_measurements']=endpoint_measurements
        choice=report.get('guarded_contact_candidate')
        checks['contact_preparation_observes_fresh_postyaw_native_state']=bool(isinstance(choice,dict) and np.isfinite(float(choice.get('observed_time_s',np.nan))) and abs(float(choice['observed_time_s'])-expected_end)<=1e-8)
        preparation=choice.get('report',{}) if isinstance(choice,dict) else {}
        checks['preparation_nested_observation_matches_actual_endpoint']=bool(isinstance(preparation,dict) and _yaw_finite_tree(preparation) and
            np.isfinite(float(preparation.get('observed_time_s',np.nan))) and abs(float(preparation['observed_time_s'])-expected_end)<=1e-8 and
            np.asarray(preparation.get('observed_right_tcp_world_m',[])).shape==(3,) and np.linalg.norm(np.asarray(preparation['observed_right_tcp_world_m'])-ep)<=1e-9)
        if checks['preparation_nested_observation_matches_actual_endpoint']:
            prep_checks,_=_yaw_actual_soft_state(run,config,rest,end,preparation['physical_state'],planner)
            checks.update({'preparation_'+k:v for k,v in prep_checks.items()})
            tets=np.asarray(rest['bear_tets'],int);tet=np.asarray(rest['bear'])[tets]
            weights=np.abs(np.linalg.det(np.stack([tet[:,i]-tet[:,0] for i in (1,2,3)],axis=-1)))
            actual_com=np.average(np.asarray(end['bear'])[tets].mean(axis=1),weights=weights,axis=0)
            stored_com=np.asarray(preparation.get('observed_center_of_mass_world_m',[]),float)
            checks['preparation_com_matches_actual_rest_mass_com']=stored_com.shape==(3,) and np.isfinite(stored_com).all() and np.linalg.norm(stored_com-actual_com)<=1e-9
        if isinstance(choice,dict) and choice.get('status')=='candidate':
            route=choice.get('route',{})
            checks['accepted_preparation_route_is_fresh_actual_endpoint']=bool(isinstance(route,dict) and _yaw_finite_tree(route) and
                np.isfinite(float(route.get('observed_time_s',np.nan))) and abs(float(route['observed_time_s'])-expected_end)<=1e-8 and
                (Rotation.from_quat(route['rotation_xyzw'])*end_r.inv()).magnitude()<=1e-9)
        if isinstance(choice,dict) and (choice.get('status')=='candidate' or 'predicted_targets' in preparation):
            predicted=preparation.get('predicted_targets',{})
            checks['preparation_predicted_full_rotation_comes_from_actual_endpoint']=bool(isinstance(predicted,dict) and
                np.asarray(predicted.get('rotation_xyzw',[])).shape==(4,) and
                (Rotation.from_quat(predicted['rotation_xyzw'])*end_r.inv()).magnitude()<=1e-9)
        checks['preparation_keeps_planning_separate_from_execution']=bool(isinstance(choice,dict) and choice.get('status') in ('candidate','rejected') and
            choice.get('physical_success') is False and choice.get('runtime_execution_performed') is False)
        trace=np.asarray(trace,float)
        if trace.ndim!=2 or trace.shape[1]!=19 or not np.isfinite(trace).all():raise ValueError('Actual joint trace must have 19 finite columns')
        mask=(trace[:,0]>=observed-1e-8)&(trace[:,0]<=expected_end+1e-8);actual_trace=trace[mask]
        checks['actual_joint_trace_covers_yaw_integration_steps']=bool(len(actual_trace)>=steps and len(actual_trace)<=steps+1 and
            np.allclose(actual_trace[-steps:,0],observed+np.arange(1,steps+1)*dt,rtol=0,atol=1e-8))
        consistency=[]
        for k in np.flatnonzero((times>=observed-1e-8)&(times<=expected_end+1e-8)):
            ids=np.flatnonzero(np.abs(trace[:,0]-times[k])<=1e-8)
            if not len(ids) and abs(times[k]-observed)<=1e-8:continue  # preceding source sample may be outside a phase-only trace
            consistency.append(len(ids)==1 and np.allclose(trace[ids[0],1:10],np.asarray(states['q_left'])[k],rtol=0,atol=1e-9) and
                np.allclose(trace[ids[0],10:19],np.asarray(states['q_right'])[k],rtol=0,atol=1e-9))
        checks['actual_trace_joint_values_match_all_saved_yaw_frames']=bool(consistency and all(consistency))
        result.update(independent_reference_metrics=dict(minimum_sampled_rigid_clearance_m=rigid_min,maximum_sampled_tcp_error_m=pmax,maximum_sampled_slerp_error_rad=rmax),
            actual_endpoint_position_error_m=pe,actual_endpoint_rotation_error_rad=re)
        result.update(status='passed' if all(checks.values()) and not incomplete else 'failed',passed=all(checks.values()) and not incomplete)
    except (ValueError,KeyError,IndexError,TypeError,OSError,AttributeError) as error:
        result.update(status='failed',passed=False,error=type(error).__name__+': '+str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    for name in ("report.json", "events.json", "states.npz", "rest.npz", "joint_trace.npy",
                 "soft_metrics.npy", "coupling_trace.json"):
        if not (run / name).exists():
            raise SystemExit(f"Run is not fully saved: missing {name}")
    report = json.loads((run / "report.json").read_text())
    config = json.loads((run / "config.json").read_text())
    states, rest = np.load(run / "states.npz"), np.load(run / "rest.npz")
    trace = np.load(run / "joint_trace.npy")
    validator = load_module(run / "source/validate.py", "frozen_postrun_validator")
    audit = validator.audit_arrays(rest, states, config,
        task_completed=bool(report.get("sequence_completed", False)),
        joint_trace=trace, events=report["events"], soft_metrics=np.load(run / "soft_metrics.npy"))
    (run / "independent-audit.json").write_text(json.dumps(audit, indent=2))
    reactions = audit_reactions(run, report, trace)
    (run / "independent-guard-reactions.json").write_text(json.dumps(reactions, indent=2))
    yaw=audit_postlift_yaw(run,report,states,trace,config,reactions)
    if yaw["required"] or yaw["status"]!="not_configured":
        (run / "independent-postlift-yaw-sequence.json").write_text(json.dumps(yaw,indent=2,default=_json_scalar,allow_nan=False))
    staged=config.get("task",{}).get("contact_trial",{}).get("insertion_strategy")=="lower_then_center"
    if staged:
        corrections=audit_staged_sequence(run,report,states,trace,config,reactions)
        (run / "independent-staged-contact-sequence.json").write_text(json.dumps(corrections,indent=2,default=_json_scalar))
    else:
        corrections=audit_corrections(report,states,float(config["simulation"]["dt"]))
        (run / "independent-correction-sequence.json").write_text(json.dumps(corrections,indent=2))
    richer_required="handle_grip_trace" in (run / "source/run.py").read_text()
    richer=None
    if richer_required:
        from audit_richer_contact_traces import audit as audit_richer
        richer=audit_richer(run)
        (run / "independent-richer-contact-review.json").write_text(json.dumps(richer,indent=2))
    input_audit=None
    if (run / "input_sha256.json").exists():
        from audit_input_manifest import audit as audit_inputs
        input_audit=audit_inputs(run)
        (run / "independent-input-manifest-review.json").write_text(json.dumps(input_audit,indent=2))
    source_hashes = {}
    for path in (run / "source").glob("*.py"):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    passed = bool(audit.get("task_success") and reactions["passed"] and corrections["passed"] and
                  report.get("sequence_completed") and not report.get("error"))
    if richer_required:
        passed=bool(passed and richer["held_handle_monitor"]["passed"] and
                    richer["per_link_wrench_reconstruction"]["passed"])
    if input_audit is not None:
        passed=bool(passed and input_audit["all_declared_hashes_match"] and input_audit["all_snapshot_files_in_manifest"])
    passed=bool(passed and yaw["passed"])
    result = {"run": str(run), "independent_task_success": passed,
        "runtime_status": report["status"], "runtime_error": report.get("error"),
        "sequence_completed": bool(report.get("sequence_completed", False)),
        "geometry_and_soft_audit_status": audit["status"],
        "guard_reaction_status": reactions["status"],
        "correction_sequence_status":"not_applicable_staged_trial" if staged else corrections["status"],
        "staged_sequence_status":corrections["status"] if staged else "not_applicable",
        "staged_native_hand_height_status":corrections.get("native_hand_height_audit",{}).get("status","not_entered") if staged else "not_applicable",
        "richer_contact_evidence_required":richer_required,
        "held_handle_monitor_status":richer["held_handle_monitor"]["status"] if richer else "not_recorded_in_this_version",
        "per_link_wrench_reconstruction_status":richer["per_link_wrench_reconstruction"]["status"] if richer else "not_recorded_in_this_version",
        "input_manifest_matches":input_audit["all_declared_hashes_match"] if input_audit else None,
        "failed_geometry_or_soft_checks": [k for k,v in audit.get("checks", {}).items() if not v],
        "failed_guard_checks": [k for k,v in reactions.get("checks", {}).items() if not v],
        "failed_correction_checks":[] if staged else [k for k,v in corrections.get("checks",{}).items() if not v],
        "failed_staged_checks":[k for k,v in corrections.get("checks",{}).items() if not v] if staged else [],
        "final_containment": audit.get("final_containment"),
        "runtime_success_claim_agrees": bool(bool(report.get("requested_task_passed")) == passed),
        "frozen_source_sha256": source_hashes,
        "limits": ["No success follows merely from a completed script or planned geometric candidate.",
            "Every-step summaries are sampled integration states, not a continuous-time nonpenetration certificate.",
            "Full geometry, final bag-bear penetration, and final retention use saved-frame samples.",
            "Material/friction values are engineering assumptions, not physical calibration."]}
    if yaw["required"] or yaw["status"]!="not_configured":
        result.update(postlift_yaw_status=yaw["status"],failed_postlift_yaw_checks=[k for k,v in yaw["checks"].items() if not v])
    (run / "independent-final-review.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
