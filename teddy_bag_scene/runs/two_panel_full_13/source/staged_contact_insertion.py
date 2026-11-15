"""Observed, finite-force contact trial in three independently gated stages.

CPU planning only. Frozen cloth intersections are contact-risk diagnostics, not
permission to alter any simulation node. Native contact/CCD, grasp, wrench,
strain, table and final containment checks must remain active in the runner.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from contact_insertion import (_emit, _policy as base_policy, _state_context,
    _mouth, _section_envelopes, _evaluate_offset)
from contact_robot_geometry import check_translation_sweep, left_collision_solids
from contact_slice_geometry import convex_hull_metrics
from insertion_candidates import projected_inside
from motion import minimum_jerk

PHASE_NAMES = {"entry":"lower_teddy_to_rim", "center":"center_teddy_at_rim", "insert":"insert_teddy"}


def staged_policy(config):
    settings=config.get("task",{}).get("contact_trial",{})
    if settings.get("insertion_strategy")!="lower_then_center":
        raise ValueError("Staged trial requires explicit insertion_strategy=lower_then_center")
    s=settings.get("staged",{})
    gap=float(s.get("hand_clearance_m",.01))
    ratio=float(s.get("maximum_rim_stretch_ratio",1.01))
    lateral=float(s.get("entry_max_lateral_excursion_fraction_of_mouth_width",1.))
    overlap=float(s.get("entry_minimum_section_envelope_overlap_fraction",0.))
    delta=np.asarray(s.get("center_offset_xy_m",[-.027,-.010]),float)
    if not np.isfinite([gap,ratio,lateral,overlap]).all() or not .01<=gap<=.03 \
            or not 1.<=ratio<=1.01 or not 0.<=lateral<=1. or not 0.<=overlap<=1.:
        raise ValueError("Invalid bounded staged contact engineering policy")
    if delta.shape!=(2,) or not np.isfinite(delta).all() or not 1e-6<=np.linalg.norm(delta)<=.03:
        raise ValueError("Staged centering must declare a finite nonzero XY displacement at most 30 mm")
    return dict(planned_hand_clearance_m=gap,observed_required_hand_clearance_m=.005,
        center_offset_xy_m=delta.tolist(),maximum_rim_stretch_ratio=ratio,
        entry_max_lateral_excursion_fraction_of_mouth_width=lateral,
        entry_minimum_section_envelope_overlap_fraction=overlap,
        maximum_individual_xy_m=.03,maximum_cumulative_xy_m=.05,
        maximum_original_rotation_error_rad=.015,maximum_previous_endpoint_error_m=.002,
        rigid_clearance_m=.008,left_robot_sweep_clearance_m=.005,
        maximum_joint_velocity_rad_s=.35,maximum_joint_acceleration_rad_s2=.5,
        minimum_downward_duration_s=6.,minimum_center_duration_s=4.,
        each_stage_requires_strictly_new_actual_observation=True,
        wall_contact_is_risk_not_success=True,physical_limits_unchanged=True)


def left_hand_minimum_z(q_left,config,*,planner=None):
    """Measured-q collision geometry only; native affine state is a runner check."""
    if planner is None:
        from motion import Planner
        planner=Planner(config)
    solids=left_collision_solids(planner,np.asarray(q_left,float))
    selected=[s for s in solids if s["body"] in
              ("left_hand","left_left_finger","left_right_finger")]
    if {s["body"] for s in selected}!={"left_hand","left_left_finger","left_right_finger"}:
        raise ValueError("Missing left palm or finger collision geometry")
    return float(min(s["vertices"][:,2].min() for s in selected))


def _validated_checkpoint(previous,expected):
    checkpoint=previous["checkpoint"]
    start=np.asarray(checkpoint["observed_tcp_world_m"],float)
    target=np.asarray(checkpoint["stage_target_tcp_world_m"],float)
    stage_target=np.asarray(previous["stage"]["right_tcp_m"],float)
    origin=np.asarray(checkpoint["original_transport_tcp_xy_m"],float)
    values=np.asarray([checkpoint["cumulative_xy_m_before"],checkpoint["planned_xy_distance_m"],
                       checkpoint["planned_cumulative_xy_m_after"]],float)
    reported=np.asarray([previous["cumulative_xy_m_before"],previous["cumulative_xy_m_after"]],float)
    if start.shape!=(3,) or target.shape!=(3,) or stage_target.shape!=(3,) or origin.shape!=(2,) \
            or not all(np.isfinite(v).all() for v in (start,target,stage_target,origin,values,reported)) \
            or np.any(values<0.) or values[0]>.05 or values[1]>.03 or values[2]>.05:
        raise ValueError("Invalid prior staged position/budget record")
    if np.linalg.norm(target-stage_target)>1e-10 or abs(np.linalg.norm(target[:2]-start[:2])-values[1])>1e-9 \
            or abs(values[0]+values[1]-values[2])>1e-9 \
            or abs(float(previous["cumulative_xy_m_before"])-values[0])>1e-9 \
            or abs(float(previous["cumulative_xy_m_after"])-values[2])>1e-9:
        raise ValueError("Prior staged targets and cumulative budget disagree")
    times=np.asarray(checkpoint["actual_stage_times_s"],float)
    if times.shape!=((1,) if expected=="entry" else (2,)) or not np.isfinite(times).all() \
            or np.any(np.diff(times)<=0) or times[-1]!=float(previous["observed_time_s"]):
        raise ValueError("Damaged staged observation chronology")
    if expected=="entry" and values[0]!=0.:
        raise ValueError("First staged entry cannot inherit or reset a prior correction budget")
    return checkpoint


def _rim_envelope_review(bear,tets,start_displacement,end_displacement,mouth,polygon,triangles,
                         config_policy,policy,phase):
    """Covers every actual rim height and every point of the stated translation."""
    start_displacement=np.asarray(start_displacement,float)
    end_displacement=np.asarray(end_displacement,float)
    levels=np.asarray([mouth["rim_z_min_m"],mouth["rim_z_max_m"],mouth["available_nonstretch_rim_perimeter_m"]],float)
    if phase not in PHASE_NAMES or start_displacement.shape!=(3,) or end_displacement.shape!=(3,) \
            or not all(np.isfinite(v).all() for v in (start_displacement,end_displacement,levels)) \
            or levels[0]>levels[1] or levels[2]<=0.:
        raise ValueError("Invalid phase, finite translation or actual rim bounds for section review")
    _,envs=_section_envelopes(bear,tets)
    lo=mouth["rim_z_min_m"]-max(start_displacement[2],end_displacement[2])
    hi=mouth["rim_z_max_m"]-min(start_displacement[2],end_displacement[2])
    relevant=[e for e in envs if e["z_max"]>=lo and e["z_min"]<=hi]
    if not relevant: raise ValueError("Declared entry path does not cross actual body-rim heights")
    envelopes=[]
    for e in relevant:
        hull=convex_hull_metrics(np.vstack((e["hull_xy"]+start_displacement[:2],
                                          e["hull_xy"]+end_displacement[:2])))
        envelopes.append(dict(z_min=e["z_min"],z_max=e["z_max"],**hull))
    trial=dict(config_policy)
    if phase in ("entry","center"):
        trial["lateral_excursion_fraction_of_mouth_width"]=policy["entry_max_lateral_excursion_fraction_of_mouth_width"]
        trial["minimum_section_envelope_overlap_fraction"]=policy["entry_minimum_section_envelope_overlap_fraction"]
    available=mouth["available_nonstretch_rim_perimeter_m"]
    trial_mouth=dict(mouth)
    trial_mouth["available_nonstretch_rim_perimeter_m"]=available*policy["maximum_rim_stretch_ratio"]
    evaluation=_evaluate_offset(envelopes,np.zeros(2),polygon,triangles,trial_mouth,trial)
    evaluation.update(actual_unextended_available_rim_perimeter_m=available,
        explicitly_allowed_rim_perimeter_m=trial_mouth["available_nonstretch_rim_perimeter_m"],
        declared_maximum_rim_stretch_ratio=policy["maximum_rim_stretch_ratio"],
        required_perimeter_ratio_to_unextended_rim=evaluation["maximum_required_enclosing_rim_perimeter_m"]/available,
        actual_rim_height_range_m=[mouth["rim_z_min_m"],mouth["rim_z_max_m"]],
        relevant_observed_teddy_height_range_m=[lo,hi],relevant_slab_count=len(envelopes),
        envelope_covers_entire_straight_segment=True,
        does_not_certify_deformable_3d_passage=True,
        wall_contact_risk="Initial entry and lateral centering may contact upper side panels, not only the free rim. Native CCD/contact must prevent actual penetration; no geometry is imposed.")
    return evaluation


def _robot_segment(context,state,config,target,phase,original_rotation):
    planner=context["planner"];tcp=np.asarray(state["tcp_right"],float)
    rotation=original_rotation;fingers=context["fingers"]
    duration=4. if phase=="center" else 6.
    plan=planner.trajectory(context["qs"],{"right":target},rotations={"right":rotation},
        duration=duration,samples=45,validate_samples=101,finger_start=fingers,finger_end=fingers,
        clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5)
    errors=[];minimum=float("inf")
    for time in np.linspace(0.,plan.duration,201):
        sample=plan.sample(float(time));position,r=planner.fk("right",sample["q"]["right"])
        u=float(minimum_jerk(time/plan.duration)[0])
        errors.append((float(np.linalg.norm(position-(tcp+u*(target-tcp)))),float((r*rotation.inv()).magnitude())))
        minimum=min(minimum,min(v["distance"] for v in planner.clearance(sample["q"],fingers).values()))
    error=np.max(errors,axis=0)
    if error[0]>.0005 or error[1]>.002:
        raise ValueError("Sampled FK deviates from the declared straight frozen-shape segment")
    evidence=dict(analytic_joint_extrema=plan.joint_extrema(),minimum_sampled_rigid_clearance_m=minimum,
        rigid_clearance_sample_count=201,maximum_sampled_tcp_position_error_m=float(error[0]),
        maximum_sampled_rotation_error_rad=float(error[1]),continuous_fk_error_bound=False,
        measured_finger_half_openings_m=fingers)
    stage=dict(name=PHASE_NAMES[phase],right_tcp_m=target.tolist(),rotation_xyzw=rotation.as_quat().tolist(),
        duration_s=plan.duration,right_q_end=plan.end_q()["right"].tolist())
    return stage,evidence,plan


def plan_staged_contact(rest,state,config,*,phase="entry",previous=None,output=None):
    """Return only one executable stage, requiring new physical state afterwards.

    entry.previous is the accepted original guarded contact candidate;
    center.previous and insert.previous are the preceding staged result.
    No result grants a simulation reset, attachment or subsequent stage.
    """
    result=dict(kind="staged_guarded_contact_trial",status="rejected",phase=phase,stage=None,
        physical_success=False,runtime_execution_performed=False,outside_tcp_m=None,
        requires_final_actual_3d_containment_audit=True)
    report=dict(rejections=[],physical_success=False)
    result["report"]=report
    try:
        if phase not in PHASE_NAMES: raise ValueError("Staged phase must be entry, center or insert")
        policy=staged_policy(config);base=base_policy(config)
        report.update(staged_policy=policy,base_contact_policy=base,
            actual_contact_force_torque_unavailable_in_saved_state=True,
            all_runtime_native_contact_grasp_wrench_strain_table_gates_required=True,
            final_containment_criteria_unchanged=True)
        context=_state_context(rest,state,config,base)
        report["physical_state"]=context["physical"]
        if context["reasons"]: raise ValueError("; ".join(context["reasons"]))
        polygon,triangles,mouth=_mouth(rest,state,config);report["mouth"]=mouth
        tcp=np.asarray(state["tcp_right"],float);bear=np.asarray(state["bear"],float);tets=np.asarray(rest["bear_tets"],int)
        now=context["time"];rotation=context["rotation"]
        result["observed_time_s"]=now
        if not isinstance(previous,dict) or previous.get("status")!="candidate":
            raise ValueError("Every staged phase needs its preceding accepted candidate")
        old=float(previous["observed_time_s"])
        if not np.isfinite(old) or now<=old: raise ValueError("Each staged phase needs a strictly newer actual observation")
        if phase=="entry":
            if previous.get("kind")!="guarded_contact_trial" or previous.get("route") is None:
                raise ValueError("Entry requires the original accepted guarded contact preparation")
            if previous.get("report",{}).get("policy")!=base:
                raise ValueError("Do not change base contact/physical policy after original preparation")
            if previous.get("is_bounded_recenter",False):
                raise ValueError("Staged entry must precede bounded corrections; it cannot reset their budget")
            original=np.asarray(previous["route"]["rotation_xyzw"],float)
            checkpoint=dict(base_policy=base,staged_policy=policy,original_transport_tcp_xy_m=tcp[:2].tolist(),
                original_rotation_xyzw=original.tolist(),actual_stage_times_s=[now],
                cumulative_xy_m_before=0.,cumulative_xy_m_after_observation=0.)
            above=float(bear[:,2].min()-mouth["rim_z_max_m"])
            report["initial_whole_bear_clearance_above_highest_rim_m"]=above
            if above<.04: raise ValueError("Initial entry must start with whole teddy at least 40 mm above actual highest body rim")
        else:
            expected="entry" if phase=="center" else "center"
            if previous.get("kind")!="staged_guarded_contact_trial" or previous.get("phase")!=expected:
                raise ValueError("Staged phase order must be entry then center then insert")
            old_check=_validated_checkpoint(previous,expected)
            if old_check["base_policy"]!=base or old_check["staged_policy"]!=policy:
                raise ValueError("Do not change staged or physical policy between observations")
            original=np.asarray(old_check["original_rotation_xyzw"],float)
            old_target=np.asarray(previous["stage"]["right_tcp_m"],float)
            error=float(np.linalg.norm(tcp-old_target));report["previous_endpoint_error_m"]=error
            if not np.isfinite(error) or error>.002: raise ValueError("Previous physical stage has not reached its complete TCP target within 2 mm")
            previous_start=np.asarray(old_check["observed_tcp_world_m"],float)
            observed_xy=float(np.linalg.norm(tcp[:2]-previous_start[:2]))
            old_total=float(old_check["cumulative_xy_m_before"])
            if not np.isfinite([observed_xy,old_total]).all() or old_total<0. or observed_xy>.03 or old_total+observed_xy>.05:
                raise ValueError("Observed staged XY displacement exceeded 30 mm individual or 50 mm cumulative budget")
            times=np.asarray(old_check["actual_stage_times_s"],float)
            if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[-1]!=old:
                raise ValueError("Damaged staged observation chronology")
            checkpoint=dict(base_policy=base,staged_policy=policy,
                original_transport_tcp_xy_m=old_check["original_transport_tcp_xy_m"],
                original_rotation_xyzw=original.tolist(),actual_stage_times_s=times.tolist()+[now],
                cumulative_xy_m_before=old_total+observed_xy,
                cumulative_xy_m_after_observation=old_total+observed_xy)
            report["observed_previous_stage_xy_displacement_m"]=observed_xy
        if original.shape!=(4,) or not np.isfinite(original).all() or abs(np.linalg.norm(original)-1.)>1e-8:
            raise ValueError("Invalid original full orientation")
        angle=float((rotation*Rotation.from_quat(original).inv()).magnitude())
        report["original_rotation_error_rad"]=angle
        if angle>.015: raise ValueError("Actual hand differs from original full orientation by over 0.015 rad")
        result["original_rotation_xyzw"]=original.tolist()
        hand_z=left_hand_minimum_z(state["q_left"],config,planner=context["planner"])
        hand_clearance=hand_z-float(bear[:,2].max())
        report.update(actual_left_hand_minimum_z_m=hand_z,actual_bear_top_z_m=float(bear[:,2].max()),
            observed_bear_top_to_left_hand_clearance_m=hand_clearance,
            planned_hand_clearance_m=policy["planned_hand_clearance_m"],
            observed_required_hand_clearance_m=policy["observed_required_hand_clearance_m"])
        target=tcp.copy()
        if phase=="entry":
            target[2]+=hand_clearance-policy["planned_hand_clearance_m"]
            if target[2]>=tcp[2]: raise ValueError("Entry target must be a finite downward move from the current observation")
        elif phase=="center":
            if hand_clearance<.005: raise ValueError("Actual entire teddy is not at least 5 mm below measured left palm and fingers")
            target[:2]+=np.asarray(policy["center_offset_xy_m"])
        else:
            if hand_clearance<.005: raise ValueError("Actual entire teddy lost its required 5 mm hand clearance before insertion")
            target[2]=float(config["table"]["height"])+base["bottom_clearance_m"]-float((bear-tcp)[:,2].min())
            if target[2]>=tcp[2]: raise ValueError("Final insertion must descend from its actual centered state")
        delta=target-tcp
        distance=float(np.linalg.norm(delta[:2]));total=checkpoint["cumulative_xy_m_before"]+distance
        if distance>.03 or total>.05: raise ValueError("Planned staged XY displacement exceeds unchanged 30/50 mm budgets")
        endpoint_bear=bear+delta
        report["target_bear_bounds_world_m"]=[endpoint_bear.min(0).tolist(),endpoint_bear.max(0).tolist()]
        if endpoint_bear[:,2].min()<float(config["table"]["height"])+base["bottom_clearance_m"]-1e-10:
            raise ValueError("Stage would lower teddy below planned table clearance")
        if phase in ("center","insert"):
            com=context["com"]+delta
            inside,dist=projected_inside(com[None,:2],polygon)
            report.update(planned_center_of_mass_world_m=com.tolist(),planned_com_inside=bool(inside[0]),
                planned_com_boundary_distance_m=float(dist[0]))
            if not inside[0] or dist[0]<.002: raise ValueError("Planned COM must remain inside actual opening with 2 mm margin")
        if phase=="insert" and endpoint_bear[:,2].max()>mouth["rim_z_min_m"]-.002:
            raise ValueError("Whole teddy endpoint does not fit below actual lowest rim with 2 mm clearance")
        sections=_rim_envelope_review(bear,tets,np.zeros(3),delta,mouth,polygon,triangles,base,policy,phase)
        report["rim_height_section_envelopes"]=sections
        if sections["section_guard_rejections"]: raise ValueError("; ".join(sections["section_guard_rejections"]))
        sweep=check_translation_sweep(bear,tets,state["q_left"],config,displacement_m=delta,
            required_gap_m=.005,planner=context["planner"])
        report["continuous_frozen_teddy_left_robot_sweep"]=sweep
        if sweep["status"]!="clear": raise ValueError("Continuous frozen teddy segment lacks the 5 mm left-robot certificate")
        command_rotation=Rotation.from_quat(original)
        stage,rigid,plan=_robot_segment(context,state,config,target,phase,command_rotation);report["robot_trajectory"]=rigid
        outside=target.copy() if phase=="center" else tcp.copy()
        if phase=="insert":
            outside=np.asarray(previous["stage"]["right_tcp_m"],float).copy();outside[:2]=tcp[:2]
            qs=plan.end_q();fingers=context["fingers"];opened=dict(fingers);opened["right"]=.04
            release_min=float("inf")
            for fraction in np.linspace(0.,1.,11):
                ff=dict(fingers);ff["right"]=(1-fraction)*ff["right"]+fraction*.04
                release_min=min(release_min,min(v["distance"] for v in context["planner"].clearance(qs,ff).values()))
            if release_min<.008: raise ValueError("Release opening fails 8 mm rigid clearance")
            withdraw=context["planner"].trajectory(qs,{"right":outside},rotations={"right":command_rotation},
                duration=3.,samples=45,validate_samples=101,finger_start=opened,finger_end=opened,
                clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5)
            report["release_withdraw_preflight"]=dict(minimum_release_opening_clearance_m=release_min,
                release_opening_samples=11,withdraw_rigid_samples=101,
                withdraw_duration_s=withdraw.duration,withdraw_analytic_joint_extrema=withdraw.joint_extrema(),
                right_finger_half_opening_m=.04,not_executable_before_actual_release=True)
        checkpoint.update(observed_tcp_world_m=tcp.tolist(),stage_target_tcp_world_m=target.tolist(),
            planned_xy_distance_m=distance,planned_cumulative_xy_m_after=total)
        result.update(status="candidate",stage=stage,outside_tcp_m=outside.tolist(),checkpoint=checkpoint,
            correction_distance_m=distance,cumulative_xy_m_before=checkpoint["cumulative_xy_m_before"],
            cumulative_xy_m_after=total,requires_new_observation_before_next_stage=True,
            next_phase={"entry":"center","center":"insert","insert":"actual_release_and_containment_audit"}[phase])
    except (ValueError,KeyError,IndexError,OSError,TypeError) as exc:
        report["rejections"].append(str(exc))
    return _emit(result,output)
