"""Two bounded CPU-only post-lift yaw hypotheses; never executes a robot motion."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motion import Planner, MotionPlan, CubicSpline, minimum_jerk
from contact_insertion import plan_guarded_contact_trial
from contact_robot_geometry import _convex, convex_clearance_certificate, left_collision_solids


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar(v): return v.tolist() if isinstance(v,np.ndarray) else v.item()
def save(p,data): p.write_text(json.dumps(data,indent=2,default=scalar,allow_nan=False)+"\n")


def rotation_left_sweep(planner,state,end_rotation,end_tcp):
    pivot=state["tcp_right"]
    _,start_rotation=planner.fk("right",state["q_right"][:7])
    rotvec=(end_rotation*start_rotation.inv()).as_rotvec()
    angle=float(np.linalg.norm(rotvec)); axis=rotvec/angle
    rel=state["bear"]-pivot
    radius=float(np.linalg.norm(rel-np.outer(rel@axis,axis),axis=1).max())
    segments=max(1,int(np.ceil(angle/np.deg2rad(5.))))
    sagitta=radius*(1-np.cos(angle/segments/2.))
    # Account for the small actual FK endpoint translation as a separate outer
    # perturbation of the ideal fixed-pivot rotation. No continuous FK claim.
    endpoint_tcp_error=float(np.linalg.norm(end_tcp-pivot))
    solids=left_collision_solids(planner,state["q_left"])
    rows=[];minimum=float("inf");passed=True
    for i in range(segments):
        points=[Rotation.from_rotvec(rotvec*f).apply(rel)+pivot for f in (i/segments,(i+1)/segments)]
        hull=_convex(np.vstack(points));worst=None
        for solid in solids:
            cert=convex_clearance_certificate(hull,solid,required_gap_m=.005+sagitta+endpoint_tcp_error)
            lower=float(cert["conservative_gap_lower_bound_m"])-sagitta-endpoint_tcp_error
            if worst is None or lower<worst["lower_bound_m"]:
                worst=dict(left_body=solid["body"],geom_id=solid["geom_id"],lower_bound_m=lower,certificate=cert)
            passed=passed and cert["status"]=="clear"
        minimum=min(minimum,worst["lower_bound_m"])
        rows.append(dict(segment_index=i,worst=worst))
    return dict(status="clear" if passed else "not_certified",physical_success=False,
        required_left_soft_clearance_m=.005,minimum_continuous_frozen_arc_lower_bound_m=minimum,
        segment_count=segments,maximum_axis_normal_vertex_radius_m=radius,
        per_segment_sagitta_bound_m=float(sagitta),endpoint_tcp_numerical_perturbation_bound_m=endpoint_tcp_error,
        rows=rows,scope="Continuous fixed-axis frozen rigid-grasp arc and tetrahedral interior, with endpoint TCP numerical perturbation; not native deformation or continuous joint-FK clearance")


def one_case(number,cfg,cfgpath):
    run=ROOT/f"runs/two_panel_full_{number}"
    out=run/"independent-postlift-yaw15";out.mkdir(exist_ok=True)
    paths=[Path(__file__).resolve(),cfgpath,run/"config.json",run/"rest.npz",run/"lift_teddy.npz",
           run/"guarded_contact_candidate.json",ROOT/"assets/panda_tactile.xml",ROOT/"assets"/cfg["bag"]["npz_file"]]
    paths+=list(ROOT.glob("*.py"))
    before={str(p):sha(p) for p in paths}
    observed_cfg=json.loads((run/"config.json").read_text())
    if any(cfg[k]!=observed_cfg[k] for k in ("bag","bear","table","robots","simulation")):
        raise ValueError("Configuration changes observed geometry/material/robot/simulation")
    state=dict(np.load(run/"lift_teddy.npz",allow_pickle=False));rest=dict(np.load(run/"rest.npz",allow_pickle=False))
    state_backup={k:v.copy() for k,v in state.items()}
    old=json.loads((run/"guarded_contact_candidate.json").read_text())
    original_path="route.rotation_xyzw" if old.get("route") else "report.predicted_targets.rotation_xyzw"
    original=Rotation.from_quat(old["route"]["rotation_xyzw"] if old.get("route") else old["report"]["predicted_targets"]["rotation_xyzw"])
    planner=Planner(cfg);tcp=state["tcp_right"].copy()
    initial_q={side:state["q_"+side][:7].copy() for side in ("left","right")}
    fingers={side:float(state["q_"+side][7:].mean()) for side in initial_q}
    fk_tcp,actual=planner.fk("right",initial_q["right"])
    target=Rotation.from_euler("z",15.,degrees=True)*original
    expected=Slerp([0.,1.],Rotation.from_quat([actual.as_quat(),target.as_quat()]))
    report=dict(kind="offline_postlift_yaw15_preparation",physical_success=False,
        source_run=str(run),source_actual_state=str(run/"lift_teddy.npz"),source_actual_time_s=float(state["time"]),
        original_command_rotation_source=original_path,original_command_rotation_xyzw=original.as_quat().tolist(),
        original_command_vs_actual_fk_rotation_error_rad=float((original*actual.inv()).magnitude()),
        target_world_z_yaw_degrees=15.,target_full_rotation_xyzw=target.as_quat().tolist(),
        initial_measured_tcp_m=tcp.tolist(),initial_fk_to_measured_tcp_error_m=float(np.linalg.norm(fk_tcp-tcp)),
        unchanged_initial_offsets_xy_m=cfg["task"]["contact_trial"]["center_offsets_xy_m"],
        config_source=str(cfgpath),runtime_execution_performed=False,gpu_used=False,
        source_config_or_state_modified=False)
    try:
        paths_plan=planner.plan_cartesian(initial_q,{"right":tcp},rotations={"right":target},samples=45)
        fractions=np.array([k["fraction"] for k in paths_plan])
        curves={side:CubicSpline(fractions,np.array([k["q"].get(side,initial_q[side]) for k in paths_plan]),axis=0) for side in initial_q}
        plan=MotionPlan(curves,6.,fingers,fingers)
        extrema=plan.validate_joint_limits(planner.limits);plan.limit_rate(.35,.5)
        worst=None;minimum=float("inf");tcp_error=0.;angle_error=0.;samples=[];peak_v=0.;peak_a=0.
        for t in np.linspace(0.,plan.duration,401):
            s=plan.sample(float(t));position,orientation=planner.fk("right",s["q"]["right"])
            phase=float(minimum_jerk(t/plan.duration)[0]);checks=planner.clearance(s["q"],s["finger"])
            category,value=min(checks.items(),key=lambda kv:kv[1]["distance"])
            if value["distance"]<minimum:
                minimum=float(value["distance"]);worst=dict(time_s=float(t),fraction=phase,category=category,**value)
            tcp_error=max(tcp_error,float(np.linalg.norm(position-tcp)))
            angle_error=max(angle_error,float((orientation*expected(phase).inv()).magnitude()))
            samples.append(np.r_[t,s["q"]["left"],s["q"]["right"],position,orientation.as_quat()])
        for t in np.linspace(0.,plan.duration,501):
            s=plan.sample(float(t))
            peak_v=max(peak_v,*[float(np.abs(s["qd"][side]).max()) for side in initial_q])
            peak_a=max(peak_a,*[float(np.abs(s["qdd"][side]).max()) for side in initial_q])
        end_q=plan.end_q();end_tcp,end_rotation=planner.fk("right",end_q["right"])
        rigid_transform=end_rotation*actual.inv()
        arc=rotation_left_sweep(planner,state,end_rotation,end_tcp)
        passed=minimum>=.008 and tcp_error<=.0005 and angle_error<=.002 and peak_v<=.35 and peak_a<=.5 and arc["status"]=="clear"
        report["yaw_preflight"]=dict(status="candidate" if passed else "rejected",physical_success=False,
            duration_s=plan.duration,analytic_joint_position_limits_passed=True,analytic_joint_extrema=extrema,
            sample_count=401,minimum_sampled_rigid_clearance_m=minimum,required_sampled_rigid_clearance_m=.008,
            worst_rigid_sample=worst,maximum_sampled_tcp_fixed_point_error_m=tcp_error,
            maximum_sampled_slerp_rotation_error_rad=angle_error,
            command_velocity_limit_rad_s=.35,command_acceleration_limit_rad_s2=.5,
            rate_sample_count=501,sampled_peak_velocity_rad_s=peak_v,sampled_peak_acceleration_rad_s2=peak_a,
            rotation_left_soft_sweep=arc,continuous_robot_rigid_clearance_certified=False)
        np.savez(out/"yaw_path_samples.npz",samples=np.array(samples))
        if passed:
            dt=float(cfg["simulation"]["dt"]);steps=int(np.ceil(plan.duration/dt))
            hypothetical={k:state[k].copy() for k in ("bag","bear","q_left","q_right","tcp_left","tcp_right","time")}
            hypothetical["q_right"][:7]=end_q["right"]
            hypothetical["q_left"][:7]=end_q["left"]
            hypothetical["tcp_right"]=end_tcp.copy()
            hypothetical["tcp_left"]=planner.fk("left",end_q["left"])[0]
            hypothetical["bear"]=rigid_transform.apply(state["bear"]-tcp)+end_tcp
            hypothetical["time"]=np.array(float(state["time"])+steps*dt)
            np.savez(out/"hypothetical_after_yaw.npz",**hypothetical)
            report["hypothetical_construction"]=dict(physical_success=False,
                path=str(out/"hypothetical_after_yaw.npz"),sha256=sha(out/"hypothetical_after_yaw.npz"),
                input_is_actual=False,interpretation="Explicit CPU hypothesis, never a simulator checkpoint",
                rotation_formula="bear_end = FK_R_end * inverse(FK_R_start) * (actual_bear - measured_TCP_start) + FK_TCP_end",
                actual_start_fk_rotation_xyzw=actual.as_quat().tolist(),end_fk_rotation_xyzw=end_rotation.as_quat().tolist(),
                rigid_grasp_relative_rotation_xyzw=rigid_transform.as_quat().tolist(),
                end_fk_to_target_rotation_error_rad=float((end_rotation*target.inv()).magnitude()),
                right_q_end=end_q["right"].tolist(),left_q_end=end_q["left"].tolist(),
                hypothetical_time_s=float(hypothetical["time"]),clock_increment_steps=steps,clock_dt_s=dt,
                rounded_duration_s=steps*dt,maximum_bear_vertex_displacement_m=float(np.linalg.norm(hypothetical["bear"]-state["bear"],axis=1).max()),
                tcp_endpoint_displacement_m=float(np.linalg.norm(end_tcp-tcp)),
                original_bag_unchanged=bool(np.array_equal(hypothetical["bag"],state["bag"])),
                left_q_unchanged_to_1e12=bool(np.allclose(hypothetical["q_left"],state["q_left"],rtol=0,atol=1e-12)),
                no_measured_forces_carried_into_hypothesis=True)
            print(f"full{number}: yaw passed, running unchanged 12-offset preparation",flush=True)
            preparation=plan_guarded_contact_trial(rest,hypothetical,cfg)
            save(out/"preparation.json",preparation)
            selected=preparation.get("report",{}).get("selected_sections",{})
            report["preparation"]=dict(status=preparation["status"],physical_success=False,
                output=str(out/"preparation.json"),sha256=sha(out/"preparation.json"),
                rejections=preparation.get("report",{}).get("rejections",[]),
                selected_sections=selected,predicted_targets=preparation.get("report",{}).get("predicted_targets"),
                stages_to_execute_on_hypothesis_only=preparation.get("stages_to_execute",[]),
                no_real_native_state_executed=True)
            if preparation["status"]=="candidate":
                from insertion_candidates import polygon_centroid
                from contact_insertion import _mouth
                polygon,_,_=_mouth(rest,hypothetical,cfg)
                offset=np.asarray(selected["center_of_mass_xy_target_m"])-polygon_centroid(polygon)
                report["preparation"]["selected_initial_com_offset_xy_m"]=offset.tolist()
                outside=np.asarray(preparation["route"]["outside_tcp_m"])
                stages=preparation["stages_to_execute"]
                pts=np.vstack([end_tcp,*[np.asarray(s["right_tcp_m"]) for s in stages]])
                report["preparation"].update(outside_tcp_total_translation_m=(outside-end_tcp).tolist(),
                    outside_tcp_endpoint_distance_m=float(np.linalg.norm(outside-end_tcp)),
                    declared_transport_tcp_polyline_length_m=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()))
    except Exception as error:
        report["evaluation_exception"]=dict(type=type(error).__name__,message=str(error))
    report["input_sha256"]=before
    report["all_input_hashes_unchanged"]=before=={str(p):sha(p) for p in paths}
    report["all_actual_state_arrays_unchanged"]=all(np.array_equal(state[k],state_backup[k]) for k in state)
    report["limitations"]=["Only these two actual lift observations are used, once each. No entry/body geometry is reused.",
        "The yaw endpoint and all subsequent geometry are frozen rigid-grasp hypotheses. Native cloth, grasp, force and deformation evolution is unknown.",
        "Analytic joint position extrema apply to the continuous spline. Robot clearances and Cartesian/Slerp errors are sampled only; no continuous robot clearance certificate.",
        "Preparation uses the original 12 offsets and unchanged gates, and its tentative final insert/release/withdraw are planning evidence only.",
        "No downstream entry, adaptive centering, real insertion, release, settling or task success follows from preparation feasibility."]
    save(out/"review.json",report)
    print(f"full{number}: "+json.dumps({k:report.get(k) for k in ("evaluation_exception","all_input_hashes_unchanged","all_actual_state_arrays_unchanged")})+" prep="+report.get("preparation",{}).get("status","not_entered"),flush=True)


if __name__=="__main__":
    path=ROOT/"config_guarded_adaptive_alignment_trial.json"
    config=json.loads(path.read_text())
    for case in (17,18): one_case(case,config,path)
