"""Read-only CPU shape, grip and STC moment decomposition for full08."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parents[1]; RUN=HERE/"runs/two_panel_full_08"
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
motion=module("frozen_torque_motion",RUN/"source/motion.py")
grasp=module("frozen_torque_grasp",RUN/"source/contact_grasp.py")
validate=module("frozen_torque_validate",RUN/"source/validate.py")
c=json.loads((RUN/"config.json").read_text());rest=np.load(RUN/"rest.npz")
trace=np.load(RUN/"joint_trace.npy"); guards=json.loads((RUN/"contact_guard_trace.json").read_text())
planner=motion.Planner(c);tets=rest["bear_tets"]; weights=np.abs(validate.signed_volumes(rest["bear"],tets))*c["bear"]["rho"]
mass=weights.sum();nodal=np.zeros(len(rest["bear"]));np.add.at(nodal,tets.ravel(),np.repeat(weights/4,4))
reference=np.asarray(c["bear"]["position"])+np.asarray(c["task"]["bear_grasp_local"])
material=grasp.tetra_material_point(rest["bear"],tets,reference)
local=np.flatnonzero(np.linalg.norm(rest["bear"]-reference,axis=1)<.032)
names=("link7","left_finger","right_finger")

def pose(qleft,qright):
    for side,q in (("left",qleft),("right",qright)):
        planner.d.qpos[planner.qindices[side]]=q[:7]
        for j in (1,2):planner.d.qpos[planner.m.joint(f"{side}_finger_joint{j}").qposadr[0]]=q[6+j]
    mujoco.mj_forward(planner.m,planner.d)
    hand=planner.m.body("right_hand").id; R=planner.d.xmat[hand].reshape(3,3).copy()
    tcp=planner.d.xpos[hand]+R@motion.TCP
    origins=np.array([planner.d.xpos[planner.m.body("right_"+name).id] for name in names])
    return tcp,R,origins

def moments(points):
    tv=points[tets]; sums=tv.sum(1);com=np.einsum("n,ni->i",weights,tv.mean(1))/mass
    second=(np.einsum("ni,nj->nij",sums,sums)+np.einsum("nki,nkj->nij",tv,tv))/20
    cov=np.einsum("n,nij->ij",weights,second)/mass-np.outer(com,com)
    eigen,axes=np.linalg.eigh(cov);return com,eigen[::-1],axes[:,::-1]

def fitted_rotation(a,b,w):
    w=w/w.sum();a=a-np.sum(a*w[:,None],0);b=b-np.sum(b*w[:,None],0)
    u,_,vt=np.linalg.svd(a.T@(w[:,None]*b));d=np.eye(3);d[-1,-1]=np.linalg.det(vt.T@u.T)
    return vt.T@d@u.T

def wrench(row):
    index=int(np.argmin(abs(trace[:,0]-row["time_s"])))
    if abs(trace[index,0]-row["time_s"])>1e-8 or index==0:raise ValueError("Missing previous-step q")
    previous=trace[index-1];tcp,R,origins=pose(previous[1:10],previous[10:19])
    f=np.asarray(row["right_link_force_world_n"]);tau=np.asarray(row["right_link_torque_at_aim_origin_world_nm"])
    shifted=tau+np.cross(origins-tcp,f)
    return tcp,R,shifted,float(np.linalg.norm(shifted.sum(0)-np.asarray(row["torque_at_aim_tcp_world_nm"]))),float(np.linalg.norm(tcp-np.asarray(row["aim_tcp_world_m"])))

reconstruction=[wrench(row)[3:] for row in guards]
stage_states=[]
for name in ("lift_teddy","transport_teddy","recenter_teddy_xy_1"):
    stage_states.append((name,dict(np.load(RUN/(name+".npz")))))
failed=dict(np.load(RUN/"failed_soft_state.npz"));q=trace[-1]
failed.update(q_left=q[1:10],q_right=q[10:19]);failed["tcp_right"]=pose(q[1:10],q[10:19])[0]
stage_states.append(("failure",failed))
base=stage_states[0][1];basecom,baseeval,baseaxes=moments(base["bear"])
basetcp,baseR,_=pose(base["q_left"],base["q_right"])
baselocal=(base["bear"]-basetcp)@baseR
output=[]
for name,state in stage_states:
    t=float(state["time"]); row=min(guards,key=lambda x:abs(x["time_s"]-t))
    tcp,R,_=pose(state["q_left"],state["q_right"]);com,eigen,axes=moments(state["bear"])
    aim,aimR,parts,error,tcperror=wrench(row)
    gravity=np.array([0.,0.,-mass*9.81]);gtau=np.cross(com-aim,gravity)
    localpoints=(state["bear"]-tcp)@R
    matpos=material.evaluate(state["bear"])
    wholefit=fitted_rotation(baselocal,localpoints,nodal)
    patchfit=fitted_rotation(baselocal[local],localpoints[local],nodal[local])
    output.append({"stage":name,"time_s":t,"actual_tcp_world_m":tcp.tolist(),"aim_tcp_world_m":aim.tolist(),
        "com_world_m":com.tolist(),"com_relative_to_tcp_hand_m":((com-tcp)@R).tolist(),
        "principal_variances_m2":eigen.tolist(),"principal_axes_world_columns":axes.tolist(),
        "long_axis_angle_from_lift_deg":float(np.degrees(np.arccos(np.clip(abs(axes[:,0]@baseaxes[:,0]),-1,1)))),
        "whole_shape_rigid_fit_rotation_relative_to_hand_from_lift_deg":float(np.degrees(Rotation.from_matrix(wholefit).magnitude())),
        "local_grasp_patch_rigid_fit_rotation_relative_to_hand_from_lift_deg":float(np.degrees(Rotation.from_matrix(patchfit).magnitude())),
        "same_material_grasp_point_relative_to_tcp_hand_m":((matpos-tcp)@R).tolist(),
        "same_material_grasp_point_distance_tcp_m":float(np.linalg.norm(matpos-tcp)),
        "finger_positions_m":state["q_right"][-2:].tolist(),
        "minimum_tetra_volume_ratio":float((validate.signed_volumes(state["bear"],tets)/validate.signed_volumes(rest["bear"],tets)).min()),
        "gravity_force_n":gravity.tolist(),"gravity_moment_at_aim_tcp_nm":gtau.tolist(),"gravity_moment_norm_nm":float(np.linalg.norm(gtau)),
        "observed_stc_net_force_n":row["force_world_n"],"observed_stc_tcp_moment_nm":row["torque_at_aim_tcp_world_nm"],
        "observed_moment_minus_gravity_nm":(np.asarray(row["torque_at_aim_tcp_world_nm"])-gtau).tolist(),
        "link_wrench_contributions":{n:{"force_world_n":row["right_link_force_world_n"][i],
            "moment_at_aim_tcp_world_nm":parts[i].tolist(),"moment_norm_nm":float(np.linalg.norm(parts[i]))} for i,n in enumerate(names)},
        "reconstructed_moment_error_nm":error,"reconstructed_aim_tcp_error_m":tcperror})

# Finite-window endpoint acceleration sensitivity: useful for force-balance
# inference, not a native pair/contact-force measurement.
states=np.load(RUN/"states.npz");keep=np.r_[np.diff(states["time"])>1e-10,True]
times=np.r_[states["time"][keep],float(failed["time"])];points=np.concatenate([states["bear"][keep],failed["bear"][None]])
force_balance=[];com,eigen,axes=moments(failed["bear"]);tau=np.asarray(guards[-1]["torque_at_aim_tcp_world_nm"])
aim=np.asarray(guards[-1]["aim_tcp_world_m"]);force=np.asarray(guards[-1]["force_world_n"]);gravity=np.array([0,0,-mass*9.81])
for window in (.15,.25,.35):
    mask=times>=times[-1]-window-1e-9;tt=times[mask]-times[-1]
    coeff=np.linalg.lstsq(np.column_stack([np.ones(len(tt)),tt,.5*tt**2]),points[mask].reshape(len(tt),-1),rcond=None)[0]
    acceleration=coeff[2].reshape(-1,3);a_com=np.einsum("n,ni->i",nodal,acceleration)/mass
    rr=failed["bear"][tets]-com;aa=acceleration[tets]
    dH=np.einsum("n,ni->i",weights,(np.cross(rr.sum(1),aa.sum(1))+np.cross(rr,aa).sum(1))/20)
    inferred_other_force=mass*a_com+force-gravity
    inferred_other_torque=tau-np.cross(com-aim,gravity)+dH+np.cross(com-aim,mass*a_com)
    force_balance.append({"trailing_window_s":window,"samples":int(mask.sum()),"com_acceleration_m_s2":a_com.tolist(),
        "inferred_other_external_force_on_bear_n":inferred_other_force.tolist(),"inferred_other_external_moment_about_tcp_nm":inferred_other_torque.tolist()})
report={"bear_mass_kg":float(mass),"stages":output,"all_guard_wrench_reconstruction":{
    "rows":len(guards),"maximum_moment_vector_error_nm":float(np.max(np.asarray(reconstruction)[:,0])),
    "maximum_aim_tcp_error_m":float(np.max(np.asarray(reconstruction)[:,1])),
    "method":"Use actual joint_trace one integration step before each guard sample; shift link-origin torques to reconstructed aim TCP."},
    "failure_force_balance_sensitivity":force_balance,
    "limitations":["STC reactions include native proxy dynamics/compliance and do not identify contacted materials.",
        "PCA axes and rigid fits summarize a deforming body, not a rigid orientation measurement.",
        "Force-balance estimates use 20Hz saved full geometry and short trailing quadratic fits; they are diagnostic, not exact native contact forces.",
        "0.15Nm is an uncalibrated engineering stop, not a material failure limit."]}
path=RUN/"independent-torque-decomposition.json";path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
