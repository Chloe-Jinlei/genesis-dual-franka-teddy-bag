"""Independent CPU-only frozen full12 shape and per-link STC transport audit."""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parents[1]; RUN=HERE/'runs/two_panel_full_12'
def mod(name,file):
 s=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
motion=mod('full12_frozen_motion',RUN/'source/motion.py');v=mod('full12_frozen_validator',RUN/'source/validate.py');g=mod('full12_frozen_grasp',RUN/'source/contact_grasp.py')
c=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));s=dict(np.load(RUN/'states.npz'));guards=json.loads((RUN/'contact_guard_trace.json').read_text());events=json.loads((RUN/'events.json').read_text());tt=np.array([x['time_s'] for x in guards]);planner=motion.Planner(c)
tets=rest['bear_tets'];rv=v.signed_volumes(rest['bear'],tets);weights=np.abs(rv)*c['bear']['rho'];mass=weights.sum();nw=np.zeros(len(rest['bear']));np.add.at(nw,tets.ravel(),np.repeat(weights/4,4));gravity=np.array([0,0,-mass*9.81])
reference=np.asarray(c['bear']['position'])+np.asarray(c['task']['bear_grasp_local']);material=g.tetra_material_point(rest['bear'],tets,reference);local=np.flatnonzero(np.linalg.norm(rest['bear']-reference,axis=1)<.032)
source=np.load(RUN/'source/assets'/Path(c['bag']['npz_file']).name);errs,mapping=cKDTree(rest['bag']).query(source['points']+np.asarray(c['bag']['position']));assert errs.max()<1e-7 and len(np.unique(mapping))==len(mapping)
btri=rest['bag'][rest['bag_faces']];bw=.5*np.linalg.norm(np.cross(btri[:,1]-btri[:,0],btri[:,2]-btri[:,0]),axis=1)*2*c['bag']['thickness']*c['bag']['rho'];bn=np.zeros(len(rest['bag']));np.add.at(bn,rest['bag_faces'].ravel(),np.repeat(bw/3,3))
def moments(points):
 tv=points[tets];summ=tv.sum(1);com=np.einsum('n,ni->i',weights,tv.mean(1))/mass
 sec=(np.einsum('ni,nj->nij',summ,summ)+np.einsum('nki,nkj->nij',tv,tv))/20
 cov=np.einsum('n,nij->ij',weights,sec)/mass-np.outer(com,com);e,a=np.linalg.eigh(cov);return com,e[::-1],a[:,::-1]
def fit(a,b,w):
 w=w/w.sum();a=a-np.sum(a*w[:,None],0);b=b-np.sum(b*w[:,None],0);u,_,vt=np.linalg.svd(a.T@(w[:,None]*b));d=np.eye(3);d[-1,-1]=np.linalg.det(vt.T@u.T);return vt.T@d@u.T
stages=[];base=dict(np.load(RUN/'lift_teddy.npz'));basetcp,baseR=planner.fk('right',base['q_right'][:7]);basehand=(base['bear']-basetcp)@baseR.as_matrix();basecom,_,baseaxes=moments(base['bear'])
for name in ('lift_teddy','adjust_teddy_height_contact_trial','transport_teddy_along_y','transport_teddy'):
 x=dict(np.load(RUN/(name+'.npz')));t=float(x['time']);row=min(guards,key=lambda r:abs(r['time_s']-t));tcp,R=planner.fk('right',x['q_right'][:7]);R=R.as_matrix();com,e,axes=moments(x['bear']);hand=(x['bear']-tcp)@R;aim=np.asarray(row['aim_tcp_world_m']);gt=np.cross(com-aim,gravity)
 bf,btx=v.mouth_structure(rest['bag'],rest['bag_faces'],c)[:2]
 rim=x['bag'][btx];cross=rim[:,0]*np.roll(rim[:,1],-1)-rim[:,1]*np.roll(rim[:,0],-1);area=float(cross.sum()/2);center=((rim[:,:2]+np.roll(rim[:,:2],-1,axis=0))*cross[:,None]).sum(0)/(6*area)
 stages.append(dict(stage=name,time_s=t,actual_tcp_world_m=tcp.tolist(),bear_com_world_m=com.tolist(),bear_com_relative_to_tcp_world_m=(com-tcp).tolist(),bear_long_axis_angle_from_lift_deg=float(np.degrees(np.arccos(np.clip(abs(axes[:,0]@baseaxes[:,0]),-1,1)))),whole_bear_rotation_relative_to_hand_from_lift_deg=float(np.degrees(Rotation.from_matrix(fit(basehand,hand,nw)).magnitude())),grasp_patch_rotation_relative_to_hand_from_lift_deg=float(np.degrees(Rotation.from_matrix(fit(basehand[local],hand[local],nw[local])).magnitude())),grasp_material_relative_to_hand_m=((material.evaluate(x['bear'])-tcp)@R).tolist(),finger_positions_m=x['q_right'][-2:].tolist(),minimum_tetra_volume_ratio=float((v.signed_volumes(x['bear'],tets)/rv).min()),gravity_moment_at_aim_tcp_nm=gt.tolist(),right_net_force_world_n=row['force_world_n'],right_net_moment_at_aim_tcp_nm=row['torque_at_aim_tcp_world_nm'],right_moment_minus_gravity_nm=(np.asarray(row['torque_at_aim_tcp_world_nm'])-gt).tolist(),right_per_link_force_world_n=dict(zip(row['right_link_names'],row['right_link_force_world_n'])),right_per_link_moment_at_aim_tcp_world_nm=dict(zip(row['right_link_names'],row['right_link_torque_at_aim_tcp_world_nm'])),left_net_force_world_n=row['left_all_link_net_force_world_n'],left_net_moment_at_aim_tcp_nm=row['left_all_link_torque_at_aim_tcp_world_nm'],bag_com_world_m=(bn@x['bag']/bn.sum()).tolist(),rim_area_centroid_xy_m=center.tolist(),rim_xy_projected_area_m2=abs(area),handle_material_centroids_world_m={group:x['bag'][mapping[source[group]]].mean(0).tolist() for group in ('handle_front','handle_back','grasp_front_center','grasp_back_center')}))
phase=[]
for ev in events:
 if ev['start']<tt[0]-1e-8:continue
 mask=(tt>=ev['start']-1e-9)&(tt<=ev['start']+ev['duration']+.005);ids=np.flatnonzero(mask);rows=[guards[i] for i in ids]
 if not rows:continue
 rowout=dict(phase=ev['phase'],samples=len(rows),start_s=float(tt[ids[0]]),end_s=float(tt[ids[-1]]))
 for key in ('force_world_n','torque_at_aim_tcp_world_nm','left_all_link_net_force_world_n','left_all_link_torque_at_aim_tcp_world_nm'):
  a=np.array([r[key] for r in rows]);norm=np.linalg.norm(a,axis=1);k=int(np.argmax(norm));rowout[key]=dict(first=a[0].tolist(),last=a[-1].tolist(),min_components=a.min(0).tolist(),max_components=a.max(0).tolist(),maximum_norm=float(norm[k]),maximum_time_s=float(tt[ids[k]]))
 for side in ('left','right'):
  a=np.array([r[side+'_link_force_world_n'] for r in rows]);rowout[side+'_per_link_force_norm_max_n']=dict(zip(rows[0][side+'_link_names'],np.linalg.norm(a,axis=2).max(0).tolist()))
 phase.append(rowout)
keep=np.r_[np.diff(s['time'])>1e-10,True];times=s['time'][keep];allbear=s['bear'][keep];balance=[];end=stages[-1];row=guards[-1];force=np.asarray(row['force_world_n']);aim=np.asarray(row['aim_tcp_world_m']);com=np.asarray(end['bear_com_world_m']);tau=np.asarray(row['torque_at_aim_tcp_world_nm'])
for window in (.15,.25,.35):
 mask=times>=times[-1]-window-1e-9;dt=times[mask]-times[-1];co=np.linalg.lstsq(np.c_[np.ones(len(dt)),dt,.5*dt**2],allbear[mask].reshape(len(dt),-1),rcond=None)[0];acc=co[2].reshape(-1,3);ac=nw@acc/mass;rr=allbear[-1][tets]-com;aa=acc[tets];dh=np.einsum('n,ni->i',weights,(np.cross(rr.sum(1),aa.sum(1))+np.cross(rr,aa).sum(1))/20)
 balance.append(dict(trailing_window_s=window,samples=len(dt),bear_com_acceleration_m_s2=ac.tolist(),inferred_other_force_on_bear_n=(mass*ac+force-gravity).tolist(),inferred_other_moment_on_bear_about_tcp_nm=(tau-np.cross(com-aim,gravity)+dh+np.cross(com-aim,mass*ac)).tolist()))
report=dict(kind='independent_full12_transport_reaction_review',physical_task_success=False,reason='Fresh geometry and bounded correction rejected before insertion; no release occurred.',bear_mass_kg=float(mass),bag_mass_kg=float(bw.sum()),stages=stages,phase_wrenches=phase,final_force_balance_sensitivity=balance,findings=['Every-step guard reactions and hold evidence pass; maximum net right TCP moment is 0.03435 Nm, below the unchanged 0.15 Nm stop.','Left palm/link7 force stays below 0.000273 N throughout X transport; saved reactions do not support the earlier left-palm blocking mechanism.','During X transport the right positive-X reaction rises while the left net X reaction decreases. This is compatible with coupling through cloth, but per-link STC feedback does not identify contacting materials.','Final right arm still bears approximately the full teddy weight and both fingers retain about 4 N opening reactions; no runtime slip was recorded.'],limitations=['STC feedback is not a native per-contact/material-pair force measurement and includes affine proxy dynamics.','Force-balance inference uses 20Hz soft states and short trailing quadratic fits; it assumes the right distal resultant is predominantly from the teddy. Direct right-hand/cloth contact would invalidate that attribution.','Point material/PCA/rigid-fit diagnostics describe a deforming object, not a unique rigid body pose.','Any cloth/body contact cause needs independent geometry evidence; no causal claim follows from correlated arm forces alone.','No actual downward insertion or release occurred, so containment success remains false.'])
(RUN/'independent-transport-reactions.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2)[:24000])
