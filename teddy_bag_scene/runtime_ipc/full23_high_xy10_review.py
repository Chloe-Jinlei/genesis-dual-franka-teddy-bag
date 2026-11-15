"""One bounded high-position XY backup from actual full23 transport state."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_23';OUT=RUN/'independent-high-xy10-backup';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from contact_robot_geometry import check_translation_sweep
from validate import mouth_structure
from insertion_candidates import projected_inside
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['config.json','rest.npz','transport_teddy.npz','states.npz','source/motion.py','source/contact_robot_geometry.py']
hashes={f:sha(RUN/f) for f in files}
cfg=json.loads((RUN/'config.json').read_text());r=dict(np.load(RUN/'rest.npz'));s=dict(np.load(RUN/'transport_teddy.npz'));hist=dict(np.load(RUN/'states.npz'));meta=dict(np.load(RUN/'source/assets'/cfg['bag']['npz_file']))
body,rim,_=mouth_structure(r['bag'],r['bag_faces'],cfg);delta=np.array([.010,-.0025,0.]);tet=s['bear'][r['bear_tets']];prisms=np.concatenate([tet,tet+delta],axis=1);lo=prisms.min(1);hi=prisms.max(1)
def bounds_check(face_ids):
 tri=s['bag'][r['bag_faces'][face_ids]];tlo=tri.min(1);thi=tri.max(1);minimum=np.inf;pair=None
 for i in range(0,len(tet),128):
  gap=np.linalg.norm(np.maximum(np.maximum(tlo[None]-hi[i:i+128,None],lo[i:i+128,None]-thi[None]),0),axis=2)
  k=np.unravel_index(gap.argmin(),gap.shape)
  if gap[k]<minimum:minimum=float(gap[k]);pair=[int(i+k[0]),int(face_ids[k[1]])]
 return {'all_tetra_count':len(tet),'cloth_face_count':len(face_ids),'all_pairs_covered':len(tet)*len(face_ids),'frozen_whole_segment_aabb_distance_lower_bound_m':minimum,'minimum_bound_pair_tet_globalface':pair,'two_mm_geometric_band_clear':minimum-1e-7>=.002,'scope':'Axis-aligned boxes enclose every point of each tetrahedron throughout this one exact linear translation; positive lower bound is conservative. Frozen cloth only; not dynamic deformation or FK guarantee.'}
faces=r['bag_faces'];front=np.flatnonzero(np.isin(faces,meta['handle_front']).all(1));back=np.flatnonzero(np.isin(faces,meta['handle_back']).all(1));bodyidx=meta['body_face_indices']
cloth={name:bounds_check(ids) for name,ids in [('held_front_handle',front),('free_back_handle',back),('body',bodyidx)]}
planner=Planner(cfg);qs={side:s['q_'+side][:7] for side in ['left','right']};finger={side:float(np.clip(s['q_'+side][7:].mean(),0,.04)) for side in qs};tcp,rot=planner.fk('right',qs['right']);target=s['tcp_right']+delta
result={'scope':'One hypothetical +10mmX/-2.5mmY high-position preparation only, from actual 40.280s transport shape. No lowering path or actual execution certified.','actual_source_time_s':float(s['time']),'delta_world_m':delta.tolist(),'actual_tcp_world_m':s['tcp_right'].tolist(),'target_tcp_world_m':target.tolist(),'full_R_xyzw':rot.as_quat().tolist(),'cloth_coarse_certificates':cloth,'physical_success':False}
soft=check_translation_sweep(s['bear'],r['bear_tets'],s['q_left'],cfg,displacement_m=delta,required_gap_m=.005,planner=planner);(OUT/'frozen-bear-left.json').write_text(json.dumps(soft,indent=2)+'\n');result['whole_bear_left_robot']={k:soft.get(k) for k in ['status','conservative_gap_lower_bound_m','rejections','complete_robot_certificate']}
try:
 plan=planner.trajectory(qs,{'right':target},rotations={'right':rot},samples=45,duration=2.,finger_start=finger,finger_end=finger,clearance_margin=.008,max_joint_velocity=.35,max_joint_acceleration=.5,validate_samples=401)
 rows=[];minimum=np.inf;witness=None;maxpos=0.;maxrot=0.
 for time in np.linspace(0,plan.duration,401):
  sample=plan.sample(float(time));raw=planner.clearance(sample['q'],sample['finger']);dist=np.array([x['distance'] for x in raw.values()]);assert dist.size and np.isfinite(dist).all()
  if dist.min()<minimum:minimum=float(dist.min());key=list(raw)[int(dist.argmin())];witness={'time_s':float(time),'category':key,**raw[key]}
  pp,rr=planner.fk('right',sample['q']['right']);maxrot=max(maxrot,float((rr*rot.inv()).magnitude()));rows.append(np.r_[time,pp,*sample['q']['right']])
 endp,endr=planner.fk('right',plan.end_q()['right']);rates=plan.continuous_rate_extrema()
 result['robot_high_shift']={'status':'passed_sampled_rigid_and_continuous_command_rates','duration_s':plan.duration,'samples':401,'minimum_rigid_gap_m':minimum,'minimum_witness':witness,'endpoint_tcp_error_m':float(np.linalg.norm(endp-target)),'maximum_sampled_rotation_error_rad':maxrot,'continuous_rate_extrema':rates,'actual_individual_finger_to_paired_model_error_m':{side:float(np.max(abs(s['q_'+side][7:]-finger[side]))) for side in qs},'whole_curve_archive':'high-shift-plan.npz'}
 np.savez_compressed(OUT/'high-shift-plan.npz',duration_s=plan.duration,right_x=plan.curves['right'].x,right_c=plan.curves['right'].c,left_x=plan.curves['left'].x,left_c=plan.curves['left'].c,full_rotation_xyzw=rot.as_quat(),delta=delta,samples=np.array(rows))
except Exception as exc:result['robot_high_shift']={'status':'rejected','exception':type(exc).__name__,'reason':str(exc)}
point=[]
for targettime in [40.28,42.98,43.03,43.08]:
 i=int(np.argmin(abs(hist['time']-targettime)));bag=hist['bag'][i];bear=hist['bear'][i];foot=bear[66];tri=bag[r['bag_faces'][1407]];normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);normal/=np.linalg.norm(normal)
 ins,dd=projected_inside(np.array([foot[:2],(foot+delta)[:2]]),bag[rim,:2]);point.append({'actual_time_s':float(hist['time'][i]),'bear_material_node':66,'actual_foot_world_m':foot.tolist(),'front_body_face':1407,'actual_front_triangle_oriented_unit_normal':normal.tolist(),'original_projected_inside':bool(ins[0]),'original_projected_rim_distance_m':float(dd[0]),'hypothetical_shifted_point_projected_inside':bool(ins[1]),'hypothetical_shifted_point_rim_distance_m':float(dd[1]),'scope':'Same saved cloth/point plus diagnostic XY offset. This is not the actual outcome of moving earlier and not a whole-object clearance test.'})
result['observed_first_body_contact_and_point_direction']=point
result['overall_scope_passed']=bool(all(v['two_mm_geometric_band_clear'] for v in cloth.values()) and soft['status']=='clear' and result['robot_high_shift']['status'].startswith('passed'))
result['not_certified']=['No complete corrected descent or final bag fit.','No continuous physical robot/cloth clearance, tracking error, grip retention or deformation prediction.','Need fresh actual shape after any high-position correction and original native force/J/stretch/torque/containment guards.']
result['input_hashes_before']=hashes;result['input_hashes_after']={f:sha(RUN/f) for f in files};assert result['input_hashes_before']==result['input_hashes_after']
def serial(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 raise TypeError(type(v).__name__)
out=OUT/'review.json';out.write_text(json.dumps(result,indent=2,default=serial,allow_nan=False)+'\n');print(json.dumps({'report':str(out),'sha256':sha(out),'cloth':cloth,'soft':result['whole_bear_left_robot'],'robot':{k:v for k,v in result['robot_high_shift'].items() if k!='continuous_rate_extrema'},'point':point,'overall_scope_passed':result['overall_scope_passed']},indent=2,default=serial))
