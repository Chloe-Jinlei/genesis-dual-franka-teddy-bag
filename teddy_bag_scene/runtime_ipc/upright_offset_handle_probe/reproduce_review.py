"""One CPU-only upright offset handle candidate; no Genesis runtime."""
import json,hashlib,inspect,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import contact_robot_geometry as crg
from motion import Planner,DOWN,hold_trajectory
from contact_grasp import cloth_grasp_frame
root=Path(__file__).resolve().parents[2];out=root/'runtime_ipc/upright_offset_handle_probe';out.mkdir(exist_ok=True)
cfg=json.loads((root/'config_upright_offset_handle_probe.json').read_text());base=json.loads((root/'config_upright_handle_probe.json').read_text())
a=np.load(root/'assets'/cfg['bag']['npz_file']);points=a['points']+cfg['bag']['position'];faces=a['faces'];p=Planner(cfg)
j={'kind':'one_upright_offset_handle_geometric_prefix_candidate','physical_success':False,'gpu_used':False,'production_edited':False,'input_sha256':{str(x.relative_to(root)):hashlib.sha256(x.read_bytes()).hexdigest()for x in [root/'config_upright_handle_probe.json',root/'config_upright_offset_handle_probe.json',root/'motion.py',root/'contact_grasp.py',root/'contact_robot_geometry.py',root/'assets'/cfg['bag']['npz_file']]},'initial':{},'paths':[]}
def save(): (out/'review.json').write_text(json.dumps(j,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
def diff(x,y,prefix=''):
 if isinstance(x,dict) and isinstance(y,dict):
  return sum((diff(x.get(k),y.get(k),prefix+'/'+k)for k in sorted(x.keys()|y.keys())),[])
 return [] if x==y else [{'path':prefix,'old':x,'new':y}]
j['config_diff']=diff(base,cfg)
qs={};fr={};gap={}
for s,h in [('left','front'),('right','back')]:
 spec=cfg['task']['handle_grasps'][s];ids=a[spec['group']];frame=cloth_grasp_frame(a['points'],faces,ids,shell_radius=cfg['bag']['thickness'],preferred_approach=spec['preferred_approach'])
 q,e,ang=p.solve(s,frame.position+cfg['bag']['position'],frame.rotation)
 qs[s]=q;fr[s]=frame;gap[s]=max(.001,frame.minimum_half_gap)
 j['initial'][s]={'group':spec['group'],'ids':ids.tolist(),'all_material_nodes_on_correct_handle':bool(np.isin(ids,a['handle_'+h]).all()),'shared_with_physical_body_nodes':np.intersect1d(ids,a['body']).tolist(),'local_mean_m':frame.position.tolist(),'world_TCP_m':(frame.position+cfg['bag']['position']).tolist(),'frame':frame.as_dict(),'q_rad':q.tolist(),'initial_half_gap_m':gap[s],'IK_position_error_m':e,'IK_rotation_error_rad':ang}
li=a[cfg['task']['handle_grasps']['left']['group']];ri=a[cfg['task']['handle_grasps']['right']['group']]
j['initial_material_separation']={'minimum_node_pair_m':float(cdist(points[li],points[ri]).min()),'centroid_distance_m':float(np.linalg.norm(points[li].mean(0)-points[ri].mean(0))),'both_groups_disjoint':not bool(np.intersect1d(li,ri).size)}
j['initial_clearance']=p.clearance(qs,gap)
j['initial_rigid_pass']=bool(all(v['distance']>=.008 and np.isfinite(v['distance']) for v in j['initial_clearance'].values()))
save();print('INITIAL',j['initial_rigid_pass'],min(v['distance']for v in j['initial_clearance'].values()),'half_gap',gap,flush=True)
# Same fixed extraction algorithm for each side. No source file is changed.
ns=crg.__dict__.copy();exec(inspect.getsource(crg.left_collision_solids).replace('left','right'),ns)
solids=[]
for s in ['left','right']:
 fn=crg.left_collision_solids if s=='left' else ns['right_collision_solids']
 solids.extend(fn(p,np.r_[qs[s],gap[s],gap[s]]))
tri=points[faces]
collision=[];near=0
for solid in solids:
 lo,hi=solid['bounds'];rad=float(cfg['bag']['thickness']);eq=solid['equations']
 cand=np.flatnonzero(np.all(tri.max(1)>=lo-rad,axis=1)&np.all(tri.min(1)<=hi+rad,axis=1))
 for k in cand:
  t=tri[k];s=eq[:,:3]@t.T+eq[:,3,None]
  # Entire triangle separated from inflated solid by one defining plane.
  if np.any(s.min(1)>rad+1e-9):continue
  near+=1
  # Maximize minimum interior plane slack on the true complete triangle.
  # lambda>=0,sumlambda=1; each plane sum(lambda*s)+depth<=0.
  res=linprog([0.,0.,0.,-1.],A_ub=np.column_stack([s,np.ones(len(s))]),b_ub=np.zeros(len(s)),A_eq=[[1.,1.,1.,0.]],b_eq=[1.],bounds=[(0,None)]*3+[(None,None)],method='highs')
  if not res.success:raise ValueError('Triangle-versus-convex LP failed')
  pt=res.x[:3]@t;depth=float(np.min(-(eq[:,:3]@pt+eq[:,3])))
  if depth>-rad-1e-8:collision.append({'body':solid['body'],'geom_id':solid['geom_id'],'cloth_face':int(k),'cloth_vertices':faces[k].tolist(),'maximum_feasible_inward_plane_slack_m':depth,'triangle_barycentric':res.x[:3].tolist(),'world_point_m':pt.tolist(),'center_surface_inside_convex_solid':bool(depth>1e-7),'within_shell_radius_of_solid_halfspaces':bool(depth>-rad)})
j['initial_cloth_robot_convex_screen']={'robot_geom_count':len(solids),'full_cloth_triangle_count':len(faces),'near_tri_geom_pairs_optimized':near,'surface_intrusion_pairs':[v for v in collision if v['center_surface_inside_convex_solid']],'shell_radius_near_pairs':collision,'surface_intrusion_detected':any(v['center_surface_inside_convex_solid']for v in collision),'scope':'Whole triangle linear-program clipping against every nonbase MuJoCo convex collision volume. Positive depth is an interior witness for that convex representation, not exact native nonconvex penetration depth. Shell inflation uses each halfspace and can overestimate Euclidean offset at corners. No native sanity check performed.'}
save();print('CLOTH',len(collision),'surface_inside',sum(v['center_surface_inside_convex_solid']for v in collision),flush=True)
def summary(plan):
 mins={}
 for t in np.linspace(0,plan.duration,201):
  sample=plan.sample(float(t))
  for name,value in p.clearance(sample['q'],sample['finger']).items():
   if not np.isfinite(value['distance']):raise ValueError('Nonfinite raw distance')
   if name not in mins or value['distance']<mins[name]['distance']:mins[name]={**value,'time_s':float(t)}
 if any(v['distance']<.008 for v in mins.values()):raise ValueError('8mm rigid gate failed')
 return {'duration_s':float(plan.duration),'minimum_rigid_clearance_m':min(v['distance']for v in mins.values()),'clearance_minima':mins,'sample_count':201,'continuous_reference_rates':plan.continuous_rate_extrema(),'end_q':{s:q.tolist()for s,q in plan.end_q().items()}}
try:
 if any(v['IK_position_error_m']>.0005 or v['IK_rotation_error_rad']>.002 for v in j['initial'].values()):raise ValueError('Initial IK failed0.5mm/2mrad')
 if not j['initial_rigid_pass']:raise ValueError('Initial rigid8mm gate failed')
 if j['initial_cloth_robot_convex_screen']['surface_intrusion_detected']:raise ValueError('Initial cloth triangle inside robot convex solid')
 speed=float(cfg['task']['motion_speed_scale'])
 for h in [0.,.0015]:
  case={'held_half_gap_m':h,'stages':[]};j['paths'].append(case);g={'left':h,'right':h}
  close=hold_trajectory(qs,.5,gap,g);case['stages'].append({'phase':'close_both_handles',**summary(close)})
  spread=p.trajectory(qs,cfg['task']['open_targets'],rotations={s:fr[s].rotation for s in fr},duration=2.5/speed,finger_start=g,finger_end=g,clearance_margin=.008,max_joint_velocity=.55*speed,max_joint_acceleration=speed**2,validate_samples=201)
  case['stages'].append({'phase':'spread_bag',**summary(spread)});end=spread.end_q()
  release=hold_trajectory(end,.4,g,{'left':h,'right':.04});case['stages'].append({'phase':'release_right_handle',**summary(release)})
  clear=p.trajectory(end,{'right':[.12,.20,1.32]},rotations={'right':DOWN},duration=2./speed,finger_start={'left':h,'right':.04},finger_end={'left':h,'right':.04},clearance_margin=.008,max_joint_velocity=.55*speed,max_joint_acceleration=speed**2,validate_samples=201)
  case['stages'].append({'phase':'right_clear_handle',**summary(clear)});case['passed']=True;save()
  print('PATH',h,[(x['phase'],x['minimum_rigid_clearance_m'])for x in case['stages']],flush=True)
 j['status']='CPU_geometric_prefix_passed_native_pending'
except Exception as e:
 j['status']='rejected';j['reason']=str(e);print('REJECTED',str(e),flush=True)
j['limits']={'rigid_clearance_m':.008,'IK_position_m':.0005,'IK_rotation_rad':.002,'unchanged_native_prefix_joint_velocity_rad_s':.55*float(cfg['task']['motion_speed_scale']),'unchanged_native_prefix_joint_acceleration_rad_s2':float(cfg['task']['motion_speed_scale'])**2}
j['limitations']=['All arm configurations are new IK/trajectory results; no old35/85degree q states are used.','201time rigid checks and two held equal halfgap cases are not a continuous collision certificate nor allactual asymmetric finger openings.','Native prefix .715rad/s and1.69rad/s2 command limits preserved; later guarded insertion .35/.5 policies unaffected.','Finiteforce grasp retention, cloth/bear deformation and dynamic handoff shape are untested. Source cloth is unchanged, fully flexible and unconstrained.','CPU initial contact screen cannot replace native geometry sanity, STC physical reactions or initialization audit.','No robot transport/pickup/insertion path is licensed by this opening and rightretreat prefix result.']
save();print('REPORT',out/'review.json',j['status'],flush=True)
