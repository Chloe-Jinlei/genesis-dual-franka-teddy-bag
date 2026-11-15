"""One CPU-only -165 degree postlift hypothesis, outside current policy.
Only an in-memory clone changes the magnitude guard 15->170. No runtime writes.
"""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys,types,time
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_20';sys.path[:0]=[str(RUN/'source'),str(ROOT/'runtime_ipc')]
import postlift_yaw as original
import audit_guarded_run as auditor
from motion import Planner,minimum_jerk
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tracked=[ROOT/'postlift_yaw.py',ROOT/'run.py',ROOT/'config_guarded_postlift_yaw_trial.json',RUN/'source/postlift_yaw.py',RUN/'lift_teddy.npz',RUN/'postlift_yaw_plan.npz']
before={str(p):sha(p) for p in tracked}
rest=dict(np.load(RUN/'rest.npz',allow_pickle=False));state=dict(np.load(RUN/'lift_teddy.npz',allow_pickle=False));cfg=json.loads((RUN/'config.json').read_text())
old_choice=json.loads((RUN/'postlift_yaw_candidate.json').read_text());command={s:np.asarray(old_choice['command_source_poses'][s]['q'],float) for s in ['left','right']}
with np.load(RUN/'postlift_yaw_plan.npz') as prior:
 from scipy.interpolate import PPoly
 assert all(np.array_equal(PPoly(prior[s+'_c'],prior[s+'_x'])(0.),command[s]) for s in command)
assert float(state['time'])==old_choice['observed_time_s']
settings=copy.deepcopy(cfg);settings['task']['contact_trial']['postlift_yaw']={'world_z_degrees':-165.,'duration_s':6.}
source=(RUN/'source/postlift_yaw.py').read_text();needle='if not 0 < abs(degrees) <= 15 or duration < 6:';assert source.count(needle)==1
modified=source.replace(needle,'if not 0 < abs(degrees) <= 170 or duration < 6:')
module=types.ModuleType('diagnostic_postlift_yaw_170');module.__file__=str(RUN/'source/postlift_yaw.py');exec(compile(modified,'<in_memory_diagnostic_postlift_yaw_170>','exec'),module.__dict__)
input_copies=copy.deepcopy((rest,state,settings,command));policy_result,policy_plan=original.plan_postlift_yaw(rest,state,settings,command)
t0=time.monotonic();result,plan=module.plan_postlift_yaw(rest,state,settings,command)
review=dict(kind='independent_one_antipodal_postlift_yaw_diagnostic',physical_success=False,native_execution_performed=False,current_production_policy_admissible=False,runtime_execution_authorized=False,
 current_policy_original_rejection=policy_result['rejections'],current_policy_rejects_and_returns_no_plan=policy_result['status']=='rejected' and policy_plan is None,
 diagnostic_only_change='In-memory numeric yaw magnitude guard 15 -> 170 degrees; source and physical/clearance/rate gates unchanged.',
 original_helper_sha256=sha(RUN/'source/postlift_yaw.py'),inmemory_diagnostic_source_sha256=hashlib.sha256(modified.encode()).hexdigest(),source_time_s=float(state['time']),source_run=str(RUN),source_input_hashes=before,
 diagnostic_parameters=settings['task']['contact_trial']['postlift_yaw'],computation_wall_seconds=time.monotonic()-t0,result=result)
if plan is not None:
 planner=Planner(cfg);A=planner.fk('right',state['q_right'][:7])[1];T=Rotation.from_quat(result['target_rotation_xyzw']);relative=T*A.inv();rv=relative.as_rotvec()
 expected=Rotation.from_rotvec([0,0,np.deg2rad(-165.)]);end=plan.end_q();end_p,end_R=planner.fk('right',end['right']);samples=[]
 for u in np.linspace(0,1,401):
  v=plan.sample(float(u*plan.duration));p,R=planner.fk('right',v['q']['right']);rel=R*A.inv()
  samples.append(dict(t=float(u*plan.duration),fraction=float(minimum_jerk(u)[0]),relative_rotvec_rad=rel.as_rotvec().tolist(),q_right=v['q']['right'].tolist(),tcp=p.tolist()))
 knots=plan.curves['right'].x;qs=plan.curves['right'](knots);rotations=[planner.fk('right',q)[1] for q in qs]
 gaps=[float((b*a.inv()).magnitude()) for a,b in zip(rotations[:-1],rotations[1:])]
 arcs=auditor._yaw_verify_frozen_arc(result['frozen_bear_left_rotation_sweep'],state,command,-165.,planner,sys.modules['motion'].mujoco)
 review['independent_checks']=dict(exact_prior_command_matches_archived_full20_curve_start=True,target_is_signed_minus165_world_left_multiply=np.linalg.norm(rv-[0,0,np.deg2rad(-165.)])<1e-10,
 target_error_against_expected_world_rotation_rad=float((relative*expected.inv()).magnitude()),actual_source_to_target_shortest_angle_degrees=float(np.rad2deg(relative.magnitude())),
 sampled_signed_rotation_z_range_degrees=np.rad2deg([min(x['relative_rotvec_rad'][2] for x in samples),max(x['relative_rotvec_rad'][2] for x in samples)]).tolist(),
 ik_knot_count=len(knots),maximum_consecutive_knot_orientation_angle_degrees=float(np.rad2deg(max(gaps))),full_frozen_tetra_rotation_axes_and_sagitta_reverified=arcs,
 exact_plan_command_start_preserved=all(np.array_equal(plan.sample(0.)['q'][s],command[s]) for s in command),
 left_entire_spline_constant=bool(np.all(plan.curves['left'].c[:3]==0) and np.all(plan.curves['left'].c[3]==command['left'])),
 endpoint_tcp_error_m=float(np.linalg.norm(end_p-state['tcp_right'])),endpoint_rotation_error_rad=float((end_R*T.inv()).magnitude()),right_joint7_start_rad=float(command['right'][-1]),right_joint7_endpoint_rad=float(end['right'][-1]))
 outdir=RUN/'independent-antipodal-postlift-yaw';outdir.mkdir(exist_ok=True)
 np.savez_compressed(outdir/'exact_plan.npz',duration=plan.duration,left_x=plan.curves['left'].x,left_c=plan.curves['left'].c,right_x=plan.curves['right'].x,right_c=plan.curves['right'].c,source_time=float(state['time']))
 (outdir/'samples.json').write_text(json.dumps(samples,indent=2,default=original._serial)+'\n')
 review['exact_plan_archive']=str(outdir/'exact_plan.npz');review['exact_plan_archive_sha256']=sha(outdir/'exact_plan.npz')
review['production_and_original_inputs_unchanged']=before=={str(p):sha(p) for p in tracked}
review['limitations']=['A candidate is outside current accepted yaw policy; no native execution is authorized.', '45 IK nodes and 401 original FK/rigid samples can certify the stated numerical gates, not continuous FK collision clearance.', 'The 33 subsegment hull/sagitta proof encloses every point in every frozen teddy tetrahedron under ideal fixed-TCP world-Z rotation, not native deformable motion.', 'Original native force, torque, two-sided grip, strain/J/table gates and fresh postyaw planning remain mandatory.']
path=RUN/'independent-antipodal-postlift-yaw-review.json';path.write_text(json.dumps(review,indent=2,default=original._serial,allow_nan=False)+'\n')
print(json.dumps({'status':result['status'],'rejections':result['rejections'],'physical_success':False,'review_path':str(path),'review_sha256':sha(path),'independent_checks':review.get('independent_checks'),'robot_preflight':{k:v for k,v in result.get('robot_preflight',{}).items() if k not in ['continuous_rate_bounds','analytic_joint_extrema']},'frozen_arc':{k:v for k,v in result.get('frozen_bear_left_rotation_sweep',{}).items() if k!='per_pose'}},indent=2,default=original._serial))
