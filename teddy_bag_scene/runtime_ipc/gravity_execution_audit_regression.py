"""Small independent auditor tests; synthetic records never imply native success."""
import copy,hashlib,json,sys,tempfile
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
import audit_gravity_execution as a
checks={}
def check(name,value):
 checks[name]=bool(value)
 if not value:raise AssertionError(name)
def rejects(name,call):
 try:call()
 except (ValueError,TypeError,AssertionError,KeyError,IndexError):check(name,True)
 else:check(name,False)
cfg=json.loads((ROOT/'config_gravity_seating_trial.json').read_text());p=a.policy(cfg)
check('valid_exact_reviewed_gravity_policy',p['tcp_height_above_table_m']==.345)
check('legacy_absence_stays_off',a.policy({'task':{'contact_trial':{}}}) is None)
for key in p:
 b=copy.deepcopy(cfg);del b['task']['contact_trial']['gravity_release'][key];rejects('missing_policy_'+key,lambda:a.policy(b))
for value in [False,[],1,'gravity_seating']:
 b=copy.deepcopy(cfg);b['task']['contact_trial']['gravity_release']=value;rejects('malformed_policy_'+repr(value),lambda:a.policy(b))
for key in ['tcp_height_above_table_m','minimum_inside_vertex_fraction','settle_seconds']:
 for value in [True,'0.5',float('nan'),float('inf'),[],None,-1,100]:
  b=copy.deepcopy(cfg);b['task']['contact_trial']['gravity_release'][key]=value;rejects('invalid_'+key+'_'+repr(value),lambda:a.policy(b))
for key,value in [('mode','other'),('extra',0)]:
 b=copy.deepcopy(cfg);b['task']['contact_trial']['gravity_release'][key]=value;rejects('unknown_policy_'+key,lambda:a.policy(b))
for value in [True,0,None,'false']:
 b=copy.deepcopy(cfg);b['task']['contact_trial']['recenter_enabled']=value;rejects('invalid_recenter_'+repr(value),lambda:a.policy(b))
b=copy.deepcopy(cfg);b['task']['contact_trial']['insertion_strategy']='staged_contact';rejects('nonvertical_policy',lambda:a.policy(b))
for name in ['two_panel_full_20','two_panel_full_24']:
 run=ROOT/'runs'/name;r=a.audit(run);check(name+'_legacy_not_configured_no_success',r['passed'] is True and r['status']=='not_configured' and r['physical_success'] is False)
contract=a.runner_contract(ROOT/'run.py',ROOT/'runs/two_panel_full_20/source/run.py')
check('reviewed_source_exact_contract',contract['passed'] and contract['mode'] in ('exact_reviewed_gravity_release_extension','exact_reviewed_gravity_serialization_only_extension'))
with tempfile.TemporaryDirectory(prefix='gravity-auditor-test-') as temp:
 d=Path(temp);(d/'config.json').write_text(json.dumps(cfg));r=a.audit(d)
 check('unentered_configured_run_never_passes',r['passed'] is False and r['physical_success'] is False and r['status']=='incomplete')
 legacy={'task':{'contact_trial':{}}};(d/'config.json').write_text(json.dumps(legacy))
 for name in ('gravity_release_candidate.json','gravity_insert_plan.npz','gravity_release_actual_review.json','gravity_withdraw_plan.npz'):
  (d/name).write_text('{}');r=a.audit(d);check('removed_policy_with_'+name+'_rejected',r['passed'] is False and r['status']=='unexpected_evidence_without_policy');(d/name).unlink()
 for key in ('gravity_release_candidate','gravity_release_actual_review'):
  (d/'report.json').write_text(json.dumps({key:{}}));r=a.audit(d);check('removed_policy_with_report_'+key+'_rejected',r['passed'] is False and r['status']=='unexpected_evidence_without_policy')
 (d/'report.json').write_text(json.dumps({'events':[{'phase':'insert_to_gravity_release'}]}));r=a.audit(d);check('removed_policy_with_gravity_event_rejected',r['passed'] is False)
 (d/'report.json').unlink();(d/'config.json').write_text(json.dumps(cfg))
 (d/'run.py').write_bytes((ROOT/'run.py').read_bytes())
 check('missing_helper_blocks_source_adapter',not a.runner_contract(d/'run.py',ROOT/'runs/two_panel_full_20/source/run.py')['passed'])
 (d/'gravity_release.py').write_bytes((ROOT/'gravity_release.py').read_bytes()+b'\n# synthetic mutation\n')
 check('changed_helper_blocks_source_adapter',not a.runner_contract(d/'run.py',ROOT/'runs/two_panel_full_20/source/run.py')['passed'])
 (d/'gravity_release.py').write_bytes((ROOT/'gravity_release.py').read_bytes())
 (d/'run.py').write_bytes((ROOT/'run.py').read_bytes()+b'\n# synthetic mutation\n')
 check('changed_gravity_runner_blocks_source_adapter',not a.runner_contract(d/'run.py',ROOT/'runs/two_panel_full_20/source/run.py')['passed'])
 # A synthetic curve has no robot or physics; it exercises archive arithmetic only.
 motion=a.old.load_module(ROOT/'motion.py','synthetic_gravity_audit_motion')
 class Planner:
  limits={s:np.tile([-5.,5.],(7,1)) for s in ['left','right']}
  raw_mode='normal'
  def fk(self,side,q):return np.asarray(q)[:3],Rotation.identity()
  def clearance(self,qs,fingers):
   rows={key:{'distance':.02} for key in ['left_self','left_table','left_mounts','right_self','right_table','right_mounts','arms']}
   if self.raw_mode=='nan':rows['arms']['distance']=np.nan
   if self.raw_mode=='hidden_nan':rows['arms']['hidden']=np.nan
   if self.raw_mode=='low':rows['arms']['distance']=.007999
   if self.raw_mode=='missing':del rows['arms']
   return rows
 planner=Planner();x=np.linspace(0.,1.,45);v=np.zeros((45,7));v[:,0]=.01*x
 curves={'left':CubicSpline(x,np.zeros_like(v),axis=0),'right':CubicSpline(x,v,axis=0)}
 raw={'duration':np.array(10.),**{s+'_x':c.x.copy() for s,c in curves.items()},**{s+'_c':c.c.copy() for s,c in curves.items()},'finger_left':np.array(0.),'finger_right':np.array(0.)}
 source={'q_left':np.zeros(9),'q_right':np.zeros(9)};start={s:c(0.) for s,c in curves.items()};end={s:c(1.) for s,c in curves.items()};target=np.array([.01,0,0]);proof={'rigid_sample_count':401,'minimum_sampled_rigid_clearance_m':.02,'maximum_sampled_position_error_m':0.,'maximum_sampled_rotation_error_rad':0.}
 def curve_case(data=raw,start_q=start,end_q=end,proof_q=proof):
  path=d/'curve.npz';np.savez(path,**data)
  return a.archive_review(path,planner,motion,source,start_q,end_q,target,Rotation.identity(),proof_q)[0]
 check('synthetic_constant_left_linear_right_archive',curve_case()['passed'])
 for mode in ['nan','hidden_nan','low','missing']:
  planner.raw_mode=mode;rejects('raw401_'+mode+'_fail_closed',curve_case)
 planner.raw_mode='normal'
 data=copy.deepcopy(raw);data['right_c'][0,1,2]=np.nan;rejects('nonfinite_archive_fail_closed',lambda:curve_case(data))
 data=copy.deepcopy(raw);data['extra']=np.array(0);rejects('extra_archive_field_fail_closed',lambda:curve_case(data))
 data=copy.deepcopy(raw);data['right_x'][3]=data['right_x'][2];rejects('duplicate_knot_fail_closed',lambda:curve_case(data))
 data=copy.deepcopy(raw);data['right_c'][3,5,0]+=.01
 try:r=curve_case(data)
 except ValueError:r={'passed':False}
 check('discontinuous_curve_rejected',not r['passed'])
 b=copy.deepcopy(start);b['right'][0]+=.01;check('recorded_command_start_mismatch_rejected',not curve_case(start_q=b)['passed'])
 b=copy.deepcopy(end);b['right'][0]+=.01;check('recorded_command_end_mismatch_rejected',not curve_case(end_q=b)['passed'])
 b=copy.deepcopy(proof);b['rigid_sample_count']=400;check('reported_sample_count_mismatch_rejected',not curve_case(proof_q=b)['passed'])
 data=copy.deepcopy(raw);data['duration']=np.array(.01);check('continuous_rate_excess_rejected',not curve_case(data)['passed'])
 data=copy.deepcopy(raw);data['left_c'][2,:,1]=.01
 try:r=curve_case(data)
 except ValueError:r={'passed':False}
 check('changing_left_curve_rejected',not r['passed'])
cap={'virtual_closed_body_valid':True,'bear_center_of_mass_inside':True,'bear_vertex_inside_fraction':.5};guard={'right_two_sided_contact_required':True,'stop_reason':None,'right_finger_opening_projection_n':[.10001,.2]};velocity=np.zeros(7)
check('interim_gate_threshold_is_inclusive_at_half',a.release_decision(cap,p,velocity,guard))
for key,value in [('virtual_closed_body_valid',False),('bear_center_of_mass_inside',False),('bear_vertex_inside_fraction',.499999),('bear_vertex_inside_fraction',np.nan)]:
 b=copy.deepcopy(cap);b[key]=value;check('release_reject_'+key+'_'+repr(value),not a.release_decision(b,p,velocity,guard))
for key,value in [('right_two_sided_contact_required',False),('stop_reason','lost grip'),('right_finger_opening_projection_n',[.1,.2]),('right_finger_opening_projection_n',[.2]),('right_finger_opening_projection_n',[np.nan,.2])]:
 b=copy.deepcopy(guard);b[key]=value;check('release_reject_'+key+'_'+repr(value),not a.release_decision(cap,p,velocity,b))
for value in [np.full(7,.050001),np.full(7,np.nan),np.zeros(6)]:check('release_velocity_reject_'+str(value),not a.release_decision(cap,p,value,guard))
check('interim_true_is_not_task_success_evidence','physical_success' not in cap)
for phase,minimum in [('insert_teddy',6.),('withdraw_right_arm',4.)]:
 check(phase+'_minimum_duration_inclusive',a.duration_valid(minimum,phase))
 for value in (minimum-1e-8,np.nan,np.inf,True,'6'):
  check(phase+'_duration_reject_'+repr(value),not a.duration_valid(value,phase))
# A real moving passage above 50% must not be relabeled an admissible stopped release.
prior=ROOT/'runs/two_panel_full_24';rows=json.loads((prior/'independent-release-window-containment-review.json').read_text())['rows'];row=min(rows,key=lambda v:abs(v['time_s']-42.665))
trace=np.load(prior/'joint_trace.npy');index=int(np.argmin(abs(trace[:,0]-row['time_s'])));velocity=(trace[index,10:17]-trace[index-1,10:17])/(trace[index,0]-trace[index-1,0])
guard=min(json.loads((prior/'contact_guard_trace.json').read_text()),key=lambda v:abs(v['time_s']-row['time_s']))
capture={'virtual_closed_body_valid':row['original_mean_fan_cap_valid'],'bear_center_of_mass_inside':row['original_cap_COM_inside'],'bear_vertex_inside_fraction':row['original_cap_node_inside_fraction_not_volume']}
check('real_full24_half_capture_is_not_stopped_release',capture['bear_center_of_mass_inside'] and capture['bear_vertex_inside_fraction']>=.5 and max(abs(velocity))>.05 and not a.release_decision(capture,p,velocity,guard))
historical={'run':'two_panel_full_24','time_s':row['time_s'],'capture':capture,'maximum_actual_joint_speed_rad_s':float(max(abs(velocity))),'interim_release_decision':False,'actual_motion_was_not_changed':True}
result={'kind':'independent_gravity_execution_auditor_regression','passed':all(checks.values()),'count':len(checks),'checks':checks,'synthetic_cases_marked':True,'real_historical_negative_case':historical,'physical_success':False,'native_execution_performed':False,'auditor_sha256':a.sha(HERE/'audit_gravity_execution.py'),'approved_production_hashes':a.APPROVED,'limits':['No native run is executed or successful by these tests.','The complete actual full25 execution chain remains pending until final evidence exists.']}
out=HERE/'gravity-execution-audit-regression.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':result['passed'],'count':len(checks),'path':str(out),'sha256':a.sha(out)}))
