"""Independent saved-evidence audit for the one reviewed gravity-release extension.

Never imports run.py/gravity_release.py or plans a motion. Old auditors are not
edited. A narrowly hash-pinned static adapter is available for the unchanged yaw
checks, whose original complete-method recognizer predates this extension.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,json,runpy,sys
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
import audit_guarded_run as old
APPROVED={'run.py':'f774826d687eb6662bd1d4c491382214eac2fce22e554fd0ebdf9bf6fba0a2a6','gravity_release.py':'d88f4a7cd34853fb8f07eb34d42e137007cd553f69ddc920a0856a0710c96dee'}
SERIALIZATION_ONLY_RUN_SHA256='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
APPROVED_RUN_SHA256={APPROVED['run.py'],SERIALIZATION_ONLY_RUN_SHA256}
LOW_RELEASE_HELPER_SHA256='3cb7ea9a10390bcb473d353863917d39cc67a05766b2da3bfc38ed28bd26915f'
HEIGHT_RANGE_OLD='for key, lo, hi in (("tcp_height_above_table_m", .25, .45),'
HEIGHT_RANGE_NEW='for key, lo, hi in (("tcp_height_above_table_m", .15, .45),'
SERIALIZATION_OLD='        (self.out/"report.json").write_text(json.dumps(self.report,indent=2))\n'
SERIALIZATION_NEW='        from contact_insertion import _serial\n        (self.out/"report.json").write_text(json.dumps(_serial(self.report),indent=2,allow_nan=False))\n'
LEGACY_CONTRACT=old.frozen_runner_yaw_contract
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(v):
 if isinstance(v,dict):return all(finite(x) for x in v.values())
 if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
 if isinstance(v,np.ndarray):return bool(np.isfinite(v).all()) if v.dtype.kind in 'fiu' else False
 if isinstance(v,(float,int,np.number)) and not isinstance(v,(bool,np.bool_)):return bool(np.isfinite(v))
 return v is None or isinstance(v,(str,bool,np.bool_))
def helper_contract(path):
 """Read frozen bytes only; never execute a helper to discover its policy."""
 path=Path(path)
 if not path.is_file():return {'passed':False,'minimum_height_m':None,'mode':'missing_frozen_helper'}
 source=path.read_bytes();digest=hashlib.sha256(source).hexdigest()
 if digest==APPROVED['gravity_release.py']:
  return {'passed':True,'sha256':digest,'minimum_height_m':.25,'mode':'original_height_range'}
 equivalent=bool(digest==LOW_RELEASE_HELPER_SHA256 and source.count(HEIGHT_RANGE_NEW.encode())==1 and hashlib.sha256(source.replace(HEIGHT_RANGE_NEW.encode(),HEIGHT_RANGE_OLD.encode())).hexdigest()==APPROVED['gravity_release.py'])
 return {'passed':equivalent,'sha256':digest,'minimum_height_m':.15 if equivalent else None,'mode':'exact_height_lower_bound_only_extension' if equivalent else 'unrecognized_frozen_helper','reverse_literal_replacement_byte_equivalent':equivalent}
def runner_contract(path,baseline_path):
 path=Path(path)
 digest=sha(path)
 if digest not in APPROVED_RUN_SHA256:return LEGACY_CONTRACT(path,baseline_path)
 helper=helper_contract(path.parent/'gravity_release.py')
 checks={'exact_reviewed_gravity_runner_hash':True,'exact_reviewed_gravity_helper_hash':helper['passed']}
 if digest==SERIALIZATION_ONLY_RUN_SHA256:
  source=path.read_text();checks['exact_serialization_only_whole_source_equivalence']=source.count(SERIALIZATION_NEW)==1 and hashlib.sha256(source.replace(SERIALIZATION_NEW,SERIALIZATION_OLD).encode()).hexdigest()==APPROVED['run.py']
 names=['begin_guarded_contact_monitor','run_postlift_yaw','observe_physics','execute']
 def methods(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(Path(p).read_text())) if isinstance(n,ast.FunctionDef) and n.name in names}
 a,b=methods(path),methods(baseline_path)
 checks.update({name:a.get(name)==b.get(name) and name in a for name in names})
 return {'passed':all(checks.values()),'mode':'exact_reviewed_gravity_release_extension' if digest==APPROVED['run.py'] else 'exact_reviewed_gravity_serialization_only_extension','recognized_runner_sha256':digest,'helper_source_contract':helper,'checks':checks,'scope':'Only two explicitly reviewed runner hashes and two explicitly reviewed helper hashes are accepted. The second runner reverses only exact final report serialization lines; the new helper reverses only the height policy lower bound .15 to .25, both byte-for-byte to the originals. Four original guard/yaw/execute methods remain exact baseline AST. This is source dataflow evidence, not an observed per-step command log.'}
def approved_manifest_sources(run,manifest):
 return bool(sha(run/'source/run.py') in APPROVED_RUN_SHA256 and sha(run/'source/run.py')==manifest.get('source/run.py') and helper_contract(run/'source/gravity_release.py')['passed'] and sha(run/'source/gravity_release.py')==manifest.get('source/gravity_release.py'))
def policy(cfg,helper_path=None):
 p=cfg.get('task',{}).get('contact_trial',{}).get('gravity_release')
 if p is None:return None
 if not isinstance(p,dict) or set(p)!={'mode','tcp_height_above_table_m','minimum_inside_vertex_fraction','settle_seconds'} or p['mode']!='gravity_seating':raise ValueError('Malformed explicit gravity policy')
 minimum_height=.25
 if helper_path is not None:
  helper=helper_contract(helper_path)
  if not helper['passed']:raise ValueError('Unrecognized frozen gravity helper policy')
  minimum_height=helper['minimum_height_m']
 for key,lo,hi in [('tcp_height_above_table_m',minimum_height,.45),('minimum_inside_vertex_fraction',.5,1.),('settle_seconds',2.,10.)]:
  v=p[key]
  if isinstance(v,(bool,str)) or not isinstance(v,(int,float)) or not np.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid policy '+key)
 t=cfg['task']['contact_trial']
 if t.get('insertion_strategy','vertical')!='vertical' or t.get('recenter_enabled') is not False:raise ValueError('Gravity policy must use fixed vertical route and disabled recenter')
 return p

def duration_valid(duration,phase):
 minimum={'insert_teddy':6.,'withdraw_right_arm':4.}[phase]
 return bool(not isinstance(duration,(bool,np.bool_)) and isinstance(duration,(int,float,np.number)) and np.isfinite(duration) and duration>=minimum)

def archive_review(path,planner,motion,source,start_q,end_q,target,R,proof,opening_right=False):
 a=dict(np.load(path,allow_pickle=False));required={'duration','left_x','left_c','right_x','right_c','finger_left','finger_right'}
 if set(a)!=required or not finite(a):raise ValueError('Malformed/nonfinite curve archive')
 T=float(a['duration']);assert T>0;curves={};bounds={};checks={};result={'checks':checks,'duration_s':T,'archive_sha256':sha(path)}
 for side in ['left','right']:
  x,c=a[side+'_x'],a[side+'_c'];assert x.shape==(45,) and c.shape==(4,44,7) and x[0]==0 and x[-1]==1 and np.all(np.diff(x)>0)
  curve=PPoly(c,x);curves[side]=curve
  checks[side+'_recorded_command_start_end_match_archive']=np.array_equal(curve(0.),start_q[side]) and np.array_equal(curve(1.),end_q[side])
  joins=[]
  for k,h in enumerate(np.diff(x)[:-1]):
   # Evaluate the two one-sided polynomial limits exactly at the knot.
   for order in range(3):
    for joint in range(7):joins.append(abs(np.polyval(np.polyder(c[:,k,joint],order),h)-np.polyval(np.polyder(c[:,k+1,joint],order),0.)))
  checks[side+'_cubic_continuity']=bool(max(joins,default=0.)<=1e-8)
  bounds[side]=old._yaw_curve_bounds(curve,T);bb=bounds[side];lim=planner.limits[side]
  checks[side+'_independent_joint_and_continuous_rate_bounds']=bool(np.all(bb['minimum']>=lim[:,0]) and np.all(bb['maximum']<=lim[:,1]) and np.all(bb['velocity']<=.35) and np.all(bb['acceleration']<=.5))
 checks['left_curve_constant']=bool(np.all(curves['left'].c[:3]==0) and np.all(curves['left'].c[3]==np.asarray(start_q['left'])))
 fingers={side:float(np.clip(source['q_'+side][7:].mean(),0,.04)) for side in curves}
 if opening_right:fingers['right']=.04
 minimum=np.inf;pmax=rmax=0.;start=planner.fk('right',np.asarray(start_q['right']))[0]
 for u in np.linspace(0.,1.,401):
  s=10*u**3-15*u**4+6*u**5;qr=curves['right'](s)
  for ql in [np.asarray(source['q_left'])[:7],curves['left'](s)]:
   raw=planner.clearance({'left':ql,'right':qr},fingers)
   if not old._yaw_rigid_rows(raw):raise ValueError('Archived path raw rigid check failed')
   minimum=min(minimum,*[float(v['distance']) for v in raw.values()])
  p,r=planner.fk('right',qr);pe=float(np.linalg.norm(p-(start+s*(np.asarray(target)-start))));re=float((r*R.inv()).magnitude())
  assert np.isfinite([pe,re]).all();pmax=max(pmax,pe);rmax=max(rmax,re)
 checks['raw401_finite_rigid_and_FK']=bool(minimum>=.008 and pmax<=.0005 and rmax<=.002)
 checks['reported_raw401_metrics_match_archive']=bool(proof['rigid_sample_count']==401 and abs(proof['minimum_sampled_rigid_clearance_m']-minimum)<=1e-9 and abs(proof['maximum_sampled_position_error_m']-pmax)<=1e-9 and abs(proof['maximum_sampled_rotation_error_rad']-rmax)<=1e-9)
 plan=motion.MotionPlan(curves,T,{s:float(a['finger_'+s]) for s in curves},{s:float(a['finger_'+s]) for s in curves})
 checks['zero_reference_endpoint_velocity_acceleration']=all(np.max(abs(plan.sample(t)[key][side]))<1e-10 for t in [0.,T] for key in ['qd','qdd'] for side in curves)
 result.update(passed=all(checks.values()),bounds=bounds,minimum_rigid_gap_m=minimum,maximum_FK_position_error_m=pmax,maximum_FK_full_R_error_rad=rmax,command_finger_targets={s:float(a['finger_'+s]) for s in curves})
 return result,curves

def actual_state_review(run,cfg,rest,state,planner):
 """Recompute original physical/grasp/rigid limits without motion planning."""
 from scipy.spatial import cKDTree
 bag,bear=np.asarray(state['bag']),np.asarray(state['bear']);tets=np.asarray(rest['bear_tets'],int);faces=np.asarray(rest['bag_faces'],int)
 if bag.shape!=rest['bag'].shape or bear.shape!=rest['bear'].shape or not np.isfinite(bag).all() or not np.isfinite(bear).all():raise ValueError('Invalid actual endpoint soft geometry')
 def determinant(p):
  t=p[tets];return np.linalg.det(np.stack([t[:,i]-t[:,0] for i in (1,2,3)],axis=-1))
 refs=determinant(rest['bear']);edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
 lengths=np.linalg.norm(rest['bag'][edges[:,1]]-rest['bag'][edges[:,0]],axis=1)
 if np.any(abs(refs)<6e-14) or np.any(lengths<1e-10):raise ValueError('Degenerate reference geometry')
 values={'minimum_tetra_volume_ratio':float((determinant(bear)/refs).min()),'maximum_cloth_edge_stretch_ratio':float((np.linalg.norm(bag[edges[:,1]]-bag[edges[:,0]],axis=1)/lengths).max())}
 half=np.asarray(cfg['table']['size'][:2])/2;penetrations=[]
 for points in (bag,bear):
  covered=np.all(abs(points[:,:2])<=half,axis=1);penetrations.append(max(0.,cfg['table']['height']-float(points[covered,2].min())) if covered.any() else 0.)
 with np.load(run/'source/assets'/Path(cfg['bag']['npz_file']).name,allow_pickle=False) as data:
  group=cfg['task']['handle_grasps']['left']['group'];material=data['points'][data[group]]+np.asarray(cfg['bag']['position'])
 error,ids=cKDTree(rest['bag']).query(material)
 if error.max()>1e-6:raise ValueError('Cannot map designated held material')
 values.update(table_penetrations_m=penetrations,left_material_distance_m=float(np.linalg.norm(bag[np.unique(ids)]-state['tcp_left'],axis=1).min()),right_material_distance_m=float(np.linalg.norm(bear-state['tcp_right'],axis=1).min()))
 qs={};fingers={};valid=True
 for side in ('left','right'):
  q=np.asarray(state['q_'+side]);tcp=np.asarray(state['tcp_'+side]);valid=valid and q.shape==(9,) and tcp.shape==(3,) and np.isfinite(q).all() and np.isfinite(tcp).all()
  if not valid:raise ValueError('Invalid endpoint robot data')
  valid=valid and np.all(q[:7]>=planner.limits[side][:,0]) and np.all(q[:7]<=planner.limits[side][:,1]) and np.all(q[7:]>=-.0005) and np.all(q[7:]<=.0405) and np.linalg.norm(planner.fk(side,q[:7])[0]-tcp)<=.002
  qs[side]=q[:7];fingers[side]=float(np.clip(q[7:].mean(),0.,.04))
 raw=planner.clearance(qs,fingers)
 values['current_raw_rigid_clearances']=raw
 ok=valid and finite(values) and values['minimum_tetra_volume_ratio']>=.05 and values['maximum_cloth_edge_stretch_ratio']<=1.15 and max(penetrations)<=.002 and values['left_material_distance_m']<=.018 and values['right_material_distance_m']<=.045 and old._yaw_rigid_rows(raw)
 return {'passed':bool(ok),'measurements':values,'scope':'Original actual-state geometry checks. Paired-mean fingers are the original planning approximation; raw native reactions remain separately mandatory.'}

def opening_review(planner,source,command,record):
 minimum=np.inf
 for gap in np.linspace(0.,.04,21):
  fingers={'left':float(np.clip(source['q_left'][7:].mean(),0.,.04)),'right':float(gap)}
  raw=planner.clearance(command,fingers)
  if not old._yaw_rigid_rows(raw):raise ValueError('Actual release opening raw rigid check failed')
  minimum=min(minimum,*[float(v['distance']) for v in raw.values()])
 return {'passed':bool(record['opening_samples']==21 and abs(record['minimum_opening_rigid_clearance_m']-minimum)<=1e-9),'minimum_opening_rigid_clearance_m':minimum,'sample_count':21,'scope':'21 original paired-finger opening states; no continuous opening certificate.'}

def release_decision(captured,p,velocity,guard):
 """Independent interim decision; a true result is never final task success."""
 values=np.asarray(velocity,float);opening=np.asarray(guard.get('right_finger_opening_projection_n'),float)
 return bool(captured.get('virtual_closed_body_valid') is True and captured.get('bear_center_of_mass_inside') is True and np.isfinite(captured.get('bear_vertex_inside_fraction',np.nan)) and captured['bear_vertex_inside_fraction']>=p['minimum_inside_vertex_fraction'] and values.shape==(7,) and np.isfinite(values).all() and np.max(abs(values))<=.05 and guard.get('right_two_sided_contact_required') is True and not guard.get('stop_reason') and opening.shape==(2,) and np.isfinite(opening).all() and min(opening)>.1)

def audit(run):
 run=Path(run).resolve();result={'kind':'independent_gravity_release_execution_audit','status':'incomplete','passed':False,'physical_success':False,'checks':{},'incomplete_evidence':[],'curve_evidence_scope':'Hash-reviewed same-object source dataflow plus saved curve/event/actual tracking. There is no independent per-step command log.'};checks=result['checks']
 def ck(k,v):checks[k]=bool(v)
 def read(name):return json.loads((run/name).read_text())
 try:
  cfg=read('config.json')
  if cfg.get('task',{}).get('contact_trial',{}).get('gravity_release') is not None and not (run/'source/gravity_release.py').is_file():
   result['incomplete_evidence']=['source/gravity_release.py'];return result
  p=policy(cfg,run/'source/gravity_release.py')
  if p is None:
   evidence=[name for name in ('gravity_release_candidate.json','gravity_insert_plan.npz','gravity_release_actual_review.json','gravity_withdraw_plan.npz') if (run/name).exists()]
   prior=read('report.json') if (run/'report.json').exists() else {}
   if not isinstance(prior,dict):raise ValueError('Invalid legacy report')
   evidence += ['report.'+key for key in ('gravity_release_candidate','gravity_release_actual_review') if prior.get(key) is not None]
   evidence += ['unexpected gravity event'] if any(str(e.get('phase','')).startswith('gravity_') or e.get('phase')=='insert_to_gravity_release' for e in prior.get('events',[])) else []
   ck('absent_policy_has_no_gravity_evidence',not evidence);result['unexpected_evidence']=evidence
   result.update(status='not_configured' if not evidence else 'unexpected_evidence_without_policy',passed=not evidence);return result
  required=['report.json','events.json','states.npz','rest.npz','joint_trace.npy','soft_metrics.npy','contact_guard_trace.json','input_sha256.json','gravity_release_candidate.json','gravity_insert_plan.npz','guarded_contact_reobserved.json']
  missing=[name for name in required if not (run/name).exists()]
  if missing:result['incomplete_evidence']=missing;return result
  inputs={str(run/n):sha(run/n) for n in required};report=read('report.json');events=read('events.json');states=dict(np.load(run/'states.npz'));rest=dict(np.load(run/'rest.npz'));trace=np.load(run/'joint_trace.npy');guards=read('contact_guard_trace.json');manifest=read('input_sha256.json');c=read('gravity_release_candidate.json');full=read('guarded_contact_reobserved.json');dt=float(cfg['simulation']['dt'])
  contract=runner_contract(run/'source/run.py',ROOT/'runs/two_panel_full_20/source/run.py');result['runner_static_contract']=contract;ck('recognized_exact_reviewed_runner_and_helper',contract['passed'])
  ck('reviewed_module_hashes_match_manifest',approved_manifest_sources(run,manifest))
  if not all(checks.values()):return result
  ck('events_equal_report',events==report['events']);ck('candidate_equal_report_and_finite',c==report['gravity_release_candidate'] and finite(c) and c['status']=='candidate');ck('declared_policy_exact',c['policy']==p)
  ck('full_depth_original_fresh_preflight_retained',full['status']=='candidate' and not full['report']['rejections'] and full['observed_time_s']==c['observed_time_s'] and full==report['guarded_contact_reobserved'] and full['report']['policy']['rigid_clearance_m']==.008 and full['report']['policy']['left_robot_soft_sweep_clearance_m']==.005)
  def frame(t):
   ids=np.flatnonzero(abs(states['time']-float(t))<=1e-8)
   if not len(ids):raise LookupError('No actual complete frame at '+str(t))
   return {k:np.asarray(v)[ids[-1]] for k,v in states.items()}
  def ev(name):
   found=[e for e in events if e.get('phase')==name]
   if len(found)!=1:raise LookupError('Expected one '+name+' event')
   return found[0]
  source=frame(c['observed_time_s']);low=np.asarray(full['insert_stage']['right_tcp_m']);target=np.asarray(c['insert_stage']['right_tcp_m']);R=Rotation.from_quat(c['insert_stage']['rotation_xyzw']);outside=np.asarray(full['outside_tcp_m'])
  ck('source_arm_records_exact',all(np.array_equal(c['actual_source_arms'][side]['q'],source['q_'+side]) and np.array_equal(c['actual_source_arms'][side]['tcp_m'],source['tcp_'+side]) for side in ['left','right']))
  ck('short_target_explicit_fixed_XY_between_original_endpoints',c['insert_stage']['name']=='insert_to_gravity_release' and np.max(abs(target[:2]-source['tcp_right'][:2]))<=1e-9 and np.max(abs(low[:2]-source['tcp_right'][:2]))<=1e-9 and np.max(abs(low-c['original_full_depth_tcp_m']))<=1e-12 and np.array_equal(c['outside_tcp_m'],outside) and abs(target[2]-cfg['table']['height']-p['tcp_height_above_table_m'])<=1e-12 and low[2]+.02<target[2]<source['tcp_right'][2]-.04)
  ck('no_partial_success_claim_or_final_gate_override',c['physical_success'] is False and c['requires_actual_partial_capture_before_release'] is True and c['requires_unchanged_final_full_containment_and_retention'] is True)
  motion=old.load_module(run/'source/motion.py','gravity_audit_frozen_motion');pc=copy.deepcopy(cfg);pc['robots']['mjcf_path']=str(run/'source/assets'/Path(cfg['robots'].get('mjcf_path','panda_tactile.xml')).name);planner=motion.Planner(pc)
  source_review=actual_state_review(run,cfg,rest,source,planner);result['independent_source_physical_review']=source_review;ck('source_original_physical_limits',source_review['passed'])
  ck('full_actual_orientation_and_command_source_consistency',bool((planner.fk('right',source['q_right'][:7])[1]*R.inv()).magnitude()<=1e-12 and all(np.linalg.norm(planner.fk(s,np.asarray(c['command_q_start'][s]))[0]-planner.fk(s,source['q_'+s][:7])[0])<=.0005 and (planner.fk(s,np.asarray(c['command_q_start'][s]))[1]*planner.fk(s,source['q_'+s][:7])[1].inv()).magnitude()<=.002 for s in ('left','right'))))
  sweeps=c['left_robot_soft_sweeps'];sweep_checks=[]
  for label in ('actual','commanded'):
   sw=sweeps[label];q=np.asarray(source['q_left']).copy()
   if label=='commanded':q[:7]=c['command_q_start']['left']
   sweep_checks.append(finite(sw) and sw['status']=='clear' and sw['physical_success'] is False and sw['complete_robot_certificate'] is True and not sw['rejections'] and sw['required_gap_m']==.006 and np.array_equal(sw['q_left_measured'],q) and np.array_equal(sw['translation_m'],np.zeros(3)) and np.max(abs(np.asarray(sw['displacement_m'])-(target-source['tcp_right'])))<=1e-12 and sw['input_vertex_count']==len(source['bear']) and sw['tetrahedron_count']==len(rest['bear_tets']) and sw['conservative_gap_lower_bound_m']>=.006)
  ck('saved_complete_teddy_sweep_contract_and_actual_source',set(sweeps)=={'actual','commanded'} and all(sweep_checks));result['soft_sweep_scope']='Source/pose/displacement/count/clear-status/6mm certificate metadata are checked. This audit does not rerun the convex separation search.'
  ins,icurves=archive_review(run/'gravity_insert_plan.npz',planner,motion,source,c['command_q_start'],c['command_q_end'],target,R,c['insert_curve_review']);result['insert_archive']=ins;ck('insert_archive_checks',ins['passed'])
  ck('insert_original_minimum_six_seconds',duration_valid(ins['duration_s'],'insert_teddy'))
  ck('insert_original_closed_finger_commands',ins['command_finger_targets']=={'left':0.,'right':cfg['task'].get('bear_finger_target',.027)})
  event=ev('insert_teddy');end=float(event['start'])+int(np.ceil(ins['duration_s']/dt))*dt;result['expected_insert_endpoint_time_s']=end
  ck('insert_event_matches_source_and_archive',abs(event['start']-float(source['time']))<=1e-8 and abs(event['duration']-ins['duration_s'])<=1e-12 and abs(event['duration']-c['insert_stage']['duration_s'])<=1e-12)
  reactions=old.audit_reactions(run,report,trace);result['every_step_reaction_audit']=reactions;ck('native_reaction_guards_and_original_release_flag',reactions['passed'])
  if report['simulated_seconds']<end-1e-8:
   result['incomplete_evidence'].append('Native insertion stopped before its declared short endpoint');return result
  actual=frame(end);ck('actual_endpoint_matches_short_target',np.linalg.norm(actual['tcp_right']-target)<=.002 and (planner.fk('right',actual['q_right'][:7])[1]*R.inv()).magnitude()<=.015)
  endpoint_review=actual_state_review(run,cfg,rest,actual,planner);result['independent_endpoint_physical_review']=endpoint_review;ck('endpoint_original_physical_limits',endpoint_review['passed'])
  if not (run/'gravity_release_actual_review.json').exists():result['incomplete_evidence'].append('No saved actual release decision');return result
  actual_review=read('gravity_release_actual_review.json');inputs[str(run/'gravity_release_actual_review.json')]=sha(run/'gravity_release_actual_review.json');ck('actual_review_equal_report_finite',actual_review==report.get('gravity_release_actual_review') and finite(actual_review) and abs(actual_review['actual_time_s']-end)<=1e-8)
  ck('reported_actual_endpoint_errors_match_geometry',abs(actual_review['actual_endpoint_position_error_m']-np.linalg.norm(actual['tcp_right']-target))<=1e-12 and abs(actual_review['actual_orientation_error_rad']-(planner.fk('right',actual['q_right'][:7])[1]*R.inv()).magnitude())<=1e-12)
  opening_actual=opening_review(planner,actual,{s:actual['q_'+s][:7] for s in ('left','right')},actual_review['actual_opening_and_withdrawal_preflight'])
  opening_command=opening_review(planner,actual,{s:np.asarray(actual_review['withdraw_command_q_start'][s]) for s in ('left','right')},actual_review['commanded_opening_and_withdrawal_preflight'])
  result['independent_actual_release_opening']={'actual':opening_actual,'commanded':opening_command};ck('actual_and_commanded_opening_samples_match',opening_actual['passed'] and opening_command['passed'])
  validator=old.load_module(run/'source/validate.py','gravity_audit_frozen_validate');body,rim,topology=validator.mouth_structure(rest['bag'],rest['bag_faces'],cfg);w=abs(validator.signed_volumes(rest['bear'],rest['bear_tets']));captured=validator.containment(actual['bag'],actual['bear'],body,rim,w,rest['bear_tets']);result['recomputed_actual_release_containment_all_nodes']=captured
  ck('actual_release_capture_uses_all_nodes_and_rest_volume_COM',captured==actual_review['actual_containment'])
  assert trace.ndim==2 and trace.shape[1]==19 and np.isfinite(trace).all()
  ids=np.flatnonzero(abs(trace[:,0]-end)<=1e-8);assert len(ids) and ids[-1]>=1;j=int(ids[-1]);assert 0<trace[j,0]-trace[j-1,0]<=.02;velocity=(trace[j,10:17]-trace[j-1,10:17])/(trace[j,0]-trace[j-1,0]);gi=np.flatnonzero(abs(np.array([g['time_s'] for g in guards])-end)<=1e-8);assert len(gi);g=guards[gi[-1]]
  decision_ok=release_decision(captured,p,velocity,g)
  ck('actual_joint_endpoint_and_velocity_evidence_matches',np.array_equal(trace[j,1:10],actual['q_left']) and np.array_equal(trace[j,10:19],actual['q_right']) and np.max(abs(velocity-np.array(actual_review['actual_right_endpoint_joint_velocity_rad_s'])))<=1e-12)
  ck('reported_release_checks_have_exact_known_keys',set(actual_review['checks'])=={'actual_state_has_no_original_preflight_rejection','actual_time_after_candidate','actual_endpoint_within_2mm','actual_full_orientation_retained','actual_joint_trace_matches_endpoint_time','actual_joint_trace_matches_saved_endpoint_joints','actual_right_joint_velocity_below_0_05_rad_s','contact_row_is_current_and_still_holding','both_right_opening_reactions_above_original_0_1N','actual_analysis_body_valid','actual_mass_COM_inside','actual_all_node_fraction_at_least_declared_minimum'})
  ck('release_decision_consistent_with_recorded_checks',actual_review['status']==('admissible_release' if all(v is True for v in actual_review['checks'].values()) else 'rejected'))
  released=[e for e in events if e['phase']=='release_teddy']
  if actual_review['status']!='admissible_release':
   ck('rejected_partial_gate_not_released',not released);result['status']='actual_release_rejected';return result
  ck('partial_capture_and_original_hold_guard_precede_intent',decision_ok and len(released)==1 and abs(released[0]['start']-end)<=1e-8)
  ck('explicit_half_second_release',len(released)==1 and released[0]['duration']==.5)
  if not (run/'gravity_withdraw_plan.npz').exists():result['incomplete_evidence'].append('No withdrawal curve archived');return result
  inputs[str(run/'gravity_withdraw_plan.npz')]=sha(run/'gravity_withdraw_plan.npz')
  wd,wcurves=archive_review(run/'gravity_withdraw_plan.npz',planner,motion,actual,actual_review['withdraw_command_q_start'],actual_review['withdraw_command_q_end'],outside,R,actual_review['commanded_opening_and_withdrawal_preflight']['withdrawal_curve_review'],True);result['withdraw_archive']=wd;ck('withdraw_archive_checks',wd['passed'])
  ck('withdraw_original_minimum_four_seconds',duration_valid(wd['duration_s'],'withdraw_right_arm'))
  ck('withdraw_command_continues_same_insert_q',all(np.array_equal(actual_review['withdraw_command_q_start'][side],icurves[side](1.)) for side in ['left','right']))
  ck('withdraw_original_open_right_command',wd['command_finger_targets']=={'left':0.,'right':.04})
  withdraw=ev('withdraw_right_arm');release_end=end+int(np.ceil(.5/dt))*dt;ck('withdraw_only_after_complete_intentional_opening',abs(withdraw['start']-release_end)<=1e-8 and abs(withdraw['duration']-wd['duration_s'])<=1e-12)
  wdend=withdraw['start']+int(np.ceil(withdraw['duration']/dt))*dt;wend=frame(wdend);ck('actual_withdraw_endpoint_matches_original_outside',np.linalg.norm(wend['tcp_right']-outside)<=.002)
  settle=ev('final_settle');stop=settle['start']+int(np.ceil(settle['duration']/dt))*dt
  ck('full_declared_settle_after_withdraw',abs(settle['start']-wdend)<=1e-8 and settle['duration']==p['settle_seconds'] and report['simulated_seconds']>=stop-1e-8)
  final=validator.audit_arrays(rest,states,cfg,task_completed=bool(report.get('sequence_completed')),joint_trace=trace,events=events,soft_metrics=np.load(run/'soft_metrics.npy'));result['original_final_validator']=final;ck('unchanged_final_ALL_COM_retention_and_physics',final['task_success'])
  old.frozen_runner_yaw_contract=runner_contract
  yaw=old.audit_postlift_yaw(run,report,states,trace,cfg,reactions);result['original_yaw_audit_with_pinned_extension_contract']=yaw;ck('original_yaw_native_audit',yaw['passed'])
  corr=old.audit_corrections(report,states,dt);rimcheck=old.audit_rim_perimeter_policy(run,report,cfg);recenter=old.audit_recenter_policy(cfg,report);result['original_sequence_policy_checks']={'corrections':corr,'rim':rimcheck,'recenter':recenter};ck('original_policy_sequence_checks',corr['passed'] and rimcheck['passed'] and recenter['passed'])
  from audit_input_manifest import audit as input_audit
  from audit_richer_contact_traces import audit as richer_audit
  ia=input_audit(run);ra=richer_audit(run);result['original_input_manifest_audit']=ia;result['original_richer_contact_audit']=ra;ck('manifest_and_richer_native_evidence',ia['all_declared_hashes_match'] and ia['all_snapshot_files_in_manifest'] and ra['held_handle_monitor']['passed'] and ra['per_link_wrench_reconstruction']['passed'])
  ck('complete_run_without_error',bool(report.get('sequence_completed')) and not report.get('error'))
  ck('all_audited_inputs_unchanged',all(sha(Path(path))==h for path,h in inputs.items()));result['input_sha256']=inputs
  result.update(status='passed' if all(checks.values()) else 'failed',passed=all(checks.values()),independent_task_success=all(checks.values()))
 except (KeyError,TypeError,ValueError,AssertionError,OSError,LookupError,IndexError,AttributeError) as exc:
  result['error']=type(exc).__name__+': '+str(exc);result['status']='incomplete' if isinstance(exc,(OSError,LookupError)) else 'failed'
 finally:
  old.frozen_runner_yaw_contract=LEGACY_CONTRACT
  result['failed_checks']=[k for k,v in checks.items() if not v]
  if result['failed_checks'] and result['status']=='incomplete':result['status']='failed_or_incomplete'
  result['limits']=['The original cap validity test is not a general body embedding/self-intersection certificate; all-node count is not whole-tetrahedron or volume containment.','The 50% interim criterion never replaces original final all-node, COM, released-finger and retention checks.','401 reference/FK checks are sampled; continuous reference rates concern commands, not every native dynamic velocity.','Failed or incomplete execution never becomes successful because a planning candidate exists.']
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('--yaw-prefix',action='store_true');args=ap.parse_args();run=args.run.resolve()
 if args.yaw_prefix:
  contract=runner_contract(run/'source/run.py',ROOT/'runs/two_panel_full_20/source/run.py')
  if not contract['passed']:raise SystemExit('Unrecognized frozen runner/helper contract')
  old.frozen_runner_yaw_contract=runner_contract;sys.argv=[str(HERE/'postlift_yaw_prefix_review.py'),str(run)]
  try:runpy.run_path(str(HERE/'postlift_yaw_prefix_review.py'),run_name='__main__')
  finally:old.frozen_runner_yaw_contract=LEGACY_CONTRACT
  return
 result=audit(run);out=run/'independent-gravity-execution-review.json'
 if out.exists():raise SystemExit('Refusing to overwrite existing independent gravity audit')
 out.write_text(json.dumps(result,indent=2,default=old._json_scalar,allow_nan=False)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'status':result['status'],'passed':result['passed'],'failed_checks':result.get('failed_checks',[]),'incomplete':result['incomplete_evidence'],'error':result.get('error')},indent=2))
if __name__=='__main__':main()
