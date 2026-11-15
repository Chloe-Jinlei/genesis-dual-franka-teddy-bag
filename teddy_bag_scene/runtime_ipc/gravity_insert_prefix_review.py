"""One bounded CPU audit of a saved gravity-insert source/candidate/curve.
No final-report requirement, no native execution, no new motion planning.
"""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
import audit_gravity_execution as audit
from audit_input_manifest import audit as manifest_audit

def review(run):
 run=Path(run).resolve();names=['config.json','input_sha256.json','rest.npz','transport_teddy.npz','guarded_contact_candidate.json','guarded_contact_reobserved.json','gravity_release_candidate.json','gravity_insert_plan.npz']
 before={name:audit.sha(run/name) for name in names};read=lambda n:json.loads((run/n).read_text());cfg=read('config.json');manifest=read('input_sha256.json');c=read('gravity_release_candidate.json');full=read('guarded_contact_reobserved.json');prep=read('guarded_contact_candidate.json');source=dict(np.load(run/'transport_teddy.npz',allow_pickle=False));rest=dict(np.load(run/'rest.npz',allow_pickle=False));checks={}
 def ck(k,v):checks[k]=bool(v)
 result={'kind':'independent_actual_gravity_insert_prefix_review','physical_success':False,'native_execution_performed_by_review':False,'overall_execution_audit_passed':False,'checks':checks,'scope':'Exactly the saved actual transport observation and the already-created short-insert cubic archive. No final report, actual release or future endpoint is inferred.'}
 contract=audit.runner_contract(run/'source/run.py',ROOT/'runs/two_panel_full_20/source/run.py');result['runner_static_contract']=contract;ck('exact_reviewed_runner_and_helper',contract['passed'])
 ck('approved_production_hashes_in_manifest',audit.approved_manifest_sources(run,manifest))
 ia=manifest_audit(run);result['input_manifest_review']=ia;ck('all_frozen_sources_assets_config_match_manifest',ia['all_declared_hashes_match'] and ia['all_snapshot_files_in_manifest'] and not ia['missing_external_model_files'])
 if not all(checks.values()):raise ValueError('Frozen source contract rejected')
 p=audit.policy(cfg);ck('explicit_gravity_policy_exact',p is not None and c['policy']==p)
 ck('candidate_finite_and_not_physical_success',audit.finite(c) and c['kind']=='explicit_gravity_seating_release' and c['status']=='candidate' and c['physical_success'] is False)
 ck('actual_source_time_matches_all_candidates',float(source['time'])==c['observed_time_s']==full['observed_time_s'])
 ck('actual_source_q_and_TCP_exact',all(np.array_equal(c['actual_source_arms'][s]['q'],source['q_'+s]) and np.array_equal(c['actual_source_arms'][s]['tcp_m'],source['tcp_'+s]) for s in ('left','right')))
 ck('original_full_depth_fresh_preflight_retained',full['status']=='candidate' and not full['report']['rejections'] and full['physical_success'] is False and full['requires_final_actual_3d_containment_audit'] is True and full['report']['policy']==prep['report']['policy'])
 pol=full['report']['policy'];ck('original_rigid_soft_material_thresholds_unchanged',pol['rigid_clearance_m']==.008 and pol['left_robot_soft_sweep_clearance_m']==.005 and pol['minimum_tetra_volume_ratio']==.05 and pol['maximum_cloth_edge_stretch_ratio']==1.15 and pol['maximum_table_penetration_m']==.002)
 motion=audit.old.load_module(run/'source/motion.py','gravity_prefix_frozen_motion');pc=copy.deepcopy(cfg);pc['robots']['mjcf_path']=str(run/'source/assets'/Path(cfg['robots'].get('mjcf_path','panda_tactile.xml')).name);planner=motion.Planner(pc)
 actual=audit.actual_state_review(run,cfg,rest,source,planner);result['actual_source_physical_review']=actual;ck('actual_source_original_physical_checks',actual['passed'])
 low=np.asarray(full['insert_stage']['right_tcp_m']);target=np.asarray(c['insert_stage']['right_tcp_m']);R=Rotation.from_quat(c['insert_stage']['rotation_xyzw']);actual_R=planner.fk('right',source['q_right'][:7])[1]
 ck('target_exact_1p085m_and_fixed_actual_XY',target.shape==(3,) and target[2]==1.085 and abs(target[2]-cfg['table']['height']-p['tcp_height_above_table_m'])<=1e-12 and np.array_equal(target[:2],source['tcp_right'][:2]))
 ck('short_target_strictly_between_original_depth_and_source',np.array_equal(low,c['original_full_depth_tcp_m']) and np.max(abs(low[:2]-target[:2]))<=1e-9 and low[2]+.02<target[2]<source['tcp_right'][2]-.04)
 ck('full_actual_orientation_and_outside_target_preserved',(actual_R*R.inv()).magnitude()<=1e-12 and c['insert_stage']['rotation_xyzw']==full['insert_stage']['rotation_xyzw'] and c['outside_tcp_m']==full['outside_tcp_m'])
 command_errors={}
 for s in ('left','right'):
  q=np.asarray(c['command_q_start'][s]);ap,ar=planner.fk(s,source['q_'+s][:7]);cp,cr=planner.fk(s,q)
  command_errors[s]={'position_m':float(np.linalg.norm(cp-ap)),'orientation_rad':float((cr*ar.inv()).magnitude())}
  ck(s+'_command_start_matches_actual_pose',q.shape==(7,) and np.isfinite(q).all() and command_errors[s]['position_m']<=.0005 and command_errors[s]['orientation_rad']<=.002)
 result['command_actual_source_pose_errors']=command_errors
 ar,curves=audit.archive_review(run/'gravity_insert_plan.npz',planner,motion,source,c['command_q_start'],c['command_q_end'],target,R,c['insert_curve_review']);result['saved_insert_archive_review']=ar;ck('archived_same_curve_raw401_continuous_rates_full_R_and_endpoints',ar['passed'])
 ck('archived_duration_equals_candidate_and_at_least_6s',ar['duration_s']==c['insert_stage']['duration_s'] and audit.duration_valid(ar['duration_s'],'insert_teddy'))
 ck('closed_original_motor_targets',ar['command_finger_targets']=={'left':0.,'right':cfg['task'].get('bear_finger_target',.027)})
 ck('zero_reference_endpoint_velocity_acceleration',c['continuous_reference_zero_endpoint_velocity_acceleration'] is True and ar['checks']['zero_reference_endpoint_velocity_acceleration'])
 sweeps=c['left_robot_soft_sweeps'];ck('both_actual_and_commanded_left_sweeps_present',set(sweeps)=={'actual','commanded'});summaries={}
 for label in ('actual','commanded'):
  sw=sweeps[label];q=np.asarray(source['q_left']).copy()
  if label=='commanded':q[:7]=c['command_q_start']['left']
  ok=audit.finite(sw) and sw['status']=='clear' and sw['physical_success'] is False and sw['complete_robot_certificate'] is True and not sw['rejections'] and sw['required_gap_m']==.006 and np.array_equal(sw['q_left_measured'],q) and np.array_equal(sw['translation_m'],np.zeros(3)) and np.max(abs(np.asarray(sw['displacement_m'])-(target-source['tcp_right'])))<=1e-12 and sw['input_vertex_count']==len(source['bear']) and sw['tetrahedron_count']==len(rest['bear_tets']) and sw['conservative_gap_lower_bound_m']>=.006
  ok=ok and len(sw['per_geom'])==sw['left_collision_geom_count'] and len({v['geom_id'] for v in sw['per_geom']})==len(sw['per_geom']) and all(v['sweep_certificate']['status']=='clear' and v['sweep_certificate']['conservative_gap_lower_bound_m']>=.006 for v in sw['per_geom'])
  ck(label+'_left_whole_teddy_sweep_metadata_exact',ok);summaries[label]={'minimum_certificate_gap_m':sw['conservative_gap_lower_bound_m'],'required_gap_m':sw['required_gap_m'],'left_geom_count':sw['left_collision_geom_count'],'input_vertex_count':sw['input_vertex_count'],'tetrahedron_count':sw['tetrahedron_count'],'displacement_m':sw['displacement_m'],'per_geom_statuses':[v['sweep_certificate']['status'] for v in sw['per_geom']]}
 result['saved_sweep_metadata_review']=summaries
 proof=c['insert_curve_review'];rate_match=[]
 for side,b in ar['bounds'].items():
  declared=proof['continuous_rate_bounds'][side]
  rate_match.append(np.all(np.asarray(declared['velocity_upper_bound_rad_s'])>=b['velocity']-1e-12) and np.all(np.asarray(declared['acceleration_upper_bound_rad_s2'])>=b['acceleration']-1e-12) and max(declared['velocity_upper_bound_rad_s'])<=.35 and max(declared['acceleration_upper_bound_rad_s2'])<=.5)
 ck('reported_continuous_bounds_cover_independent_bounds',all(rate_match))
 ck('partial_capture_not_replacing_final_ALL_and_retention',c['requires_actual_partial_capture_before_release'] is True and c['requires_unchanged_final_full_containment_and_retention'] is True)
 ck('read_inputs_unchanged',all(audit.sha(run/n)==h for n,h in before.items()))
 dt=float(cfg['simulation']['dt']);steps=int(np.ceil(ar['duration_s']/dt));result.update(status='available_prefix_checks_passed' if all(checks.values()) else 'failed',available_checks_passed=all(checks.values()),passed_check_count=sum(checks.values()),check_count=len(checks),failed_checks=[k for k,v in checks.items() if not v],actual_source_time_s=float(source['time']),actual_source_TCP_m=source['tcp_right'],actual_source_q_right=source['q_right'],short_target_TCP_m=target,retained_original_full_depth_TCP_m=low,planned_duration_s=ar['duration_s'],fixed_dt_s=dt,planned_integration_steps=steps,hypothetical_endpoint_time_s=float(source['time'])+steps*dt,input_sha256=before,auditor_sha256=audit.sha(HERE/'audit_gravity_execution.py'),script_sha256=audit.sha(Path(__file__)),pending=['Native short-insert endpoint and physical tracking','Fresh actual capture, stop velocity and original two-sided grip conditions before deliberate release','Actual opening, withdrawal, final ALL-node/COM containment and released retention'],limitations=['Frozen same-object source dataflow and archive are checked; no independent per-step commanded-joint log exists.','401 rigid/FK checks are sampled; continuous bounds apply to the reference joint curve.','Soft-sweep source/pose/segment/count/complete-status metadata are checked, not a rerun of its separation-axis search.','Frozen shape translation cannot predict native deformation, coupled cloth contact or success.'])
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();out=args.run/'independent-gravity-insert-prefix-review.json'
 if out.exists():raise SystemExit('Refusing overwrite')
 result=review(args.run);out.write_text(json.dumps(result,indent=2,default=audit.old._json_scalar,allow_nan=False)+'\n');print(json.dumps({'path':str(out.resolve()),'sha256':audit.sha(out),'status':result['status'],'checks':[result['passed_check_count'],result['check_count']],'failed':result['failed_checks'],'duration_s':result['planned_duration_s'],'expected_end_s':result['hypothetical_endpoint_time_s'],'min_rigid_m':result['saved_insert_archive_review']['minimum_rigid_gap_m']},indent=2))
