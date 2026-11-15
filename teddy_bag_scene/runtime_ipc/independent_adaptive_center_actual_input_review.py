"""Read-only CPU regression of adaptive centering against full17 actual states.

No physics backend is initialized. Transparent core-call observation preserves
arguments, results and exception behavior and records otherwise-omitted rejects.
"""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_full_17'
sys.path.insert(0,str(ROOT))
import staged_contact_insertion as staged
from contact_insertion import _policy

config_path=ROOT/'config_guarded_adaptive_alignment_trial.json'
cfg=json.loads(config_path.read_text())
rest=dict(np.load(RUN/'rest.npz'))
transport=dict(np.load(RUN/'transport_teddy.npz'))
entry_state=dict(np.load(RUN/'lower_teddy_to_rim.npz'))
prep=json.loads((RUN/'guarded_contact_candidate.json').read_text())
prior_actual_entry=json.loads((RUN/'staged_contact_entry.json').read_text())

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_hashes(state):
 return {k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in state.items()}
before={name:array_hashes(v) for name,v in [('rest',rest),('transport',transport),('entry',entry_state)]}
cfg_before=json.dumps(cfg,sort_keys=True);prep_before=json.dumps(prep,sort_keys=True)
files=[config_path,RUN/'rest.npz',RUN/'transport_teddy.npz',RUN/'lower_teddy_to_rim.npz',
 RUN/'guarded_contact_candidate.json',RUN/'staged_contact_entry.json',ROOT/'staged_contact_insertion.py',
 ROOT/'contact_insertion.py',ROOT/'contact_robot_geometry.py',ROOT/'motion.py',ROOT/'run.py',
 ROOT/'assets/panda_tactile.xml',ROOT/'assets/two_panel_bag_wide_tall_coarse.npz']
file_hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
def differences(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:out.append(dict(path=path+'/'+k,old=a.get(k),new=b.get(k)))
   else:out.extend(differences(a[k],b[k],path+'/'+k))
  return out
 return [] if a==b else [dict(path=path,old=a,new=b)]
report=dict(kind='independent_adaptive_center_full17_actual_input_review',physical_success=False,
 runtime_execution_performed=False,source_config_or_state_files_modified=False,
 base_policy_matches_old_accepted_preparation=prep['report']['policy']==_policy(cfg),
 config_differences_from_full18=differences(json.loads((ROOT/'runs/two_panel_full_18/config.json').read_text()),cfg),
 input_sha256=file_hashes,initial_offset_count=len(cfg['task']['contact_trial']['center_offsets_xy_m']),
 low_center_offset_count=len(cfg['task']['contact_trial']['staged']['center_offsets_xy_m']),
 actual_transport_time_s=float(transport['time']),actual_entry_endpoint_time_s=float(entry_state['time']),
 actual_inputs_before=before,core_calls=[],candidate_summaries=[],
 limitations=['Uses saved full17 states; does not claim that a fresh native run reproduces this geometry.',
 'Old accepted guarded preparation is reused only because base policy is identical. New 12-offset initial alignment search is not exercised here.',
 'Only entry and center commands are planned; insertion/release/withdraw are frozen hypothetical preflights and do not authorize actual execution.',
 'No force, soft-body dynamics, release stability or final settled containment success is established.'])
out=RUN/'independent-adaptive-center-actual-input-review.json'
def save():out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
save()
assert report['base_policy_matches_old_accepted_preparation']
real_core=staged._plan_staged_contact_single
counts={'entry':0,'center':0,'insert':0}
def observed_core(*args,**kwargs):
 phase=kwargs.get('phase','entry');counts[phase]+=1
 delta=kwargs.get('center_delta_override');start=time.monotonic()
 print(json.dumps(dict(event='begin',phase=phase,index=counts[phase]-1,delta=delta)),flush=True)
 result=real_core(*args,**kwargs)
 duration=time.monotonic()-start
 filename=f'independent-adaptive-actual-{phase}-{counts[phase]-1:02d}.json'
 (RUN/filename).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 cr=result['report']
 row=dict(phase=phase,phase_call_index=counts[phase]-1,center_delta_override=delta,status=result['status'],
  observation_time_s=float(args[1]['time']),elapsed_cpu_wall_s=duration,rejections=cr['rejections'],
  minimum_sampled_rigid_clearance_m=cr.get('robot_trajectory',{}).get('minimum_sampled_rigid_clearance_m'),
  planned_com_boundary_distance_m=cr.get('planned_com_contact_envelope_boundary_distance_m',cr.get('planned_com_boundary_distance_m')),
  continuous_frozen_teddy_sweep_status=cr.get('continuous_frozen_teddy_left_robot_sweep',{}).get('status'),
  report_file=filename)
 report['core_calls'].append(row);save()
 print(json.dumps(dict(event='complete',**row)),flush=True)
 return result
staged._plan_staged_contact_single=observed_core

started=time.monotonic()
entry=staged.plan_staged_contact(rest,transport,cfg,phase='entry',previous=prep,
 output=RUN/'independent-adaptive-actual-entry.json')
report['entry_status']=entry['status']
if entry['status']=='candidate':
 report['entry_target_difference_from_original_m']=float(np.linalg.norm(np.asarray(entry['stage']['right_tcp_m'])-prior_actual_entry['stage']['right_tcp_m']))
 report['entry_new_policy_frozen']=entry['checkpoint']['staged_policy']==staged.staged_policy(cfg)
 center=staged.plan_staged_contact(rest,entry_state,cfg,phase='center',previous=entry,
  output=RUN/'independent-adaptive-actual-center.json')
 report['center_status']=center['status'];report['center_rejections']=center['report']['rejections']
 selection=center.get('center_selection',{})
 for attempt in selection.get('attempts',[]):
  cr=next(r for r in report['core_calls'] if r['phase']=='center' and r['phase_call_index']==attempt['index'])
  report['candidate_summaries'].append(dict(index=attempt['index'],offset_xy_m=attempt['center_offset_xy_m'],
   center_status=attempt['center_status'],center_rejections=attempt['center_rejections'],
   center_minimum_sampled_rigid_clearance_m=cr['minimum_sampled_rigid_clearance_m'],
   center_com_boundary_margin_m=cr['planned_com_boundary_distance_m'],
   frozen_remainder_status=attempt.get('frozen_remainder_status'),
   frozen_remainder_rejections=attempt.get('frozen_remainder_rejections'),
   minimum_center_insert_release_rigid_clearance_m=attempt.get('minimum_center_insert_release_rigid_clearance_m'),
   score_m=attempt.get('score_m')))
 report['selected_index']=selection.get('selected_index')
 report['selected_offset_xy_m']=selection.get('selected_offset_xy_m')
 report['returned_only_actual_center_stage']=bool(center.get('phase')=='center' and center.get('stage',{}).get('name')=='center_teddy_at_rim' and center.get('observed_time_s')==float(entry_state['time'])) if center['status']=='candidate' else None
 report['no_executable_remainder_stage_exposed']=not any('stage' in a or 'stages' in a for a in selection.get('attempts',[]))
 report['real_insert_requires_new_actual_observation']=bool(center.get('requires_new_observation_before_next_stage') and center.get('next_phase')=='insert' and selection.get('later_insert_requires_fresh_actual_observation'))
 if center['status']=='candidate':
  selected=selection['attempts'][selection['selected_index']]
  report['selected_frozen_remainder_is_feasible']=selected.get('frozen_remainder_status')=='candidate'
  report['selected_frozen_remainder_is_explicit_hypothesis']=selected.get('frozen_remainder_is_hypothesis_not_actual_observation') is True
  score=max((a['score_m'],-a['index']) for a in selection['attempts'] if 'score_m' in a)
  report['ranking_and_deterministic_tie_break_correct']=selection['selected_index']==-score[1]
else:report['entry_rejections']=entry['report']['rejections']
after={name:array_hashes(v) for name,v in [('rest',rest),('transport',transport),('entry',entry_state)]}
report['all_actual_input_arrays_unchanged']=before==after
report['config_and_previous_preparation_unchanged']=cfg_before==json.dumps(cfg,sort_keys=True) and prep_before==json.dumps(prep,sort_keys=True)
report['all_input_and_source_file_hashes_unchanged']=file_hashes=={str(p.relative_to(ROOT)):sha(p) for p in files}
report['elapsed_wall_s']=time.monotonic()-started
save()
print(json.dumps({k:report[k] for k in ['entry_status','center_status','selected_index','selected_offset_xy_m',
 'returned_only_actual_center_stage','selected_frozen_remainder_is_feasible','all_actual_input_arrays_unchanged',
 'config_and_previous_preparation_unchanged','all_input_and_source_file_hashes_unchanged','elapsed_wall_s'] if k in report}),flush=True)
print(str(out),flush=True)
