"""Bounded CPU review of the explicit low-phase diagnostic section policy.

Actual full14 transport and entry-end geometry are inputs. The later insert
state is a frozen-translation hypothesis and is never called a physical run.
"""
from pathlib import Path
import sys,json,copy,time,hashlib
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import staged_contact_insertion as sc
P=B/'runs/two_panel_full_14';config_path=B/'config_guarded_native_low_trial.json'
config=json.loads(config_path.read_text());oldconfig=json.loads((P/'config.json').read_text())
rest=dict(np.load(P/'rest.npz'));transport=dict(np.load(P/'transport_teddy.npz'));actual=dict(np.load(P/'lower_teddy_to_rim.npz'));original=json.loads((P/'guarded_contact_candidate.json').read_text());oldentry=json.loads((P/'staged_contact_entry.json').read_text());oldcenter=json.loads((P/'staged_contact_center.json').read_text())
checks=[];stages=[]
def check(name,value):
 checks.append(dict(name=name,passed=bool(value)));print(name,bool(value),flush=True)
def run(phase,state,previous):
 start=time.monotonic();o=sc.plan_staged_contact(rest,state,config,phase=phase,previous=previous,output=P/f'native_low_CPU_{phase}.json')
 row=dict(phase=phase,status=o['status'],rejections=o['report']['rejections'],wall_seconds=time.monotonic()-start,state_origin='saved full14 transport' if phase=='entry' else 'saved full14 actual entry endpoint' if phase=='center' else 'frozen center translation with planned q; NOT a physical observation')
 report=o['report'];row['physical_state_rejections']=report.get('physical_state',{}).get('physical_state_rejections')
 for key in ['observed_bear_top_to_left_hand_clearance_m','planned_com_contact_envelope_boundary_distance_m','target_bear_bounds_world_m']:
  if key in report:row[key]=report[key]
 e=report.get('rim_height_section_envelopes',{});row['section_diagnostics']={k:e.get(k)for k in ['section_guard_enforced','section_policy_role','minimum_section_slab_overlap_fraction','maximum_outside_excursion_bound_m','allowed_outside_excursion_m','required_perimeter_ratio_to_unextended_rim','section_heuristic_violations']}
 row['left_sweep_status']=report.get('continuous_frozen_teddy_left_robot_sweep',{}).get('status');row['rigid_path_minimum_m']=report.get('robot_trajectory',{}).get('minimum_sampled_rigid_clearance_m');row['release_withdraw_preflight_present']='release_withdraw_preflight'in report
 stages.append(row);print(json.dumps(row),flush=True);return o
check('explicit_diagnostic_only',sc.staged_policy(config)['low_phase_section_gate']=='diagnostic_only')
check('default_strict',sc.staged_policy(oldconfig)['low_phase_section_gate']=='strict')
check('base_physical_policy_unchanged',sc.base_policy(config)==sc.base_policy(oldconfig))
bad=copy.deepcopy(config);bad['task']['contact_trial']['staged']['low_phase_section_gate']='bad'
try:sc.staged_policy(bad);rejected=False
except ValueError:rejected=True
check('unknown_section_gate_rejected',rejected)
check('actual_full14_strict_rejection_retained_as_evidence',oldcenter['status']=='rejected' and bool(oldcenter['report']['rim_height_section_envelopes']['section_guard_rejections']))
entry=run('entry',transport,original);check('actual_transport_entry_candidate',entry['status']=='candidate')
if entry['status']=='candidate':
 check('entry_still_enforces_section_gate',entry['report']['rim_height_section_envelopes']['section_guard_enforced'])
 check('entry_target_rotation_duration_unchanged',np.allclose(entry['stage']['right_tcp_m'],oldentry['stage']['right_tcp_m'],atol=1e-12,rtol=0) and np.allclose(entry['stage']['rotation_xyzw'],oldentry['stage']['rotation_xyzw'],atol=1e-12,rtol=0) and abs(entry['stage']['duration_s']-oldentry['stage']['duration_s'])<1e-12)
 center=run('center',actual,entry);check('actual_entry_endpoint_center_candidate',center['status']=='candidate')
 if center['status']=='candidate':
  e=center['report']['rim_height_section_envelopes'];check('center_violations_saved_nonblocking',not e['section_guard_enforced'] and bool(e['section_heuristic_violations']) and e['section_heuristic_violations']==e['section_guard_rejections'])
  check('actual_com_loop_robot_gates_retained',center['report']['planned_com_in_contact_envelope'] and center['report']['planned_com_contact_envelope_boundary_distance_m']>=.002 and center['report']['mouth']['minimum_nonadjacent_3d_edge_distance_lower_bound_m']>center['report']['mouth']['required_nonadjacent_3d_edge_separation_m'] and center['report']['continuous_frozen_teddy_left_robot_sweep']['status']=='clear' and center['report']['robot_trajectory']['minimum_sampled_rigid_clearance_m']>=.008)
  hypothetical=copy.deepcopy(actual);target=np.asarray(center['stage']['right_tcp_m']);hypothetical['bear']+=target-hypothetical['tcp_right'];hypothetical['tcp_right']=target;hypothetical['q_right'][:7]=center['stage']['right_q_end'];hypothetical['time']=float(actual['time'])+center['stage']['duration_s']
  insert=run('insert',hypothetical,center);check('frozen_hypothetical_insert_candidate',insert['status']=='candidate')
  if insert['status']=='candidate':
   check('insert_diagnostic_and_release_withdraw_retained',not insert['report']['rim_height_section_envelopes']['section_guard_enforced'] and 'release_withdraw_preflight'in insert['report'] and insert['report']['final_containment_criteria_unchanged'])
report=dict(kind='actual_full14_low_phase_section_diagnostic_CPU_review',physical_success=False,actual_center_not_executed=True,hypothetical_insert_not_physical_observation=True,all_checks_passed=all(v['passed']for v in checks),checks=checks,stages=stages,module_sha256=hashlib.sha256((B/'staged_contact_insertion.py').read_bytes()).hexdigest(),config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
(P/'native_low_CPU_review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in ['all_checks_passed','module_sha256','config_sha256']}),flush=True)
