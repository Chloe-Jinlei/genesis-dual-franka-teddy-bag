"""CPU review: actual full13 entry endpoint; later insert is frozen hypothesis."""
from pathlib import Path
import sys,json,copy,time,hashlib
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
import staged_contact_insertion as sc
P=B/'runs/two_panel_full_13';rest=dict(np.load(P/'rest.npz'));transport=dict(np.load(P/'transport_teddy.npz'));actual=dict(np.load(P/'lower_teddy_to_rim.npz'));config=json.load(open(B/'config_guarded_spatial_rim_trial.json'));oldconfig=json.load(open(P/'config.json'));original=json.load(open(P/'guarded_contact_candidate.json'));oldentry=json.load(open(P/'staged_contact_entry.json'))
checks=[];rows=[]
def check(name,passed):
 checks.append({'name':name,'passed':bool(passed)});print(name, bool(passed),flush=True)
def run(phase,state,previous):
 t=time.monotonic();result=sc.plan_staged_contact(rest,state,config,phase=phase,previous=previous,output=P/f'spatial_trial_CPU_{phase}.json');row={'phase':phase,'state_origin':'saved actual full13 transport' if phase=='entry' else 'saved actual full13 lower_teddy_to_rim' if phase=='center' else 'frozen translation after CPU center; NOT physical observation','status':result['status'],'rejections':result['report']['rejections'],'wall_seconds':time.monotonic()-t}
 for name in ['mouth','rim_height_section_envelopes','robot_trajectory','continuous_frozen_teddy_left_robot_sweep','release_withdraw_preflight']:
  if name in result['report']:
   ev=result['report'][name];row[name]={k:v for k,v in ev.items()if k not in ['sections','ordered_rim_world_m','ordered_rim_indices','contact_envelope_xy_m','analytic_joint_extrema','withdraw_analytic_joint_extrema','solids']}
 rows.append(row);print(phase,result['status'],result['report']['rejections'],flush=True);return result
check('new_policy_explicit',sc.staged_policy(config)['low_rim_geometry']=='spatial_loop_contact_envelope')
check('default_remains_ordered_vertical_projection',sc.staged_policy(oldconfig)['low_rim_geometry']=='ordered_vertical_projection')
try:sc._staged_mouth(rest,actual,config,'entry',sc.staged_policy(config));entry_reject=False
except ValueError as e:entry_reject='self-crossed' in str(e)
check('entry_cannot_bypass_crossed_vertical_projection',entry_reject)
oldlow=sc.plan_staged_contact(rest,actual,oldconfig,phase='center',previous=oldentry)
check('legacy_actual_low_state_still_rejected',oldlow['status']=='rejected' and any('self-crossed'in s for s in oldlow['report']['rejections']))
bad=copy.deepcopy(config);bad['task']['contact_trial']['staged']['low_rim_geometry']='bad'
try:sc.staged_policy(bad);rejected=False
except ValueError:rejected=True
check('unknown_mode_rejected',rejected)
entry=run('entry',transport,original);check('actual_transport_entry_candidate',entry['status']=='candidate')
if entry['status']=='candidate':
 check('entry_target_and_duration_unchanged',np.allclose(entry['stage']['right_tcp_m'],oldentry['stage']['right_tcp_m'],atol=1e-12,rtol=0) and abs(entry['stage']['duration_s']-oldentry['stage']['duration_s'])<1e-12 and np.allclose(entry['stage']['rotation_xyzw'],oldentry['stage']['rotation_xyzw'],atol=1e-12,rtol=0))
 stale=dict(actual);stale['time']=entry['observed_time_s'];sr=sc.plan_staged_contact(rest,stale,config,phase='center',previous=entry);check('new_mode_still_requires_strictly_new_observation',sr['status']=='rejected' and any('newer'in s for s in sr['report']['rejections']))
 mismatch=copy.deepcopy(entry);mismatch['checkpoint']['staged_policy']['maximum_rim_stretch_ratio']=1.;mr=sc.plan_staged_contact(rest,actual,config,phase='center',previous=mismatch);check('policy_change_between_phases_rejected',mr['status']=='rejected' and any('policy between'in s for s in mr['report']['rejections']))
 center=run('center',actual,entry);check('actual_lowered_state_center_candidate',center['status']=='candidate')
 if center['status']=='candidate':
  ev=center['report'];check('envelope_com_and_sections_explicitly_named',ev.get('com_membership_is_not_actual_opening_containment') and 'planned_com_inside'not in ev and 'planned_com_in_contact_envelope'in ev and all('opening_overlap_fraction'not in x and 'contact_envelope_overlap_fraction'in x for x in ev['rim_height_section_envelopes']['sections']))
  check('actual_3d_rim_edges_checked',ev['mouth']['nonadjacent_3d_pair_count']==2627 and ev['mouth']['minimum_nonadjacent_3d_edge_distance_lower_bound_m']>.0006 and ev['mouth']['actual_rim_xy_projection_crossings']==2)
  hypothetical=copy.deepcopy(actual);target=np.asarray(center['stage']['right_tcp_m']);hypothetical['bear']+=target-hypothetical['tcp_right'];hypothetical['tcp_right']=target;hypothetical['q_right'][:7]=center['stage']['right_q_end'];hypothetical['time']=float(actual['time'])+center['stage']['duration_s']
  insert=run('insert',hypothetical,center);check('frozen_hypothetical_insert_candidate',insert['status']=='candidate')
  if insert['status']=='candidate':check('final_release_withdraw_rechecked','release_withdraw_preflight'in insert['report'] and insert['report']['final_containment_criteria_unchanged'])
report={'kind':'actual_full13_spatial_loop_trial_CPU_review','physical_success':False,'actual_center_not_executed':True,'hypothetical_insert_not_physical_observation':True,'checks':checks,'all_checks_passed':all(x['passed']for x in checks),'stages':rows,'module_sha256':hashlib.sha256((B/'staged_contact_insertion.py').read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((B/'config_guarded_spatial_rim_trial.json').read_bytes()).hexdigest()}
(P/'spatial_trial_CPU_review.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item())+'\n');print(json.dumps({'all_checks_passed':report['all_checks_passed'],'check_count':len(checks),'module_sha256':report['module_sha256']}))
