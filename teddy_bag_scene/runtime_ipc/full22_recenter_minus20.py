"""One explicit saved-state correction query; no simulation or source/config mutation."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_22';OUT=RUN/'independent-actual-recenter-minus20';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import recenter_guarded_contact_trial,_policy
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
paths=[RUN/n for n in ['rest.npz','transport_teddy.npz','config.json','guarded_contact_candidate.json','source/contact_insertion.py','source/motion.py','source/contact_robot_geometry.py']]+[ROOT/'run.py',ROOT/'config_guarded_upright_offset_trial.json',Path(__file__)]
hashes={str(p):sha(p) for p in paths}
assert sha(ROOT/'contact_insertion.py')==sha(RUN/'source/contact_insertion.py')
assert sha(ROOT/'motion.py')==sha(RUN/'source/motion.py')
rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'transport_teddy.npz'));cfg=json.loads((RUN/'config.json').read_text());selected=json.loads((RUN/'guarded_contact_candidate.json').read_text());trial=copy.deepcopy(cfg);trial['task']['contact_trial']['recenter_offsets_xy_m']=[[-.020,-.020]]
assert _policy(trial)==_policy(cfg)==selected['report']['policy']
assert abs(float(state['time'])-40.735)<1e-8
original_arrays={k:v.copy() for k,v in state.items()};original_rest={k:v.copy() for k,v in rest.items()}
dump(OUT/'diagnostic_config.json',trial)
report={'kind':'one_actual_full22_fixed_minus20mm_xy_recenter','physical_success':False,'native_execution':False,'actual_source_time_s':float(state['time']),'actual_state_source':str(RUN/'transport_teddy.npz'),'original_selected_candidate':str(RUN/'guarded_contact_candidate.json'),'only_config_difference':{'task.contact_trial.recenter_offsets_xy_m':[[-.020,-.020]]},'runtime_recenter_enabled_unchanged':trial['task']['contact_trial']['recenter_enabled'],'attempt_index':1,'cumulative_xy_m_before':0.,'input_sha256':hashes,'status':'planning','limitations':['The saved actual state is queried without native execution. Runtime recenter_enabled remains false in this diagnostic copy; this query alone does not authorize an execution.','Only returned preparation stages could execute in a separately authorized enabled trial; provisional insert/release/withdraw are frozen-state hypotheses and require a fresh later observation.']}
dump(OUT/'review.json',report);start=time.monotonic();result=recenter_guarded_contact_trial(rest,state,trial,selected,attempt_index=1,cumulative_xy_m=0.,output=OUT/'candidate.json')
report.update(status='bounded_cpu_query_finished',candidate_status=result['status'],planning_wall_s=time.monotonic()-start,rejections=result['report']['rejections'],all_inputs_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),actual_arrays_unchanged=all(np.array_equal(v,original_arrays[k]) for k,v in state.items()) and all(np.array_equal(v,original_rest[k]) for k,v in rest.items()),candidate_sha256=sha(OUT/'candidate.json'))
if result['status']=='candidate':
 report.update(correction_xy_m=result['correction_xy_m'],correction_distance_m=result['correction_distance_m'],cumulative_xy_m_after=result['cumulative_xy_m_after'],stages_to_execute=[s['name'] for s in result['stages_to_execute']],frozen_all_stages=[s['name'] for s in result['route']['stages']],selected_sections={k:v for k,v in result['report']['selected_sections'].items() if k!='sections'},robot_stages_summary=[{'name':s['name'],'duration_s':s['duration_s'],'right_tcp_m':s['right_tcp_m']} for s in result['route']['stages']])
else:
 report['candidates']=[{k:v for k,v in c.items() if k!='planning_report'} for c in result['report'].get('candidates',[])]
 report['candidate_failures']=[{'xy':c.get('correction_xy_m'),'rejections':c.get('planning_report',{}).get('rejections'),'attempts':[{'rejection':a['rejection'],'stages_completed':a['stages_completed']} for a in c.get('planning_report',{}).get('robot_centering_attempts',[])]} for c in result['report'].get('candidates',[])]
dump(OUT/'review.json',report)
print(json.dumps({'output':str(OUT/'review.json'),'sha256':sha(OUT/'review.json'),'status':result['status'],'rejections':report['rejections'],'all_inputs_unchanged':report['all_inputs_unchanged'],'candidate_failures':report.get('candidate_failures')},indent=2))
