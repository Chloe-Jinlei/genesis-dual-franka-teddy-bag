"""One +15 degree / original COM-offset frozen candidate, no native execution."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_25';OUT=RUN/'independent-plus15-original-offset-review';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from postlift_yaw import plan_postlift_yaw
from contact_insertion import plan_guarded_contact_trial,reobserve_guarded_contact_trial,_serial
from motion import Planner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads((RUN/'config.json').read_text());config=copy.deepcopy(cfg);config['task']['contact_trial']['postlift_yaw']={'world_z_degrees':15.,'duration_s':6.}
assert config['task']['contact_trial']['center_offsets_xy_m']==[[.015,.03075]]
config['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for k in ['npz_file','metadata']:config['bag'][k]=str(RUN/'source/assets'/Path(config['bag'][k]).name)
(OUT/'diagnostic_config.json').write_text(json.dumps(config,indent=2)+'\n')
rest=dict(np.load(RUN/'rest.npz',allow_pickle=False));actual=dict(np.load(RUN/'lift_teddy.npz',allow_pickle=False));archive=dict(np.load(RUN/'postlift_yaw_plan.npz',allow_pickle=False));saved=json.loads((RUN/'postlift_yaw_candidate.json').read_text());commands={s:PPoly(archive[s+'_c'],archive[s+'_x'])(0.) for s in ['left','right']}
assert all(np.array_equal(commands[s],saved['command_source_poses'][s]['q']) for s in commands)
assert abs(float(actual['time'])-saved['observed_time_s'])<1e-9
np.savez_compressed(OUT/'rest.npz',**rest)
report=dict(status='yaw_pending',physical_success=False,native_execution_performed=False,source_lift_path=str(RUN/'lift_teddy.npz'),source_lift_sha256=sha(RUN/'lift_teddy.npz'),source_config_sha256=sha(RUN/'config.json'),source_time_s=float(actual['time']),sole_yaw_degrees=15.,sole_COM_offset_xy_m=[.015,.03075],command_source='Archived actual full25 postlift first-knot command, cross-checked to saved command_source_poses.',limitations=['Only lift source is actual native data. Subsequent states are frozen rigid-grasp hypotheses, not deformation/contact/slip predictions.','All original helper gates retained. No handle clearance or actual final containment follows.','Provisional preparation clock uses helper-reported duration and integer dt; native move_guarded may retime/replan from its command source and remeasure fingers.','No new angles or COM offsets beyond this one candidate.'])
def save(): (OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
def emit(tag,data):print(tag+' '+json.dumps(_serial(data),allow_nan=False),flush=True)
save();yaw,plan=plan_postlift_yaw(rest,actual,config,commands);(OUT/'yaw_candidate.json').write_text(json.dumps(yaw,indent=2,allow_nan=False)+'\n');report['yaw_status']=yaw['status'];report['yaw_rejections']=yaw['rejections'];save();emit('YAW',{'status':yaw['status'],'rejections':yaw['rejections']})
if plan is None:report['status']='rejected_yaw';save();raise SystemExit(0)
np.savez_compressed(OUT/'hypothetical_yaw_plan.npz',duration=plan.duration,**{s+'_x':plan.curves[s].x for s in commands},**{s+'_c':plan.curves[s].c for s in commands})
planner=Planner(config);_,R0=planner.fk('right',actual['q_right'][:7]);end=plan.end_q();p1,R1=planner.fk('right',end['right']);hyp={k:np.array(actual[k],copy=True) for k in ['bag','bear','q_left','q_right','tcp_left','tcp_right','time']}
for s in end:hyp['q_'+s][:7]=end[s];hyp['tcp_'+s]=planner.fk(s,end[s])[0]
hyp['bear']=(R1*R0.inv()).apply(actual['bear']-actual['tcp_right'])+p1;dt=float(cfg['simulation']['dt']);steps=int(np.ceil(plan.duration/dt));hyp['time']=np.array(float(actual['time'])+steps*dt)
np.savez_compressed(OUT/'hypothetical_after_yaw.npz',**hyp)
report['hypothetical_yaw_construction']=dict(formula='(endFK_R * inverse(actualLiftFK_R)).apply(actualBear - actualLiftTCP) + endFK_TCP',integer_steps=steps,time_increment_s=steps*dt,time_s=float(hyp['time']),bag_unchanged=np.array_equal(hyp['bag'],actual['bag']),no_force_arrays=True)
report['status']='preparation_pending';save();choice=plan_guarded_contact_trial(rest,hyp,config,output=OUT/'preparation.json');report['preparation_status']=choice['status'];report['preparation_rejections']=choice['report'].get('rejections',[]);report['robot_attempts']=[{k:r.get(k) for k in ['rejection','stages_completed']} for r in choice['report'].get('robot_centering_attempts',[])];save();emit('PREP',{'status':choice['status'],'rejections':report['preparation_rejections'],'robot_attempts':report['robot_attempts']})
if choice['status']!='candidate':report['status']='rejected_preparation';save();raise SystemExit(0)
prepared={k:v.copy() for k,v in hyp.items()};clocks=[];steps=0
for stage in choice['stages_to_execute']:
 n=int(np.ceil(stage['duration_s']/dt));steps+=n;clocks.append(dict(stage=stage['name'],reported_duration_s=stage['duration_s'],integer_steps=n));prepared['q_right'][:7]=stage['right_q_end']
pend,Rend=planner.fk('right',prepared['q_right'][:7]);prepared['tcp_right']=pend;prepared['bear']=(Rend*R1.inv()).apply(hyp['bear']-hyp['tcp_right'])+pend;prepared['time']=np.array(float(hyp['time'])+steps*dt)
np.savez_compressed(OUT/'hypothetical_transport_endpoint.npz',**prepared);report.update(status='hypothetical_fresh_pending',preparation_clock=clocks,hypothetical_transport_time_s=float(prepared['time']),hypothetical_transport_path=str(OUT/'hypothetical_transport_endpoint.npz'));save();emit('TRANSPORT_READY',{'path':report['hypothetical_transport_path'],'source':str(OUT/'hypothetical_after_yaw.npz'),'config':str(OUT/'diagnostic_config.json'),'preparation':str(OUT/'preparation.json')})
fresh=reobserve_guarded_contact_trial(rest,prepared,config,choice,output=OUT/'hypothetical_fresh_full_depth.json');report.update(status='candidate_frozen_only' if fresh['status']=='candidate' else 'rejected_hypothetical_fresh',hypothetical_fresh_status=fresh['status'],hypothetical_fresh_rejections=fresh['report'].get('rejections',[]),fresh_robot_attempts=[{k:r.get(k) for k in ['rejection','stages_completed']} for r in fresh['report'].get('robot_centering_attempts',[])]);save();emit('FINAL',{'status':report['status'],'rejections':report['hypothetical_fresh_rejections'],'path':str(OUT/'review.json'),'sha256':sha(OUT/'review.json')})
