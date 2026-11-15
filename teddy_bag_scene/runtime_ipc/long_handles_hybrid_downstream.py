"""Prepared single-candidate CPU entry. Default only describes the interface.

Execute only after the parent authorizes the finalized handoff state. No native
engine is imported; two actual sources are explicitly combined hypothetically.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
from scipy.interpolate import PPoly
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-run',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--candidate-config',type=Path,help='Optional sole authorized variant: X offset .015 to .005 only')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'status':'interface_only_no_inputs_read_no_planning','required_after_parent_authorizes':['--execute','--handoff-run FINALIZED_RUN','--output NEW_DIRECTORY'],'candidate':{'yaw_degrees':15,'COM_offset_xy_m':[.015,.03075]},'sequence':['two-source hybrid','original yaw helper','original preparation','command-continuous frozen preparation proxy','original fresh full-depth','original short gravity planner'],'excluded':'No actual release eligibility, native execution, dynamic shape prediction or parameter search.'},indent=2));return
    if args.handoff_run is None or args.output is None:parser.error('--execute requires --handoff-run and --output')
    run=args.handoff_run.resolve();old=ROOT/'runs/two_panel_full_25';out=args.output.resolve()
    if out.exists():raise FileExistsError('Refusing to overwrite prior diagnostic')
    required=[run/n for n in ['report.json','validation.json','config.json','rest.npz','settle_handoff.npz','input_sha256.json']]
    for p in required:
        if not p.is_file():raise ValueError('Final handoff input missing: '+str(p))
    actual_report=json.loads((run/'report.json').read_text())
    if actual_report.get('status')!='completed' or actual_report.get('requested_task')!='handoff' or actual_report.get('requested_task_passed') is not True or actual_report.get('handoff_probe_passed') is not True:
        raise ValueError('Final handoff report does not certify its limited requested scope')
    sys.path.insert(0,str(run/'source'))
    from scipy.spatial.transform import Rotation
    from motion import Planner
    from contact_insertion import plan_guarded_contact_trial,reobserve_guarded_contact_trial,_serial
    from postlift_yaw import plan_postlift_yaw,_clearance_rows
    from gravity_release import plan_gravity_release
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cfg=json.loads((run/'config.json').read_text());oldcfg=json.loads((old/'config.json').read_text());config=copy.deepcopy(cfg)
    if args.candidate_config is not None:
        candidate=json.loads(args.candidate_config.read_text());expected=copy.deepcopy(cfg);expected['task']['contact_trial']['center_offsets_xy_m']=[[.005,.03075]]
        if candidate!=expected:raise ValueError('Only the explicit X offset .015 to .005 change is authorized')
        config=candidate
    assert cfg['robots']==oldcfg['robots'] and cfg['bear']==oldcfg['bear']
    assert cfg['task']['contact_trial']['postlift_yaw']=={'world_z_degrees':15.,'duration_s':6.}
    assert cfg['task']['contact_trial']['center_offsets_xy_m']==[[.015,.03075]]
    assert cfg['task']['contact_trial']['gravity_release']==oldcfg['task']['contact_trial']['gravity_release']
    newrest=dict(np.load(run/'rest.npz',allow_pickle=False));oldrest=dict(np.load(old/'rest.npz',allow_pickle=False));bag=dict(np.load(run/'settle_handoff.npz',allow_pickle=False));bear=dict(np.load(old/'lift_teddy.npz',allow_pickle=False))
    assert np.array_equal(newrest['bear'],oldrest['bear']) and np.array_equal(newrest['bear_tets'],oldrest['bear_tets'])
    # Real simultaneous sources do not exist: these keys form a labelled hybrid.
    state=dict(time=np.asarray(max(float(bag['time']),float(bear['time']))),bag=bag['bag'].copy(),bear=bear['bear'].copy(),q_left=bag['q_left'].copy(),tcp_left=bag['tcp_left'].copy(),q_right=bear['q_right'].copy(),tcp_right=bear['tcp_right'].copy())
    rest=dict(bag=newrest['bag'].copy(),bag_faces=newrest['bag_faces'].copy(),bear=oldrest['bear'].copy(),bear_tets=oldrest['bear_tets'].copy())
    leftarchive=ROOT/'runtime_ipc/long-handles-spread-plan.npz';leftreport=json.loads((ROOT/'runtime_ipc/long-handles-cpu-validation.json').read_text())
    assert sha(ROOT/'config_long_handles_gravity_trial.json')==leftreport['asset_sha256']['config_long_handles_gravity_trial.json'], 'CPU left-command config hash changed'
    assert cfg==json.loads((ROOT/'config_long_handles_gravity_trial.json').read_text()), 'CPU command configuration differs from frozen native configuration'
    assert sha(run/'source/motion.py')==sha(ROOT/'motion.py')
    la=dict(np.load(leftarchive,allow_pickle=False));ra=dict(np.load(old/'postlift_yaw_plan.npz',allow_pickle=False));commands={'left':PPoly(la['left_c'],la['left_x'])(1.),'right':PPoly(ra['right_c'],ra['right_x'])(0.)}
    assert np.array_equal(commands['left'],leftreport['open_route']['end_q']['left'])
    assert np.array_equal(commands['right'],json.loads((old/'postlift_yaw_candidate.json').read_text())['command_source_poses']['right']['q'])
    config['robots']['mjcf_path']=str(run/'source/assets/panda_tactile.xml')
    for k in ['npz_file','metadata']:config['bag'][k]=str(run/'source/assets'/Path(config['bag'][k]).name)
    tracked=required+[old/n for n in ['lift_teddy.npz','rest.npz','postlift_yaw_plan.npz','postlift_yaw_candidate.json']]+[leftarchive,ROOT/'runtime_ipc/long-handles-cpu-validation.json',ROOT/'config_long_handles_gravity_trial.json']+[run/'source'/n for n in ['motion.py','contact_insertion.py','postlift_yaw.py','gravity_release.py','contact_robot_geometry.py','validate.py']]
    if args.candidate_config is not None:tracked.append(args.candidate_config.resolve())
    before={str(p):sha(p) for p in tracked};out.mkdir(parents=True)
    np.savez_compressed(out/'hybrid_source.npz',**state);np.savez_compressed(out/'rest.npz',**rest);(out/'diagnostic_config.json').write_text(json.dumps(config,indent=2)+'\n')
    report=dict(status='yaw_pending',physical_success=False,native_execution_performed=False,not_a_joint_native_snapshot=True,sole_COM_offset_xy_m=config['task']['contact_trial']['center_offsets_xy_m'][0],candidate_config_path=None if args.candidate_config is None else str(args.candidate_config.resolve()),candidate_config_scope='Only COM X .015 to .005 differs from source handoff config' if args.candidate_config is not None else 'Original source handoff config unchanged',source_bag_left={'path':str(run/'settle_handoff.npz'),'time_s':float(bag['time'])},source_bear_right={'path':str(old/'lift_teddy.npz'),'time_s':float(bear['time'])},synthetic_start_time_s=float(state['time']),timestamp_scope='max actual source clocks, then integer dt bookkeeping; no synchronized physical history',command_sources={'left':'CPU long-handle spread command endpoint, left stays constant through right clearance/hold; actual compatibility checked by original yaw helper','right':'full25 native postlift archived command start, equal to lift command endpoint'},input_sha256=before,limitations=['Bag and teddy are from separate native runs; combined robot/soft state is hypothetical and may not be reachable dynamically.','No force arrays inherited, no grip/force provenance asserted for combined state.','Rigid-grasp frozen transforms are planning assumptions, not future cloth/teddy dynamics.','No actual partial-capture/release check is called on synthetic data; original final native ALL-inside/retention remains mandatory.','Only +15 degrees and the one explicitly recorded COM offset, with all original helper gates; first rejection stops.'])
    def save():
        report['inputs_unchanged']=all(sha(p)==h for p,h in before.items());(out/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
    def record(name,value):(out/(name+'.json')).write_text(json.dumps(_serial(value),indent=2,allow_nan=False)+'\n')
    def archive(name,plan):np.savez_compressed(out/(name+'.npz'),duration=plan.duration,**{s+'_x':plan.curves[s].x for s in commands},**{s+'_c':plan.curves[s].c for s in commands})
    def advance(origin,qs,time_s,planner):
        h={k:v.copy() for k,v in origin.items()};p0,R0=planner.fk('right',origin['q_right'][:7]);p1,R1=planner.fk('right',qs['right']);h['bear']=(R1*R0.inv()).apply(origin['bear']-origin['tcp_right'])+p1
        for s in qs:h['q_'+s][:7]=qs[s];h['tcp_'+s]=planner.fk(s,qs[s])[0]
        h['time']=np.asarray(time_s);return h
    save();stage='yaw'
    try:
        yaw,plan=plan_postlift_yaw(rest,state,config,commands);record('yaw_candidate',yaw)
        if yaw['status']!='candidate' or plan is None:raise ValueError('Yaw rejected: '+str(yaw.get('rejections')))
        planner=Planner(config);dt=float(config['simulation']['dt']);archive('yaw_plan',plan);commands=plan.end_q();state=advance(state,commands,float(state['time'])+int(np.ceil(plan.duration/dt))*dt,planner);np.savez_compressed(out/'hypothetical_after_yaw.npz',**state)
        stage='preparation';choice=plan_guarded_contact_trial(rest,state,config,output=out/'preparation.json')
        if choice['status']!='candidate':raise ValueError('Preparation rejected: '+str(choice['report'].get('rejections'))+' '+str([{k:r.get(k) for k in ['rejection','stages_completed']} for r in choice['report'].get('robot_centering_attempts',[])]))
        stage='preparation_command_proxy';report['preparation_command_proxy']=[];time_s=float(state['time']);fingers={s:float(np.clip(state['q_'+s][7:].mean(),0,.04)) for s in commands}
        for i,st in enumerate(choice['stages_to_execute']):
            plan=planner.trajectory(commands,{'right':st['right_tcp_m']},duration=st['duration_s'],rotations={'right':Rotation.from_quat(st['rotation_xyzw'])},finger_start=fingers,finger_end=fingers,clearance_margin=.008,validate_samples=201,max_joint_velocity=.35,max_joint_acceleration=.5)
            minimum=float('inf')
            for t in np.linspace(0,plan.duration,201):
                sample=plan.sample(float(t));_,r=_clearance_rows(planner.clearance(sample['q'],sample['finger']),st['name']);minimum=min(minimum,r['distance'])
            n=int(np.ceil(plan.duration/dt));time_s+=n*dt;commands=plan.end_q();archive('preparation_proxy_'+str(i),plan);report['preparation_command_proxy'].append(dict(name=st['name'],duration_s=plan.duration,integer_steps=n,minimum_rigid_clearance_m=minimum,rates=plan.continuous_rate_extrema()))
        state=advance(state,commands,time_s,planner);np.savez_compressed(out/'hypothetical_transport_endpoint.npz',**state);report['status']='fresh_pending';save();print('TRANSPORT_READY',out/'hypothetical_transport_endpoint.npz',flush=True)
        stage='fresh_full_depth';fresh=reobserve_guarded_contact_trial(rest,state,config,choice,output=out/'hypothetical_fresh_full_depth.json')
        if fresh['status']!='candidate':raise ValueError('Fresh rejected: '+str(fresh['report'].get('rejections')))
        stage='short_gravity';gravity,plan=plan_gravity_release(rest,state,config,fresh,command_qs=commands,return_plan=True);record('gravity_candidate',gravity)
        if gravity['status']!='candidate' or plan is None:raise ValueError('Gravity planner did not return a candidate plan')
        archive('gravity_insert_plan',plan);report.update(status='all_original_helper_stages_candidate_frozen_only',gravity_target_m=gravity['insert_stage']['right_tcp_m'],gravity_duration_s=plan.duration)
    except Exception as exc:report.update(status='rejected',rejected_stage=stage,rejection_type=type(exc).__name__,rejection=str(exc))
    save();print(json.dumps({k:report.get(k) for k in ['status','rejected_stage','rejection','gravity_target_m','gravity_duration_s','inputs_unchanged']},indent=2));print('REPORT',out/'review.json',sha(out/'review.json'),flush=True)
if __name__=='__main__':main()
