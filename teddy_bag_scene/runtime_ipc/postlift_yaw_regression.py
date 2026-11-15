"""Bounded CPU postlift helper regression: two actual lifts and invalid inputs."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import sys
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import postlift_yaw as helper
from motion import Planner, DOWN
from contact_grasp import cloth_grasp_frame


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')


def replay_lift_command(run,cfg):
    """Rebuild original finite reference commands; does not advance physics."""
    spec=importlib.util.spec_from_file_location('postlift_regression_historical_motion',run/'source/motion.py')
    frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
    planner=frozen.Planner(cfg);initial=dict(np.load(run/'initial.npz',allow_pickle=False))
    qs={s:initial['q_'+s][:7].copy() for s in ('left','right')}
    finger={s:float(initial['q_'+s][7:].mean()) for s in qs}
    with np.load(run/'source/assets'/cfg['bag']['npz_file'],allow_pickle=False) as bag:
        rotations={s:cloth_grasp_frame(bag['points'],bag['faces'],bag[cfg['task']['handle_grasps'][s]['group']],
            shell_radius=cfg['bag']['thickness'],preferred_approach=cfg['task']['handle_grasps'][s]['preferred_approach']).rotation for s in qs}
        hold=cloth_grasp_frame(bag['points'],bag['faces'],bag[cfg['task']['handle_grasps']['left']['group']],
            shell_radius=cfg['bag']['thickness'],preferred_approach=cfg['task']['left_hold_preferred_approach']).rotation
    original=json.loads((run/'report.json').read_text());target=np.asarray(original['teddy_grasp_target'])
    previous='initial';rows=[]
    for event in original['events']:
        name=event['phase']
        if name in ('close_both_handles','drape_under_gravity','settle_oriented_bag','release_right_handle','close_teddy_gripper'):
            if name=='close_both_handles':finger=dict(left=0.,right=0.)
            if name=='release_right_handle':finger['right']=.04
            if name=='close_teddy_gripper':finger['right']=cfg['task']['bear_finger_target']
            previous=name;continue
        actual=np.load(run/(previous+'.npz'),allow_pickle=False)
        measured={s:float(np.clip(actual['q_'+s][7:].mean(),0.,.04)) for s in qs}
        if name=='orient_held_bag':
            plan=planner.trajectory(qs,{'left':cfg['task']['open_targets']['left']},duration=cfg['task']['left_hold_rotation_duration_s'],
                rotations={'left':hold},finger_start=measured,finger_end=measured,clearance_margin=.008,max_joint_velocity=.25,max_joint_acceleration=.4)
            rotations['left']=hold
        else:
            if name=='spread_bag':targets=cfg['task']['open_targets'];duration=2.5
            elif name=='right_clear_handle':targets={'right':[.12,.20,1.32]};duration=2.;rotations['right']=DOWN
            elif name=='teddy_pregrasp':targets={'right':target+[0,0,.15]};duration=3.;rotations['right']=DOWN
            elif name=='teddy_grasp_approach':targets={'right':target};duration=2.;rotations['right']=DOWN
            elif name=='lift_teddy':targets={'right':[target[0],target[1],1.37]};duration=4.
            else:raise ValueError('Unexpected original command phase '+name)
            speed=cfg['task']['motion_speed_scale']
            plan=planner.trajectory(qs,targets,duration=duration/speed,rotations={s:rotations[s] for s in targets},
                finger_start=finger,finger_end=finger,clearance_margin=.008,max_joint_velocity=.55*speed,max_joint_acceleration=speed**2)
        error=float(plan.duration-event['duration'])
        if abs(error)>1e-8:raise ValueError('Original duration did not reconstruct: '+name+' '+str(error))
        qs=plan.end_q();rows.append(dict(phase=name,duration_s=plan.duration,recorded_duration_s=event['duration'],difference_s=error))
        previous=name
        if name=='lift_teddy':return qs,rows
    raise ValueError('Actual original lift event missing')


def main():
    base=json.loads((ROOT/'config_guarded_adaptive_alignment_trial.json').read_text())
    cfg=copy.deepcopy(base);cfg['task']['contact_trial']['postlift_yaw']={'world_z_degrees':15.,'duration_s':6.}
    source_hash=sha(ROOT/'postlift_yaw.py');checks=[];cases=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    fixtures=[]
    for number in (17,18):
        run=ROOT/f'runs/two_panel_full_{number}'
        original=json.loads((run/'config.json').read_text())
        command,replay=replay_lift_command(run,original)
        state=dict(np.load(run/'lift_teddy.npz',allow_pickle=False));rest=dict(np.load(run/'rest.npz',allow_pickle=False))
        before={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()}
        out,plan=helper.plan_postlift_yaw(rest,state,cfg,command)
        check(f'full{number}_actual_lift_and_rebuilt_commands_accepted',out['status']=='candidate' and plan is not None)
        check(f'full{number}_report_serializes_finite',bool(json.dumps(out,allow_nan=False)))
        check(f'full{number}_actual_state_unchanged',before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in state.items()})
        if plan is not None:
            check(f'full{number}_exact_command_start',all(np.array_equal(plan.sample(0.)['q'][s],command[s]) for s in command))
            check(f'full{number}_returned_same_plan_end_matches_report',all(np.array_equal(plan.end_q()[s],out[s+'_q_end']) for s in command))
            check(f'full{number}_both_left_poses_checked',out['frozen_bear_left_rotation_sweep']['left_poses_checked']==['actual','commanded'] and out['robot_preflight']['left_poses_checked']==['actual','commanded'])
            # Dense observations are an independent sanity check of the analytic bound,
            # never the reason for accepting continuous rates.
            rates=out['robot_preflight']['continuous_rate_bounds']
            below=True
            for t in np.linspace(0,plan.duration,10001):
                sample=plan.sample(float(t))
                for side in command:
                    below=below and np.all(np.abs(sample['qd'][side])<=np.asarray(rates[side]['velocity_upper_bound_rad_s'])) and np.all(np.abs(sample['qdd'][side])<=np.asarray(rates[side]['acceleration_upper_bound_rad_s2']))
            check(f'full{number}_dense_rates_below_analytic_bounds',below)
        evidence=dict(kind='postlift_yaw_actual_lift_helper_regression',physical_success=False,
            helper_source_sha256=source_hash,input_sha256={n:sha(run/n) for n in ('config.json','rest.npz','lift_teddy.npz','initial.npz','report.json')},
            command_qs={s:q.tolist() for s,q in command.items()},reconstructed_command_prefix=replay,helper_report=out)
        evidence_path=ROOT/f'runtime_ipc/postlift-yaw30-full{number}-helper-review.json'
        save(evidence_path,evidence)
        cases.append(dict(run=number,status=out['status'],duration_s=out.get('duration_s'),rejections=out['rejections'],report=str(evidence_path)))
        fixtures.append((rest,state,command))
    rest,state,command=fixtures[0]
    def reject(name,cfg_override=None,state_override=None,cmd_override=None,patcher=None):
        args=(rest,state if state_override is None else state_override,cfg if cfg_override is None else cfg_override,command if cmd_override is None else cmd_override)
        if patcher is None:r,p=helper.plan_postlift_yaw(*args)
        else:
            with patcher:r,p=helper.plan_postlift_yaw(*args)
        check(name,r['status']=='rejected' and p is None and bool(r['rejections']) and bool(json.dumps(r,allow_nan=False)))
    for field,value in [('world_z_degrees',0),('world_z_degrees',30.001),('world_z_degrees',-30.001),
                        ('world_z_degrees',float('nan')),('world_z_degrees',float('inf')),('world_z_degrees',True),
                        ('world_z_degrees','15'),('duration_s',5.999),('duration_s',float('inf')),('duration_s',float('nan'))]:
        c=copy.deepcopy(cfg);c['task']['contact_trial']['postlift_yaw'][field]=value
        reject('invalid_'+field+'_'+str(value),cfg_override=c)
    for value in (None,[],{},dict(world_z_degrees=15.,duration_s=6.,extra=1)):
        c=copy.deepcopy(cfg);c['task']['contact_trial']['postlift_yaw']=value
        reject('malformed_option_'+str(value),cfg_override=c)
    reject('missing_option_fails_if_explicitly_called',cfg_override=base)
    for value in (None,[],{},dict(left=[0]*7),dict(left=[0]*7,right=[0]*6),dict(left=[0]*7,right=[0]*8),
                  dict(left=[0]*7,right=[float('nan')]*7),dict(left=[0]*7,right=[float('inf')]*7)):
        if value is None:
            r,p=helper.plan_postlift_yaw(rest,state,cfg,None);check('command_none_rejected',r['status']=='rejected' and p is None)
        else:reject('invalid_command_'+str(value),cmd_override=value)
    bad=copy.deepcopy(command);bad['right'][0]+=.10;reject('command_actual_pose_mismatch',cmd_override=bad)
    bad=copy.deepcopy(command);bad['left'][0]+=.10;reject('left_command_actual_pose_mismatch',cmd_override=bad)
    bad={k:v.copy() for k,v in state.items()};bad['bear'][0,0]=float('nan');reject('actual_soft_nan',state_override=bad)
    bad={k:v.copy() for k,v in state.items()};bad['time']=np.array(float('inf'));reject('actual_time_inf',state_override=bad)
    bad={k:v.copy() for k,v in state.items()};bad['bear'][:,2]-=.6;reject('not_lifted_physical_state',state_override=bad)
    original_clearance=Planner.clearance
    for value in (float('nan'),float('inf'),.007999):
        counter=[0]
        def injected(self,*args,**kwargs):
            rows=original_clearance(self,*args,**kwargs);counter[0]+=1
            if counter[0]>=2:rows['arms']['distance']=value
            return rows
        reject('later_raw_rigid_'+str(value)+'_fails_closed',patcher=patch.object(Planner,'clearance',injected))
    for cert in (dict(status='clear',conservative_gap_lower_bound_m=float('nan')),
                 dict(status='clear',conservative_gap_lower_bound_m=.01,hidden_raw=float('inf')),
                 dict(status='clear',conservative_gap_lower_bound_m=.004),
                 dict(status='not_certified',conservative_gap_lower_bound_m=.10)):
        reject('rotation_certificate_'+str(cert)+'_fails_closed',patcher=patch.object(helper,'convex_clearance_certificate',lambda *args,**kwargs:cert))
    def missing_arms(self,*args,**kwargs):
        rows=original_clearance(self,*args,**kwargs);del rows['arms'];return rows
    reject('missing_rigid_category_fails_closed',patcher=patch.object(Planner,'clearance',missing_arms))
    original_cartesian=Planner.plan_cartesian
    for mode in ('short_domain','duplicate_fraction','nan_fraction'):
        def bad_knots(self,*args,**kwargs):
            rows=original_cartesian(self,*args,**kwargs)
            if mode=='short_domain':
                for row in rows:row['fraction']=.1+.8*row['fraction']
            elif mode=='duplicate_fraction':rows[1]['fraction']=rows[0]['fraction']
            else:rows[1]['fraction']=float('nan')
            return rows
        reject('ik_'+mode+'_fails_closed',patcher=patch.object(Planner,'plan_cartesian',bad_knots))
    from types import SimpleNamespace
    for knots,width in [([.1,.9],7),([0.,1.],6)]:
        curve=SimpleNamespace(x=np.asarray(knots),c=np.zeros((4,1,width)))
        try:helper.continuous_rate_bounds(SimpleNamespace(duration=6.,curves={'left':curve,'right':curve}));rejected=False
        except ValueError:rejected=True
        check('rate_bound_rejects_bad_domain_or_dimension_'+str((knots,width)),rejected)
    for angle in (-15.,-30.,30.):
        r,p=helper.plan_postlift_yaw(rest,state,{**cfg,'task':{**cfg['task'],'contact_trial':{**cfg['task']['contact_trial'],'postlift_yaw':{'world_z_degrees':angle,'duration_s':6.}}}},command)
        check(str(angle)+'_degree_boundary_supported',r['status']=='candidate' and p is not None)
        if abs(angle)==30 and p is not None:
            save(ROOT/f'runtime_ipc/postlift-yaw30-angle{angle:+g}-helper-review.json',dict(helper_report=r,command_qs=command,physical_success=False))
    check('helper_source_unchanged_during_regression',source_hash==sha(ROOT/'postlift_yaw.py'))
    result=dict(kind='postlift_yaw_helper_regression',physical_success=False,all_checks_passed=all(c['passed'] for c in checks),
        check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,cases=cases,
        helper_source_sha256=source_hash,scope='CPU helper contracts and frozen geometry only; no physics, force closure or successful insertion is certified')
    save(ROOT/'runtime_ipc/postlift-yaw30-regression.json',result)
    print(json.dumps({k:result[k] for k in ('check_count','all_checks_passed','failed_checks','cases')},indent=2))
    if not result['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
