"""Independent CPU audits of held-strap monitoring and per-link IPC wrench records."""
from pathlib import Path
import argparse
import json
import numpy as np


def audit_handle_trace(rows, events, joint_trace, dt):
    start=next((float(e['start']) for e in events if e['phase']=='spread_bag'),None)
    end=next((float(e['start']) for e in events if e['phase']=='release_right_handle'),None)
    if start is None:
        return dict(status='not_entered',passed=False,reason='No physical spread phase was recorded.')
    if not rows:
        return dict(status='missing',passed=False,reason='No every-step held-handle evidence.')
    time=np.asarray([r['time_s'] for r in rows],float)
    forces=np.asarray([[*r['left_opening_reactions_n'],*r['right_opening_reactions_n']] for r in rows],float)
    distance=np.asarray([[r['left_material_distance_m'],r['right_material_distance_m']] for r in rows],float)
    stop=float(joint_trace[-1,0]) if end is None else end
    expected=joint_trace[(joint_trace[:,0]>start+1e-9)&(joint_trace[:,0]<=stop+1e-9),0]
    checks=dict(all_recorded_values_finite=bool(all(np.isfinite(a).all() for a in (time,forces,distance))),
        exact_four_individual_fingers_and_two_material_groups=forces.shape==(len(rows),4) and distance.shape==(len(rows),2),
        strictly_increasing_fixed_step_times=bool(np.all(np.diff(time)>0) and np.allclose(np.diff(time),dt,rtol=0,atol=1e-9)),
        every_integration_step_from_spread_until_handoff=bool(len(time)==len(expected) and np.allclose(time,expected,rtol=0,atol=1e-9)),
        all_four_opening_reactions_above_0p1n=bool(np.all(forces>.1)),
        both_designated_material_groups_within_18mm=bool(np.all(distance<=.018)))
    phases=[]
    for e in events:
        selected=(time>float(e['start'])+1e-9)&(time<=float(e['start'])+np.ceil(float(e['duration'])/dt)*dt+1e-9)
        if not selected.any():continue
        phases.append(dict(phase=e['phase'],samples=int(selected.sum()),
            min_opening_reactions_n=forces[selected].min(0).tolist(),
            max_opening_reactions_n=forces[selected].max(0).tolist(),
            max_designated_material_distance_m=distance[selected].max(0).tolist()))
    return dict(status='passed' if all(checks.values()) else 'failed',passed=all(checks.values()),checks=checks,
        samples=len(rows),start_s=float(time[0]),end_s=float(time[-1]),
        minimum_opening_reactions_n=forces.min(0).tolist(),maximum_designated_material_distance_m=distance.max(0).tolist(),
        phases=phases,limitations=['Reactions are IPC soft-coupling feedback, not contact forces identified by individual material.',
            'Material proximity and four positive opening-axis reactions support held-handle evidence; they do not prove force closure.',
            'Checks cover recorded integration states only.'])


def audit_wrench_rows(rows,events):
    if not rows:
        return dict(status='not_entered',passed=False,reason='No distal-hand guard trace.')
    keys=('right_link_aim_origin_world_m','right_link_torque_at_aim_tcp_world_nm',
        'right_finger_aim_opening_axis_world','left_link_names','left_link_force_world_n',
        'left_link_torque_at_aim_origin_world_nm','left_link_aim_origin_world_m',
        'left_link_torque_at_left_aim_tcp_world_nm','left_aim_tcp_world_m',
        'left_all_link_net_force_world_n','left_all_link_torque_at_aim_tcp_world_nm',
        'left_finger_aim_opening_axis_world')
    if not all(all(k in r for k in keys) for r in rows):
        return dict(status='missing',passed=False,reason='New per-link geometry diagnostics are incomplete.')
    names=rows[0]['left_link_names']
    needed=[*[f'link{i}' for i in range(8)],'left_finger','right_finger']
    if len(names)!=len(needed) or set(names)!=set(needed):
        return dict(status='failed',passed=False,reason='Unexpected or duplicate left link mapping.')
    finger_ids=[names.index(n) for n in ('left_finger','right_finger')]
    def arr(k):return np.asarray([r[k] for r in rows],float)
    time=arr('time_s');rf=arr('right_link_force_world_n');rt=arr('right_link_torque_at_aim_origin_world_nm')
    ro=arr('right_link_aim_origin_world_m');rp=arr('aim_tcp_world_m');ra=arr('right_finger_aim_opening_axis_world')
    lf=arr('left_link_force_world_n');lt=arr('left_link_torque_at_aim_origin_world_nm')
    lo=arr('left_link_aim_origin_world_m');lp=arr('left_aim_tcp_world_m');la=arr('left_finger_aim_opening_axis_world')
    right_moment=rt+np.cross(ro-rp[:,None,:],rf)
    left_moment=lt+np.cross(lo-lp[:,None,:],lf)
    compare={
        'right_per_link_shifted_moment':(right_moment,arr('right_link_torque_at_aim_tcp_world_nm')),
        'right_total_shifted_moment':(right_moment.sum(1),arr('torque_at_aim_tcp_world_nm')),
        'right_total_force':(rf.sum(1),arr('force_world_n')),
        'right_finger_axis_projection':(np.einsum('ijk,ijk->ij',rf[:,1:],ra),arr('right_finger_opening_projection_n')),
        'left_per_link_shifted_moment':(left_moment,arr('left_link_torque_at_left_aim_tcp_world_nm')),
        'left_total_shifted_moment':(left_moment.sum(1),arr('left_all_link_torque_at_aim_tcp_world_nm')),
        'left_total_force':(lf.sum(1),arr('left_all_link_net_force_world_n')),
        'left_finger_axis_projection':(np.einsum('ijk,ijk->ij',lf[:,finger_ids],la),arr('left_finger_opening_projection_n')),
        'left_finger_force_norm':(np.linalg.norm(lf[:,finger_ids],axis=2),arr('left_finger_force_norm_n'))}
    errors={name:float(np.max(np.abs(a-b))) for name,(a,b) in compare.items()}
    checks={name+'_reconstructs':np.isfinite(e) and e<1e-9 for name,e in errors.items()}
    checks.update(
        stable_left_link_mapping=all(r['left_link_names']==names for r in rows),
        stable_right_link_mapping=all(r['right_link_names']==['link7','left_finger','right_finger'] for r in rows),
        raw_geometry_and_wrenches_finite=all(np.isfinite(a).all() for a in (time,rf,rt,ro,rp,ra,lf,lt,lo,lp,la)),
        opening_axes_unit_length=bool(np.allclose(np.linalg.norm(ra,axis=2),1.,rtol=0,atol=1e-9) and
            np.allclose(np.linalg.norm(la,axis=2),1.,rtol=0,atol=1e-9)))
    link_stats=[]
    for i,name in enumerate(names):
        magnitudes=np.linalg.norm(lf[:,i],axis=1);peak=int(np.argmax(magnitudes))
        link_stats.append(dict(link=name,max_raw_force_n=float(magnitudes[peak]),peak_time_s=float(time[peak]),
            force_at_peak_world_n=lf[peak,i].tolist(),max_torque_at_left_tcp_nm=float(np.linalg.norm(left_moment[:,i],axis=1).max()),
            first_force_world_n=lf[0,i].tolist(),last_force_world_n=lf[-1,i].tolist()))
    phases=[];palm=names.index('link7')
    for index,e in enumerate(events):
        start=float(e['start']);end=float(events[index+1]['start']) if index+1<len(events) else float(time[-1])
        selected=(time>=start-1e-9)&(time<=end+1e-9)
        if not selected.any():continue
        force=lf[selected,palm];moment=left_moment[selected,palm]
        phases.append(dict(phase=e['phase'],sample_count=int(selected.sum()),
            left_palm_force_norm_min_max_n=[float(np.linalg.norm(force,axis=1).min()),float(np.linalg.norm(force,axis=1).max())],
            left_palm_force_world_first_last_n=[force[0].tolist(),force[-1].tolist()],
            left_palm_torque_at_left_tcp_norm_max_nm=float(np.linalg.norm(moment,axis=1).max()),
            right_palm_force_norm_max_n=float(np.linalg.norm(rf[selected,0],axis=1).max())))
    return dict(status='passed' if all(checks.values()) else 'failed',passed=all(checks.values()),
        checks=checks,reconstruction_max_absolute_errors=errors,samples=len(rows),
        left_link_diagnostics=link_stats,phase_palm_diagnostics=phases,
        limitations=['Exact algebraic reconstruction checks stored reaction geometry, not independent native solver pair forces.',
            'Left link7 contains the welded palm. Its nonzero reaction alone cannot identify teddy versus cloth contact.',
            'No new force or torque limits are imposed by these diagnostic summaries.'])


def audit(run):
    run=Path(run);report=json.loads((run/'report.json').read_text());config=json.loads((run/'config.json').read_text())
    def read(name):return json.loads((run/name).read_text()) if (run/name).exists() else []
    handles=audit_handle_trace(read('handle_grip_trace.json'),report['events'],np.load(run/'joint_trace.npy'),config['simulation']['dt'])
    try:wrenches=audit_wrench_rows(read('contact_guard_trace.json'),report['events'])
    except (ValueError,KeyError,IndexError,TypeError) as exc:wrenches=dict(status='failed',passed=False,reason=str(exc))
    return dict(kind='independent_richer_contact_trace_review',physical_task_success=None,
        held_handle_monitor=handles,per_link_wrench_reconstruction=wrenches,
        note='Only independent-task containment, physics, sequencing and force-limit audits can establish simulated task success.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);args=parser.parse_args()
    result=audit(args.run);out=args.run/'independent-richer-contact-review.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]['status'] for k in ('held_handle_monitor','per_link_wrench_reconstruction')},indent=2))
