"""Read-only evidence extraction from the two archived native head-grasp probes."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs={}
def record(p):
    inputs[str(p.relative_to(ROOT))]=sha(p)
def loadj(p):
    record(p)
    return json.loads(p.read_text())
def loadn(p):
    record(p)
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
related={name:loadj(ROOT/name) for name in ['config_head_grasp_candidate.json','config_head_grasp_following_3n.json','config_head_grasp_tall.json']}
current=loadj(ROOT/'runs/two_panel_full_22/config.json')
rows=[]
for name in ['two_panel_head_grasp_01','two_panel_head_grasp_02']:
    p=ROOT/'runs'/name
    report=loadj(p/'report.json'); validation=loadj(p/'validation.json'); cfg=loadj(p/'config.json')
    rest=loadn(p/'rest.npz'); states=loadn(p/'states.npz'); close=loadn(p/'close_teddy_gripper.npz')
    for file in ['source/run.py','source/scene.py','source/motion.py','source/contact_grasp.py','source/validate.py']:
        record(p/file)
    tets=rest['bear_tets']; points=rest['bear'][tets]
    vols=np.abs(np.linalg.det(np.stack([points[:,1]-points[:,0],points[:,2]-points[:,0],points[:,3]-points[:,0]],axis=-1))/6)
    center=lambda x: (x[tets].mean(1)*vols[:,None]).sum(0)/vols.sum()
    com_close=center(close['bear']);com_final=center(states['bear'][-1])
    nearest=np.linalg.norm(states['bear']-states['tcp_right'][:,None,:],axis=2).min(1)
    lift=next(e for e in report['events'] if e['phase']=='lift_teddy')
    after=states['time']>=float(close['time'])
    metrics=validation['soft_metric_sampling']
    rows.append({
        'run':name,
        'matching_related_config_semantic_not_byte_identity':[n for n,c in related.items() if c==cfg],
        'reported_status':report['status'],'reported_error':report['error'],
        'grasp_probe_passed':report['grasp_probe_passed'],
        'requested_task_passed_if_present':report.get('requested_task_passed'),
        'simulation_stop_s':report['simulated_seconds'],
        'lift_start_s':lift['start'],'requested_lift_plan_duration_s':lift['duration'],
        'actual_lift_attempt_elapsed_s':report['simulated_seconds']-lift['start'],
        'lift_endpoint_snapshot_exists':(p/'lift_teddy.npz').exists(),
        'hold_phase_recorded':any(e['phase']=='grasp_hold' for e in report['events']),
        'grasp_local_m':cfg['task']['bear_grasp_local'],
        'grasp_rotation_xyzw':cfg['task']['bear_grasp_rotation_xyzw'],
        'commanded_per_finger_force_limit_n':cfg['task']['bear_finger_force_limit'],
        'commanded_per_finger_target_m':cfg['task']['bear_finger_target'],
        'force_scope':'Configured actuator clamp; not a measured normal contact or force-closure certificate.',
        'actual_finger_q_at_close_m':close['q_right'][-2:].tolist(),
        'actual_finger_q_at_stop_m':states['q_right'][-1,-2:].tolist(),
        'nearest_any_teddy_vertex_to_tcp_at_close_m':float(np.linalg.norm(close['bear']-close['tcp_right'],axis=1).min()),
        'last_pre_failure_saved_time_s':float(states['time'][-2]),
        'last_pre_failure_saved_nearest_vertex_to_tcp_m':float(nearest[-2]),
        'stop_saved_nearest_vertex_to_tcp_m':float(nearest[-1]),
        'tcp_displacement_close_to_stop_m':(states['tcp_right'][-1]-close['tcp_right']).tolist(),
        'rest_volume_weighted_com_displacement_close_to_stop_m':(com_final-com_close).tolist(),
        'maximum_saved_whole_bear_bottom_height_during_lift_attempt_m':float(states['bear'][after,:,2].min(1).max()),
        'actual_tabletop_m':cfg['table']['height'],
        'bear_lifted_clear_of_table':validation['checks']['bear_lifted_clear_of_table'],
        'minimum_per_step_tet_J':metrics['minimum_tetrahedron_volume_ratio'],
        'minimum_J_time_s':metrics['worst_volume_ratio_time_s'],
        'per_step_metric_dt_max_s':metrics['maximum_interval_s'],
        'no_inversion':validation['checks']['no_tetrahedron_inversion'],
        'no_J_below_five_percent':validation['checks']['no_tetrahedron_volume_below_5_percent'],
        'max_bag_edge_stretch_ratio':metrics['maximum_bag_edge_stretch_ratio'],
        'max_table_penetration_m':metrics['maximum_table_penetration_m'],
        'maximum_saved_principal_stretch':validation['bear_deformation']['maximum_principal_stretch'],
        'derived_mass_kg':float(cfg['bear']['rho']*vols.sum()),
        'physical_success':False,
    })
result={
    'scope':'Bounded read-only extraction of archived actual native grasp probes 01/02; no new planning, simulation, source/config/material edit or change to active full22.',
    'runs':rows,
    'historical_gate_semantics':{
        'source':'Both archived source/run.py capture methods',
        'predicate':'After close_teddy_gripper enables must_hold_teddy, capture rejects if min distance of any teddy vertex to actual right TCP exceeds 0.045 m.',
        'cadence':'Saved captures about every 0.05 s, plus phase endpoints; this is a coarse proximity loss guard, not direct two-pad force closure.',
        'both_stops_verified_from_actual_states':all(r['stop_saved_nearest_vertex_to_tcp_m']>.045 for r in rows),
        'lift_event_duration_is_not_completion':True,
    },
    'conclusion_cn':[
        '两次头端试验均完成接近与夹爪合拢，但没有完成抬升或到达持握停留阶段；不能记录为动态抓持成功。',
        '01在3.61秒、抬升开始0.95秒后失持；02改为持续合拢仍在3.56秒、抬升开始0.90秒后失持。两次手腕向上走了约67.3/59.1毫米，而熊质心高度基本未升、整体底部仍在桌面约0.93毫米处。',
        '02更深合拢产生更强局部压缩和拉伸，未改善抬起结果。两次停止均由失持保护触发，而非J<0.05、翻转、布拉伸或桌面穿透保护。',
        '因此原头端坐标、原姿态、每指3N配17毫米或0目标的直接重复没有既有成功依据；更深合拢在这两例中已试过，不能视作尚未尝试的修复。',
    ],
    'related_tall_config_scope':{
        'matches_either_archived_trial':False,
        'difference_from_trial_02':'Only bag asset family/height .34 to .40 m and both open target Z 1.20 to 1.26 m. Head target, orientation, 3 N force limit and zero finger target are unchanged.',
        'evidence_limit':'A configuration file is not a successful native grasp record; this review makes no claim about an unexamined tall run.'},
    'transfer_limits':{
        'historical_bear_mesh':related['config_head_grasp_candidate.json']['bear']['mesh'],
        'current_full22_bear_mesh':current['bear']['mesh'],
        'historical_bear_material':{k:related['config_head_grasp_candidate.json']['bear'][k] for k in ['E','nu','rho','friction_mu']},
        'current_full22_bear_material':{k:current['bear'][k] for k in ['E','nu','rho','friction_mu']},
        'statement':'These probes used the full-size teddy, old bag/control snapshots and no bag-insertion sequence. They do not prove all head grasps physically impossible or test the current 85%-size teddy at a new head grasp. Do not infer the active full22 outcome or change its policy from them.'},
    'input_hashes_before':inputs.copy(),
}
result['input_hashes_after']={p:sha(ROOT/p) for p in inputs}
result['all_input_bytes_unchanged']=result['input_hashes_before']==result['input_hashes_after']
assert result['all_input_bytes_unchanged']
out=ROOT/'runtime_ipc/independent-historical-head-grasp-review.json'
out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'all_inputs_unchanged':True,'runs':[{'run':r['run'],'stop_s':r['simulation_stop_s'],'grasp_probe_passed':r['grasp_probe_passed'],'min_J':r['minimum_per_step_tet_J']} for r in rows]},ensure_ascii=False,indent=2))
