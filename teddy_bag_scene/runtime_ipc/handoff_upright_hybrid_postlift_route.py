"""A single bounded hybrid diagnostic: actual final handoff bag + old actual held teddy."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OLD=ROOT/'runs/two_panel_full_21';OUT=RUN/'independent-hybrid-postlift-route';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import plan_guarded_contact_trial,_state_context,_policy,_mouth,_serial
from motion import Planner
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
input_paths=[RUN/'settle_handoff.npz',RUN/'rest.npz',RUN/'config.json',OLD/'orient_teddy_postlift.npz',OLD/'rest.npz',OLD/'config.json',*[RUN/'source'/n for n in ['motion.py','contact_insertion.py','contact_robot_geometry.py','contact_slice_geometry.py','physical_monitor.py','validate.py']],Path(__file__)]
hashes={str(p):sha(p) for p in input_paths}
cfg=json.loads((RUN/'config.json').read_text());oldcfg=json.loads((OLD/'config.json').read_text());nr=dict(np.load(RUN/'rest.npz'));oldrest=dict(np.load(OLD/'rest.npz'));bag=dict(np.load(RUN/'settle_handoff.npz'));bear=dict(np.load(OLD/'orient_teddy_postlift.npz'))
assert cfg['robots']==oldcfg['robots'] and cfg['bear']==oldcfg['bear'] and cfg['bag']==oldcfg['bag']
assert np.array_equal(nr['bag_faces'],oldrest['bag_faces']) and np.array_equal(nr['bear_tets'],oldrest['bear_tets'])
assert np.array_equal(nr['bag'],oldrest['bag']) and np.array_equal(nr['bear'],oldrest['bear'])
# Explicit synthetic timestamp, never a joint native snapshot. Force arrays are deliberately absent.
state=dict(time=np.asarray(max(float(bag['time']),float(bear['time']))),bag=bag['bag'].copy(),bear=bear['bear'].copy(),q_left=bag['q_left'].copy(),tcp_left=bag['tcp_left'].copy(),q_right=bear['q_right'].copy(),tcp_right=bear['tcp_right'].copy())
rest=dict(bag=nr['bag'].copy(),bag_faces=nr['bag_faces'].copy(),bear=oldrest['bear'].copy(),bear_tets=oldrest['bear_tets'].copy())
np.savez_compressed(OUT/'hybrid_hypothetical_observation.npz',**state);np.savez_compressed(OUT/'rest.npz',**rest)
(OUT/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
config_for_planner=copy.deepcopy(cfg);config_for_planner['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for key in ['npz_file','metadata']:config_for_planner['bag'][key]=str(RUN/'source/assets'/cfg['bag'][key])
metadata=dict(kind='two_source_hybrid_frozen_geometry_only',physical_success=False,native_execution=False,not_a_joint_native_snapshot=True,
 source_actual_bag_and_left_arm=dict(path=str(RUN/'settle_handoff.npz'),time_s=float(bag['time']),keys=['bag','q_left','tcp_left']),
 source_actual_bear_and_right_arm=dict(path=str(OLD/'orient_teddy_postlift.npz'),time_s=float(bear['time']),keys=['bear','q_right','tcp_right']),
 hybrid_timestamp_s=float(state['time']),timestamp_scope='max(source timestamps), synthetic planner bookkeeping only; no physical transition/time synchronization claimed.',
 no_force_arrays_inherited=True,rest_meshes_and_topology_exactly_identical=True,robot_bases_materials_scale_unchanged=True,
 full_actual_postyaw_rotation_preserved=True,no_additional_yaw_or_pitch=True,declared_four_offsets_unchanged=cfg['task']['contact_trial']['center_offsets_xy_m'],
 input_sha256=hashes,limits=['This is a composition of two separately observed physical states, not an actual simultaneously observed or reachable state.',
 'Bag shape and teddy shape stay frozen only for geometry prediction. Actual cloth forces, grasp retention, deformation, released bear motion and final stability are unknown.',
 'The stored config is an unchanged copy. In-memory absolute paths bind planner assets to the new probe frozen source assets.',
 'A candidate remains provisional: native implementation must lift/grasp/reorient then freshly observe again, and original per-step/final gates remain mandatory.'])
(OUT/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
context=_state_context(rest,state,config_for_planner,_policy(config_for_planner));print('context',context['reasons'],'synthetic_t',float(state['time']),flush=True)
try:
 _,_,mouth=_mouth(rest,state,config_for_planner);print('mouth',mouth['rim_z_min_m'],mouth['rim_z_max_m'],mouth['projected_area_m2'],flush=True)
except Exception as exc:print('mouth_error',type(exc).__name__,str(exc),flush=True)
started=time.monotonic();result=plan_guarded_contact_trial(rest,state,config_for_planner)
result['hybrid_diagnostic_provenance']=metadata;result['not_a_joint_native_snapshot']=True
(OUT/'guarded_candidate.json').write_text(json.dumps(_serial(result),indent=2,allow_nan=False)+'\n')
report=dict(kind='hybrid_guarded_route_screen',physical_success=False,native_execution=False,status=result['status'],candidate_path=str(OUT/'guarded_candidate.json'),provenance_path=str(OUT/'provenance.json'),rejections=result['report'].get('rejections',[]),actual_source_context_numeric_rejections=context['reasons'],planning_elapsed_s=time.monotonic()-started,
 candidate_sections=[dict(center=c.get('center_of_mass_xy_target_m'),rejections=c.get('section_guard_rejections')) for c in result['report'].get('centering_candidates',[])],
 route_attempts=[dict(center=x['center_of_mass_xy_target_m'],rejection=x['rejection'],stages_completed=x['stages_completed'],soft_stages=[dict(stage=s['stage'],status=s['status'],minimum=s.get('minimum_clearance_lower_bound_m')) for s in x.get('left_robot_teddy_sweep_segments',[])]) for x in result['report'].get('robot_centering_attempts',[])],
 selected_route=result['route'],inputs_unchanged=all(sha(Path(p))==h for p,h in hashes.items()))
(OUT/'review.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n');print('RESULT',json.dumps({k:report[k] for k in ['status','rejections','route_attempts','planning_elapsed_s']}),flush=True)
