"""Authorized one-step native constructor smoke and read-only pose replay."""
from pathlib import Path
import os,sys,json,hashlib,time
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE),str(HERE/'vendor')]
os.environ.setdefault('GS_CACHE_FILE_PATH',str(HERE/'.cache/genesis'))
os.environ.setdefault('XDG_CACHE_HOME',str(HERE/'.cache'))
import numpy as np
from scene import build,read_config,positions
from staged_hand_monitor import NativeHandHeightMonitor
from motion import Planner
from contact_robot_geometry import left_collision_solids

started=time.perf_counter()
config=read_config(HERE/'config_guarded_staged_trial.json')
initial=np.load(HERE/'runs/two_panel_full_12/initial.npz')
scene,robots,bag,bear,cameras=build(config,width=160,height=120,
    initial_q={s:initial['q_'+s] for s in ('left','right')})
monitor=NativeHandHeightMonitor(scene)
scene.step()
row,reason=monitor.measure(config['simulation']['dt'],positions(bear))
planner=Planner(config)
q=robots['left'].get_qpos().cpu().numpy().reshape(9)
solids=left_collision_solids(planner,q)
selected=[s for s in solids if s['body'] in ('left_hand','left_left_finger','left_right_finger')]
rigid_min=min(s['vertices'][:,2].min() for s in selected)
merged_points=np.asarray(scene.sim.coupler._abd_slots_by_link[monitor.links[0]][0].geometry().positions().view()).reshape(-1,3)
matrix=np.asarray(scene.sim.coupler._coupling_data.ipc_transforms[0,monitor.indices[0]])
whole_min=float(np.min(merged_points@matrix[2,:3]+matrix[2,3]))
snapshot=np.load(HERE/'runs/two_panel_full_12/transport_teddy.npz')
replayed=left_collision_solids(planner,snapshot['q_left'])
hand_min=min(s['vertices'][:,2].min() for s in replayed if s['body']=='left_hand')
arm_min=min(s['vertices'][:,2].min() for s in replayed if s['body']=='left_link7')
# Use the identical geometry at the saved q for a CPU-only world transform.
planner.fk('left',snapshot['q_left'][:7])
body=planner.m.body('left_link7').id
r=planner.d.xmat[body].reshape(3,3);p=planner.d.xpos[body]
native_reference_replay_min=float(np.min(monitor.vertices[0]@r[2,:]+p[2]))
checks={'constructor_and_completed_step_sample_succeeded':True,
    'palm_is_strict_subset_of_link7':len(monitor.vertices[0])<len(merged_points),
    'all_palm_points_match_native':monitor.geometry_metadata[0]['selection']['native_subset_max_error_m']<=1e-7,
    'smoke_native_vs_measured_fk_below_1mm':abs(row['left_distal_minimum_z_m']-rigid_min)<.001,
    'full12_palm_geometry_matches_mujoco_below_1um':abs(native_reference_replay_min-hand_min)<1e-6,
    'full12_link7_lower_point_is_excluded':hand_min-arm_min>.01}
result={'kind':'one_step_native_hand_mapping_regression','passed':all(checks.values()),'checks':checks,
    'monitor':monitor.report(),'sample':row,'sample_stop_reason':reason,
    'initial_native_entire_link7_minimum_z_m':whole_min,
    'initial_actual_q_hand_minimum_z_m':float(rigid_min),
    'full12_pose_replay':{'palm_minimum_z_m':float(hand_min),'arm_link7_minimum_z_m':float(arm_min),
        'cached_native_palm_at_measured_q_minimum_z_m':native_reference_replay_min},
    'source_sha256':hashlib.sha256((HERE/'staged_hand_monitor.py').read_bytes()).hexdigest(),
    'wall_seconds':time.perf_counter()-started,
    'scope':'One native integration step plus saved-pose CPU geometry replay; no insertion or material calibration.'}
np.savez_compressed(HERE/'runtime_ipc/staged-native-hand-reference.npz',**monitor.reference_geometry())
(HERE/'runtime_ipc/staged-native-hand-smoke.json').write_text(json.dumps(result,indent=2,default=lambda value:value.item())+'\n')
print(json.dumps(result,indent=2,default=lambda value:value.item()),flush=True)
raise SystemExit(0 if result['passed'] else 2)
