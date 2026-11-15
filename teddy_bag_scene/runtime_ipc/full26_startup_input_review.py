"""One read-only startup audit, adapted from the existing full25 input review."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_26';PROBE=ROOT/'runs/two_panel_long_handles_handoff_01';OLD=ROOT/'runs/two_panel_full_25'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def differences(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum((differences(a.get(k),b.get(k),path+'.'+k) for k in sorted(a.keys()|b.keys())),[])
 return [] if a==b else [dict(path=path,old=a,new=b)]
launch=read(ROOT/'runtime_ipc/full26-launch.json');cfg=read(RUN/'config.json');probe_cfg=read(PROBE/'config.json');old_cfg=read(OLD/'config.json');manifest=read(RUN/'input_sha256.json');pm=read(PROBE/'input_sha256.json');om=read(OLD/'input_sha256.json');livecfg=Path(launch['command'][launch['command'].index('--config')+1]);initial=dict(np.load(RUN/'initial.npz',allow_pickle=False));rest=dict(np.load(RUN/'rest.npz',allow_pickle=False));pi=dict(np.load(PROBE/'initial.npz',allow_pickle=False));pr=dict(np.load(PROBE/'rest.npz',allow_pickle=False));orr=dict(np.load(OLD/'rest.npz',allow_pickle=False));asset=RUN/'source/assets'/cfg['bag']['npz_file'];src=dict(np.load(asset,allow_pickle=False));oldsrc=dict(np.load(OLD/'source/assets'/old_cfg['bag']['npz_file'],allow_pickle=False));meta=read(RUN/'source/assets'/cfg['bag']['metadata'])
protected=[RUN/k for k in manifest]+[RUN/'rest.npz',RUN/'initial.npz',ROOT/'config.json',livecfg,ROOT/'runtime_ipc/full26-launch.json'];before={str(p):sha(p) for p in protected}
source_names={k for k in manifest if k.startswith('source/')};actual_files={str(p.relative_to(RUN)) for p in (RUN/'source').rglob('*') if p.is_file()}|{'config.json'}
launch_sources=launch['source_sha256'];pid=int(launch['pid']);proc=Path('/proc')/str(pid)/'cmdline';cmd=[v.decode() for v in proc.read_bytes().split(b'\0') if v] if proc.exists() else []
changes=differences(probe_cfg,cfg)
dist,idx=cKDTree(rest['bag']).query(src['points']+cfg['bag']['position']);mapped=np.sort(idx[src['faces']],axis=1);native=np.sort(rest['bag_faces'],axis=1);face_set=lambda f:{tuple(map(int,x)) for x in f};tet=rest['bear'][rest['bear_tets']];vol=np.linalg.det(np.stack([tet[:,1]-tet[:,0],tet[:,2]-tet[:,0],tet[:,3]-tet[:,0]],axis=2))/6;tri=rest['bag'][rest['bag_faces']];area=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)
physical_files=['scene.py','motion.py','validate.py','physics_config.py','contact_grasp.py','contact_insertion.py','contact_guard.py','physical_monitor.py','contact_robot_geometry.py','postlift_yaw.py','gravity_release.py','staged_contact_insertion.py','staged_hand_monitor.py']
default_ref=read(OLD/'independent-startup-input-review.json')['input_sha256'][str(ROOT/'config.json')]
checks={
 'launch_expected_configuration_hash':launch['config_sha256']=='177af420224b1729cfe424fa62ce20ff95393f31443d7d922200dd276bd8731d' and sha(livecfg)==launch['config_sha256'],
 'live_and_frozen_configuration_equal':read(livecfg)==cfg,
 'only_X_offset_differs_from_actual_handoff_configuration':changes==[dict(path='.task.contact_trial.center_offsets_xy_m',old=[[.015,.03075]],new=[[.005,.03075]])],
 'all_frozen_manifest_checksums_match':all(sha(RUN/k)==h for k,h in manifest.items()),
 'manifest_covers_all_frozen_inputs':set(manifest)==actual_files,
 'all_frozen_sources_and_assets_equal_completed_handoff':source_names=={k for k in pm if k.startswith('source/')} and all(manifest[k]==pm[k] for k in source_names),
 'all_frozen_python_sources_match_launch':all(launch_sources[Path(k).name]==h for k,h in manifest.items() if k.startswith('source/') and k.endswith('.py')),
 'all_frozen_sources_and_assets_match_live':all(sha(ROOT/k.removeprefix('source/'))==manifest[k] for k in source_names),
 'all_original_physical_planning_validation_sources_equal_full25':all(manifest['source/'+n]==om['source/'+n] for n in physical_files),
 'reviewed_serialization_only_runner':manifest['source/run.py']=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9',
 'launch_evidence_hashes_match':all(sha(ROOT/k)==h for k,h in launch['evidence_sha256'].items()),
 'launch_additional_generator_config_archives_match':all(sha(ROOT/k)==h for k,h in launch['additional_archived_inputs'].items()),
 'default_configuration_unchanged':sha(ROOT/'config.json')==default_ref,
 'actual_native_process_full_task_and_expected_configuration':bool(cmd) and '--task'in cmd and cmd[cmd.index('--task')+1]=='full' and '--config'in cmd and Path(cmd[cmd.index('--config')+1])==livecfg and '--output'in cmd and cmd[cmd.index('--output')+1]==RUN.name,
 'fresh_zero_time':float(initial['time'])==0.,
 'initial_all_arrays_finite':all(np.isfinite(v).all() for v in initial.values()),
 'rest_all_arrays_finite':all(np.isfinite(v).all() for v in rest.values()),
 'rest_all_arrays_exact_completed_handoff_initial_rest':set(rest)==set(pr) and all(np.array_equal(v,pr[k]) for k,v in rest.items()),
 'initial_all_arrays_exact_completed_handoff_initial_state':set(initial)==set(pi) and all(np.array_equal(v,pi[k]) for k,v in initial.items()),
 'initial_soft_positions_exact_own_rest':np.array_equal(initial['bag'],rest['bag']) and np.array_equal(initial['bear'],rest['bear']),
 'actual_long_bag_topology_2148_4112':rest['bag'].shape==(2148,3) and rest['bag_faces'].shape==(4112,3),
 'actual_bear_topology_1450_5486':rest['bear'].shape==(1450,3) and rest['bear_tets'].shape==(5486,4),
 'bear_rest_exact_full25_with_all_positive_tetrahedra':np.array_equal(rest['bear'],orr['bear']) and np.array_equal(rest['bear_tets'],orr['bear_tets']) and bool(np.all(vol>0)),
 'source_native_cloth_mapping_bijective_under_1um':len(np.unique(idx))==len(rest['bag'])==len(src['points']) and float(dist.max())<1e-6,
 'all_source_native_cloth_faces_bidirectional_exact':len(face_set(mapped))==len(mapped)==len(native)==len(face_set(native)) and face_set(mapped)==face_set(native),
 'body_points_faces_U_seam_exact_short_handle_asset':np.array_equal(src['points'][src['body']],oldsrc['points'][oldsrc['body']]) and np.array_equal(src['faces'][src['body_face_indices']],oldsrc['faces'][oldsrc['body_face_indices']]) and np.array_equal(src['u_seam'],oldsrc['u_seam']),
 'source_two_panel_zero_bottom_depth_no_support':meta['bag']['width_m']==.46 and meta['bag']['height_m']==.40 and meta['bag']['depth_m']==0 and meta['bag']['mesh']['connected_components']==1 and meta['bag']['mesh']['nonmanifold_edges']==0 and meta['bag']['mesh']['boundary_loops']==3 and meta['bag']['fixed_particles']==0 and meta['bag']['hidden_rigid_support'] is False,
 'long_handle_straight_legs_018':all(meta['bag']['handles'][s]['vertical_leg_extra_m']==.18 for s in ['front','back']),
 'original_finite_force_3N_zero_target_upperarm_grasp':cfg['task']['bear_finger_force_limit']==3. and cfg['task']['bear_finger_target']==0. and cfg['task']['bear_grasp_local']==old_cfg['task']['bear_grasp_local'] and cfg['robots']==old_cfg['robots'],
 'original_material_and_bear_parameters_unchanged':cfg['bear']==old_cfg['bear'] and all(cfg['bag'][k]==old_cfg['bag'][k] for k in ['E','nu','rho','thickness','bending_stiffness','friction_mu','shear_stiffness','handle_shear_stiffness']) and cfg['bag']['shear_stiffness']==50. and cfg['bag']['handle_shear_stiffness']==250.,
 'declared_yaw15_offset005_03075_gravity_and_no_recenter':cfg['task']['contact_trial']['postlift_yaw']=={'world_z_degrees':15.,'duration_s':6.} and cfg['task']['contact_trial']['center_offsets_xy_m']==[[.005,.03075]] and cfg['task']['contact_trial']['gravity_release']==old_cfg['task']['contact_trial']['gravity_release'] and cfg['task']['contact_trial']['recenter_enabled'] is False,
 'both_open_targets_height140':all(cfg['task']['open_targets'][s][2]==1.40 for s in ['left','right']),
}
checks['inputs_unchanged_during_review']=all(sha(p)==h for p,h in before.items())
result=dict(kind='independent_full26_startup_input_review',created_utc=datetime.now(timezone.utc).isoformat(),all_checks_passed=all(checks.values()),checks=checks,physical_task_success=False,native_validation_pending=True,launch_pid=pid,observed_process_command=cmd,config_diff_from_completed_handoff=changes,config_diff_from_full25=differences(old_cfg,cfg),topology={'bag_nodes':len(rest['bag']),'bag_faces':len(rest['bag_faces']),'bear_nodes':len(rest['bear']),'bear_tetrahedra':len(rest['bear_tets'])},source_to_native_max_distance_m=float(dist.max()),initial_clock_s=float(initial['time']),rest_derived_mass={'bag_area_m2':area,'bag_mass_kg':area*cfg['bag']['rho']*2*cfg['bag']['thickness'],'bear_volume_m3':float(vol.sum()),'bear_mass_kg':float(vol.sum()*cfg['bear']['rho']),'minimum_positive_tetra_volume_m3':float(vol.min()),'mass_scope':'Derived from saved reference geometry/config; no native mass-buffer readback.'},material_and_force_scope={'configured_bear_force_limit_per_finger_N':3.,'bear_motor_target_m':0.,'configured_bag_grasp_limit_N':cfg['robots']['finger_force_limit'],'cloth_body_shear_N_per_m':50.,'cloth_handles_shear_N_per_m':250.,'native_material_buffer_readback':'pending final native report','bear_3N_application_scope':'Source-pinned close_teddy_gripper installs 3N when teddy grasp begins; startup saved initial is the two-handle grasp, not proof of already applied teddy force.'},input_sha256=before,limitations=['Fresh t=0/rest equality certifies saved initial inputs, not later native behavior or task success.','No new physics stepping, planning, GPU operation, state injection or production edit occurred.','Native per-step grip/force/torque/J/strain and final material-buffer readback/whole-teddy containment remain pending.'])
output=RUN/'independent-startup-input-review.json'
if output.exists():raise FileExistsError('Do not replace a completed startup report')
output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(output),'sha256':sha(output),'passed':sum(checks.values()),'total':len(checks),'all_checks_passed':all(checks.values()),'failed':[k for k,v in checks.items() if not v],'topology':result['topology'],'masses':result['rest_derived_mass']},indent=2))
if not all(checks.values()):raise SystemExit(1)
