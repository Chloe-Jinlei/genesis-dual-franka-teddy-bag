"""Bounded, read-only full28 startup review; no simulation or dynamic claims."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/two_panel_full_28'
OLD = ROOT / 'runs/two_panel_full_27'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def diff(a, b, p=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return sum((diff(a.get(k), b.get(k), p+'.'+k) for k in sorted(a.keys() | b.keys())), [])
    return [] if a == b else [{'path': p, 'old': a, 'new': b}]
def obj(path):
    lines = Path(path).read_text().splitlines()
    return (np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')]),
            np.array([[int(x.split('/')[0])-1 for x in l.split()[1:]] for l in lines if l.startswith('f ')]))
def faceset(f): return {tuple(map(int, row)) for row in np.sort(f, axis=1)}

launch = read(ROOT/'runtime_ipc/full28-launch.json')
cfg, oldcfg = read(RUN/'config.json'), read(OLD/'config.json')
manifest, oldmanifest = read(RUN/'input_sha256.json'), read(OLD/'input_sha256.json')
livecfg = ROOT/'config_bear_s663_long_handles_trial.json'
rest, initial = dict(np.load(RUN/'rest.npz')), dict(np.load(RUN/'initial.npz'))
oldrest, oldinitial = dict(np.load(OLD/'rest.npz')), dict(np.load(OLD/'initial.npz'))
protected = [RUN/k for k in manifest] + [RUN/'rest.npz', RUN/'initial.npz', ROOT/'config.json', livecfg, ROOT/'runtime_ipc/full28-launch.json']
before = {str(p): sha(p) for p in protected}
src = dict(np.load(RUN/'source/assets'/cfg['bag']['npz_file']))
dist, ids = cKDTree(rest['bag']).query(src['points'] + cfg['bag']['position'])
v, f = obj(RUN/'source/assets'/cfg['bear']['mesh'])
vd, vi = cKDTree(rest['bear']).query(v + cfg['bear']['position'])
t = rest['bear_tets']
tf = np.sort(np.concatenate([t[:, x] for x in ([0,1,2],[0,1,3],[0,2,3],[1,2,3])]), axis=1)
unique, count = np.unique(tf, axis=0, return_counts=True)
boundary = unique[count == 1]
tet = rest['bear'][t]
vol = np.linalg.det(np.stack([tet[:,1]-tet[:,0], tet[:,2]-tet[:,0], tet[:,3]-tet[:,0]], axis=2))/6
tri = rest['bag'][rest['bag_faces']]
area = float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1).sum()/2)
roi_dist = np.linalg.norm(rest['bear']-cfg['bear']['position']-cfg['task']['bear_grasp_local'], axis=1)
roi = np.flatnonzero(roi_dist < cfg['task']['bear_grasp_tracking_radius_m'])
changes = diff(oldcfg, cfg)
expected = {
    '.bear.mesh': 'bear_uniform_coarse_s663.obj',
    '.bear.maxvolume': oldcfg['bear']['maxvolume']*1.3**3,
    '.task.bear_grasp_local': (np.array(oldcfg['task']['bear_grasp_local'])*1.3).tolist(),
    '.task.bear_grasp_tracking_radius_m': oldcfg['task']['bear_grasp_tracking_radius_m']*1.3,
    '.task.bear_finger_force_limit': 1.8252,
}
sources = {k for k in manifest if k.startswith('source/')}
py = {k for k in sources if k.endswith('.py')}
bagfiles = {k for k in sources if '/assets/' in k and 'bear_uniform_coarse_' not in k}
default_ref = read(OLD/'independent-startup-input-review.json')['input_sha256'][str(ROOT/'config.json')]
checks = {
    'expected_launch_config_sha256': launch['config_sha256'] == 'ebded9bb993ba886e24259a21d6b554c091afd9feb7b144f1d480ba3f062a1ea' == sha(livecfg),
    'live_frozen_config_equal': cfg == read(livecfg),
    'exactly_five_authorized_scale_and_force_changes': {x['path']: x['new'] for x in changes} == expected,
    'all_manifest_hashes_match': all(sha(RUN/k) == h for k,h in manifest.items()),
    'manifest_covers_frozen_inputs': set(manifest) == {str(p.relative_to(RUN)) for p in (RUN/'source').rglob('*') if p.is_file()} | {'config.json'},
    'all_frozen_python_unchanged_from_full27': py == {k for k in oldmanifest if k.startswith('source/') and k.endswith('.py')} and all(manifest[k] == oldmanifest[k] for k in py),
    'python_sources_match_launch': all(manifest[k] == launch['source_sha256'][Path(k).name] for k in py),
    'reviewed_runner_hash': manifest['source/run.py'] == '3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9',
    'all_frozen_sources_and_assets_match_live': all(sha(ROOT/k.removeprefix('source/')) == manifest[k] for k in sources),
    'all_nonbear_assets_unchanged_from_full27': all(manifest[k] == oldmanifest[k] for k in bagfiles),
    'new_s663_mesh_expected_hash': manifest['source/assets/bear_uniform_coarse_s663.obj'] == '0e4c2bd67b51b741b18530979e4d22fd1d718ff302ea0901bf9356d01c46dacd',
    'launch_asset_review_hash_matches': sha(ROOT/'runtime_ipc/bear-s663-long-handles-asset-review.json') == launch['asset_review_sha256'],
    'additional_archived_inputs_match': all(sha(ROOT/k) == h for k,h in launch['additional_archived_inputs'].items()),
    'default_configuration_unchanged': sha(ROOT/'config.json') == default_ref,
    'fresh_zero_clock_and_finite_initial': float(initial['time']) == 0 and all(np.isfinite(x).all() for x in initial.values()),
    'finite_rest_with_valid_tetra_indices': all(np.isfinite(x).all() for x in rest.values()) and t.ndim == 2 and t.shape[1] == 4 and t.min() >= 0 and t.max() < len(rest['bear']),
    'initial_soft_positions_equal_own_rest': np.array_equal(initial['bag'],rest['bag']) and np.array_equal(initial['bear'],rest['bear']),
    'initial_nonbear_arrays_equal_full27': set(initial) == set(oldinitial) and all(np.array_equal(x, oldinitial[k]) for k,x in initial.items() if k != 'bear'),
    'actual_long_bag_rest_and_topology_unchanged': np.array_equal(rest['bag'],oldrest['bag']) and np.array_equal(rest['bag_faces'],oldrest['bag_faces']) and rest['bag'].shape == (2148,3) and rest['bag_faces'].shape == (4112,3),
    'cloth_source_native_bijection_and_faces_exact': len(np.unique(ids)) == len(rest['bag']) == len(src['points']) and float(dist.max()) < 1e-6 and faceset(ids[src['faces']]) == faceset(rest['bag_faces']),
    'all_observed_bear_tetrahedra_positive': bool(np.all(vol > 0)),
    'native_boundary_matches_s663_source_surface': len(np.unique(vi)) == len(v) and float(vd.max()) < 1e-6 and faceset(vi[f]) == faceset(boundary),
    'actual_tetra_mass_near_scaled_mesh_mass': abs(float(vol.sum()*cfg['bear']['rho'])-.06377953685647826) < 1e-8,
    'actual_bag_mass_near_120079644mg': abs(area*cfg['bag']['rho']*2*cfg['bag']['thickness']-.1200796444646929) < 1e-10,
    'new_roi_has_at_least_three_actual_nodes': len(roi) >= 3,
    'finite_18252N_force_and_zero_target': cfg['task']['bear_finger_force_limit'] == 1.8252 and cfg['task']['bear_finger_target'] == 0.,
    'yaw15_center005_03075_no_recenter': cfg['task']['contact_trial']['postlift_yaw'] == {'world_z_degrees':15.,'duration_s':6.} and cfg['task']['contact_trial']['center_offsets_xy_m'] == [[.005,.03075]] and cfg['task']['contact_trial']['recenter_enabled'] is False,
    'all_bag_material_and_trial_gates_unchanged': cfg['bag'] == oldcfg['bag'] and cfg['task']['contact_trial'] == oldcfg['task']['contact_trial'],
}
pid = int(launch['pid']); proc = Path('/proc')/str(pid)/'cmdline'
try: cmd = [x.decode() for x in proc.read_bytes().split(b'\0') if x]
except FileNotFoundError: cmd = []
process = {'pid':pid, 'observed_utc':datetime.now(timezone.utc).isoformat(), 'state':'running' if cmd else 'not_running_at_review', 'command':cmd}
if cmd:
    process['expected_command_matches'] = all(flag in cmd and cmd[cmd.index(flag)+1] == value for flag,value in [('--task','full'),('--config',str(livecfg)),('--output',RUN.name)])
    checks['observed_running_command_matches_if_present'] = process['expected_command_matches']
checks['all_protected_inputs_unchanged_during_review'] = all(sha(p) == h for p,h in before.items())
checks = {k: bool(v) for k,v in checks.items()}
result = dict(kind='independent_full28_startup_input_review', created_utc=datetime.now(timezone.utc).isoformat(), all_checks_passed=all(checks.values()), checks=checks,
    physical_task_success=False, native_validation_pending=True, process_observation=process, config_diff_from_full27=changes,
    observed_topology={'bag_nodes':len(rest['bag']), 'bag_faces':len(rest['bag_faces']), 'bear_nodes':len(rest['bear']), 'bear_tetrahedra':len(t), 'bear_boundary_nodes':len(np.unique(boundary)), 'bear_boundary_faces':len(boundary)},
    reference_geometry={'bear_volume_m3':float(vol.sum()), 'bear_mass_kg':float(vol.sum()*cfg['bear']['rho']), 'minimum_positive_tetra_volume_m3':float(vol.min()), 'bag_area_m2':area, 'bag_mass_kg':area*cfg['bag']['rho']*2*cfg['bag']['thickness'], 'bag_physical_thickness_m':2*cfg['bag']['thickness'], 'bear_source_native_max_distance_m':float(vd.max()), 'mass_scope':'Derived from actual saved rest geometry and configured density; not a native mass-buffer readback.'},
    actual_grasp_roi={'radius_m':cfg['task']['bear_grasp_tracking_radius_m'], 'local_center_m':cfg['task']['bear_grasp_local'], 'node_count':len(roi), 'node_indices':roi.tolist(), 'nearest_node_distance_m':float(roi_dist.min()), 'grasp_floor_above_table_m':.025},
    material_force_scope={'configured_bear_force_per_finger_N':1.8252, 'bear_motor_target_m':0., 'initial_handle_force_limit_per_finger_N':cfg['robots']['finger_force_limit'], 'cloth_body_shear_N_per_m':cfg['bag']['shear_stiffness'], 'cloth_handles_shear_N_per_m':cfg['bag']['handle_shear_stiffness'], 'native_material_buffer_readback':'pending final report', 'force_application':'Pinned runner installs 1.8252 N when teddy grasp begins; initial state is the two-handle grasp.'},
    input_sha256=before, limitations=['Startup saved t=0 inputs only; subsequent actual grasp, lift, physical gates, release and final containment remain unverified here.', 'Observed saved tetra topology is measured, not assumed from the older bear; equal counts do not imply identical interior IDs.', 'Old s85/s51 dynamic or hybrid evidence does not certify this s663 bear.', 'No native stepping, planning, GPU, state injection or production edit performed.'])
out = RUN/'independent-startup-input-review.json'
if out.exists(): raise FileExistsError('Do not overwrite completed report')
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'passed':sum(checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v], 'topology':result['observed_topology'],'mass':result['reference_geometry'],'roi_count':len(roi),'process':process['state']},indent=2))
if not all(checks.values()): raise SystemExit(1)
