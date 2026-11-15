"""One bounded read-only comparison of frozen full25 startup inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/two_panel_full_25'
OLD = ROOT/'runs/two_panel_full_24'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def differences(old, new, path=''):
    if isinstance(old, dict) and isinstance(new, dict):
        return sum((differences(old.get(k), new.get(k), path+'.'+k)
                    for k in sorted(old.keys() | new.keys())), [])
    return [] if old == new else [dict(path=path, old=old, new=new)]


launch = read(ROOT/'runtime_ipc/full25-launch.json')
decision_path = ROOT/launch['pretrial_decision']
decision = read(decision_path)
config = read(RUN/'config.json'); previous = read(OLD/'config.json')
manifest = read(RUN/'input_sha256.json'); old_manifest = read(OLD/'input_sha256.json')
review = read(ROOT/'runtime_ipc/independent-gravity-release-review.json')
initial = dict(np.load(RUN/'initial.npz', allow_pickle=False))
rest = dict(np.load(RUN/'rest.npz', allow_pickle=False))
old_initial = dict(np.load(OLD/'initial.npz', allow_pickle=False))
old_rest = dict(np.load(OLD/'rest.npz', allow_pickle=False))
live_config_path = Path(launch['command'][launch['command'].index('--config')+1])
protected = [RUN/name for name in manifest]+[RUN/'rest.npz', RUN/'initial.npz',
              ROOT/'config.json', live_config_path, decision_path, ROOT/'runtime_ipc/full25-launch.json']
before = {str(path): sha(path) for path in protected}
changes = differences(previous, config)
new_sources = {name: digest for name, digest in manifest.items() if name.startswith('source/')}
old_sources = {name: digest for name, digest in old_manifest.items() if name.startswith('source/')}
added = sorted(new_sources.keys()-old_sources.keys())
removed = sorted(old_sources.keys()-new_sources.keys())
changed = sorted(name for name in new_sources.keys() & old_sources.keys() if new_sources[name] != old_sources[name])
pid = int(launch['pid']); proc = Path('/proc')/str(pid)
cmd = proc.joinpath('cmdline').read_bytes().split(b'\0') if proc.exists() else []
cmd = [item.decode() for item in cmd if item]
gravity = dict(mode='gravity_seating', tcp_height_above_table_m=.345,
               minimum_inside_vertex_fraction=.5, settle_seconds=4.)
expected_sources = {
    'source/gravity_release.py': 'd88f4a7cd34853fb8f07eb34d42e137007cd553f69ddc920a0856a0710c96dee',
    'source/run.py': 'f774826d687eb6662bd1d4c491382214eac2fce22e554fd0ebdf9bf6fba0a2a6',
}
checks = dict(
    config_only_adds_gravity_policy=changes == [dict(path='.task.contact_trial.gravity_release', old=None, new=gravity)],
    launch_config_hash_exact=launch['config_sha256'] == 'a07e36055578a5aeb7bd27262112a6dbbdf88c0db41ecc8a7c578113bf33a861'
        and sha(live_config_path) == launch['config_sha256'],
    live_and_frozen_config_semantically_identical=read(live_config_path) == config,
    frozen_manifest_all_matches=all(sha(RUN/name) == digest for name, digest in manifest.items()),
    exact_reviewed_gravity_and_runner_sources=all(new_sources.get(name) == digest for name, digest in expected_sources.items()),
    only_runner_changed_and_gravity_added_to_frozen_sources=added == ['source/gravity_release.py'] and not removed and changed == ['source/run.py'],
    all_other_frozen_sources_assets_match_full24=all(sha(OLD/name) == digest for name, digest in new_sources.items() if name not in expected_sources),
    frozen_sources_assets_match_current_live=all(sha(ROOT/name.removeprefix('source/')) == digest for name, digest in new_sources.items()),
    rest_arrays_identical_full24=set(rest) == set(old_rest) and all(np.array_equal(value, old_rest[name]) for name, value in rest.items()),
    initial_arrays_identical_full24=set(initial) == set(old_initial) and all(np.array_equal(value, old_initial[name]) for name, value in initial.items()),
    initial_physical_clock_zero=float(initial['time']) == 0.,
    startup_coordinates_and_robot_states_finite=all(np.isfinite(initial[name]).all() for name in ['bag', 'bear', 'q_left', 'q_right', 'tcp_left', 'tcp_right']),
    decision_exact_frozen_hash=sha(decision_path) == launch['pretrial_decision_sha256'] == '522c36304c8bb273bebab9090e573f8f8f163c2f5a22414768653574b043d097',
    decision_production_source_hashes_match=all(sha(ROOT/name) == digest for name, digest in decision['source_sha256'].items()),
    decision_evidence_hashes_match=all(sha(ROOT/name) == digest for name, digest in decision['evidence_sha256'].items()),
    default_config_unchanged_from_prelaunch_review=sha(ROOT/'config.json') == review['production_source_sha256']['config.json'],
    actual_native_process_uses_full25_config=bool(cmd) and '--task' in cmd and cmd[cmd.index('--task')+1] == 'full'
        and '--config' in cmd and Path(cmd[cmd.index('--config')+1]) == live_config_path
        and '--output' in cmd and cmd[cmd.index('--output')+1] == 'two_panel_full_25',
    original_upper_arm_3N_grasp_yaw_and_fixed_XY_policy_preserved=config['task']['bear_finger_force_limit'] == 3.
        and config['task']['bear_finger_target'] == 0.
        and config['task']['bear_grasp_local'] == previous['task']['bear_grasp_local']
        and config['task']['contact_trial']['postlift_yaw'] == {'world_z_degrees': -30., 'duration_s': 6.}
        and config['task']['contact_trial']['center_offsets_xy_m'] == [[.015, .03075]]
        and config['task']['contact_trial']['recenter_enabled'] is False,
)
checks['startup_inputs_unchanged_during_review'] = all(sha(path) == digest for path, digest in before.items())
result = dict(kind='independent_full25_startup_input_review', created_utc=datetime.now(timezone.utc).isoformat(),
              all_checks_passed=all(checks.values()), checks=checks, physical_task_success=False,
              native_validation_pending=True, config_diff=changes, source_changes=dict(added=added, removed=removed, changed=changed),
              expected_reviewed_sources=expected_sources, launch_pid=pid, observed_process_command=cmd,
              initial_clock_s=float(initial['time']), topology=dict(bear_nodes=len(rest['bear']), bear_tetrahedra=len(rest['bear_tets']),
              bag_nodes=len(rest['bag']), bag_faces=len(rest['bag_faces'])),
              input_sha256=before,
              scope='Frozen files, live launch identity and exact initial/rest array equality. No native stepping or planning was performed.',
              limitations=['Equal t=0/rest arrays demonstrate the same saved original initial state; no successful later grasp, release or containment is implied.',
                           'Source and material configuration are unchanged except the explicitly reviewed runner/new release helper. No final native material readback is claimed.',
                           'The initialization review does not certify all installed simulator binaries or future native trajectory. Original physical gates and final full containment remain pending.'])
output = RUN/'independent-startup-input-review.json'
if output.exists(): raise RuntimeError('Do not overwrite the completed startup report')
output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(path=str(output), sha256=sha(output), checks=len(checks), all_checks_passed=all(checks.values()),
                     failed_checks=[name for name, passed in checks.items() if not passed]), indent=2))
if not all(checks.values()): raise SystemExit(1)
