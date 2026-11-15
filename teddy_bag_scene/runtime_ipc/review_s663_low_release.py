"""One lower-release candidate against full28's actual pre-insertion state."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_insertion import reobserve_guarded_contact_trial, _serial
from gravity_release import plan_gravity_release

run = ROOT / 'runs/two_panel_full_28'
out = ROOT / 'runtime_ipc/s663-low-release-preflight'
assert not out.exists()
out.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg_path = ROOT / 'config_bear_s663_low_release_trial.json'
cfg = json.loads(cfg_path.read_text())
old_cfg = json.loads((run / 'config.json').read_text())
expected = json.loads(json.dumps(old_cfg))
expected['task']['contact_trial']['gravity_release']['tcp_height_above_table_m'] = .175
assert cfg == expected
tracked = [cfg_path, ROOT / 'gravity_release.py', run / 'rest.npz',
           run / 'transport_teddy.npz', run / 'guarded_contact_candidate.json',
           run / 'gravity_release_candidate.json']
before = {str(p): sha(p) for p in tracked}
cfg['robots']['mjcf_path'] = str(run / 'source/assets/panda_tactile.xml')
for key in ('npz_file', 'metadata'):
    cfg['bag'][key] = str(run / 'source/assets' / Path(cfg['bag'][key]).name)
rest = dict(np.load(run / 'rest.npz', allow_pickle=False))
state = dict(np.load(run / 'transport_teddy.npz', allow_pickle=False))
choice = json.loads((run / 'guarded_contact_candidate.json').read_text())
old_gravity = json.loads((run / 'gravity_release_candidate.json').read_text())
commands = {s: np.array(q) for s, q in old_gravity['command_q_start'].items()}
report = {'status': 'pending', 'physical_success': False,
          'source_native_time_s': float(state['time']), 'input_sha256': before,
          'scope': 'Replanning from one real full28 simultaneous state, with no geometry mutation. Frozen checks do not predict subsequent cloth dynamics or certify success.',
          'sole_config_change': 'TCP release height above table .345 to .175 m',
          'sole_source_change': 'Allow input heights from .15 instead of .25 m; physical and geometric execution checks unchanged.'}
try:
    fresh = reobserve_guarded_contact_trial(rest, state, cfg, choice,
                                           output=out / 'fresh_full_depth.json')
    assert fresh['status'] == 'candidate', fresh.get('report', {}).get('rejections')
    candidate, plan = plan_gravity_release(rest, state, cfg, fresh, commands, return_plan=True)
    assert candidate['status'] == 'candidate' and plan is not None
    (out / 'gravity_candidate.json').write_text(json.dumps(_serial(candidate), indent=2, allow_nan=False) + '\n')
    np.savez_compressed(out / 'gravity_insert_plan.npz', duration=plan.duration,
                        **{s+'_x': plan.curves[s].x for s in commands},
                        **{s+'_c': plan.curves[s].c for s in commands})
    target = np.array(candidate['insert_stage']['right_tcp_m'])
    shift = target - state['tcp_right']
    frozen_bottom = float((state['bear'] + shift)[:, 2].min())
    report.update(status='candidate', target_tcp_m=target.tolist(),
                  duration_s=plan.duration, frozen_bear_bottom_m=frozen_bottom,
                  frozen_bear_bottom_above_table_m=frozen_bottom-cfg['table']['height'],
                  full_depth_tcp_m=fresh['insert_stage']['right_tcp_m'],
                  native_execution_performed=False)
except Exception as exc:
    report.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
report['inputs_unchanged'] = all(sha(Path(p)) == h for p, h in before.items())
(out / 'review.json').write_text(json.dumps(_serial(report), indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
