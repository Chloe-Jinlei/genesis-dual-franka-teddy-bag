"""Single -15 mm X placement candidate from an actual post-yaw native state."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from contact_insertion import plan_guarded_contact_trial, _serial

run = ROOT / 'runs/two_panel_full_29'
out = ROOT / 'runtime_ipc/s663-offset-left-preflight'
assert not out.exists()
out.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg_path = ROOT / 'config_bear_s663_offset_left_trial.json'
cfg = json.loads(cfg_path.read_text())
expected = json.loads((run / 'config.json').read_text())
expected['task']['contact_trial']['center_offsets_xy_m'] = [[-.01, .03075]]
assert cfg == expected
tracked = [cfg_path, ROOT / 'run.py', ROOT / 'gravity_release.py',
           ROOT / 'contact_insertion.py', run / 'rest.npz',
           run / 'orient_teddy_postlift.npz']
before = {str(p): sha(p) for p in tracked}
cfg['robots']['mjcf_path'] = str(run / 'source/assets/panda_tactile.xml')
for key in ('npz_file', 'metadata'):
    cfg['bag'][key] = str(run / 'source/assets' / Path(cfg['bag'][key]).name)
rest = dict(np.load(run / 'rest.npz', allow_pickle=False))
state = dict(np.load(run / 'orient_teddy_postlift.npz', allow_pickle=False))
candidate = plan_guarded_contact_trial(rest, state, cfg, output=out / 'initial_candidate.json')
report = dict(status=candidate['status'], physical_success=False,
              native_execution_performed=False,
              source_native_time_s=float(state['time']), input_sha256=before,
              sole_config_change='COM X offset +5 mm to -10 mm; Y offset and release height unchanged.',
              scope='Single candidate from a real simultaneous post-yaw state. Frozen geometry planning does not predict new cloth motion or future contact forces.',
              rejections=candidate.get('report', {}).get('rejections'),
              inputs_unchanged=all(sha(Path(p)) == h for p, h in before.items()))
if candidate['status'] == 'candidate':
    report.update(outside_tcp_m=candidate['route']['outside_tcp_m'],
                  full_depth_tcp_m=candidate['route']['insert_tcp_m'],
                  stages=[dict(name=s['name'], duration_s=s['duration_s'])
                          for s in candidate['stages_to_execute']])
(out / 'review.json').write_text(json.dumps(_serial(report), indent=2, allow_nan=False) + '\n')
print(json.dumps(report, indent=2))
