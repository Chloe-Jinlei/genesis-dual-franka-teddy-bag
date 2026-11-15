"""Fresh s663 run with one lateral approach adjustment to clear the held strap."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
config = ROOT / 'config_bear_s663_offset_left_trial.json'
assert sha(config) == '31c08d28f64834dcef550bdad32d25d016c014d277784428f469cf0430eddcdf'
assert sha(ROOT / 'run.py') == '3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
assert sha(ROOT / 'gravity_release.py') == '3cb7ea9a10390bcb473d353863917d39cc67a05766b2da3bfc38ed28bd26915f'
assert sha(ROOT / 'assets/bear_uniform_coarse_s663.obj') == '0e4c2bd67b51b741b18530979e4d22fd1d718ff302ea0901bf9356d01c46dacd'
assert sha(ROOT / 'runtime_ipc/audit_gravity_execution.py') == '93d3b6a33cacdcb08f7969fdd1e5e7db463ac33e7b99aae726bb439f8d021177'
assert sha(ROOT / 'runtime_ipc/independent-low-release-policy-review.json') == '8bdcdb2756465a70fe5e13727f29247b061ecbdb5b8fe123307b6dc7d21d6dc7'
assert sha(ROOT / 'runtime_ipc/independent-s663-offset-left-input-review.json') == 'd3df84ed7a5dafeee235ebcd74855123ca721e4933a120a2631b2bf1eacc3a40'
assert sha(ROOT / 'runs/two_panel_full_29/independent-minus-x15-palm-held-front-review-v2.json') == 'aecfadad451b0f3bc254263d0f8a34b0e4a296d0cd826649e5d292135c450044'
expected = read(ROOT / 'runs/two_panel_full_29/config.json')
expected['task']['contact_trial']['center_offsets_xy_m'] = [[-.01, .03075]]
assert expected == read(config)
preflight = ROOT / 'runtime_ipc/s663-offset-left-preflight/review.json'
p = read(preflight)
assert p['status'] == 'candidate' and p['inputs_unchanged'] and not p['physical_success']
assert read(config)['task']['contact_trial']['gravity_release']['tcp_height_above_table_m'] == .175
for path, digest in p['input_sha256'].items():
    assert sha(Path(path)) == digest
out = ROOT / 'runs/two_panel_full_30'
log = out.with_suffix('.log')
launch = ROOT / 'runtime_ipc/full30-launch.json'
archive = ROOT / 'runtime_ipc/full30-offset-left-launch-inputs'
assert not any(p.exists() for p in (out, log, launch, archive))
archive.mkdir()
for name in ('scale_bear_asset.py', 'build_long_handle_bag.py', config.name,
             'runtime_ipc/audit_gravity_execution.py',
             'runtime_ipc/independent-low-release-policy-review.json',
             'runtime_ipc/independent-s663-offset-left-input-review.json',
             'runs/two_panel_full_29/independent-minus-x15-palm-held-front-review-v2.json',
             'runtime_ipc/s663-offset-left-preflight/review.json',
             'runtime_ipc/s663-offset-left-preflight/initial_candidate.json'):
    target = archive / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / name, target)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
command = ['bash', str(ROOT / 'run.sh'), '--headless', '--task', 'full',
           '--config', str(config), '--output', out.name]
with log.open('x') as f:
    process = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
launch.write_text(json.dumps({
    'started_utc': now, 'pid': process.pid, 'command': command, 'output': str(out), 'log': str(log),
    'config_sha256': sha(config), 'source_sha256': {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))},
    'additional_archived_inputs': {str(p.relative_to(ROOT)): sha(p) for p in archive.rglob('*') if p.is_file()},
    'user_selected_size': 's51 enlarged by 30 percent, s663, unchanged from full28',
    'physical_success_established': False,
    'change': 'COM target X offset +5 mm to -10 mm, Y +30.75 mm unchanged. Same .915 m low release, current s663 teddy and all physics/acceptance limits.',
    'reason': 'Full29 palm loaded the held front strap near its right root and exceeded the original TCP torque limit. Shift -15 mm X follows the measured separation direction.',
    'scope': 'Fresh native simulation from time zero; every measured-state plan and release check recomputed. Full29 frozen geometry preflight is not future dynamic proof.'
}, indent=2) + '\n')
path = ROOT / 'dependency_manifest.json'
d = read(path)
d.update(active_run=str(out.relative_to(ROOT)), last_attempt=str(out.relative_to(ROOT)),
         latest_full_run=str(out.relative_to(ROOT)), current_size_candidate=config.name,
         updated_utc=now, latest_full_task_passed=False,
         next_trial_scope='Full30 fresh native lateral-alignment run, same current s663 size and low release; dynamic outcome pending.')
path.write_text(json.dumps(d, indent=2) + '\n')
print(json.dumps({'pid': process.pid, 'launch': str(launch), 'output': str(out)}))
