"""Run the accepted task with only three bag material coefficients changed."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
config = ROOT / 'config_cotton_linen_material.json'
baseline = ROOT / 'runs/two_panel_full_30'
expected = read(baseline / 'config.json')
expected['bag'].update(E=20000000.0, shear_stiffness=1000.0,
                       handle_shear_stiffness=2000.0)
assert read(config) == expected
assert sha(config) == 'bb5fa17e4a7725ef2efcba05aaaebf4538b8fc32996891ca5020d0813068c85a'
for name, digest in read(baseline / 'input_sha256.json').items():
    if name.startswith('source/'):
        assert sha(ROOT / name.removeprefix('source/')) == digest, name
out = ROOT / 'runs/two_panel_full_31'
log = out.with_suffix('.log')
launch = ROOT / 'runtime_ipc/full31-launch.json'
assert not any(p.exists() for p in (out, log, launch))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
command = ['bash', str(ROOT / 'run.sh'), '--headless', '--task', 'full',
           '--config', str(config), '--output', out.name]
with log.open('x') as stream:
    process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
launch.write_text(json.dumps({
    'started_utc': now, 'pid': process.pid, 'command': command,
    'output': str(out), 'log': str(log), 'config_sha256': sha(config),
    'source_sha256': {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))},
    'baseline_run': str(baseline),
    'change': 'Only bag E 4e6 to 20e6 Pa, body shear 50 to 1000 N/m, handle shear 250 to 2000 N/m.',
    'unchanged': 'All motion, task, geometry, teddy, bending, mass, thickness, friction, solver and acceptance settings.',
    'physical_success_established': False,
}, indent=2) + '\n')
path = ROOT / 'dependency_manifest.json'
d = read(path)
d.update(active_run=str(out.relative_to(ROOT)), last_attempt=str(out.relative_to(ROOT)),
         latest_full_run=str(out.relative_to(ROOT)), current_candidate=config.name,
         updated_utc=now, latest_full_task_passed=False,
         next_trial_scope='Full31 native material regression; unchanged accepted motion, only three bag elastic coefficients changed. Outcome pending; default still full30.')
path.write_text(json.dumps(d, indent=2) + '\n')
print(json.dumps({'pid': process.pid, 'launch': str(launch), 'output': str(out)}))
