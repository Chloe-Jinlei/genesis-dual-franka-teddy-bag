"""Fresh s663 run with lower physical placement before deliberate release."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
config = ROOT / 'config_bear_s663_low_release_trial.json'
assert sha(config) == 'c04a219b22ccbaa045437b3bf8351442d9afd5101a00fac025584884f4e9cd86'
assert sha(ROOT / 'run.py') == '3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
assert sha(ROOT / 'gravity_release.py') == '3cb7ea9a10390bcb473d353863917d39cc67a05766b2da3bfc38ed28bd26915f'
assert sha(ROOT / 'assets/bear_uniform_coarse_s663.obj') == '0e4c2bd67b51b741b18530979e4d22fd1d718ff302ea0901bf9356d01c46dacd'
assert sha(ROOT / 'runtime_ipc/audit_gravity_execution.py') == '93d3b6a33cacdcb08f7969fdd1e5e7db463ac33e7b99aae726bb439f8d021177'
assert sha(ROOT / 'runtime_ipc/independent-low-release-policy-review.json') == '8bdcdb2756465a70fe5e13727f29247b061ecbdb5b8fe123307b6dc7d21d6dc7'
expected = read(ROOT / 'runs/two_panel_full_28/config.json')
expected['task']['contact_trial']['gravity_release']['tcp_height_above_table_m'] = .175
assert expected == read(config)
preflight = ROOT / 'runtime_ipc/s663-low-release-preflight/review.json'
p = read(preflight)
assert p['status'] == 'candidate' and p['inputs_unchanged'] and not p['physical_success']
assert p['target_tcp_m'][2] == .915
for path, digest in p['input_sha256'].items():
    assert sha(Path(path)) == digest
out = ROOT / 'runs/two_panel_full_29'
log = out.with_suffix('.log')
launch = ROOT / 'runtime_ipc/full29-launch.json'
archive = ROOT / 'runtime_ipc/full29-low-release-launch-inputs'
assert not any(p.exists() for p in (out, log, launch, archive))
archive.mkdir()
for name in ('scale_bear_asset.py', 'build_long_handle_bag.py', config.name,
             'runtime_ipc/audit_gravity_execution.py',
             'runtime_ipc/independent-low-release-policy-review.json',
             'runtime_ipc/s663-low-release-preflight/review.json',
             'runtime_ipc/s663-low-release-preflight/gravity_candidate.json'):
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
    'change': 'Release TCP 1.085 to .915 m. Policy accepts heights >= .15 m above table; all geometric, contact, material, grip and final acceptance conditions unchanged.',
    'reason': 'Full28 released successfully but a foot element compressed at lower bag/table impact after unloading the fingers.',
    'scope': 'Fresh native simulation from time zero; every measured-state plan and release check recomputed. Full28 frozen geometry preflight is not future dynamic proof.'
}, indent=2) + '\n')
path = ROOT / 'dependency_manifest.json'
d = read(path)
d.update(active_run=str(out.relative_to(ROOT)), last_attempt=str(out.relative_to(ROOT)),
         latest_full_run=str(out.relative_to(ROOT)), current_size_candidate=config.name,
         updated_utc=now, latest_full_task_passed=False,
         next_trial_scope='Full29 fresh native lower-release run, same current s663 size; dynamic outcome pending.')
path.write_text(json.dumps(d, indent=2) + '\n')
print(json.dumps({'pid': process.pid, 'launch': str(launch), 'output': str(out)}))
