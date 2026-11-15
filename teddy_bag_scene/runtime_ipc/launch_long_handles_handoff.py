"""Start the single native long-cloth-handle handoff probe with frozen provenance."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess
ROOT = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
config = ROOT / 'config_long_handles_gravity_trial.json'
delivery = read(ROOT / 'runtime_ipc/long-handles-delivery.json')
for name, digest in delivery['sha256'].items():
    assert sha(ROOT/name) == digest, name
assert sha(ROOT/'run.py') == '3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
clear_path = ROOT/'runtime_ipc/long-handles-release-clear-review/review.json'
clear = read(clear_path)
assert clear['status'] == 'original_release_clear_robot_preflight_passed_native_pending' and clear['inputs_unchanged'] and not clear['physical_success'], clear
out = ROOT/'runs/two_panel_long_handles_handoff_01'
log = out.with_suffix('.log')
record = ROOT/'runtime_ipc/two_panel_long_handles_handoff_01-launch.json'
archive = ROOT/'runtime_ipc/long-handles-handoff-launch-inputs'
assert not any(p.exists() for p in (out,log,record,archive))
archive.mkdir()
for name in ('build_long_handle_bag.py','two_panel_builder.py','config_long_handles_gravity_trial.json'):
    shutil.copy2(ROOT/name, archive/name)
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
command = ['bash',str(ROOT/'run.sh'),'--headless','--task','handoff','--config',str(config),'--output',out.name]
with log.open('x') as stream:
    process = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record.write_text(json.dumps({
    'started_utc':now,'pid':process.pid,'command':command,'log':str(log),'output':str(out),
    'task_scope':'Native two-hand opening, right handle release and clearance, then ten seconds of left-only cloth holding. No teddy handling or insertion.',
    'physical_success_established':False,'full_task_success_established':False,
    'config_sha256':sha(config),'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
    'delivery_sha256':sha(ROOT/'runtime_ipc/long-handles-delivery.json'),
    'clearance_review_sha256':sha(clear_path),
    'additional_archived_inputs':{str(p.relative_to(ROOT)):sha(p) for p in sorted(archive.iterdir())},
    'change':'Only optional cloth-handle straight-leg length .04 to .18m, actual grasp heights 1.26 to1.40m, and postlift yaw -30 to+15deg. Body points, faces and shared U-seam are unchanged. Postlift yaw is not executed by this probe.',
    'limits':'Robot-only preflight does not predict cloth draping or guarantee dynamic contact. Original finite-force, no-pins model and all physical limits remain. Default desktop config is unchanged.'
},indent=2)+'\n')
manifest = ROOT/'dependency_manifest.json'
data = read(manifest)
data.update(active_run=str(out.relative_to(ROOT)),updated_utc=now,long_handles_candidate=config.name,
    next_trial_scope='Native longer cloth handles opening and one-hand hold probe; no full insertion success established.',latest_full_task_passed=False)
manifest.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'pid':process.pid,'launch':str(record),'output':str(out)}))
