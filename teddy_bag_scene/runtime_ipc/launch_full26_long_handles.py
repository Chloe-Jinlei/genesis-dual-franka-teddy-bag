"""Fresh full task after the single long-handle native probe and frozen route screen."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
config=ROOT/'config_long_handles_centered_gravity_trial.json'
assert sha(config)=='177af420224b1729cfe424fa62ce20ff95393f31443d7d922200dd276bd8731d'
assert sha(ROOT/'run.py')=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
probe=ROOT/'runs/two_panel_long_handles_handoff_01'
manifest=read(probe/'input_sha256.json')
for name,digest in manifest.items():
    if name.startswith('source/'):
        assert sha(ROOT/name.removeprefix('source/'))==digest,name
handoff=read(probe/'independent-handoff-final-review.json')
assert handoff['independent_requested_handoff_probe_passed'] is True and all(handoff['checks'].values()),handoff
hybrid_path=ROOT/'runtime_ipc/long-handles-centered-hybrid-review/review.json'
hybrid=read(hybrid_path)
assert hybrid['status']=='all_original_helper_stages_candidate_frozen_only' and hybrid['inputs_unchanged'] and not hybrid['physical_success']
for name,digest in hybrid['input_sha256'].items():assert sha(Path(name))==digest,name
# This evidence file is supplied only after the independent all-face handle check.
handles_path=ROOT/'runtime_ipc/long-handles-full26-admission.json'
handles=read(handles_path)
assert handles['admitted_for_native_contact_trial'] is True and handles['physical_success'] is False
for name,digest in handles['evidence_sha256'].items():assert sha(ROOT/name)==digest,name
out=ROOT/'runs/two_panel_full_26';log=out.with_suffix('.log')
launch=ROOT/'runtime_ipc/full26-launch.json';archive=ROOT/'runtime_ipc/full26-long-handles-launch-inputs'
assert not any(p.exists() for p in (out,log,launch,archive))
archive.mkdir()
for name in ('build_long_handle_bag.py','two_panel_builder.py','config_long_handles_centered_gravity_trial.json'):
    shutil.copy2(ROOT/name,archive/name)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
command=['bash',str(ROOT/'run.sh'),'--headless','--task','full','--config',str(config),'--output',out.name]
with log.open('x') as stream:
    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
launch.write_text(json.dumps({'started_utc':now,'pid':process.pid,'command':command,'output':str(out),'log':str(log),
    'config_sha256':sha(config),'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
    'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [probe/'independent-handoff-final-review.json',hybrid_path,handles_path]},
    'additional_archived_inputs':{str(p.relative_to(ROOT)):sha(p) for p in archive.iterdir()},
    'physical_success_established':False,'scope':'One complete from-time-zero native long-cloth-handle task. New 14cm straight legs and 1.40m handle hold; +15deg teddy yaw; COM offset[.005,.03075]m, original 3N grasp and gravity release conditions.',
    'limitations':['The successful handoff proves only its own limited actions. The hybrid geometry combines separately observed cloth and teddy shapes and does not predict this run.',
    'No state injection, cloth pins, artificial attachment, kinematic teddy placement or relaxed physical/containment gates.',
    'Actual fresh full-depth preflight and short-route checks are recomputed online. Actual stopped partial capture, both-sided finger contact and low endpoint velocity are mandatory before release.',
    'The unchanged final audit requires all teddy nodes and its mass center inside and retained after withdrawal. Partial capture is not success.']},indent=2)+'\n')
p=ROOT/'dependency_manifest.json';d=read(p);d.update(active_run=str(out.relative_to(ROOT)),last_attempt=str(out.relative_to(ROOT)),latest_full_run=str(out.relative_to(ROOT)),updated_utc=now,latest_full_task_passed=False,next_trial_scope='Full26 native long-handle from-time-zero execution in progress; no success established.');p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'pid':process.pid,'launch':str(launch),'output':str(out)}))
