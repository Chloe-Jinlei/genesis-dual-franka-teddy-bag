"""Run the user's forty-percent smaller teddy from its own initial geometry."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
config=ROOT/'config_small_bear40_long_handles_trial.json'
assert sha(config)=='2be492d560204446948e7a51cb8bbf848c1464be4d9db28a224367d383fcb2b3'
assert sha(ROOT/'run.py')=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
audit=ROOT/'runtime_ipc/small-bear40-long-handles-asset-review.json';a=read(audit)
assert all(a['checks'].values()) and a['passed']==a['total']==18 and not a['physical_success']
assert sha(ROOT/'assets/bear_uniform_coarse_s51.obj')=='932a92ff1d497bd793390cb00c670440ec7e1f241858d04138b92b00eb017e12'
old=read(ROOT/'config_long_handles_centered_gravity_trial.json');new=read(config)
expected=read(ROOT/'config_long_handles_centered_gravity_trial.json')
expected['bear']['mesh']='bear_uniform_coarse_s51.obj';expected['bear']['maxvolume']*=.6**3
expected['task']['bear_grasp_local']=[v*.6 for v in expected['task']['bear_grasp_local']]
expected['task']['bear_grasp_tracking_radius_m']*=.6;expected['task']['bear_finger_force_limit']=1.08
assert expected==new
out=ROOT/'runs/two_panel_full_27';log=out.with_suffix('.log');launch=ROOT/'runtime_ipc/full27-launch.json';archive=ROOT/'runtime_ipc/full27-small-bear40-launch-inputs'
assert not any(p.exists() for p in [out,log,launch,archive]);archive.mkdir()
for name in ['scale_bear_asset.py','build_long_handle_bag.py','two_panel_builder.py',config.name,'assets/bear_uniform_coarse_s51.json']:
    p=archive/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
command=['bash',str(ROOT/'run.sh'),'--headless','--task','full','--config',str(config),'--output',out.name]
with log.open('x') as f:process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch.write_text(json.dumps({'started_utc':now,'pid':process.pid,'command':command,'output':str(out),'log':str(log),'user_request':'把熊的尺寸变小一点；可以缩小40%','interpretation':'Uniform linear scale0.6 from the current s85 teddy, yielding original scale0.51.','physical_success_established':False,'config_sha256':sha(config),'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},'asset_review_sha256':sha(audit),'additional_archived_inputs':{str(p.relative_to(ROOT)):sha(p) for p in archive.rglob('*') if p.is_file()},'changes':{'linear_size_ratio_from_current':.6,'same_density_mass_ratio':.6**3,'grip_force_cap_per_finger_n':1.08,'force_rationale':'Original3N multiplied by linear scale squared to preserve approximate pressure; actual native contact must still validate both-sided grip.','grasp_floor_m_above_table':.025},'scope':'Fresh complete native trial with actual new volume mesh. Actual grasp/lift, yaw/preparation/full-depth/short-release planning recompute from this run. This is not a reuse of old teddy state or proof of success.','limitations':['Old s85 frozen shape/strap reports are historical only and are not claimed to certify the new teddy.','Finite-force PD, materials, native contacts, physical guards and original final ALL-node/COM containment/retention remain unchanged.','Actual size-dependent grasp may fail; original controller stops on any original protection. No soft pins, attachment or pose injection.','Default desktop config is not switched before complete validated success.']},indent=2,ensure_ascii=False)+'\n')
p=ROOT/'dependency_manifest.json';d=read(p);d.update(active_run=str(out.relative_to(ROOT)),last_attempt=str(out.relative_to(ROOT)),latest_full_run=str(out.relative_to(ROOT)),updated_utc=now,small_bear40_candidate=config.name,latest_full_task_passed=False,next_trial_scope='Full27 actual native execution of40percent-smaller teddy; all new contact and final task outcomes unverified.');p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'pid':process.pid,'launch':str(launch),'output':str(out)}))
