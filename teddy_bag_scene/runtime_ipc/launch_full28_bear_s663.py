"""Fresh native run for the user's latest teddy size (+30% from s51)."""
from pathlib import Path
import datetime,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
config=ROOT/'config_bear_s663_long_handles_trial.json'
assert sha(config)=='ebded9bb993ba886e24259a21d6b554c091afd9feb7b144f1d480ba3f062a1ea'
assert sha(ROOT/'run.py')=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
audit=ROOT/'runtime_ipc/bear-s663-long-handles-asset-review.json';a=read(audit)
assert all(a['checks'].values()) and a['passed']==a['total']==18 and not a['physical_success']
assert sha(ROOT/'assets/bear_uniform_coarse_s663.obj')=='0e4c2bd67b51b741b18530979e4d22fd1d718ff302ea0901bf9356d01c46dacd'
expected=read(ROOT/'config_small_bear40_long_handles_trial.json')
expected['bear']['mesh']='bear_uniform_coarse_s663.obj';expected['bear']['maxvolume']*=1.3**3
expected['task']['bear_grasp_local']=[v*1.3 for v in expected['task']['bear_grasp_local']]
expected['task']['bear_grasp_tracking_radius_m']*=1.3;expected['task']['bear_finger_force_limit']=1.8252
assert expected==read(config)
out=ROOT/'runs/two_panel_full_28';log=out.with_suffix('.log');launch=ROOT/'runtime_ipc/full28-launch.json';archive=ROOT/'runtime_ipc/full28-bear-s663-launch-inputs'
assert not any(p.exists() for p in [out,log,launch,archive]);archive.mkdir()
for name in ['scale_bear_asset.py','build_long_handle_bag.py','two_panel_builder.py',config.name,'assets/bear_uniform_coarse_s663.json']:
    p=archive/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
command=['bash',str(ROOT/'run.sh'),'--headless','--task','full','--config',str(config),'--output',out.name]
with log.open('x') as f:process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
launch.write_text(json.dumps({'started_utc':now,'pid':process.pid,'command':command,'output':str(out),'log':str(log),'user_request':'这个熊太小了，放大30%','interpretation':'Uniform1.3 linear scale from s51, yielding s663; this is78percent of the preceding s85 teddy.','physical_success_established':False,'config_sha256':sha(config),'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},'asset_review_sha256':sha(audit),'additional_archived_inputs':{str(p.relative_to(ROOT)):sha(p) for p in archive.rglob('*') if p.is_file()},'changes':{'linear_size_ratio_from_immediately_previous':1.3,'same_density_mass_ratio':1.3**3,'grip_force_cap_per_finger_n':1.8252,'force_rationale':'Previous1.08N multiplied by1.3 squared to preserve approximate pressure; actual native both-sided grip still must validate.','grasp_floor_m_above_table':.025},'scope':'Fresh complete native task with own initial volume mesh; actual grasp/lift, yaw/preparation/full-depth/short-release planning recomputed from this run.','limitations':['Earlier s85/s51 deformed states or geometry audits do not certify this teddy.','All finite-force/material/native contact/physical/whole-teddy containment and retention gates unchanged.','User-requested size edits caused previous full26/full27 interruptions, not recorded physics failures.','Default desktop config remains unvalidated and is not changed by this launcher.']},indent=2,ensure_ascii=False)+'\n')
p=ROOT/'dependency_manifest.json';d=read(p);d.update(active_run=str(out.relative_to(ROOT)),last_attempt=str(out.relative_to(ROOT)),latest_full_run=str(out.relative_to(ROOT)),interrupted_run='runs/two_panel_full_27/external_interruption.json',updated_utc=now,current_size_candidate=config.name,latest_full_task_passed=False,next_trial_scope='Full28 actual native execution of latest user-selected s663 teddy; dynamic outcomes still unverified.');p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'pid':process.pid,'launch':str(launch),'output':str(out)}))
