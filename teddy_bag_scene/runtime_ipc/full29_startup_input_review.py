"""One bounded native-input review, using full28 as the unchanged rest baseline."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_29';OLD=ROOT/'runs/two_panel_full_28'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
launch=read(ROOT/'runtime_ipc/full29-launch.json');cfg=read(RUN/'config.json');oc=read(OLD/'config.json');manifest=read(RUN/'input_sha256.json');om=read(OLD/'input_sha256.json')
rest=dict(np.load(RUN/'rest.npz'));initial=dict(np.load(RUN/'initial.npz'));orr=dict(np.load(OLD/'rest.npz'));oi=dict(np.load(OLD/'initial.npz'))
livecfg=ROOT/'config_bear_s663_low_release_trial.json';protected=[RUN/k for k in manifest]+[RUN/'input_sha256.json',RUN/'rest.npz',RUN/'initial.npz',livecfg,ROOT/'config.json',ROOT/'runtime_ipc/full29-launch.json'];before={str(p):sha(p) for p in protected}
expected=copy.deepcopy(oc);expected['task']['contact_trial']['gravity_release']['tcp_height_above_table_m']=.175
source={k for k in manifest if k.startswith('source/')};oldsource={k for k in om if k.startswith('source/')};changed=[k for k in sorted(source&oldsource) if manifest[k]!=om[k]]
helper=(RUN/'source/gravity_release.py').read_bytes();oldhelper=(OLD/'source/gravity_release.py').read_bytes();needle=b'for key, lo, hi in (("tcp_height_above_table_m", .15, .45),';prior=needle.replace(b'.15',b'.25')
tet=rest['bear'][rest['bear_tets']];vol=np.linalg.det(np.stack([tet[:,i]-tet[:,0] for i in [1,2,3]],axis=2))/6
tri=rest['bag'][rest['bag_faces']];area=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)
roi_dist=np.linalg.norm(rest['bear']-cfg['bear']['position']-cfg['task']['bear_grasp_local'],axis=1);roi=np.flatnonzero(roi_dist<cfg['task']['bear_grasp_tracking_radius_m'])
default_ref=read(OLD/'independent-startup-input-review.json')['input_sha256'][str(ROOT/'config.json')]
checks={
 'expected_config_hash':sha(livecfg)==launch['config_sha256']=='c04a219b22ccbaa045437b3bf8351442d9afd5101a00fac025584884f4e9cd86',
 'live_and_frozen_configuration_equal':cfg==read(livecfg),
 'only_release_height_config_changed':cfg==expected,
 'all_frozen_manifest_hashes_match':all(sha(RUN/k)==h for k,h in manifest.items()),
 'manifest_covers_all_frozen_inputs':set(manifest)=={str(p.relative_to(RUN)) for p in (RUN/'source').rglob('*') if p.is_file()}|{'config.json'},
 'source_inventory_unchanged':source==oldsource,
 'only_gravity_helper_source_or_asset_changed':changed==['source/gravity_release.py'],
 'reviewed_new_helper_exact_hash':manifest['source/gravity_release.py']=='3cb7ea9a10390bcb473d353863917d39cc67a05766b2da3bfc38ed28bd26915f',
 'helper_single_height_literal_reverse_byte_identical':helper.count(needle)==1 and helper.replace(needle,prior)==oldhelper,
 'runner_unchanged_3454':manifest['source/run.py']==om['source/run.py']=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9',
 'all_frozen_python_matches_launch':all(manifest[k]==launch['source_sha256'][Path(k).name] for k in source if k.endswith('.py')),
 'all_frozen_sources_assets_match_live':all(sha(ROOT/k.removeprefix('source/'))==manifest[k] for k in source),
 'additional_launch_archives_match':all(sha(ROOT/k)==h for k,h in launch['additional_archived_inputs'].items()),
 'default_configuration_unchanged':sha(ROOT/'config.json')==default_ref,
 'fresh_zero_time':float(initial['time'])==0.,
 'all_initial_and_rest_arrays_finite':all(np.isfinite(v).all() for d in [rest,initial] for v in d.values()),
 'all_rest_arrays_exact_full28':set(rest)==set(orr) and all(np.array_equal(v,orr[k]) for k,v in rest.items()),
 'all_initial_arrays_exact_full28':set(initial)==set(oi) and all(np.array_equal(v,oi[k]) for k,v in initial.items()),
 'initial_soft_positions_equal_own_rest':np.array_equal(initial['bear'],rest['bear']) and np.array_equal(initial['bag'],rest['bag']),
 'actual_tetrahedra_all_positive':bool(np.all(vol>0)),
 'reference_bear_mass_63779537g':abs(float(vol.sum()*cfg['bear']['rho'])-.06377953685647826)<1e-10,
 'reference_bag_mass_120079644g':abs(area*cfg['bag']['rho']*2*cfg['bag']['thickness']-.1200796444646929)<1e-10,
 'same_s663_finite_force_zero_target_tracking_and_roi':cfg['bear']['mesh']=='bear_uniform_coarse_s663.obj' and cfg['task']['bear_finger_force_limit']==1.8252 and cfg['task']['bear_finger_target']==0. and cfg['task']['bear_grasp_tracking_radius_m']==.021216 and len(roi)>=3,
 'same_cloth_material_50_250':cfg['bag']==oc['bag'] and cfg['bag']['shear_stiffness']==50. and cfg['bag']['handle_shear_stiffness']==250.,
 'actual_release_target_0915_above_table_0175':abs(cfg['table']['height']+cfg['task']['contact_trial']['gravity_release']['tcp_height_above_table_m']-.915)<1e-15,
}
pid=int(launch['pid']);proc=Path('/proc')/str(pid)/'cmdline'
try:cmd=[v.decode() for v in proc.read_bytes().split(b'\0') if v]
except FileNotFoundError:cmd=[]
process={'pid':pid,'observed_utc':datetime.now(timezone.utc).isoformat(),'state':'running' if cmd else 'not_running_at_review','command':cmd}
if cmd:
 process['matches_expected']=all(flag in cmd and cmd[cmd.index(flag)+1]==value for flag,value in [('--task','full'),('--config',str(livecfg)),('--output',RUN.name)])
 checks['observed_running_command_matches_if_present']=process['matches_expected']
checks['reviewed_inputs_unchanged']=all(sha(Path(p))==h for p,h in before.items());checks={k:bool(v) for k,v in checks.items()}
result={'kind':'independent_full29_startup_input_review','created_utc':datetime.now(timezone.utc).isoformat(),'all_checks_passed':all(checks.values()),'checks':checks,'physical_task_success':False,'native_validation_pending':True,'process_observation':process,'config_change':{'path':'task.contact_trial.gravity_release.tcp_height_above_table_m','old':.345,'new':.175},'source_changes':changed,'observed_topology':{'bear_nodes':len(rest['bear']),'bear_tetrahedra':len(rest['bear_tets']),'bag_nodes':len(rest['bag']),'bag_faces':len(rest['bag_faces'])},'reference_mass':{'bear_kg':float(vol.sum()*cfg['bear']['rho']),'bag_kg':area*cfg['bag']['rho']*2*cfg['bag']['thickness'],'minimum_tetra_volume_m3':float(vol.min()),'scope':'Computed from this saved native rest and configured density; not native mass-buffer readback.'},'grasp_roi':{'radius_m':cfg['task']['bear_grasp_tracking_radius_m'],'count':len(roi),'node_indices':roi.tolist(),'local_center_m':cfg['task']['bear_grasp_local'],'configured_force_per_finger_n':1.8252},'input_sha256':before,'limitations':['This verifies actual initial time-zero inputs, not dynamic success or future release admissibility.','Recorded teddy force is the configuration/source-proven force installed when bear grasp begins; initial state is handle grasp.','Native final material-buffer readback and final original task acceptance remain pending.','No native stepping, planning, GPU, source/config/asset/default edit, or full final audit.']}
out=RUN/'independent-startup-input-review.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(out),'sha256':sha(out),'passed':sum(checks.values()),'total':len(checks),'failed':[k for k,v in checks.items() if not v],'topology':result['observed_topology'],'mass':result['reference_mass'],'process':process['state']},indent=2))
if not all(checks.values()):raise SystemExit(1)
