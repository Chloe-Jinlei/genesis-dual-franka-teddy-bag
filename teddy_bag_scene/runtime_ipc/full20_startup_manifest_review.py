"""One read-only full20 startup audit. Does not initialize Genesis or poll a run."""
from pathlib import Path
import hashlib,json,sys,subprocess,datetime,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_20';PREV=ROOT/'runs/two_panel_full_19'
sys.path[:0]=[str(RUN/'source'),str(ROOT/'runtime_ipc')]
from audit_input_manifest import audit
from validate import mouth_structure
from physics_config import _map_shell_material_regions
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=audit(RUN);cfg=json.loads((RUN/'config.json').read_text());old=json.loads((PREV/'config.json').read_text());launch=json.loads((ROOT/'runtime_ipc/full20-launch.json').read_text());checks={}
def require(name,value):checks[name]=bool(value)
def diff(a,b,prefix=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(set(a)|set(b)):
   path=prefix+'.'+k if prefix else k
   if k not in a:out.append(dict(path=path,before='__missing__',after=b[k]))
   elif k not in b:out.append(dict(path=path,before=a[k],after='__missing__'))
   else:out+=diff(a[k],b[k],path)
  return out
 return [] if a==b else [dict(path=prefix,before=a,after=b)]
differences=diff(old,cfg)
require('all_30_frozen_hashes_match',manifest['manifest_entry_count']==30 and manifest['all_declared_hashes_match'])
require('all_frozen_snapshot_files_manifested',manifest['all_snapshot_files_in_manifest'])
require('all_external_robot_mesh_files_exist',not manifest['missing_external_model_files'])
require('config_only_adds_requested_postlift_yaw',differences==[dict(path='task.contact_trial.postlift_yaw',before='__missing__',after={'world_z_degrees':15.,'duration_s':6.})])
require('launch_source_config_hash_matches',sha(Path(launch['config']))==launch['config_sha256'])
require('frozen_config_semantically_matches_launch_config',json.loads(Path(launch['config']).read_text())==cfg)
expected={'run.py':'dde3320d8b13e7370d61079614e0cc6fe7029256de6bd589f48ebe16d099eb9b','postlift_yaw.py':'dfd8c08c8d63516cdbd6c58e466b1497114a4ec58a2390fa77906d7cd0f5928c'}
require('reviewed_production_and_frozen_run_helper_hashes_match',all(sha(ROOT/name)==sha(RUN/'source'/name)==digest==launch['source_frozen'][name] for name,digest in expected.items()))
source_rows=[]
for name in json.loads((RUN/'input_sha256.json').read_text()):
 if not name.startswith('source/'):continue
 local=ROOT/name.removeprefix('source/');frozen=RUN/name
 source_rows.append(dict(path=name,current_matches_frozen=local.is_file() and sha(local)==sha(frozen)))
require('all_29_production_source_assets_match_snapshot',len(source_rows)==29 and all(v['current_matches_frozen'] for v in source_rows))
previous_manifest=json.loads((PREV/'input_sha256.json').read_text());current_manifest=json.loads((RUN/'input_sha256.json').read_text());common=set(previous_manifest)&set(current_manifest)
changed=sorted(k for k in common if k.startswith('source/') and previous_manifest[k]!=current_manifest[k]);added=sorted(set(current_manifest)-set(previous_manifest))
require('only_run_source_changed_and_yaw_helper_added_since_full19',changed==['source/run.py'] and added==['source/postlift_yaw.py'])
asset=RUN/'source/assets'/cfg['bag']['npz_file'];a=dict(np.load(asset,allow_pickle=False));rest=dict(np.load(RUN/'rest.npz',allow_pickle=False));initial=dict(np.load(RUN/'initial.npz',allow_pickle=False));meta=json.loads((RUN/'source/assets'/cfg['bag']['metadata']).read_text())['bag']
front=set(map(int,a['panel_front']));back=set(map(int,a['panel_back']));seam=set(map(int,a['u_seam']));body_faces=set(map(int,a['body_face_indices']));pf=set(map(int,a['panel_front_face_indices']));pb=set(map(int,a['panel_back_face_indices']));fg=a['panel_front_grid'];bg=a['panel_back_grid']
require('two_panels_share_exactly_u_seam',front&back==seam and np.array_equal(fg[0],bg[0]) and np.array_equal(fg[:,0],bg[:,0]) and np.array_equal(fg[:,-1],bg[:,-1]))
require('exactly_two_panel_face_sets_cover_body',not pf&pb and pf|pb==body_faces and len(pf)==1480 and len(pb)==1480)
zero_width=float(np.ptp(a['points'][a['bottom_seam'],1]));require('zero_bottom_width_no_bottom_panel_or_support',zero_width==0 and meta['depth_m']==0 and meta['bottom_seam_transverse_depth_m']==0 and meta['bottom_seam_has_area'] is False and meta['hidden_rigid_support'] is False and meta['fixed_particles']==0)
body,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg)
require('native_import_is_open_two_panel_u_seam_body',topology['actual_open_body_boundary'] and topology['two_panels_directly_sewn_along_u'] and topology['body_triangle_count']==2960 and topology['independent_flat_bottom_triangles']==0 and topology['independent_flat_side_triangles']==0)
regions,mapping=_map_shell_material_regions(asset,cfg['bag']['position'],rest['bag'],rest['bag_faces'])
require('native_rest_material_partition_exact_body2960_handles352_each',{k:len(v) for k,v in regions.items()}=={'body':2960,'handle_front':352,'handle_back':352} and mapping['body_and_handles_disjoint_complete_face_partition'])
require('same_native_material_implementation_as_full19',all(sha(RUN/'source'/f)==sha(PREV/'source'/f) for f in ['scene.py','physics_config.py']))
require('declared_effective_body50_handle250_shear',cfg['bag']['shear_stiffness']==50. and cfg['bag']['handle_shear_stiffness']==250.)
require('all_material_and_physics_settings_unchanged',all(cfg[k]==old[k] for k in ['bag','bear','simulation','robots','table']))
require('initial_native_geometry_is_finite',all(np.isfinite(initial[k]).all() for k in ['bag','bear','q_left','q_right','tcp_left','tcp_right']) and float(initial['time'])==0.)
genesis=ROOT.parent/'genesis-world';commit=subprocess.run(['git','rev-parse','HEAD'],cwd=genesis,text=True,capture_output=True,check=True).stdout.strip();status=subprocess.run(['git','status','--porcelain'],cwd=genesis,text=True,capture_output=True,check=True).stdout
require('genesis_worktree_clean_at_review',not status.strip())
dependency=json.loads((ROOT/'dependency_manifest.json').read_text());require('genesis_commit_matches_recorded_environment',commit==dependency['genesis_source']['commit']['stdout'].strip())
previous_external=json.loads((PREV/'independent-input-manifest-review.json').read_text())['external_model_files'];old_external={x['path']:x['sha256_at_review'] for x in previous_external}
require('external_robot_meshes_unchanged_from_prior_audit',all(old_external.get(x['path'])==x['sha256_at_review'] for x in manifest['external_model_files']))
log=Path(launch['log']).read_text();require('native_initialization_completed_and_physics_began','BUILD_DONE' in log and 'PHASE close_both_handles' in log)
proc=Path('/proc')/str(launch['pid']);process=dict(pid=launch['pid'],exists=proc.exists(),command_line=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode() if proc.exists() else None)
report_path=RUN/'report.json';native_status='pending_final_report';native_evidence=None
if report_path.exists():
 final=json.loads(report_path.read_text());native_evidence=final.get('shell_conversion');native_status='report_available_not_inferred_from_config'
manifest.update(kind='independent_full20_startup_input_manifest_review',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physical_success=False,runtime_execution_performed_by_review=False,
 startup_checks_passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],launch_record=launch,process_observation=process,
 config_diff_from_full19=differences,config_serialization=dict(launch_source_sha256=sha(Path(launch['config'])),frozen_sha256=sha(RUN/'config.json'),byte_difference_is_only_trailing_newline=Path(launch['config']).read_bytes().rstrip(b'\n')==(RUN/'config.json').read_bytes()),changed_common_source_files_from_full19=changed,added_source_files_from_full19=added,production_source_asset_comparison=source_rows,
 native_imported_rest_topology=topology,native_rest_material_mapping=dict(maximum_vertex_mapping_error_m=mapping['maximum_vertex_mapping_error_m'],vertex_bijection=mapping['vertex_bijection'],face_bijection=mapping['face_bijection'],face_counts={k:len(v) for k,v in regions.items()}),
 panel_construction=dict(front_faces=len(pf),back_faces=len(pb),shared_u_seam_vertices=len(seam),bottom_transverse_width_m=zero_width,hidden_rigid_support=False,fixed_particles=0),
 material_configuration=dict(bag=cfg['bag'],bear=cfg['bear'],full_cloth_thickness_m=2*cfg['bag']['thickness'],effective_stiffness_is_uncalibrated=True),
 actual_native_material_readback=dict(status=native_status,verified_now=False,reason='run.py initializes shell_conversion in memory and writes report.json only at finish; startup config/hooks/topology are verified, actual runtime coefficient record awaits final report.',available_report_evidence=native_evidence),
 genesis_source=dict(path=str(genesis),commit=commit,git_status_porcelain=status,source_files_modified_at_review=False),
 reviewed_source_hashes=expected)
manifest['limitations'] += ['This is one startup audit only. The run remains unfinished and no native task-success conclusion is available.',
 'Body 50 and handles 250 N/m are verified configured effective shear plus unchanged initialization implementation; actual native attribute readback remains pending final report.',
 'A clean Genesis checkout does not mean the scene has no runtime adapters: the frozen scene-local initialization, cloth-shear, STC-reaction and rigid-contact adapters remain explicitly present and unchanged.',
 'Dependency status and external mesh hashes are observed at audit time, not a new complete package/binary provenance certificate.']
out=RUN/'independent-input-manifest-review.json';out.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':manifest['startup_checks_passed'],'checks':len(checks),'failed':manifest['failed_checks'],'manifest_entries':manifest['manifest_entry_count'],'native_readback':native_status,'config_diff':differences,'genesis_commit':commit},indent=2));print('report_sha256',sha(out))
