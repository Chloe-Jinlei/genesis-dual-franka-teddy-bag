"""Bounded read-only CPU review of separate s85 asset/config and radius patch."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,sys,types
import numpy as np
import trimesh,tetgen,igl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT)]
from contact_insertion import _policy
old_cfg=json.loads((ROOT/'runs/two_panel_full_20/config.json').read_text())
cfg=json.loads((ROOT/'config_guarded_small_teddy_vertical_trial.json').read_text())
old_source=(ROOT/'runs/two_panel_full_20/source/run.py').read_text();source=(ROOT/'run.py').read_text()
files=[ROOT/n for n in ['run.py','scene.py','scale_bear_asset.py','config_guarded_small_teddy_vertical_trial.json','assets/bear_uniform_coarse.obj','assets/bear_uniform_coarse.mtl','assets/bear_uniform_coarse_s85.obj','assets/bear_uniform_coarse_s85.mtl','assets/bear_uniform_coarse_s85.json','assets/teddy_fur.png']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p):sha(p) for p in files}
checks=[]
def check(name,value,**kw):checks.append(dict(name=name,passed=bool(value),**kw))
parse=lambda p:[l for l in p.read_text().splitlines()]
old_lines=parse(ROOT/'assets/bear_uniform_coarse.obj');new_lines=parse(ROOT/'assets/bear_uniform_coarse_s85.obj')
def geom(lines):
 v=np.array([[float(n) for n in l.split()[1:4]] for l in lines if l.startswith('v ')])
 f=np.array([[int(n.split('/')[0])-1 for n in l.split()[1:]] for l in lines if l.startswith('f ')])
 return v,f
v,f=geom(old_lines);w,g=geom(new_lines);m=trimesh.Trimesh(w,g,process=False)
check('all_physical_vertices_are_exactly_uniform_085_scaled',np.array_equal(w,v*.85),max_error_m=float(np.max(np.abs(w-v*.85))))
check('faces_and_all_uv_indices_exactly_preserved',[l for l in old_lines if l.startswith('f ')]==[l for l in new_lines if l.startswith('f ')])
check('uv_coordinates_exactly_preserved',[l for l in old_lines if l.startswith('vt ')]==[l for l in new_lines if l.startswith('vt ')])
check('closed_consistent_one_component_positive_volume',m.is_watertight and m.is_winding_consistent and len(m.split())==1 and m.volume>0)
check('texture_material_bytes_preserved',(ROOT/'assets/bear_uniform_coarse.mtl').read_bytes()==(ROOT/'assets/bear_uniform_coarse_s85.mtl').read_bytes())
check('obj_points_to_own_material','mtllib bear_uniform_coarse_s85.mtl' in new_lines)
check('texture_remains_shared_existing_fur', 'map_Kd teddy_fur.png' in (ROOT/'assets/bear_uniform_coarse_s85.mtl').read_text())
report=json.loads((ROOT/'assets/bear_uniform_coarse_s85.json').read_text())
check('asset_provenance_hashes_and_dimensions_match',report['source_sha256']==sha(ROOT/'assets/bear_uniform_coarse.obj') and report['output_sha256']==sha(ROOT/'assets/bear_uniform_coarse_s85.obj') and np.array_equal(report['dimensions_m'],m.extents))
# Execute only the exact radius validation AST, never import production run/Genesis.
module=ast.parse(source);cls=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='Experiment');init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
i=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='bear_grasp_tracking_radius' for t in n.targets))
fragment=ast.fix_missing_locations(ast.Module(body=copy.deepcopy(init.body[i:i+2]),type_ignores=[]));code=compile(fragment,'radius_validation_isolated','exec')
for label,value,expected in [('default',None,.032),('scaled',.0272,.0272),('minimum',.005,.005),('maximum',.05,.05),('zero',0,None),('negative',-.01,None),('just_below',.004999,None),('just_above',.050001,None),('nan',float('nan'),None),('inf',float('inf'),None),('false',False,None),('true',True,None),('list',[.0272],None),('dict',{},None),('invalid_string','oops',None),('null_explicit',None,None)]:
 c={'task':{}} if label=='default' else {'task':{'bear_grasp_tracking_radius_m':value}}
 env={'self':types.SimpleNamespace(),'config':c,'np':np,'GateFailure':RuntimeError};error=None
 try:exec(code,env)
 except (RuntimeError,ValueError,TypeError) as e:error=type(e).__name__
 check('radius_'+label,(error is None and env['self'].bear_grasp_tracking_radius==expected) if expected is not None else error is not None,error=error)
# Normalize the exact optional-radius changes back to full20 and compare whole source.
normalized=source
start=source.splitlines(keepends=True);lines=source.splitlines()
normalized=normalized.replace('\n'.join(lines[init.body[i].lineno-1:init.body[i+1].end_lineno])+'\n','',1)
normalized=normalized.replace('<self.bear_grasp_tracking_radius','<.032')
probe='            arm=np.where(np.linalg.norm(rest-center,axis=1)<.032)[0]\n            if len(arm)<3: raise GateFailure("No narrow teddy upper-arm grasp region")\n'
normalized=normalized.replace(probe,'            arm=np.where(np.linalg.norm(rest-center,axis=1)<.032)[0]\n',1)
check('legacy_run_entire_source_unchanged_except_radius_and_probe_empty_region_guard',normalized==old_source)
check('both_material_roi_sites_use_one_validated_radius',source.count('axis=1)<self.bear_grasp_tracking_radius)')==2)
rest=dict(np.load(ROOT/'runs/two_panel_full_19/rest.npz'));p=rest['bear']-old_cfg['bear']['position'];c=np.asarray(old_cfg['task']['bear_grasp_local'])
base=np.flatnonzero(np.linalg.norm(p-c,axis=1)<.032);new=np.flatnonzero(np.linalg.norm(.85*p-.85*c,axis=1)<.0272)
check('scaled_roi_preserves_legacy_material_ids',np.array_equal(base,new),count=len(new))
# Exact recursive configuration diff and unchanged policy, rather than a path broad allowlist.
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(a.keys()|b.keys()):
   q=path+'.'+k if path else k
   if k not in a:out.append(dict(path=q,old='<absent>',new=b[k]))
   elif k not in b:out.append(dict(path=q,old=a[k],new='<absent>'))
   else:out.extend(diff(a[k],b[k],q))
  return out
 return [] if a==b else [dict(path=path,old=a,new=b)]
diffs=diff(old_cfg,cfg)
allowed={'bear.mesh','bear.maxvolume','task.bear_grasp_local','task.bear_grasp_tracking_radius_m','task.bear_finger_force_limit','task.contact_trial.center_offsets_xy_m','task.contact_trial.recenter_offsets_xy_m','task.contact_trial.insertion_strategy','task.contact_trial.staged'}
check('only_declared_candidate_config_fields_differ',set(d['path'] for d in diffs)==allowed)
check('bag_robot_simulation_table_and_camera_unchanged',all(cfg[k]==old_cfg[k] for k in ('bag','robots','simulation','table','camera')))
check('elastic_material_and_density_unchanged',all(cfg['bear'][k]==old_cfg['bear'][k] for k in ('E','nu','rho','friction_mu','position','nobisect','mindihedral','minratio')))
check('native_mesh_parameters_are_cubically_scaled',cfg['bear']['mesh']=='bear_uniform_coarse_s85.obj' and abs(cfg['bear']['maxvolume']-4e-6*.85**3)<1e-20 and cfg['bear']['nobisect'] is True)
check('grasp_point_radius_force_and_motor_target_consistent',np.array_equal(cfg['task']['bear_grasp_local'],np.asarray(old_cfg['task']['bear_grasp_local'])*.85) and cfg['task']['bear_grasp_tracking_radius_m']==.032*.85 and cfg['task']['bear_finger_force_limit']==3. and cfg['task']['bear_finger_target']==old_cfg['task']['bear_finger_target']==0.)
check('original_contact_policy_exactly_unchanged',_policy(cfg)==_policy(old_cfg))
ct=cfg['task']['contact_trial'];a=np.asarray(ct['center_offsets_xy_m']);b=np.asarray(ct['recenter_offsets_xy_m'])
check('four_initial_offsets_inside_original35mm_bound',a.shape==(4,2) and np.isfinite(a).all() and np.linalg.norm(a,axis=1).max()<=.035)
check('four_nonzero_corrections_inside_original30mm_bound',b.shape==(4,2) and np.isfinite(b).all() and np.all(np.linalg.norm(b,axis=1)>0) and np.linalg.norm(b,axis=1).max()<=.03)
check('vertical_y_then_x_and_yaw_existing_paths_only',ct['insertion_strategy']=='vertical' and ct['transport_xy_strategy']=='y_then_x' and 'staged' not in ct and ct['postlift_yaw']==old_cfg['task']['contact_trial']['postlift_yaw'])
check('snapshot_will_copy_new_obj_mtl_and_shared_texture', 'bear_file,Path(bear_file).with_suffix(".mtl").name,"teddy_fur.png"' in source)
check('production_and_asset_files_unchanged_during_review',before=={str(p):sha(p) for p in files})
out=dict(kind='independent_small_teddy_vertical_pretrial_review',physical_success=False,native_execution_performed=False,all_checks_passed=all(c['passed'] for c in checks),check_count=len(checks),checks=checks,config_differences=diffs,source_sha256=before,
 required_native_checks=['Actual rest mesh volume/mass and geometry must match the new asset; CPU TetGen is not native readback.','Fresh settled grasp must retain both positive contact reactions without J<.05, with original table/cloth/handle/force gates.','15-degree yaw and all transport/reobserve/recenter/insertion planning must use each new actual state; old held-state scaled geometry is only a hypothesis.','Keep full original release, sampled penetration, all-node/COM containment and final retention validation.'],
 limitations=['3N force cap is a physically motivated trial, not calibrated stress or guaranteed force closure.','Vertical move_guarded rebuilds actual .35/.5 arm references with existing sampled rate checks; full continuous-rate proof does not follow from 501 samples.','This checks asset/config/source consistency, not dynamic feasibility or task success.'])
path=ROOT/'runtime_ipc/independent-small-teddy-pretrial-review.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'all_checks_passed':out['all_checks_passed'],'check_count':len(checks),'failed':[r for r in checks if not r['passed']],'config_diff_paths':[d['path'] for d in diffs],'sha256':sha(path)},indent=2))
