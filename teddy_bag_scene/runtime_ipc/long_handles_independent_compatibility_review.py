"""Independent CPU-only asset and source-contract review; never starts Genesis."""
from pathlib import Path
import hashlib,json,ast
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldcfg=json.loads((ROOT/'config_gravity_seating_trial.json').read_text())
cfg=json.loads((ROOT/'config_long_handles_gravity_trial.json').read_text())
paths=[ROOT/n for n in ['run.py','scene.py','validate.py','physics_config.py','motion.py','gravity_release.py','contact_insertion.py','postlift_yaw.py','two_panel_builder.py','build_long_handle_bag.py','config_gravity_seating_trial.json','config_long_handles_gravity_trial.json']]
paths += [ROOT/'assets'/c['bag'][k] for c in [oldcfg,cfg] for k in ['mesh','npz_file','metadata']]
before={str(p.relative_to(ROOT)):sha(p) for p in paths}
old=dict(np.load(ROOT/'assets'/oldcfg['bag']['npz_file'],allow_pickle=False));new=dict(np.load(ROOT/'assets'/cfg['bag']['npz_file'],allow_pickle=False))
om=json.loads((ROOT/'assets'/oldcfg['bag']['metadata']).read_text());nm=json.loads((ROOT/'assets'/cfg['bag']['metadata']).read_text())
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum((diff(a.get(k),b.get(k),path+'/'+k) for k in sorted(a.keys()|b.keys())),[])
 return [] if a==b else [dict(path=path,old=a,new=b)]
def area(data,ids=None):
 t=data['points'][data['faces'] if ids is None else data['faces'][ids]]
 return float(np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2)
def topology(data):
 faces=data['faces'];edges=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);ue,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
 graph={};
 for a,b in ue:
  graph.setdefault(int(a),set()).add(int(b));graph.setdefault(int(b),set()).add(int(a))
 def components(g):
  left=set(g);number=0
  while left:
   number+=1;pending=[left.pop()]
   while pending:
    for v in g[pending.pop()] & left:left.remove(v);pending.append(v)
  return number
 boundary={}
 for a,b in ue[counts==1]:boundary.setdefault(int(a),set()).add(int(b));boundary.setdefault(int(b),set()).add(int(a))
 tri=data['points'][faces];ar=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
 directed={}
 for a,b in edges:directed[(int(a),int(b))]=directed.get((int(a),int(b)),0)+1
 consistent=all(directed.get((int(a),int(b)),0)==1 and directed.get((int(b),int(a)),0)==1 for (a,b),n in zip(ue,counts) if n==2)
 return dict(vertices=len(data['points']),faces=len(faces),components=components(graph),nonmanifold_edges=int((counts>2).sum()),boundary_loops=components(boundary),boundary_degree_two=all(len(v)==2 for v in boundary.values()),consistent_winding=consistent,minimum_triangle_area_m2=float(ar.min()),area_m2=float(ar.sum()))
keys=['body','body_face_indices','panel_front_grid','panel_back_grid','panel_front','panel_back','panel_front_face_indices','panel_back_face_indices','u_seam','bottom_seam','seam_left','seam_right','rim','material_x','material_height']
checks={k+'_identical':bool(np.array_equal(old[k],new[k])) for k in keys}
checks.update(body_points_bit_identical=np.array_equal(old['points'][old['body']],new['points'][new['body']]),body_faces_and_orientation_bit_identical=np.array_equal(old['faces'][old['body_face_indices']],new['faces'][new['body_face_indices']]),body_uv_bit_identical=np.array_equal(old['uv'][old['body']],new['uv'][new['body']]),bottom_zero_yz_width=bool(np.ptp(new['points'][new['bottom_seam'],1])==0 and np.ptp(new['points'][new['bottom_seam'],2])==0),front_back_share_all_three_U_edges=all(np.array_equal(new['panel_front_grid'][s],new['panel_back_grid'][s]) for s in [(0,slice(None)),(slice(None),0),(slice(None),-1)]))
regions={};coverage=np.zeros(len(new['faces']),int)
for name,ids in [('body',new['body_face_indices'])]+[(s,np.flatnonzero(np.isin(new['faces'],new[s]).all(axis=1))) for s in ['handle_front','handle_back']]:
 coverage[ids]+=1;regions[name]=dict(face_count=len(ids),area_m2=area(new,ids))
checks['body_two_handles_disjoint_complete']=bool(np.all(coverage==1))
handles={}
for side in ['front','back']:
 for end in ['left','right']:
  key='handle_'+side+'_root_'+end;checks[key+'_exact']=bool(np.array_equal(new[key],old[key]) and np.array_equal(new['points'][new[key]],old['points'][old[key]]))
 grasp='grasp_'+side+('_x_minus' if side=='front' else '_x_plus');a=old['points'][old[grasp]];b=new['points'][new[grasp]]
 checks[side+'_grasp_material_patch_translated_140mm_only']=bool(a.shape==b.shape and np.allclose(b-a,[0,0,.14],atol=1e-14,rtol=0))
 meta=nm['bag']['handles'][side];c0=np.asarray(meta['root_center_left_m']);r=float(meta['semicircle_radius_m']);extra=float(meta['vertical_leg_extra_m']);spacing=float(nm['bag']['body_sampling_m']);n=max(4,2*int(np.ceil(np.pi*r/spacing/2)));nz=max(2,int(np.ceil(extra/spacing)));angles=np.linspace(np.pi,0,n+1)
 xz=np.vstack([np.column_stack([np.full(nz,-r),np.linspace(0,extra,nz+1)[:-1]]),np.column_stack([r*np.cos(angles),extra+r*np.sin(angles)]),np.column_stack([np.full(nz,r),np.linspace(extra,0,nz+1)[1:]])]);xz[:,1]+=c0[2]
 frac=(xz[:,0]+r)/(2*r);y=np.sign(c0[1])*(abs(c0[1])+(.085-abs(c0[1]))*np.sin(np.pi*frac));line=np.column_stack([xz[:,0],y,xz[:,1]])
 distances,_=cKDTree(new['points'][new['handle_'+side]]).query(line)
 length=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum());oldlength=om['bag']['handles'][side]['centerline_length_m']
 handles[side]=dict(centerline_samples=len(line),centerline_present_in_actual_mesh_max_error_m=float(distances.max()),centerline_length_m=length,old_centerline_length_m=oldlength,added_length_m=length-oldlength,centerline_bounds_m=[line.min(0).tolist(),line.max(0).tolist()],surface_bounds_m=[new['points'][new['handle_'+side]].min(0).tolist(),new['points'][new['handle_'+side]].max(0).tolist()],grasp_patch_size=len(b),grasp_local_mean_m=b.mean(0).tolist(),old_grasp_local_mean_m=a.mean(0).tolist())
 checks[side+'_length_added_280mm']=bool(abs(length-oldlength-.28)<1e-12 and distances.max()<1e-12)
# Read physical OBJ records without loader seam splitting.
objpoints=[];objfaces=[];objuv=[]
for line in (ROOT/'assets'/cfg['bag']['mesh']).read_text().splitlines():
 f=line.split()
 if not f:continue
 if f[0]=='v':objpoints.append(list(map(float,f[1:4])))
 elif f[0]=='vt':objuv.append(list(map(float,f[1:3])))
 elif f[0]=='f':objfaces.append([int(v.split('/')[0])-1 for v in f[1:]])
checks['obj_npz_physical_topology_identical']=np.array_equal(np.asarray(objfaces),new['faces'])
checks['obj_npz_points_agree_with_export_precision']=np.allclose(objpoints,new['points'],rtol=0,atol=1e-8)
checks['obj_npz_uv_agree_with_export_precision']=np.allclose(objuv,new['uv'],rtol=0,atol=1e-8)
top=topology(new);checks['connected_oriented_nondegenerate_shell_three_loops']=top['components']==1 and top['nonmanifold_edges']==0 and top['boundary_loops']==3 and top['boundary_degree_two'] and top['consistent_winding'] and top['minimum_triangle_area_m2']>0
changes=diff(oldcfg,cfg);expected={'/bag/mesh','/bag/metadata','/bag/npz_file','/task/contact_trial/postlift_yaw/world_z_degrees','/task/open_targets/left','/task/open_targets/right'};checks['only_explicit_asset_height_yaw_config_changes']={r['path'] for r in changes}==expected
checks['all_materials_physical_and_gravity_gates_unchanged']=cfg['bear']==oldcfg['bear'] and all(cfg['bag'][k]==oldcfg['bag'][k] for k in ['E','nu','rho','thickness','bending_stiffness','friction_mu','shear_stiffness','handle_shear_stiffness']) and cfg.get('validation')==oldcfg.get('validation') and cfg['task']['contact_trial']['gravity_release']==oldcfg['task']['contact_trial']['gravity_release']
checks['runner_exact_approved_serialization_only_version']=sha(ROOT/'run.py')=='3454e304fa9fb95381f843deb53e21e373900bccf235a86ff4891fd405d953f9'
oldarea=area(old);newarea=area(new);density=cfg['bag']['rho']*2*cfg['bag']['thickness'];newmass=newarea*density;oldmass=oldarea*density
checks['all_reviewed_source_config_assets_unchanged']=all(sha(ROOT/p)==h for p,h in before.items())
report=dict(kind='independent_long_handles_geometry_and_runtime_compatibility_review',status='geometry_and_generic_runtime_contract_passed_native_pending' if all(checks.values()) else 'review_checks_failed',physical_success=False,native_execution_performed=False,checks=checks,failed_checks=[k for k,v in checks.items() if not v],input_sha256=before,configuration_changes=changes,topology=top,old_topology=topology(old),body=dict(nominal_flat_width_m=nm['bag']['width_m'],nominal_flat_height_m=nm['bag']['height_m'],rim_nodes=len(new['rim']),body_nodes=len(new['body']),body_faces=len(new['body_face_indices']),area_m2=regions['body']['area_m2'],zero_bottom_panel_area=True),regions=regions,handles=handles,mass=dict(old_area_m2=oldarea,new_area_m2=newarea,added_area_m2=newarea-oldarea,rho_kg_m3=cfg['bag']['rho'],IPC_thickness_radius_m=cfg['bag']['thickness'],physical_thickness_m=2*cfg['bag']['thickness'],areal_density_kg_m2=density,old_bag_mass_kg=oldmass,new_bag_mass_kg=newmass,added_mass_kg=newmass-oldmass,relative_increase=(newmass-oldmass)/oldmass,added_weight_N=(newmass-oldmass)*9.81,native_mass_readback_pending=True,metadata_suggested_density_kg_m2=nm['bag']['suggested_material']['surface_density_kg_m2'],metadata_estimate_kg=nm['bag']['estimated_mass_kg']),source_contract_notes=[
 'run.py uses config asset paths and declared grasp groups; pre-native initial IK follows the real new grasp patches. Native named material mapping is positional, not tied to historical vertex counts.',
 'physics_config.py maps body and both handle faces by a unique source/native geometry bijection; new count is allowed but mismatch or missing partition fails closed.',
 'validate.py uses labeled body/rim from selected NPZ; handle maximum Z does not replace mouth height. Rest area and mass use actual geometry with 2*IPC radius.',
 'run.py configured spread targets become1.40m; existing fixed right_clear_handle remains[.12,.20,1.32]. This downward retreat needs its existing rigid preflight and actual new-cloth handoff verification.',
 'handoff ends after release/clear and10s native hold, before teddy grasp; original scope-specific physical, left material18mm/pose15mm, and right actual opening.038m gates remain.',
 'General input-manifest/guarded/yaw/gravity auditors use config/rest shapes, not old cloth count or1.26m. Historical independent_handoff_final_audit.py and startup scripts intentionally bind old run/config/hash and require a new explicitly bound entry for this probe.',
 'run.py snapshots selected OBJ/MTL/NPZ/metadata and two_panel_builder.py. New build_long_handle_bag.py is not on its fixed source list; preserve its exact source separately with launch provenance if desired without changing runtime.',
 'Inherited metadata.bear is historical unscaled asset information, not the current s85 physical asset. Runtime reads config.bear/rest; current reference mass/tets must be read from actual build.',
],physical_limits=[
 'Only handle cloth grows; the identical body has no additional volume, mouth perimeter or flat-panel area. Higher grasp targets plus equally longer straps do not guarantee a higher actual mouth or bag bottom.',
 'Each handle material length grows0.28m (~66%); gravitational sag, free-loop location, twisting and contact with robot/bear can change despite small total bag mass increase. Longer straps do not establish a clear insertion corridor.',
 'Added material weight is ~0.0412N; original force budgets remain unchanged and must use actual reactions. Native body50/handle250N/m shear and mass must be read back after initialization.',
 'Initial tabletop clearance and two-panel topology do not certify full triangle-triangle self-embedding or later dynamic clearance. The proposed handoff probe must preserve original J/stretch/table/hold gates and inspect actual settled rim/handles before any full insertion.',
 'This review performs no IK replanning, native integration, continuous soft collision proof, or full-task success claim.'
])
out=ROOT/'runtime_ipc/independent-long-handles-compatibility-review.json';out.write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n');print(json.dumps(dict(path=str(out),sha256=sha(out),status=report['status'],checks_passed=int(sum(checks.values())),checks_total=len(checks),failed=report['failed_checks'],mass=report['mass'],topology=top),indent=2))
