"""Read-only CPU asset, rest-volume and grasp-region scaling review.
No production asset/config/state or Genesis cache writes; no Genesis/GPU init.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh,tetgen,igl
ROOT=Path(__file__).resolve().parents[1]
config=json.loads((ROOT/'config_guarded_postlift_yaw_trial.json').read_text())
asset=ROOT/'assets'/config['bear']['mesh'];lines=asset.read_text().splitlines()
v=np.array([[float(n) for n in l.split()[1:4]] for l in lines if l.startswith('v ')])
f=np.array([[int(n.split('/')[0])-1 for n in l.split()[1:]] for l in lines if l.startswith('f ')])
rest=dict(np.load(ROOT/'runs/two_panel_full_19/rest.npz'))
p0=rest['bear']-config['bear']['position'];t0=rest['bear_tets'];center=np.asarray(config['task']['bear_grasp_local'])
tracked=[asset,asset.with_suffix('.mtl'),ROOT/'assets/teddy_fur.png',ROOT/'scene.py',ROOT/'run.py',ROOT/'config_guarded_postlift_yaw_trial.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in tracked}
def vol(p,t):return np.linalg.det(p[t[:,1:]]-p[t[:,:1]])/6
edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
def metrics(p,t):
 V=np.abs(vol(p,t));tri=p[t];E=np.linalg.norm(tri[:,edges[:,0]]-tri[:,edges[:,1]],axis=2)
 quality=12*(3*V)**(2/3)/(E**2).sum(1)
 return dict(vertices=len(p),tetrahedra=len(t),bounds_m=[p.min(0).tolist(),p.max(0).tolist()],dimensions_m=np.ptp(p,axis=0).tolist(),volume_m3=float(V.sum()),mass_kg=float(V.sum()*config['bear']['rho']),minimum_abs_tet_volume_m3=float(V.min()),quality_quantiles=np.quantile(quality,[0,.1,.5,.9,1]).tolist(),negative_tet_orientation_count=int((vol(p,t)<0).sum()))
def split_surface_tets(p,t,world_pos):
 # Mirror source algorithm exactly including its float32 centroid and positional order.
 p=p+world_pos
 F,*_=igl.boundary_facets(t);mask=np.zeros(len(p),bool);mask[F.reshape(-1)]=True;bad=mask[t].all(1)
 old=t[bad];newp=np.mean(p[old],axis=1,dtype=np.float32);new=[]
 for idx,(a,b,c,d) in enumerate(old,len(p)):new.extend([[a,b,c,idx],[a,b,idx,d],[a,idx,c,d],[idx,b,c,d]])
 return np.vstack([p,newp])-world_pos,np.vstack([t[~bad],np.asarray(new,dtype=np.int32)])
base_ids=np.flatnonzero(np.linalg.norm(p0-center,axis=1)<.032)
mesh=trimesh.Trimesh(v,f,process=False)
out=dict(kind='uniform_teddy_asset_scaling_review',physical_success=False,no_genesis_import_or_native_execution=True,source_sha256=before,
 current_surface=dict(vertices=len(v),faces=len(f),watertight=mesh.is_watertight,winding_consistent=mesh.is_winding_consistent,components=len(mesh.split()),bounds_m=mesh.bounds.tolist(),surface_volume_m3=float(mesh.volume),uv_count=sum(l.startswith('vt ') for l in lines)),current_native_rest=metrics(p0,t0),current_grasp_local_m=center.tolist(),current_grasp_radius_m=.032,current_grasp_region_count=len(base_ids),candidates=[])
for s in [0.90,0.85]:
 ps=p0*s;c=center*s;fixed=np.flatnonzero(np.linalg.norm(ps-c,axis=1)<.032);scaled=np.flatnonzero(np.linalg.norm(ps-c,axis=1)<.032*s)
 maxvol=4e-6*s**3
 points,tets,*_=tetgen.TetGen(v*s,f.astype(np.int32)).tetrahedralize(switches=f'pq1.3/15a{maxvol:.15g}YQ')
 points,tets=split_surface_tets(points,tets,np.asarray(config['bear']['position']))
 native=metrics(points,tets);newfix=np.flatnonzero(np.linalg.norm(points-c,axis=1)<.032);newscaled=np.flatnonzero(np.linalg.norm(points-c,axis=1)<.032*s)
 m=metrics(ps,t0)
 out['candidates'].append(dict(scale=s,recommended_grasp_local_m=c.tolist(),proportional_grasp_radius_m=.032*s,proportional_maxvolume_m3=maxvol,exact_scaled_existing_rest=m,
 rerun_native_algorithm_without_Genesis_cache=native,remesh_reproducibility_not_assumed=True,
 existing_topology_region=dict(original_count=len(base_ids),fixed32mm_count=len(fixed),scaled_radius_count=len(scaled),scaled_region_exact_same_ids=bool(np.array_equal(base_ids,scaled)),fixed_region_extra_count=int(len(np.setdiff1d(fixed,base_ids))),fixed_region_mean_local_m=ps[fixed].mean(0).tolist(),proportional_region_mean_local_m=ps[scaled].mean(0).tolist(),fixed_vs_proportional_mean_offset_m=(ps[fixed].mean(0)-ps[scaled].mean(0)).tolist(),fixed_region_bounds_m=[ps[fixed].min(0).tolist(),ps[fixed].max(0).tolist()]),
 new_tetgen_grasp_regions=dict(fixed32mm_count=len(newfix),proportional_radius_count=len(newscaled),fixed_vs_proportional_mean_offset_m=(points[newfix].mean(0)-points[newscaled].mean(0)).tolist()),
 undeformed_grasp_world_z_m=float(config['bear']['position'][2]+c[2]),runtime_grasp_minimum_world_z_m=config['table']['height']+.025,
 volume_mass_factor=s**3,area_factor=s**2,geometrically_similar_stress_factor_for_same_force=1/s**2,
 same_strain_force_scale_example_n_per_finger=config['task']['bear_finger_force_limit']*s**2,
 same_elastic_material_compliance_factor_same_force=1/s,
 density_remains_kg_m3=config['bear']['rho'],youngs_modulus_remains_pa=config['bear']['E']))
out['source_inputs_unchanged']=before=={str(p):sha(p) for p in tracked}
path=ROOT/'runtime_ipc/independent-teddy-uniform-scaling-review.json';path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
