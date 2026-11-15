"""Two frozen offsets,21 early-descent levels; sampled patch distances only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21';OLD=ROOT/'runs/two_panel_full_20'
OUT=RUN/'independent-positive-y20-route-review';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(RUN/'source'))
from validate import tetra_boundary,mesh_edges
from postlift_yaw import _serial
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
old=dict(np.load(OLD/'rest.npz'));oldcfg=json.loads((OLD/'config.json').read_text())
assert np.array_equal(old['bear_tets'],rest['bear_tets'])
asset_path=RUN/'source/assets'/cfg['bag']['npz_file'];asset=dict(np.load(asset_path))
asset['handle_front_face_indices']=np.flatnonzero(np.isin(asset['faces'],asset['handle_front']).all(axis=1))
assert len(asset['handle_front_face_indices'])==352
error,mapping=cKDTree(rest['bag']).query(asset['points']+cfg['bag']['position']);assert error.max()<1e-7 and len(np.unique(mapping))==len(mapping)
frontfaces=mapping[asset['faces'][asset['handle_front_face_indices']]]
bodyfaces=tetra_boundary(rest['bear'],rest['bear_tets'])
old_centers=old['bear'][bodyfaces].mean(1)-oldcfg['bear']['position']
footmask=old_centers[:,1]<-.10;footfaces=bodyfaces[footmask]
front=trimesh.Trimesh(state['bag'],frontfaces,process=False)
whole=trimesh.Trimesh(state['bear'],bodyfaces,process=False)
foot=trimesh.Trimesh(state['bear'],footfaces,process=False)

def samples(points,faces):
    edges,_,_=mesh_edges(faces)
    return np.vstack([points[np.unique(faces)],points[edges].mean(1),points[faces].mean(1)])
frontsamples=samples(state['bag'],frontfaces)
patches={'whole_bear':(whole,samples(state['bear'],bodyfaces)),'foot':(foot,samples(state['bear'],footfaces))}
targets={}
for name,file in [('zero','guarded_contact_candidate.json'),('positive_y20','independent-positive-y20-original-base.json')]:
    c=json.loads((RUN/file).read_text());assert abs(c['observed_time_s']-float(state['time']))<1e-9
    targets[name]=np.asarray(c['route']['outside_tcp_m'])
report=dict(kind='full21_0_vs_positive_y20_frozen_frontstrap_sampled_distances',physical_success=False,native_executed=False,
    source_state=str(RUN/'orient_teddy_postlift.npz'),source_time_s=float(state['time']),
    full_hand_rotation_preserved=True,original_actual_bear_and_cloth_are_frozen=True,
    query_heights_below_outside_m=np.linspace(0.,.10,21),outcomes=[],
    material_definition=dict(frontstrap_original_source_faces=asset['handle_front_face_indices'],frontstrap_face_count=len(frontfaces),
        foot_patch='Original100percent rest surface triangle centroid localY < -0.10m; actual full21 topology identity checked before reuse.',
        foot_triangle_count=len(footfaces),whole_surface_triangle_count=len(bodyfaces),whole_surface_sample_count=len(patches['whole_bear'][1]),
        foot_sample_count=len(patches['foot'][1]),frontstrap_sample_count=len(frontsamples)),
    limits=['Each query uses vertices,edge midpoints and face centroids in both patch→frontstrap and frontstrap→patch directions with exact closest-triangle distances.',
        'The minimum of finite point-to-triangle samples is an UPPER bound on exact continuous triangle-patch distance. Positive values do not prove separated surfaces; crossing edges/interiors may be missed.',
        'Only21 frozen descent displacements spaced5mm are sampled,0 through100mm. No continuous swept cloth clearance is certified.',
        'All bag and teddy shapes are from the35.060s actual post-yaw snapshot, not the later actual transport/contact state. Future deformation and cloth motion can dominate the differences.',
        'These geometric samples are not physical force,penetration depth,grasp retention,or insertion success.'],
    inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz',OLD/'rest.npz',OLD/'config.json',asset_path,Path(__file__)]})

def save():
    (OUT/'frontstrap-distances.json').write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')

for depth in np.linspace(0.,.10,21):
    for name,target in targets.items():
        delta=target-state['tcp_right']-np.array([0.,0.,depth]);row=dict(case=name,descent_depth_m=float(depth),translation_m=delta,patches={})
        for label,(mesh,points) in patches.items():
            near_bear,dist_fb,tri_fb=trimesh.proximity.closest_point(mesh,frontsamples-delta)
            near_front,dist_bf,tri_bf=trimesh.proximity.closest_point(front,points+delta)
            i=int(np.argmin(dist_fb));j=int(np.argmin(dist_bf));distance=min(float(dist_fb[i]),float(dist_bf[j]))
            details=dict(sampled_minimum_distance_m=distance,front_to_patch_minimum_m=float(dist_fb[i]),patch_to_front_minimum_m=float(dist_bf[j]),
                closest_front_sample_world_m=frontsamples[i],closest_patch_point_to_front_sample_world_m=near_bear[i]+delta,
                closest_patch_triangle_index=int(tri_fb[i]),closest_patch_sample_world_m=points[j]+delta,closest_front_triangle_point_m=near_front[j],
                closest_original_frontstrap_face_index=int(asset['handle_front_face_indices'][tri_bf[j]]))
            if label=='foot':
                bary=trimesh.triangles.points_to_barycentric(state['bear'][footfaces[[tri_fb[i]]]],near_bear[[i]])[0]
                details['closest_foot_material_coordinate_in_original100percent_rest_m']=bary@old['bear'][footfaces[tri_fb[i]]]-oldcfg['bear']['position']
                details['closest_foot_patch_global_surface_face_index']=int(np.flatnonzero(footmask)[tri_fb[i]])
            row['patches'][label]=details
        report['outcomes'].append(row);save()
        print(name,'depth_mm',round(float(depth)*1000,3),'whole_mm',row['patches']['whole_bear']['sampled_minimum_distance_m']*1000,
            'foot_mm',row['patches']['foot']['sampled_minimum_distance_m']*1000,flush=True)
report['summary']={}
for name in targets:
    rows=[r for r in report['outcomes'] if r['case']==name]
    report['summary'][name]={label:dict(minimum_sampled_distance_m=min(r['patches'][label]['sampled_minimum_distance_m'] for r in rows),
        minimum_at_depth_m=min(rows,key=lambda r:r['patches'][label]['sampled_minimum_distance_m'])['descent_depth_m']) for label in patches}
report['status']='bounded_sampled_geometry_complete';save()
print('DONE',str(OUT/'frontstrap-distances.json'),flush=True)
