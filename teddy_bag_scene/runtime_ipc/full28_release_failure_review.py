"""Bounded, read-only postmortem of actual full28 release. No physics execution."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/two_panel_full_28'
sys.path.insert(0, str(RUN / 'source'))
from validate import signed_volumes, containment, mouth_structure

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
paths = [RUN / n for n in ['config.json', 'rest.npz', 'states.npz',
    'insert_teddy.npz', 'failed_soft_state.npz', 'soft_metrics.npy',
    'joint_trace.npy', 'contact_guard_trace.json', 'coupling_trace.json',
    'gravity_release_candidate.json', 'gravity_release_actual_review.json',
    'events.json', 'report.json', 'source/validate.py', 'source/gravity_release.py']]
before = {str(p): sha(p) for p in paths}
cfg = read(paths[0]); rest = dict(np.load(paths[1])); states = dict(np.load(paths[2]))
insert = dict(np.load(paths[3])); failed = dict(np.load(paths[4]))
metrics = np.load(paths[5]); trace = np.load(paths[6]); guard = read(paths[7])
coupling = read(paths[8]); candidate = read(paths[9]); release = read(paths[10])
events = read(paths[11]); report = read(paths[12])
asset_path = RUN / 'source/assets' / cfg['bag']['npz_file']
source = dict(np.load(asset_path)); before[str(asset_path)] = sha(asset_path)
dist, mapping = cKDTree(rest['bag']).query(source['points'] + cfg['bag']['position'])
assert dist.max() < 1e-7 and len(np.unique(mapping)) == len(rest['bag'])
inverse_mapping = np.argsort(mapping)
tets = rest['bear_tets']; faces = mapping[source['faces']]
volume = signed_volumes(rest['bear'], tets); weights = abs(volume)
body, rim, topology = mouth_structure(rest['bag'], rest['bag_faces'], cfg)
table = float(cfg['table']['height'])
Jfail = signed_volumes(failed['bear'], tets) / volume
worst = int(Jfail.argmin()); ids = tets[worst]; rt = rest['bear'][ids]
reference = (rt[1:] - rt[0]).T
Dm = np.moveaxis(rest['bear'][tets][:, 1:] - rest['bear'][tets][:, :1], 1, 2)
quality_den = sum(np.sum((rest['bear'][tets[:, j]] - rest['bear'][tets[:, i]])**2, axis=1)
                  for i in range(4) for j in range(i+1, 4))
qualities = 12*(3*weights)**(2/3) / quality_den
allbad = np.flatnonzero(Jfail < .05)
part_faces = {
    'front_panel': source['panel_front_face_indices'],
    'back_panel': source['panel_back_face_indices'],
    'held_front_handle': np.flatnonzero(np.all(np.isin(source['faces'], source['handle_front']), axis=1)),
    'free_back_handle': np.flatnonzero(np.all(np.isin(source['faces'], source['handle_back']), axis=1)),
}
assert sum(len(v) for v in part_faces.values()) == len(faces)
edges4 = np.array([(i,j) for i in range(4) for j in range(i+1,4)])
guardtime = np.array([g['time_s'] for g in guard])
last = float(failed['time'])
times = [47.795, 47.945, 48.045, 48.095, 48.145, 48.195, 48.245]
observations = []
for time in times:
    i = int(np.argmin(abs(states['time'] - time)))
    assert abs(float(states['time'][i]) - time) < 1e-7
    observations.append(({k: v[i] for k,v in states.items()}, 'states.npz', i))
observations.append((failed, 'failed_soft_state.npz', None))

def sampled_tet_distance(q, bag, face_ids):
    """11 explicit tet samples against all selected physical triangles; not tet distance."""
    samples = np.vstack([q, q[edges4].mean(1), q.mean(0)])
    tri = bag[faces[face_ids]]
    best = None
    for i,p in enumerate(samples):
        close = trimesh.triangles.closest_point(tri, np.broadcast_to(p, (len(tri),3)))
        distances = np.linalg.norm(close-p, axis=1)
        j = int(distances.argmin()); d = float(distances[j])
        if best is None or d < best['distance_m']:
            fi = int(face_ids[j]); f = faces[fi]
            normal = np.cross(bag[f[1]]-bag[f[0]], bag[f[2]]-bag[f[0]])
            normal /= np.linalg.norm(normal)
            best = dict(distance_m=d, tet_sample_index=i, tet_sample_world_m=p.tolist(),
                        physical_cloth_point_m=close[j].tolist(), source_face_index=fi,
                        source_vertex_ids=source['faces'][fi].tolist(),
                        native_vertex_ids=f.tolist(), ordered_cloth_normal_world=normal.tolist(),
                        source_material_face_center_m=source['points'][source['faces'][fi]].mean(0).tolist(),
                        minimum_face_vertex_height_above_table_m=float(bag[f,2].min()-table))
    # Valid lower bound for every point in the full tet and each full triangle.
    gap = np.maximum(0, np.maximum(tri.min(1)-q.max(0), q.min(0)-tri.max(1)))
    best['whole_tet_all_triangles_AABB_lower_bound_m'] = float(np.linalg.norm(gap,axis=1).min())
    return best

rows = []
for state, source_name, index in observations:
    time = float(state['time']); bag = state['bag']; bear = state['bear']; q = bear[ids]
    J = signed_volumes(bear,tets)/volume
    F = (q[1:]-q[0]).T @ np.linalg.inv(reference)
    U,S,Vh = np.linalg.svd(F)
    c = containment(bag,bear,body,rim,weights,tets)
    # Do not turn validator invalid cap's sentinel zero into a measured fraction.
    if not c['virtual_closed_body_valid']:
        c['bear_vertex_inside_fraction'] = None
        c['bear_center_of_mass_inside'] = None
        c['all_bear_vertices_inside'] = None
    k = int(np.argmin(abs(trace[:,0]-time))); gi = int(np.argmin(abs(guardtime-time)))
    assert abs(trace[k,0]-time) < 1e-7 and abs(guardtime[gi]-time) < 1e-7
    g = guard[gi]
    row = dict(time_s=time, geometry_source=source_name, saved_frame_index=index,
        global_minimum_J=float(J.min()), global_worst_tet=int(J.argmin()),
        failed_material_tet_J=float(J[worst]), failed_material_tet_center_m=q.mean(0).tolist(),
        failed_material_tet_principal_stretches=S.tolist(),
        minimum_stretch_spatial_axis_world=U[:,-1].tolist(),
        minimum_stretch_axis_abs_Z=float(abs(U[2,-1])),
        failed_material_tet_z_range_m=[float(q[:,2].min()),float(q[:,2].max())],
        bear_z_range_m=[float(bear[:,2].min()),float(bear[:,2].max())],
        bear_minimum_material_vertex_id=int(bear[:,2].argmin()),
        bear_COM_m=np.average(bear[tets].mean(1),weights=weights,axis=0).tolist(),
        physical_body_rim_z_range_m=[float(bag[rim,2].min()),float(bag[rim,2].max())],
        bag_z_range_m=[float(bag[:,2].min()),float(bag[:,2].max())],
        actual_saved_right_TCP_m=state['tcp_right'].tolist() if 'tcp_right' in state else None,
        guard_AIM_right_TCP_reference_m=g['aim_tcp_world_m'],
        actual_finger_joint_half_openings_m=trace[k,17:19].tolist(),
        right_finger_reaction_norm_N=g['right_finger_force_norm_n'],
        right_distal_link_names=g['right_link_names'],
        right_distal_link_force_world_N=g['right_link_force_world_n'],
        right_distal_net_force_norm_N=g['net_force_norm_n'],
        right_TCP_torque_norm_Nm=g['tcp_torque_norm_nm'],
        containment_original_method=c,
        failed_tet_to_physical_cloth_11_sample_distances={name:sampled_tet_distance(q,bag,fi) for name,fi in part_faces.items()})
    if rows:
        dt=time-rows[-1]['time_s']
        row['interval_from_previous_s']=dt
        row['failed_tet_centroid_secant_velocity_m_s']=((q.mean(0)-np.array(rows[-1]['failed_material_tet_center_m']))/dt).tolist()
        row['bear_COM_secant_velocity_m_s']=((np.array(row['bear_COM_m'])-np.array(rows[-1]['bear_COM_m']))/dt).tolist()
        row['velocity_scope']='Finite difference over two saved endpoints; not instantaneous contact velocity.'
    rows.append(row)
    print(json.dumps(dict(time=time,J=row['global_minimum_J'],footz=q.mean(0)[2],minZ=row['bear_z_range_m'][0],
        inside=c['bear_vertex_inside_fraction'],COMinside=c['bear_center_of_mass_inside'],
        nearest={n:d['distance_m'] for n,d in row['failed_tet_to_physical_cloth_11_sample_distances'].items()})),flush=True)

first_bad_indices=np.flatnonzero(metrics[:,1] < .05)
release_start = next(e['start'] for e in events if e['phase']=='release_teddy')
release_window = metrics[(metrics[:,0]>=release_start-1e-8)]
short_tail = metrics[metrics[:,0]>=48.245-1e-8]
Ffail=(failed['bear'][ids][1:]-failed['bear'][ids][0]).T@np.linalg.inv(reference)
initial=rows[0]; final=rows[-1]
new_tcp=.915; old_tcp=float(candidate['insert_stage']['target_tcp_m'][2]) if 'target_tcp_m' in candidate['insert_stage'] else 1.085
full=float(candidate['original_full_depth_tcp_m'][2]); shift=new_tcp-old_tcp
current_gap=initial['bear_z_range_m'][0]-table
rigid_gap=current_gap+shift
mass=float(weights.sum()*cfg['bear']['rho'])
checks=dict(source_material_mapping_exact=bool(dist.max()<1e-7),
    true_failure_matches_report_time=abs(last-report['simulated_seconds'])<1e-8,
    true_failure_J_matches_per_step_metric=abs(Jfail.min()-metrics[-1,1])<1e-12,
    first_J_below5percent_is_true_failed_endpoint=bool(len(first_bad_indices)==1 and first_bad_indices[0]==len(metrics)-1),
    failure_after_release_intent=last>release_start,
    no_fabric_stretch_failure=bool(np.all(metrics[:,4]<=1.15)),
    no_soft_table_penetration=bool(np.all(metrics[:,2:4]<=.002)),
    all_soft_summary_finite=bool(np.isfinite(metrics).all() and np.all(metrics[:,5]==1)),
    all_referenced_inputs_unchanged=all(sha(p)==h for p,h in before.items()))
out=dict(kind='independent_full28_release_failure_review',physical_task_success=False,
    native_simulation_performed_by_reviewer=False,source_sha256=before,
    checks=checks,true_failure_time_s=last,last_complete_geometry_time_s=float(states['time'][-1]),
    release_start_time_s=release_start,release_intended_duration_s=.5,
    release_completed_snapshot_exists=(RUN/'release_teddy.npz').exists(),
    initial_actual_release_admission=release,
    worst_tetrahedron=dict(index=worst,vertex_ids=ids.tolist(),
        failed_J=float(Jfail[worst]),rest_local_centroid_m=(rt.mean(0)-cfg['bear']['position']).tolist(),
        rest_unscaled_centroid_m=((rt.mean(0)-cfg['bear']['position'])/.663).tolist(),
        approximate_anatomical_region='negative-X / left foot, distal negative-Y sole; anatomical name from original teddy asset coordinates',
        rest_volume_m3=float(weights[worst]),rest_mean_ratio_quality=float(qualities[worst]),
        rest_quality_percentile=float(100*(qualities<qualities[worst]).mean()),
        failure_world_vertices_m=failed['bear'][ids].tolist(),
        all_tets_below5percent=allbad.tolist(),all_bad_tetrahedra_vertices=tets[allbad].tolist(),
        bad_tets_reference_mass_fraction=float(weights[allbad].sum()/weights.sum()),
        failure_principal_stretches=np.linalg.svd(Ffail,compute_uv=False).tolist()),
    full_release_geometry_rows=rows,
    final_50ms_per_step_metrics_columns=['time_s','minimum_tetra_volume_ratio','bag_table_penetration_m','bear_table_penetration_m','maximum_bag_edge_stretch_ratio','all_soft_states_finite'],
    final_50ms_per_step_metrics=short_tail.tolist(),
    proposed_lower_release_necessary_geometry_only=dict(
        current_stop_TCP_Z_m=old_tcp,proposed_stop_TCP_Z_m=new_tcp,configuration_height_above_table_m=new_tcp-table,
        downward_change_m=-shift,original_full_depth_TCP_Z_m=full,
        proposed_above_original_full_depth_m=new_tcp-full,original_required_above_full_depth_m=.020,
        passes_this_one_height_inequality=bool(new_tcp>full+.020),
        actual_initial_lowest_bear_point_table_gap_m=current_gap,
        rigid_translation_counterfactual_lowest_point_table_gap_m=rigid_gap,
        translated_shape_potential_energy_to_first_table_contact_J=mass*9.81*current_gap,
        actual_failed_material_centroid_gap_at_release_start_m=initial['failed_material_tet_center_m'][2]-table,
        hypothetical_failed_material_centroid_gap_after_downward_translation_m=initial['failed_material_tet_center_m'][2]+shift-table,
        counterfactual_translation_potential_energy_to_first_table_contact_J=mass*9.81*rigid_gap,
        scope='Translation of the actual release-start shape only. Not the result of a newly executed descent, new shape, preflight, or a collision-free/support certificate.'),
    conclusion='The first 5% volume gate failure occurs in a distal foot tetrahedron near the table and free back handle / back panel, after finger contact has unloaded. The observed fast descent and near-horizontal flattening support landing impact as the leading mechanism; they do not identify individual cloth/table contact forces or exclude numerical/material discretization effects.',
    recommendation='A lower, deliberately stopped placement before the unchanged release gate is physically motivated. The proposed TCP Z=.915m leaves 24.945mm above the original full-depth target and reduces the rigid-shape nominal unsupported drop from about202mm to about32mm. It needs a new full route and fresh actual endpoint gate; no old frozen trajectory certifies it. Prefer small supported placement over increasing allowed compression or assuming a slower gripper-opening command alone controls fall.',
    limitations=[
        'Failure soft NPZ has no measured robot q/TCP; true-time joint trace and guard reactions are separate records. Guard AIM TCP is a reference, not measured IPC affine geometry.',
        'The last full coupling transform is at48.245s. It is not used as the48.295s finger pose.',
        'Cloth nearest values sample four tet vertices, six edge midpoints and its centroid (11 points). They are not exact tet-triangle distance or a penetration certificate. Full tet/triangle AABB gives only a conservative lower bound.',
        'Bag contact distances are geometry evidence, not pairwise contact forces. No pair-resolved cloth/table forces are saved here.',
        'Original virtual-cap validity is not a physical body self-intersection/embedding certificate; node inside fraction is not volume fraction or all-tetrahedron containment.',
        'A small FEM group folding under impact is a material/discretization failure to investigate, not proof that a physical plush would collapse to this degree. Material parameters have not been calibrated.',
        'The .915m alternative is a necessary-height assessment only; deeper gripper/cloth interaction and fresh grip/containment must be tested under the original gates.'
    ])
outfile=RUN/'independent-full28-release-failure-review.json'
assert not outfile.exists(), 'Do not overwrite an existing independent report.'
outfile.write_text(json.dumps(out,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')
print(json.dumps(dict(output=str(outfile),sha256=sha(outfile),checks={k:bool(v) for k,v in checks.items()},quality=float(qualities[worst]),height_check=out['proposed_lower_release_necessary_geometry_only'])),flush=True)
