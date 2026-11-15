"""Read-only native-rest mesh accounting; no Genesis import or mesher execution."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import igl

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/two_panel_full_21'
OLD = ROOT / 'runs/two_panel_full_20'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = json.loads((RUN / 'config.json').read_text())
oldcfg = json.loads((OLD / 'config.json').read_text())
rest = dict(np.load(RUN / 'rest.npz'))
old = dict(np.load(OLD / 'rest.npz'))
p, t = rest['bear'], rest['bear_tets']
log = (ROOT / 'runs/two_panel_full_21.log').read_text()
nraw = int(re.findall(r'Mesh points:\s*(\d+)', log)[0])
traw = int(re.findall(r'Mesh tetrahedra:\s*(\d+)', log)[0])
nsplit = len(p) - nraw
assert len(t) - traw == 3 * nsplit
unchanged_count = traw - nsplit
children = t[unchanged_count:].reshape(nsplit, 4, 4)
parents = children[:, 0].copy()
parents[:, 3] = children[:, 1, 3]
expected = []
for idx, (a, b, c, d) in enumerate(parents, nraw):
    expected.extend([[a,b,c,idx], [a,b,idx,d], [a,idx,c,d], [idx,b,c,d]])
assert np.array_equal(t[unchanged_count:], np.asarray(expected))
reconstructed_raw_tets = np.vstack([t[:unchanged_count], parents])
boundary, *_ = igl.boundary_facets(reconstructed_raw_tets)
surface = np.zeros(nraw, dtype=bool)
surface[boundary.reshape(-1)] = True
all_surface = surface[reconstructed_raw_tets].all(axis=1)
assert np.array_equal(np.flatnonzero(all_surface), np.arange(unchanged_count, traw))
centroids = np.mean(p[parents], axis=1, dtype=np.float32)
assert np.array_equal(p[nraw:], centroids.astype(p.dtype))

def metrics(v, q):
    a = v[q]
    volumes = np.linalg.det(a[:, 1:] - a[:, :1]) / 6
    edges = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    length = np.linalg.norm(a[:,edges[:,0]] - a[:,edges[:,1]], axis=2)
    quality = 12 * (3 * np.abs(volumes))**(2/3) / (length**2).sum(axis=1)
    return dict(vertices=len(v), tetrahedra=len(q), finite=bool(np.isfinite(v).all() and np.isfinite(volumes).all()),
                negative_tetrahedra=int((volumes < 0).sum()), zero_tetrahedra=int((volumes == 0).sum()),
                minimum_signed_volume_m3=float(volumes.min()), volume_m3=float(volumes.sum()),
                minimum_mean_ratio_shape_quality=float(quality.min()),
                shape_quality_quantiles_0_1_10_50_90_100_percent=np.quantile(quality,[0,.01,.1,.5,.9,1]).tolist(),
                minimum_edge_length_m=float(length.min()), maximum_edge_length_m=float(length.max()))

native = metrics(p, t)
raw = metrics(p[:nraw], reconstructed_raw_tets)
previous = metrics(old['bear'], old['bear_tets'])
center = np.asarray(cfg['bear']['position']) + np.asarray(cfg['task']['bear_grasp_local'])
radius = cfg['task']['bear_grasp_tracking_radius_m']
roi = np.flatnonzero(np.linalg.norm(p-center, axis=1) < radius)
old_center = np.asarray(oldcfg['bear']['position']) + np.asarray(oldcfg['task']['bear_grasp_local'])
old_roi = np.flatnonzero(np.linalg.norm(old['bear']-old_center, axis=1) < oldcfg['task'].get('bear_grasp_tracking_radius_m', .032))
matching_connectivity = np.array_equal(t, old['bear_tets'])
report = dict(kind='full21_saved_native_rest_split_accounting_and_quality', physical_success=False,
    genesis_imported=False, gpu_or_native_execution_by_review=False, mesher_executed_by_review=False,
    inputs={str(path.relative_to(ROOT)): sha(path) for path in [RUN/'rest.npz', RUN/'initial.npz', RUN/'config.json', RUN/'source/scene.py', RUN/'source/assets/bear_uniform_coarse_s85.obj']},
    source_algorithm=dict(path='/home/chloe/Desktop/genesis-local/genesis-world/genesis/utils/element.py',
        sha256=sha(Path('/home/chloe/Desktop/genesis-local/genesis-world/genesis/utils/element.py')),
        operation='split_all_surface_tets: each tet with four surface nodes becomes four tets around one float32 centroid after translation'),
    raw_tetgen_log=dict(vertices=nraw,tetrahedra=traw), native_saved_rest=native,
    verified_split=dict(parent_tetrahedra=nsplit, added_vertices=nsplit, added_tetrahedra=3*nsplit,
        unchanged_tetrahedra=unchanged_count, all_child_patterns_match_source_exactly=True,
        all_and_only_reconstructed_parents_have_four_boundary_vertices=True,
        all_added_vertices_equal_source_float32_centroids_exactly=True,
        reconstructed_raw_boundary_triangles=len(boundary), reconstructed_raw_boundary_nodes=int(surface.sum()),
        reconstructed_raw_metrics=raw, parent_to_children_total_volume_difference_m3=native['volume_m3']-raw['volume_m3']),
    previous_full20_native_rest=previous,
    roi=dict(radius_m=radius, native_point_count=len(roi), minimum_required_count=3,
        previous_full20_point_count=len(old_roi), cross_run_ids_are_not_assumed=True,
        observed_same_connectivity=matching_connectivity,
        observed_same_roi_ids_only_after_connectivity_check=bool(matching_connectivity and np.array_equal(roi, old_roi))),
    interpretation='1362/5222 are raw TetGen counts, not final entity counts. Saved 1450/5486 are explained exactly by the existing 88-parent surface-tet split. No tessellation discrepancy remains between these two observations.',
    limits=['Shape quality is mean ratio (1 regular tetrahedron, near 0 degenerate), not a guarantee of dynamic inversion resistance.',
        'TetGen mindihedral/minratio are refinement requests before the surface-tet split; no post-split angular quality guarantee is inferred.',
        'Future mesh/node IDs are not assumed stable. Runtime grasp ROI must be recomputed from actual rest coordinates.',
        'Uniform-size consistency and positive rest volumes do not establish grasping, insertion or release success.'])
out = RUN / 'independent-rest-tet-split-review.json'
out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(report=str(out),sha256=sha(out),raw=report['raw_tetgen_log'],native=native,
    split=report['verified_split'],roi=report['roi']),indent=2))
