"""CPU-only TetGen quality comparison; source teddy assets remain unchanged."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import tetgen
import trimesh

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from validate import tetra_boundary


def quantiles(a):
    return dict(zip(("min", "p01", "p10", "median", "p90", "max"),
                    np.quantile(a, [0, .01, .1, .5, .9, 1]).tolist()))


def quality(points, tets):
    t = points[tets]
    volumes = np.abs(np.linalg.det(t[:, 1:] - t[:, :1])) / 6
    l2 = np.stack([np.sum((t[:, i]-t[:, j])**2, axis=1)
                   for i in range(4) for j in range(i+1, 4)], axis=1)
    q = 12*(3*volumes)**(2/3) / l2.sum(axis=1)
    faces = tetra_boundary(points, tets)
    tri = points[faces]
    twice_area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    edge2 = sum(np.sum((tri[:, i]-tri[:, j])**2, axis=1)
                for i, j in ((0, 1), (0, 2), (1, 2)))
    tq = 2*np.sqrt(3)*twice_area / edge2
    surface_tets = np.flatnonzero(np.isin(tets, np.unique(faces)).any(axis=1))
    return faces, {
        "vertices": len(points), "tetrahedra": len(tets),
        "surface_triangles": len(faces), "volume_m3": float(volumes.sum()),
        "tet_mean_ratio_quality": quantiles(q),
        "tets_touching_boundary_mean_ratio_quality": quantiles(q[surface_tets]),
        "surface_triangle_mean_ratio_quality": quantiles(tq),
        "tet_rest_volume_m3": quantiles(volumes),
        "tet_quality_below_0_1_count": int((q < .1).sum()),
    }


def main():
    source = trimesh.load(HERE/'assets/bear_physics.obj', force='mesh', process=True)
    source.remove_unreferenced_vertices()
    saved = np.load(HERE/'runs/two_panel_grasp_04/rest.npz')
    _, baseline = quality(saved['bear'], saved['bear_tets'])
    result = {"source": str(HERE/'assets/bear_physics.obj'),
              "baseline_actual_genesis_import": baseline,
              "source_assets_modified": False, "candidates": {}}
    candidate = 'bisect_q1p3_dihedral15'
    switches = 'pq1.3/15a4e-6Q'
    start = time.monotonic()
    tet = tetgen.TetGen(source.vertices.copy(), source.faces.astype(np.int32))
    points, tets, *_ = tet.tetrahedralize(switches=switches)
    faces, stats = quality(points, tets)
    surface = np.unique(faces)
    _, distance, _ = trimesh.proximity.closest_point(source, points[surface])
    stats.update(switches=switches, seconds=time.monotonic()-start,
                 maximum_surface_vertex_distance_to_original_m=float(distance.max()),
                 surface_vertex_distance_quantiles_m=quantiles(distance))
    result['candidates'][candidate] = stats
    np.savez_compressed(HERE/'runtime_ipc'/f'{candidate}.npz', points=points, tetra=tets, surface_faces=faces)
    (HERE/'runtime_ipc/tet-quality-comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
