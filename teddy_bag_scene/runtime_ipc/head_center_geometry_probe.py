"""Nine bounded head-top candidates on saved deformed geometry, CPU only.

This is a geometric screening report, not a friction-grasp success test.
"""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
import mujoco
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from contact_grasp import _clip, track_reference_displacement
from motion import Planner, TCP
from validate import tetra_boundary


def clip_plane(poly, plane):
    out = []
    if len(poly) == 0:
        return np.empty((0, 3))
    prev = poly[-1]
    before = prev @ plane[:3] + plane[3]
    for point in poly:
        now = point @ plane[:3] + plane[3]
        if (before <= 0) != (now <= 0):
            out.append(prev + before / (before - now) * (point - prev))
        if now <= 0:
            out.append(point)
        prev, before = point, now
    return np.asarray(out).reshape(-1, 3)


def area(poly):
    return sum(np.linalg.norm(np.cross(poly[i] - poly[0], poly[i + 1] - poly[0])) / 2
               for i in range(1, len(poly) - 1))


def main():
    run = HERE / "runs/two_panel_full_05"
    rest = np.load(run / "rest.npz")
    state = np.load(run / "teddy_pregrasp.npz")
    config = json.loads((run / "config.json").read_text())
    points = state["bear"]
    origin = np.asarray(config["bear"]["position"])
    faces = tetra_boundary(points, rest["bear_tets"])
    mesh = trimesh.Trimesh(points, faces, process=False)
    rotation = Rotation.from_quat([2**-.5, -2**-.5, 0, 0]).as_matrix()
    tet = points[rest["bear_tets"]]
    volume = np.abs(np.linalg.det(np.moveaxis(tet[:, 1:] - tet[:, :1], 1, 2))) / 6
    com = np.average(tet.mean(axis=1), weights=volume, axis=0)
    planner = Planner(config)
    # Use MuJoCo's same-asset convex collision components, not a convex hull of
    # the entire hand, which would incorrectly close the empty jaw gap.
    planner.set_fingers(.04)
    mujoco.mj_forward(planner.m, planner.d)
    hand = planner.bodies["right"]
    hand_r = planner.d.xmat[hand].reshape(3, 3)
    tcp = planner.d.xpos[hand] + hand_r @ TCP
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    components = []
    for gid in planner.geoms["right"]:
        name = planner.body_name(gid)
        if not (name.endswith("_finger") or name.endswith("_hand")):
            continue
        kind = planner.m.geom_type[gid]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            local = signs * planner.m.geom_size[gid]
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = planner.m.geom_dataid[gid]
            start, count = planner.m.mesh_vertadr[mid], planner.m.mesh_vertnum[mid]
            local = planner.m.mesh_vert[start:start+count]
        else:
            raise ValueError(f"Unexpected collision geometry type {kind}")
        world = local @ planner.d.geom_xmat[gid].reshape(3, 3).T + planner.d.geom_xpos[gid]
        hand_local = (world - tcp) @ hand_r
        # The exact translational sweep of each convex component over the
        # 150 mm vertical descent is its convex hull with the translated copy.
        sweep = np.concatenate([hand_local, hand_local - [0, 0, .15]])
        hull = ConvexHull(sweep)
        components.append((gid, name, sweep.min(axis=0), sweep.max(axis=0), hull.equations))

    n = 31
    grid = (np.arange(n) + .5) / n * .017 - .0085
    xx, zz = np.meshgrid(grid, grid)
    ray_local = np.column_stack([xx.ravel(), np.full(n*n, .2), zz.ravel()])
    pixel_area = .017**2 / n**2
    candidates = [[0., y, z] for y in (.075, .080, .085) for z in (.090, .095, .100)]
    rows = []
    for candidate in candidates + [[.050, .153, .055]]:
        center = track_reference_displacement(rest["bear"], points, origin + candidate)
        tri = (points[faces] - center) @ rotation
        sweep_intersections = []
        for gid, name, low, high, planes in components:
            chosen = np.flatnonzero(np.all(tri.max(axis=1) >= low, axis=1) & np.all(tri.min(axis=1) <= high, axis=1))
            hits = []
            for index in chosen:
                poly = tri[index]
                for plane in planes:
                    poly = clip_plane(poly, plane)
                    if len(poly) < 3:
                        break
                if len(poly) >= 3 and area(poly) > 1e-14:
                    hits.append(int(index))
            if hits:
                sweep_intersections.append({"geom_id": int(gid), "body": name,
                                            "surface_triangle_count": len(hits)})
        # Exact bounds on the central-pad descent corridor, useful as a simple
        # opening margin. Separate lobes can invalidate an overall maximum;
        # the component intersection test above handles that case directly.
        corridor = []
        for poly in tri:
            for axis, bound, sign in ((0,.0085,1),(0,-.0085,-1),(2,.009358,1),(2,-.15,-1)):
                poly = _clip(poly, axis, bound, sign)
            if len(poly):
                corridor.extend(poly[:, 1])
        required_half = float(np.max(np.abs(corridor))) if corridor else None

        # Ray coverage selects the closest solid interval that fits inside the
        # jaw opening, avoiding counting the opposite ear as the same patch.
        starts = ray_local @ rotation.T + center
        directions = np.tile(-rotation[:, 1], (len(starts), 1))
        locations, ray_id, face_id = mesh.ray.intersects_location(starts, directions, multiple_hits=True)
        local_hits = (locations - center) @ rotation
        spans = []
        for index in range(len(starts)):
            ids = np.flatnonzero(ray_id == index)
            ids = ids[np.argsort(-local_hits[ids, 1])]
            choices = []
            for k in range(0, len(ids)-1, 2):
                a, b = ids[k:k+2]
                hi, lo = local_hits[a, 1], local_hits[b, 1]
                if hi > .04 or lo < -.04:
                    continue
                choices.append((max(lo, 0, -hi), hi, lo,
                    abs(mesh.face_normals[face_id[a]] @ rotation[:, 1]),
                    abs(mesh.face_normals[face_id[b]] @ rotation[:, 1])))
            if choices:
                spans.append(min(choices))
        spans = np.asarray(spans)
        rows.append({
            "local_reference_m": candidate,
            "is_existing_ear_baseline": candidate not in candidates,
            "tracked_target_world_m": center.tolist(),
            "tracked_target_local_m": (center-origin).tolist(),
            "geometry_neighborhood_tracking_radius_m": .032,
            "full_hand_and_fingers_80mm_vertical_sweep_surface_intersections": sweep_intersections,
            "full_hand_and_fingers_80mm_vertical_sweep_clear": not sweep_intersections,
            "central_pad_corridor_required_half_opening_m": required_half,
            "central_pad_corridor_margin_at_40mm_m": .04-required_half if required_half is not None else None,
            "ray_projected_two_sided_coverage_m2": len(spans)*pixel_area,
            "ray_projected_mean_abs_surface_normal_dot_closing_axis": spans[:,3:].mean(axis=0).tolist() if len(spans) else None,
            "near_first_touch_2mm_band_projected_area_m2": [float(np.sum(spans[:,1] > spans[:,1].max()-.002)*pixel_area), float(np.sum(spans[:,2] < spans[:,2].min()+.002)*pixel_area)] if len(spans) else None,
            "initial_com_closing_axis_offset_m": float((com-center) @ rotation[:,1]),
            "initial_com_horizontal_offset_m": float(np.linalg.norm((com-center)[:2])),
            "actual_friction_grasp_proven": False,
        })
    result = {
        "source_run": str(run), "source_state": "teddy_pregrasp.npz",
        "source_state_time_s": float(state["time"]), "candidate_count": 9,
        "hand_rotation_xyzw": [2**-.5,-2**-.5,0,0], "maximum_total_opening_m": .080,
        "vertical_approach_m": .15, "pad_size_m": [.017,.017],
        "ray_coverage_grid": [n,n], "actual_teddy_com_m": com.tolist(),
        "candidates_and_baseline": rows,
        "limitations": [
            "Geometry-only: no force, friction, FEM strain, IK or whole-arm path test was run; no grasp stability is claimed.",
            "The saved soft shape is held fixed during sweep. Collision components use conservative convex hulls of the same XML collision meshes.",
            "Projected ray coverage and the hypothetical 2 mm first-touch band are geometric opportunity measures, not physical contact area.",
            "A rounded upper head has inclined normals. More projected coverage does not remove downward squeeze/ejection risk.",
            "The existing ear corridor maximum also includes distant lobes; use its full component sweep, not this maximum, to judge access.",
        ],
    }
    output = run / "independent-head-center-geometry.json"
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(output)
    for row in rows:
        print(row["local_reference_m"], "clear", row["full_hand_and_fingers_80mm_vertical_sweep_clear"],
              "hits",row["full_hand_and_fingers_80mm_vertical_sweep_surface_intersections"],
              "coverage_mm2",round(row["ray_projected_two_sided_coverage_m2"]*1e6,2),
              "closing_com_offset_mm",round(row["initial_com_closing_axis_offset_m"]*1000,2))


if __name__ == "__main__":
    main()
