"""Independent saved-state audit. No Genesis import and no GPU work.

The virtual mouth cap below is an analysis surface only, never a simulation
constraint. Passing geometric samples is not a proof of continuous contact.
Task completion is never inferred from a successful solver exit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent


def signed_volumes(points, tetra):
    return np.linalg.det(points[tetra[:, 1:]] - points[tetra[:, :1]]) / 6.0


def mesh_edges(faces):
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, first, count = np.unique(np.sort(directed, axis=1), axis=0,
                                     return_index=True, return_counts=True)
    return unique, count, directed[first[count == 1]]


def boundary_loops(faces):
    _, counts, boundary = mesh_edges(faces)
    neighbours = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    if not neighbours or any(len(v) != 2 for v in neighbours.values()):
        return [], False
    loops, visited = [], set()
    for start in neighbours:
        if start in visited:
            continue
        loop, previous, current = [], None, start
        while current not in visited:
            loop.append(current)
            visited.add(current)
            nxt = next(n for n in neighbours[current] if n != previous)
            previous, current = current, nxt
        if current != start:
            return [], False
        loops.append(np.asarray(loop, dtype=int))
    return loops, bool(np.all(counts <= 2))


def winding(points, vertices, faces, chunk=32):
    """Solid-angle winding of an oriented triangle mesh, in bounded batches."""
    tri = vertices[faces]
    result = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        batch = points[start:start+chunk]
        a, b, c = (tri[None, :, k] - batch[:, None] for k in range(3))
        la, lb, lc = (np.linalg.norm(q, axis=-1) for q in (a, b, c))
        numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
        denominator = (la*lb*lc + np.einsum("...i,...i->...", a,b)*lc
                       + np.einsum("...i,...i->...", b,c)*la
                       + np.einsum("...i,...i->...", c,a)*lb)
        result[start:start+len(batch)] = np.arctan2(numerator, denominator).sum(1)/(2*np.pi)
    return result


def tetra_boundary(points, tetra):
    face = np.concatenate([tetra[:, [1,2,3]], tetra[:, [0,3,2]],
                           tetra[:, [0,1,3]], tetra[:, [0,2,1]]])
    opposite = np.concatenate([tetra[:, k] for k in range(4)])
    _, inverse, counts = np.unique(np.sort(face,axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    mask = counts[inverse] == 1
    face, opposite = face[mask].copy(), opposite[mask]
    tri = points[face]
    inward = np.einsum("ij,ij->i", np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),
                       points[opposite]-tri[:,0]) > 0
    face[inward] = face[inward][:, [0,2,1]]
    return face


def proper_segment_crossings(poly):
    count = 0
    def cross(a,b):
        return a[0]*b[1]-a[1]*b[0]
    n = len(poly)
    for i in range(n):
        a,b = poly[i],poly[(i+1)%n]
        for j in range(i+2,n):
            if i == 0 and j == n-1:
                continue
            c,d = poly[j],poly[(j+1)%n]
            if cross(b-a,c-a)*cross(b-a,d-a) < -1e-16 and cross(d-c,a-c)*cross(d-c,b-c) < -1e-16:
                count += 1
    return count


def mouth_structure(rest_bag, bag_faces, config):
    metadata_path = HERE / "assets/metadata.json"
    bag_height = config.get("bag",{}).get("height_m")
    if bag_height is None:
        bag_height = json.loads(metadata_path.read_text())["bag"]["height_m"] if metadata_path.exists() else .32
    rim_z = float(np.min(rest_bag[:,2]) + bag_height)
    body_faces = bag_faces[np.all(rest_bag[bag_faces,2] <= rim_z+1e-7,axis=1)]
    body_source="geometric body-height separation"
    source_rim=None
    source_name=config.get("bag",{}).get("npz_file")
    if source_name is None:
        source_name=str(Path(config.get("bag",{}).get("mesh","bag.obj")).with_suffix(".npz"))
    source_path=Path(source_name)
    if not source_path.is_absolute():
        source_path=HERE/"assets"/source_path
    if source_path.exists():
        with np.load(source_path,allow_pickle=False) as source:
            if "body_face_indices" in source and "rim" in source:
                from scipy.spatial import cKDTree
                world=source["points"]+np.asarray(config["bag"]["position"])
                distances,mapping=cKDTree(rest_bag).query(world)
                if distances.max()>1e-7 or len(np.unique(mapping))!=len(world):
                    body_faces=np.empty((0,3),dtype=int)
                    body_source="source-to-imported mesh mapping failed; refusing geometric fallback"
                else:
                    candidate=mapping[source["faces"][source["body_face_indices"]]]
                    imported={tuple(v) for v in np.sort(bag_faces,axis=1)}
                    if all(tuple(v) in imported for v in np.sort(candidate,axis=1)):
                        body_faces=candidate
                        source_rim=mapping[source["rim"]]
                        rim_z=float(rest_bag[source_rim,2].mean())
                        body_source="explicit physical body-face labels mapped to imported rest mesh"
                    else:
                        body_faces=np.empty((0,3),dtype=int)
                        body_source="source body triangles absent from imported topology"
    body_loops, body_manifold = boundary_loops(body_faces)
    full_loops, full_manifold = boundary_loops(bag_faces)
    mesh = trimesh.Trimesh(rest_bag, bag_faces, process=False)
    body_mesh = trimesh.Trimesh(rest_bag, body_faces, process=False)
    valid = (len(body_loops) == 1 and body_manifold and full_manifold
             and mesh.is_winding_consistent and body_mesh.is_winding_consistent)
    tri=rest_bag[body_faces]
    bottom_panels=int(np.all(np.abs(tri[:,:,2]-rest_bag[:,2].min())<1e-7,axis=1).sum())
    side_panels=int((np.all(np.abs(tri[:,:,0]-rest_bag[:,0].min())<1e-7,axis=1)
                    |np.all(np.abs(tri[:,:,0]-rest_bag[:,0].max())<1e-7,axis=1)).sum())
    top_boundary=bool(valid and (set(body_loops[0])==set(source_rim) if source_rim is not None
                                else np.all(np.abs(rest_bag[body_loops[0],2]-rim_z)<1e-7)))
    two_panel=bool(valid and top_boundary and bottom_panels==0 and side_panels==0)
    report = {"actual_open_body_boundary":bool(valid),"body_boundary_loop_count":len(body_loops),
              "complete_bag_boundary_loop_count":len(full_loops),"nonmanifold_edges":int((mesh_edges(bag_faces)[1]>2).sum()),
              "connected_components":len(mesh.split(only_watertight=False)),
              "body_triangle_count":len(body_faces),"rim_rest_z":rim_z,
              "body_face_selection":body_source,
              "independent_flat_bottom_triangles":bottom_panels,"independent_flat_side_triangles":side_panels,
              "body_boundary_only_at_top":top_boundary,"two_panels_directly_sewn_along_u":two_panel,
              "analysis_cap_is_not_simulated":True}
    return body_faces, body_loops[0] if valid else np.empty(0,dtype=int), report


def containment(bag, bear, body_faces, rim, tetra_weights, tets):
    rim_points = bag[rim]
    center = rim_points.mean(0)
    _,_,basis = np.linalg.svd(rim_points-center,full_matrices=False)
    projected = (rim_points-center)@basis[:2].T
    area = abs(np.sum(projected[:,0]*np.roll(projected[:,1],-1)-projected[:,1]*np.roll(projected[:,0],-1)))/2
    crossings = proper_segment_crossings(projected)
    vertices = np.concatenate([bag,center[None]])
    _,_,directed_boundary = mesh_edges(body_faces)
    caps = np.column_stack([directed_boundary[:,1],directed_boundary[:,0],np.full(len(directed_boundary),len(bag))])
    faces = np.concatenate([body_faces,caps])
    surface = trimesh.Trimesh(vertices, faces, process=False)
    center_of_mass = np.sum(bear[tets].mean(1)*tetra_weights[:,None],axis=0)/tetra_weights.sum()
    valid = surface.is_watertight and surface.is_winding_consistent and crossings == 0 and area > 1e-5
    if valid:
        contained = np.abs(winding(np.vstack([bear,center_of_mass]),vertices,faces)) > .5
        fraction = float(contained[:-1].mean())
        com_inside = bool(contained[-1])
    else:
        fraction,com_inside = 0.,False
    return {"mouth_projected_area_m2":float(area),"mouth_projected_crossings":crossings,
            "mouth_projected_dimensions_m":np.ptp(projected,axis=0).tolist(),
            "mouth_nonplanarity_max_m":float(np.abs((rim_points-center)@basis[2]).max()),
            "virtual_closed_body_valid":bool(valid),"bear_vertex_inside_fraction":fraction,
            "bear_center_of_mass_m":center_of_mass.tolist(),"bear_center_of_mass_inside":com_inside,
            "all_bear_vertices_inside":bool(valid and fraction == 1. and com_inside)}


def final_contact_audit(bag, bear, faces, tets, tolerance=.001):
    """Check bag vertices/edge midpoints/face centroids against the bear volume."""
    edges,_,_ = mesh_edges(faces)
    samples = np.vstack([bag,bag[edges].mean(1),bag[faces].mean(1)])
    candidate = np.all((samples >= bear.min(0)-tolerance)&(samples <= bear.max(0)+tolerance),axis=1)
    samples = samples[candidate]
    surface = trimesh.Trimesh(bear,tetra_boundary(bear,tets),process=False)
    inside = np.abs(winding(samples,surface.vertices,surface.faces)) > .5
    depth = np.empty(0)
    if inside.any():
        _,depth,_ = trimesh.proximity.closest_point(surface,samples[inside])
    return {"method":"Discrete bag vertex, edge-midpoint and triangle-centroid winding/distance samples",
            "sample_count":len(candidate),"near_bear_samples":len(samples),
            "inside_samples":int(inside.sum()),"penetration_over_tolerance":int((depth>tolerance).sum()),
            "maximum_sampled_penetration_m":float(depth.max()) if len(depth) else 0.,
            "tolerance_m":tolerance,"continuous_collision_guarantee":False}


def bear_deformation_audit(rest_points, tetra, frames, time, q_right, events=None):
    """Report FEM inversion independently of whether the gripper lifted it.

    Principal stretches come from the actual deformation gradient. Percentiles
    are weighted by rest tet volume so small slivers cannot dominate the bulk.
    Stable Neo-Hookean and surface IPC contact do not prohibit tet inversion.
    """
    tetra = np.asarray(tetra, dtype=int)
    reference = np.asarray(rest_points)[tetra]
    dm_inv = np.linalg.inv(np.moveaxis(reference[:, 1:] - reference[:, :1], 1, 2))
    volume = np.abs(signed_volumes(np.asarray(rest_points), tetra))
    weights = volume / volume.sum()
    edge_length_sq = np.stack([np.sum((reference[:, j]-reference[:, i])**2, axis=1)
                               for i in range(4) for j in range(i+1, 4)], axis=1)
    quality = 12*(3*volume)**(2/3) / edge_length_sq.sum(axis=1)
    first_inversion, worst, final_quantiles = None, None, None
    max_stretch, max_stretched_fraction = 0., 0.
    samples = []
    for i, (points, timestamp) in enumerate(zip(frames, time)):
        deformed = points[tetra]
        gradient = np.moveaxis(deformed[:, 1:] - deformed[:, :1], 1, 2) @ dm_inv
        ratio = np.linalg.det(gradient)
        stretch = np.linalg.svd(gradient, compute_uv=False)
        count = int((ratio <= 0).sum())
        sample = {"time_s": float(timestamp), "minimum_volume_ratio": float(ratio.min()),
                  "inverted_tetrahedra": count,
                  "inverted_reference_volume_fraction": float(weights[ratio <= 0].sum()),
                  "right_finger_q_m": np.asarray(q_right[i])[-2:].tolist()}
        samples.append(sample)
        if count and first_inversion is None:
            first_inversion = sample
        if worst is None or sample["minimum_volume_ratio"] < worst["minimum_volume_ratio"]:
            k = int(np.argmin(ratio))
            worst = {**sample, "tetra_index": k,
                     "rest_center_m": reference[k].mean(axis=0).tolist(),
                     "rest_volume_m3": float(volume[k]),
                     "rest_mean_ratio_quality": float(quality[k]),
                     "rest_quality_percentile": float(100*(quality < quality[k]).mean())}
        max_stretch = max(max_stretch, float(stretch.max()))
        max_stretched_fraction = max(max_stretched_fraction, float(weights[stretch[:, 0] > 2.].sum()))
        if i == len(time)-1:
            final_quantiles = {}
            for name, values in (("smallest", stretch[:, -1]), ("largest", stretch[:, 0])):
                order = np.argsort(values)
                final_quantiles[name] = dict(zip(("p01", "p50", "p99"),
                    np.interp([.01, .5, .99], np.cumsum(weights[order]), values[order]).tolist()))
    phases = {}
    for event in events or []:
        start, end = event.get("start", 0), event.get("start", 0) + event.get("duration", 0)
        window = [s for s in samples if start-1e-8 <= s["time_s"] <= end+.011]
        if window:
            phases[event.get("phase", "unknown")] = {
                "minimum_volume_ratio": min(s["minimum_volume_ratio"] for s in window),
                "maximum_inverted_tetrahedra": max(s["inverted_tetrahedra"] for s in window)}
    return {"first_observed_inversion": first_inversion,
            "worst_volume_ratio_sample": worst,
            "maximum_principal_stretch": max_stretch,
            "maximum_reference_volume_fraction_with_principal_stretch_over_2": max_stretched_fraction,
            "final_volume_weighted_principal_stretch_quantiles": final_quantiles,
            "phase_deformation": phases,
            "positive_tetra_volume_in_all_saved_states": first_inversion is None,
            "rest_mean_ratio_quality_quantiles": dict(zip(("min", "p01", "p50", "p99"),
                np.quantile(quality, [0, .01, .5, .99]).tolist())),
            "material_realism_calibrated": False}


def audit_arrays(rest, frames, config, *, task_completed=False, hold_duration=.5, joint_trace=None, events=None):
    required=("time","bear","bag","q_left","q_right","tcp_left","tcp_right")
    missing=[k for k in required if k not in frames]
    if missing:
        return {"status":"incomplete","task_success":False,"missing_frame_arrays":missing}
    time=np.asarray(frames["time"],float).reshape(-1)
    if len(time)<3 or np.any(np.diff(time)<0):
        return {"status":"incomplete","task_success":False,"reason":"Need at least three strictly increasing saved timestamps."}
    # Phase snapshots may repeat the same last video timestamp. Keep the last
    # saved value at each timestamp, without silently sorting backwards time.
    keep=np.r_[np.diff(time)>1e-10,True]
    duplicate_count=int((~keep).sum())
    frames={k:np.asarray(frames[k])[keep] for k in frames
            if np.asarray(frames[k]).shape[:1] == (len(time),)}
    time=time[keep]
    if len(time)<3:
        return {"status":"incomplete","task_success":False,"reason":"Fewer than three distinct saved timestamps."}
    finite=all(np.isfinite(np.asarray(frames[k])).all() for k in required)
    if not finite:
        return {"status":"failed","task_success":False,"reason":"Nonfinite saved state."}
    bear0,bag0=np.asarray(rest["bear"]),np.asarray(rest["bag"])
    tets,faces=np.asarray(rest["bear_tets"],int),np.asarray(rest["bag_faces"],int)
    bears,bags=np.asarray(frames["bear"]),np.asarray(frames["bag"])
    if bears.shape[1:]!=bear0.shape or bags.shape[1:]!=bag0.shape or len(bears)!=len(time) or len(bags)!=len(time):
        return {"status":"failed","task_success":False,"reason":"Saved topology/state shapes disagree."}
    rest_vol=signed_volumes(bear0,tets)
    if np.any(np.abs(rest_vol)<1e-14):
        return {"status":"failed","task_success":False,"reason":"Degenerate reference tetrahedron."}
    weights=np.abs(rest_vol)
    tetra_ratio_min=1.
    inverted_volume_fraction=0.
    table_penetration={"bear":0.,"bag":0.}
    table_height=config["table"]["height"]
    half=np.asarray(config["table"]["size"][:2])/2
    edges,_,_=mesh_edges(faces)
    rest_length=np.linalg.norm(bag0[edges[:,1]]-bag0[edges[:,0]],axis=1)
    stretch_max=1.
    for bear,bag in zip(bears,bags):
        ratios=signed_volumes(bear,tets)/rest_vol
        tetra_ratio_min=min(tetra_ratio_min,float(ratios.min()))
        inverted_volume_fraction=max(inverted_volume_fraction,float(weights[ratios<=0].sum()/weights.sum()))
        for name,p in (("bear",bear),("bag",bag)):
            on_table=np.all(np.abs(p[:,:2])<=half,axis=1)
            if on_table.any():
                table_penetration[name]=max(table_penetration[name],float(max(0.,table_height-p[on_table,2].min())))
        stretch_max=max(stretch_max,float((np.linalg.norm(bag[edges[:,1]]-bag[edges[:,0]],axis=1)/rest_length).max()))
    area=float(trimesh.Trimesh(bag0,faces,process=False).area)
    # pyuipc 0.0.29 defines the passed thickness as one-sided radius r.
    bag_mass=area*config["bag"]["rho"]*2*config["bag"]["thickness"]
    mass={"bag_kg":bag_mass,"bear_kg":float(weights.sum()*config["bear"]["rho"]),
          "bag_mass_formula":"rest area * density * 2 * IPC thickness radius",
          "derived_from_reference_geometry_and_config":True,"runtime_IPC_mass_buffer_checked":False}
    motion={}
    validation=config.get("validation",{})
    vmax=validation.get("max_joint_velocity",1.5)
    amax=validation.get("max_joint_acceleration",10.)
    joint_time=time
    joint_values={side:np.asarray(frames["q_"+side],float) for side in ("left","right")}
    motion_source="saved visualization states"
    if joint_trace is not None:
        trace=np.asarray(joint_trace,dtype=float)
        if trace.ndim!=2 or trace.shape[1]!=19 or len(trace)<3 or not np.isfinite(trace).all() or np.any(np.diff(trace[:,0])<=0):
            return {"status":"failed","task_success":False,"reason":"Invalid every-step joint_trace, expected increasing time plus nine joints per arm."}
        joint_time=trace[:,0]
        joint_values={"left":trace[:,1:10],"right":trace[:,10:19]}
        motion_source="every-physics-step actual joint_trace"
    for side in ("left","right"):
        q=joint_values[side]
        velocity=np.gradient(q[:,:7],joint_time,axis=0,edge_order=2)
        acceleration=np.gradient(velocity,joint_time,axis=0,edge_order=2)
        motion[side]={"maximum_joint_velocity_rad_s":float(np.abs(velocity).max()),
            "maximum_joint_acceleration_rad_s2":float(np.abs(acceleration).max()),
            "per_joint_peak_velocity_rad_s":np.abs(velocity).max(0).tolist(),
            "per_joint_peak_acceleration_rad_s2":np.abs(acceleration).max(0).tolist(),
            "within_configured_limits":bool(np.abs(velocity).max()<=vmax and np.abs(acceleration).max()<=amax)}
    body_faces,rim,topology=mouth_structure(bag0,faces,config)
    final_inside={"all_bear_vertices_inside":False}
    held=False
    hold_checks=[]
    if len(rim):
        final_inside=containment(bags[-1],bears[-1],body_faces,rim,weights,tets)
        hold_indices=np.flatnonzero(time>=time[-1]-hold_duration)
        if time[-1]-time[0]>=hold_duration and len(hold_indices)>=3:
            check_indices=np.unique(np.r_[hold_indices[0],hold_indices[len(hold_indices)//2],hold_indices[-1]])
            for i in check_indices:
                value=containment(bags[i],bears[i],body_faces,rim,weights,tets)
                hold_checks.append({"time":float(time[i]),**value})
            held=all(v["all_bear_vertices_inside"] for v in hold_checks)
    contact=final_contact_audit(bags[-1],bears[-1],faces,tets)
    events=events or []
    deformation=bear_deformation_audit(bear0,tets,bears,time,frames["q_right"],events)
    phase_names=[e.get("phase") for e in events]
    required_phases=("close_both_handles","spread_bag","release_right_handle",
        "teddy_pregrasp","teddy_grasp_approach","close_teddy_gripper","lift_teddy",
        "transport_teddy","insert_teddy","release_teddy","withdraw_right_arm","final_settle")
    ordered=all(p in phase_names for p in required_phases)
    if ordered:
        ordered=bool(np.all(np.diff([phase_names.index(p) for p in required_phases])>0))
    by_phase={e["phase"]:e for e in events if "phase" in e}
    lift_clear=False
    left_displacement=None
    if "lift_teddy" in by_phase:
        e=by_phase["lift_teddy"]
        index=int(np.argmin(np.abs(time-(e["start"]+e["duration"]))))
        lift_clear=bool(bears[index,:,2].min()>table_height+.04)
    if "release_right_handle" in by_phase:
        fixed_window=time>=by_phase["release_right_handle"]["start"]
        tcp_left=np.asarray(frames["tcp_left"])[fixed_window]
        if len(tcp_left):
            left_displacement=float(np.linalg.norm(tcp_left-tcp_left[0],axis=1).max())
    right_open=bool(np.min(joint_values["right"][-1,7:9])>=.025)
    checks={"finite_saved_states":finite,"actual_open_bag_topology":topology["actual_open_body_boundary"],
        "two_cloth_panels_with_direct_u_seam_and_no_bottom_or_side_panels":topology["two_panels_directly_sewn_along_u"],
        "no_tetrahedron_inversion":tetra_ratio_min>0.,
        "no_tetrahedron_volume_below_5_percent":tetra_ratio_min>.05,
        "table_penetration_below_2mm":max(table_penetration.values())<=.002,
        "cloth_edge_stretch_below_15_percent":stretch_max<=1.15,
        "joint_motion_within_configured_limits":all(v["within_configured_limits"] for v in motion.values()),
        "motion_sample_interval_at_most_20ms":float(np.diff(joint_time).max())<=.020001,
        "no_final_sampled_bag_bear_penetration_over_1mm":contact["penetration_over_tolerance"]==0,
        "bear_fully_inside_bag_final":final_inside["all_bear_vertices_inside"],
        "bear_remains_inside_during_final_hold_samples":held,
        "requested_phases_recorded_in_order":ordered,
        "bear_lifted_clear_of_table":lift_clear,
        "right_fingers_open_after_release":right_open,
        "left_gripper_holds_pose_within_15mm_after_handoff":left_displacement is not None and left_displacement<=.015,
        "controller_explicitly_reports_task_complete":bool(task_completed)}
    passed=all(checks.values())
    physical_keys=("finite_saved_states","actual_open_bag_topology","no_tetrahedron_inversion",
        "no_tetrahedron_volume_below_5_percent",
        "table_penetration_below_2mm","cloth_edge_stretch_below_15_percent",
        "joint_motion_within_configured_limits","no_final_sampled_bag_bear_penetration_over_1mm")
    physical_failure=not all(checks[k] for k in physical_keys)
    return {"status":"passed" if passed else "failed" if task_completed or physical_failure else "incomplete",
        "task_success":passed,"checks":checks,"frames":len(time),"duplicate_timestamps_removed":duplicate_count,
        "duration_s":float(time[-1]-time[0]),
        "maximum_saved_interval_s":float(np.diff(time).max()),"mass":mass,"bag_topology":topology,
        "minimum_tetrahedron_volume_ratio":tetra_ratio_min,
        "maximum_inverted_reference_volume_fraction":inverted_volume_fraction,
        "maximum_table_penetration_m":table_penetration,"maximum_bag_edge_stretch_ratio":stretch_max,
        "motion":motion,"motion_source":motion_source,"maximum_joint_trace_interval_s":float(np.diff(joint_time).max()),
        "motion_thresholds":{"velocity_rad_s":vmax,"acceleration_rad_s2":amax,
            "source":"config.validation or conservative audit defaults; not robot manufacturer limits"},
        "final_containment":final_inside,"hold_containment_samples":hold_checks,"final_contact":contact,
        "bear_deformation":deformation,
        "sequence_evidence":{"recorded_phases":phase_names,
            "maximum_left_tcp_displacement_after_handoff_m":left_displacement,
            "final_right_finger_positions_m":joint_values["right"][-1,7:9].tolist()},
        "limitations":["Saved-state sampling does not certify continuous nonpenetration or unobserved velocity peaks.",
            "Two-way IPC ABD versus rendered link transform error is unavailable in these arrays.",
            "Finger opening and event timestamps support sequence checks but do not certify force closure.",
            "Material values remain uncalibrated; no real-world success claim follows."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest",type=Path,required=True)
    parser.add_argument("--frames",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task-completed",action="store_true")
    parser.add_argument("--joint-trace",type=Path)
    parser.add_argument("--events",type=Path)
    args=parser.parse_args()
    trace_path=args.joint_trace or args.frames.parent/"joint_trace.npy"
    joint_trace=np.load(trace_path,allow_pickle=False) if trace_path.exists() else None
    events_path=args.events or args.frames.parent/"events.json"
    events=json.loads(events_path.read_text()) if events_path.exists() else []
    if not events and (args.frames.parent/"report.json").exists():
        events=json.loads((args.frames.parent/"report.json").read_text()).get("events",[])
    with np.load(args.rest,allow_pickle=False) as rest,np.load(args.frames,allow_pickle=False) as frames:
        result=audit_arrays(rest,frames,json.loads(args.config.read_text()),task_completed=args.task_completed,
                            joint_trace=joint_trace,events=events)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":result["status"],"task_success":result["task_success"],"output":str(args.output)}))
    return 0 if result["task_success"] else 2


if __name__=="__main__":
    raise SystemExit(main())
