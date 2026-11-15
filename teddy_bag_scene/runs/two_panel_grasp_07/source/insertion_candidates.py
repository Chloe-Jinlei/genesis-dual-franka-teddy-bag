"""Offline insertion candidates from an actual saved bag/teddy/robot state.

This module never initializes Genesis or modifies a simulation. It predicts
the held teddy's footprint under a rigid wrist rotation only to screen routes;
the real teddy remains deformable and needs fresh runtime contact checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from motion import DOWN, Planner
from validate import mouth_structure, proper_segment_crossings, signed_volumes


def projected_inside(points, polygon):
    """Ray casting of XY points against the actual ordered rim projection."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    inside = np.zeros(len(points), bool)
    distances = np.full(len(points), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = b-a
        if np.dot(delta, delta) < 1e-16:
            continue
        fraction = np.clip(((points-a)@delta)/np.dot(delta, delta), 0., 1.)
        distances = np.minimum(distances, np.linalg.norm(points-a-fraction[:,None]*delta, axis=1))
        crossing = (a[1] > points[:,1]) != (b[1] > points[:,1])
        if abs(delta[1]) > 1e-14:
            x_cross = a[0] + (points[:,1]-a[1])*delta[0]/delta[1]
            inside ^= crossing & (points[:,0] < x_cross)
    return inside | (distances < 1e-8), distances


def candidates(rest, state, config, *, yaws=(0., 90., -90.), clearances=(.04, .08),
               approaches=("vertical", "mouth_normal"), screen_geometry_first=False):
    bag, bear = np.asarray(state["bag"]), np.asarray(state["bear"])
    if not np.isfinite(bag).all() or not np.isfinite(bear).all():
        raise ValueError("Observed soft bodies contain nonfinite coordinates")
    _, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    if len(rim) < 4:
        raise ValueError("Could not identify a valid sewn-bag mouth")
    rp = bag[rim]
    mouth_center = rp.mean(0)
    centered = rp-mouth_center
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    normal = basis[-1].copy()
    if normal[2] < 0:
        normal = -normal
    area_xy = abs(np.sum(rp[:,0]*np.roll(rp[:,1],-1)-rp[:,1]*np.roll(rp[:,0],-1)))/2
    crossings = proper_segment_crossings(rp[:,:2])
    left_distance = float(np.linalg.norm(bag-np.asarray(state["tcp_left"]),axis=1).min())
    planner = Planner(config)
    qs = {s:np.asarray(state["q_"+s],float).reshape(-1)[:7].copy() for s in ("left","right")}
    current_tcp, current_rotation = planner.fk("right", qs["right"])
    measured_tcp = np.asarray(state["tcp_right"],float)
    relative_bear = bear-measured_tcp
    tets = np.asarray(rest["bear_tets"],int)
    weights = np.abs(signed_volumes(np.asarray(rest["bear"]),tets))
    volume_ratio = signed_volumes(bear,tets)/signed_volumes(np.asarray(rest["bear"]),tets)
    center_of_mass = np.average(bear[tets].mean(1),axis=0,weights=weights)
    offset = center_of_mass-measured_tcp
    fingers = {s:float(np.clip(np.asarray(state["q_"+s]).reshape(-1)[7:9].mean(),0.,.04)) for s in qs}
    warning = []
    if left_distance > .03:
        warning.append("Left gripper is farther than 30 mm from every bag vertex; this state does not demonstrate a held bag.")
    if volume_ratio.min() < .05:
        warning.append("Current teddy has a tetrahedron volume ratio below 0.05; this state is not an admissible insertion start.")
    limits = config.get("task", {}).get("insertion_limits", {})
    compression_limit = float(limits.get("max_planar_compression",.08))
    outside_limit = float(limits.get("max_outside_projection_m",.008))
    if not 0 <= compression_limit <= .12 or not 0 <= outside_limit <= .012:
        raise ValueError("Insertion compression allowances exceed the bounded offline screening range")
    _,_,bear_xy_basis = np.linalg.svd(bear[:,:2]-bear[:,:2].mean(0),full_matrices=False)
    aligned_yaw = (np.degrees(np.arctan2(basis[0,1],basis[0,0])
                    -np.arctan2(bear_xy_basis[0,1],bear_xy_basis[0,0]))+90)%180-90
    yaws = list(yaws)
    if all(abs(aligned_yaw-yaw)>.5 for yaw in yaws):
        yaws.append(float(aligned_yaw))
    report = {
        "kind":"offline_geometry_candidates_from_observed_state",
        "robot_planning_scope":"Only candidates passing geometric screens" if screen_geometry_first else "All requested geometric candidates",
        "observed_time_s":float(state["time"]), "warnings":warning,
        "mouth":{"center_m":mouth_center.tolist(),"rim_maximum_z_m":float(rp[:,2].max()),
            "xy_projected_area_m2":float(area_xy),"xy_projected_crossings":crossings,
            "fitted_plane_normal_abs_z":float(abs(basis[-1,2])),"topology":topology},
        "left_nearest_bag_vertex_m":left_distance,
        "observed_bear_dimensions_xyz_m":np.ptp(bear,axis=0).tolist(),
        "observed_bear_center_of_mass_offset_m":offset.tolist(),
        "current_minimum_bear_tetrahedron_volume_ratio":float(volume_ratio.min()),
        "screening_allowances":{"max_planar_compression":compression_limit,
            "max_outside_projection_m":outside_limit},
        "fk_vs_recorded_right_tcp_error_m":float(np.linalg.norm(current_tcp-measured_tcp)),
        "candidates":[],
        "limitations":[
            "This is an offline rigid-arm check, not a successful physical insertion or grasp certificate.",
            "Bear footprint predictions rotate the observed shape rigidly around the hand; real gravity, deformation and slip can change it.",
            "Rim and held bear observations become stale while the robot moves; observe again before insertion.",
            "A footprint-fit result assumes translation through the chosen projection plane; it does not establish volume containment.",
            "Sampled rigid clearance does not prove continuous collision freedom; runtime replanning and physical collision checks are required.",
        ]}
    report["mouth"].update(fitted_normal_up=normal.tolist(),
        maximum_nonplanarity_m=float(np.abs(centered@normal).max()))
    for approach in approaches:
        if approach not in ("vertical", "mouth_normal"):
            raise ValueError(f"Unknown approach {approach}")
        projection = np.eye(3)[:2] if approach == "vertical" else basis[:2]
        direction = np.array([0.,0.,1.]) if approach == "vertical" else normal
        polygon = centered@projection.T
        projected_crossings = proper_segment_crossings(polygon)
        mode_warnings = warning.copy()
        if projected_crossings:
            mode_warnings.append("Mouth contour crosses itself in this approach projection; no footprint-fit claim is made.")
        if approach == "vertical" and normal[2] < .45:
            mode_warnings.append("Mouth is too steep for a simple downward insertion.")
        if np.abs(centered@normal).max() > .025:
            mode_warnings.append("Mouth deviates from its fitted plane by over 25 mm; one fitted normal does not describe the folds.")
        for yaw in yaws:
            target_rotation = Rotation.from_euler("z",yaw,degrees=True)*DOWN
            delta = target_rotation*current_rotation.inv()
            predicted_relative = delta.apply(relative_bear)
            rotated_offset = delta.apply(offset)
            centered_prediction = predicted_relative-rotated_offset
            projected = centered_prediction@projection.T
            inside, edge_distance = projected_inside(projected,polygon)
            fit = bool(projected_crossings == 0 and inside.all() and edge_distance.min() >= .002)
            maximum_outside = float(edge_distance[~inside].max()) if (~inside).any() else 0.
            needed_compression = None
            if projected_crossings == 0 and maximum_outside <= outside_limit:
                for amount in np.linspace(0.,compression_limit,9):
                    compressed_inside, compressed_distance = projected_inside(projected*(1-amount),polygon)
                    if compressed_inside.all() and compressed_distance.min() >= .002:
                        needed_compression = float(amount)
                        break
            for clearance in clearances:
                if approach == "vertical":
                    above_z = max(float(rp[:,2].max()-predicted_relative[:,2].min()+clearance),
                                  float(config["table"]["height"]+.30))
                    above = np.r_[mouth_center[:2]-rotated_offset[:2],above_z]
                    insert = above.copy()
                    insert[2] = max(float(config["table"]["height"]+.03-predicted_relative[:,2].min()),
                                    float(mouth_center[2]-.08-rotated_offset[2]))
                else:
                    distance = max(0.,float((centered@direction).max()
                                    -(centered_prediction@direction).min()+clearance))
                    above = mouth_center-rotated_offset+direction*distance
                    max_depth = (mouth_center[2]+centered_prediction[:,2].min()
                                 -config["table"]["height"]-.03)/max(direction[2],1e-6)
                    depth = min(.08,max(0.,float(max_depth)))
                    insert = mouth_center-rotated_offset-direction*depth
                staging = np.r_[current_tcp[:2],above[2]]
                entry = {"approach":approach,"outward_approach_axis":direction.tolist(),
                    "warnings":mode_warnings,"wrist_world_z_yaw_deg":float(yaw),
                    "bear_clearance_above_rim_m":float(clearance),
                    "rotation_xyzw":target_rotation.as_quat().tolist(),
                    "predicted_rotated_bear_dimensions_xyz_m":np.ptp(predicted_relative,axis=0).tolist(),
                    "predicted_projected_vertices_inside_fraction":float(inside.mean()),
                    "predicted_footprint_fits_with_2mm_margin":fit,
                    "predicted_maximum_outside_rim_distance_m":maximum_outside,
                    "required_predicted_planar_compression":needed_compression,
                    "bounded_compression_screen_passes":needed_compression is not None,
                    "above_tcp_m":above.tolist(),"insert_tcp_m":insert.tolist(),
                    "robot_route_feasible":False,"robot_planning_attempted":False,"stages":[]}
                current = {s:q.copy() for s,q in qs.items()}
                try:
                    if screen_geometry_first and (mode_warnings or needed_compression is None):
                        raise ValueError("Observed mouth or bounded teddy compression screen rejects this candidate before robot planning")
                    if above[2]+predicted_relative[:,2].min() < config["table"]["height"]+.01:
                        raise ValueError("Predicted teddy would intersect the tabletop at the outside approach point")
                    entry["robot_planning_attempted"] = True
                    for name, target, rotation in (("height_at_current_xy",staging,current_rotation),
                            ("rotate_held_teddy",staging,target_rotation),
                            ("translate_outside_mouth",above,target_rotation),
                            ("insert_candidate",insert,target_rotation)):
                        plan = planner.trajectory(current,{"right":target},rotations={"right":rotation},
                            samples=35,validate_samples=61,duration=2.5,
                            finger_start=fingers,finger_end=fingers,clearance_margin=.008)
                        current = plan.end_q()
                        entry["stages"].append({"name":name,"right_tcp_m":target.tolist(),
                            "rotation_xyzw":rotation.as_quat().tolist(),"duration_s":plan.duration,
                            "right_q_end":current["right"].tolist()})
                    entry["robot_route_feasible"] = True
                    entry["total_motion_duration_s"] = sum(s["duration_s"] for s in entry["stages"])
                except ValueError as error:
                    entry["rejection"] = str(error)
                entry["geometry_screen_passes"] = bool(entry["robot_route_feasible"]
                    and needed_compression is not None and not mode_warnings)
                report["candidates"].append(entry)
    report["robot_feasible_candidates"] = sum(v["robot_route_feasible"] for v in report["candidates"])
    report["geometry_screen_candidates"] = sum(v["geometry_screen_passes"] for v in report["candidates"])
    return report


class NoInsertionRoute(ValueError):
    """Fail closed while retaining the complete rejection report for logging."""
    def __init__(self, report):
        super().__init__("No observed-state insertion route passed mouth, bounded-compression and rigid-motion checks")
        self.report = report


def choose_insertion_route(rest, state, config, *, output=None):
    """Return a reviewable route for fresh actual state; never executes it.

    Returned stages use world TCP positions and SciPy xyzw hand rotations.
    Runtime must reobserve the mouth immediately before insert_candidate; this
    prediction cannot keep a collapsing mouth open or guarantee contact grasp.
    """
    report = candidates(rest,state,config,screen_geometry_first=True)
    viable = [c for c in report["candidates"] if c["geometry_screen_passes"]]
    if output is not None:
        Path(output).write_text(json.dumps(report,indent=2)+"\n")
    if not viable:
        raise NoInsertionRoute(report)
    tilted = report["mouth"]["fitted_plane_normal_abs_z"] < np.cos(np.deg2rad(15))
    viable.sort(key=lambda c:(c["required_predicted_planar_compression"],
        int(tilted and c["approach"]!="mouth_normal"),c["total_motion_duration_s"]))
    return {"route":viable[0],"report":report,
        "requires_reobservation_before_insertion":True,
        "observe_before_stage":"insert_candidate",
        "compression_is_a_prediction_not_applied_deformation":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,help="Run directory containing rest.npz and config.json")
    parser.add_argument("--state",type=Path,help="A phase snapshot .npz; default is last frame of states.npz")
    parser.add_argument("--output",type=Path,help="Default: insertion_candidates.json in run directory")
    args=parser.parse_args()
    rest=dict(np.load(args.run/"rest.npz",allow_pickle=False))
    raw=dict(np.load(args.state or args.run/"states.npz",allow_pickle=False))
    state={k:raw[k][-1] if np.asarray(raw["time"]).ndim else raw[k]
           for k in ("time","bag","bear","q_left","q_right","tcp_left","tcp_right")}
    result=candidates(rest,state,json.loads((args.run/"config.json").read_text()))
    output=args.output or args.run/"insertion_candidates.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(output),"warnings":result["warnings"],
        "robot_feasible_candidates":result["robot_feasible_candidates"],
        "geometry_screen_candidates":result["geometry_screen_candidates"]}))


if __name__=="__main__":
    main()
