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
from scipy.spatial import ConvexHull, QhullError
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


def footprint_containment(points, polygon, *, margin=.002):
    """Conservatively contain the full teddy projection, not only its nodes.

    A convex envelope can reject a genuinely fitting concave teddy, but cannot
    falsely pass an edge bridging a concave/folded mouth. Segment distances
    include both directions, and crossings are checked analytically.
    """
    try:
        points = np.unique(np.asarray(points, float), axis=0)
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return False, 0., float("inf")
    polygon = np.asarray(polygon, float)
    inside, distances = projected_inside(hull, polygon)
    _, reverse_distances = projected_inside(polygon, hull)
    a, b = hull, np.roll(hull,-1,axis=0)
    c, d = polygon, np.roll(polygon,-1,axis=0)
    def cross(x,y):
        return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    first = cross((b-a)[:,None],c[None]-a[:,None])
    second = cross((b-a)[:,None],d[None]-a[:,None])
    third = cross((d-c)[None],a[:,None]-c[None])
    fourth = cross((d-c)[None],b[:,None]-c[None])
    crosses = bool(((first*second < -1e-20) & (third*fourth < -1e-20)).any())
    separation = min(float(distances.min()),float(reverse_distances.min()))
    fit = bool(inside.all() and not crosses and separation >= margin)
    if fit:
        return True, separation, 0.
    # Bound the outside excursion of hull edges, including parts between
    # mesh nodes. The distance-to-boundary function is 1-Lipschitz, so half
    # the maximum sample step bounds an unsampled excursion conservatively.
    sampled=[]
    maximum_step=0.
    for p,q in zip(a,b):
        intervals=max(1,int(np.ceil(np.linalg.norm(q-p)/.001)))
        maximum_step=max(maximum_step,float(np.linalg.norm(q-p)/intervals))
        sampled.append(p+(q-p)*np.linspace(0.,1.,intervals+1)[:,None])
    sample_inside, sample_distance=projected_inside(np.concatenate(sampled),polygon)
    outside=float(sample_distance[~sample_inside].max()) if (~sample_inside).any() else 0.
    return False, separation, outside+maximum_step/2


def minimum_nonadjacent_edge_distance(polygon):
    """Detect projected rim self-touch, including collinear overlapping edges."""
    polygon=np.asarray(polygon,float)
    minimum=float("inf")
    for i,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        for j in range(i+2,len(polygon)):
            if i==0 and j==len(polygon)-1:
                continue
            c,d=polygon[j],polygon[(j+1)%len(polygon)]
            for points,p,q in ((np.asarray([a,b]),c,d),(np.asarray([c,d]),a,b)):
                delta=q-p
                fraction=np.clip((points-p)@delta/max(float(delta@delta),1e-20),0.,1.)
                minimum=min(minimum,float(np.linalg.norm(points-p-fraction[:,None]*delta,axis=1).min()))
    return minimum


def candidates(rest, state, config, *, yaws=(0., 90., -90.), tilts=(0., 90., -90.),
               clearances=(.04, .08),
               approaches=("vertical", "mouth_normal"), screen_geometry_first=False,
               align_yaw=True, plan_mode="full"):
    if plan_mode not in ("full","insert_only"):
        raise ValueError("Unknown insertion planning mode")
    bag, bear = np.asarray(state["bag"]), np.asarray(state["bear"])
    for name,points in (("bag",bag),("bear",bear)):
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
                or points.shape != np.asarray(rest[name]).shape or not np.isfinite(points).all()
                or not np.isfinite(rest[name]).all()):
            raise ValueError(f"Observed and rest {name} must have matching finite (n,3) coordinates")
    for side in ("left","right"):
        q=np.asarray(state["q_"+side]).reshape(-1)
        tcp=np.asarray(state["tcp_"+side])
        if len(q)!=9 or not np.isfinite(q).all() or tcp.shape!=(3,) or not np.isfinite(tcp).all():
            raise ValueError(f"Invalid observed {side} robot state")
    if not np.isfinite(float(state["time"])):
        raise ValueError("Observed time must be finite")
    for key,width,count in (("bag_faces",3,len(bag)),("bear_tets",4,len(bear))):
        cells=np.asarray(rest[key])
        if (cells.ndim!=2 or cells.shape[1]!=width or not len(cells)
                or cells.dtype.kind not in "iu" or cells.min()<0 or cells.max()>=count
                or np.any(np.diff(np.sort(cells,axis=1),axis=1)==0)):
            raise ValueError(f"Invalid integer simplex topology: {key}")
    _, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    if (len(rim) < 4 or not topology["body_boundary_only_at_top"]
            or not topology["two_panels_directly_sewn_along_u"]):
        raise ValueError("Could not identify a valid sewn-bag mouth")
    rp = bag[rim]
    mouth_center = rp.mean(0)
    centered = rp-mouth_center
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1]<1e-5:
        raise ValueError("Observed mouth has no two-dimensional opening")
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
    # A sphere about the actual grasp encloses every orientation during the
    # wrist flip, including intermediate rotations. It is deliberately more
    # conservative than checking only the two endpoint teddy poses.
    held_radius = float(np.linalg.norm(relative_bear, axis=1).max())
    tets = np.asarray(rest["bear_tets"],int)
    if tets.ndim!=2 or tets.shape[1]!=4 or not len(tets) or tets.min()<0 or tets.max()>=len(bear):
        raise ValueError("Invalid teddy tetrahedron topology")
    rest_volumes=signed_volumes(np.asarray(rest["bear"]),tets)
    if np.any(np.abs(rest_volumes)<1e-15):
        raise ValueError("Rest teddy has degenerate tetrahedra")
    weights = np.abs(rest_volumes)
    volume_ratio = signed_volumes(bear,tets)/rest_volumes
    center_of_mass = np.average(bear[tets].mean(1),axis=0,weights=weights)
    offset = center_of_mass-measured_tcp
    fingers = {s:float(np.clip(np.asarray(state["q_"+s]).reshape(-1)[7:9].mean(),0.,.04)) for s in qs}
    warning = []
    if left_distance > .03:
        warning.append("Left gripper is farther than 30 mm from every bag vertex; this state does not demonstrate a held bag.")
    if volume_ratio.min() < .05:
        warning.append("Current teddy has a tetrahedron volume ratio below 0.05; this state is not an admissible insertion start.")
    tcp_errors={side:float(np.linalg.norm(planner.fk(side,qs[side])[0]-state["tcp_"+side]))
                for side in ("left","right")}
    if max(tcp_errors.values()) > .002:
        warning.append("Recorded TCP differs from the measured-joint rigid model by over 2 mm; refresh the robot state before planning.")
    raw_fingers={side:np.asarray(state["q_"+side])[7:9].tolist() for side in ("left","right")}
    if any(min(q)<-.0005 or max(q)>.0405 for q in raw_fingers.values()):
        warning.append("Observed finger position is outside the physical travel range by over 0.5 mm.")
    limits = config.get("task", {}).get("insertion_limits", {})
    compression_limit = float(limits.get("max_planar_compression",.08))
    outside_limit = float(limits.get("max_outside_projection_m",.008))
    if not 0 <= compression_limit <= .12 or not 0 <= outside_limit <= .012:
        raise ValueError("Insertion compression allowances exceed the bounded offline screening range")
    _,_,bear_xy_basis = np.linalg.svd(bear[:,:2]-bear[:,:2].mean(0),full_matrices=False)
    aligned_yaw = (np.degrees(np.arctan2(basis[0,1],basis[0,0])
                    -np.arctan2(bear_xy_basis[0,1],bear_xy_basis[0,0]))+90)%180-90
    yaws = list(yaws)
    if align_yaw and all(abs(aligned_yaw-yaw)>.5 for yaw in yaws):
        yaws.append(float(aligned_yaw))
    report = {
        "kind":"offline_geometry_candidates_from_observed_state",
        "robot_planning_scope":"Only candidates passing geometric screens" if screen_geometry_first else "All requested geometric candidates",
        "robot_planning_mode":plan_mode,
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
        "orientation_sampling":{"world_z_yaw_deg":[float(v) for v in yaws],
            "world_x_tilt_deg":[float(v) for v in tilts],
            "composition":"World-X tilt from DOWN, then world-Z yaw; rotations map hand-local vectors to world"},
        "fk_vs_recorded_right_tcp_error_m":float(np.linalg.norm(current_tcp-measured_tcp)),
        "fk_vs_recorded_tcp_error_m":tcp_errors,
        "observed_finger_q_m":raw_fingers,
        "rigid_preflight_finger_q_m":fingers,
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
        rim_clearance=minimum_nonadjacent_edge_distance(polygon)
        projected_area=abs(float(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)
                                    -polygon[:,1]*np.roll(polygon[:,0],-1))))/2
        mode_warnings = warning.copy()
        if projected_crossings:
            mode_warnings.append("Mouth contour crosses itself in this approach projection; no footprint-fit claim is made.")
        if rim_clearance < max(.0005,2*float(config["bag"].get("thickness",.0003))):
            mode_warnings.append("Nonadjacent mouth edges nearly touch in this projection; no open passage is established.")
        if projected_area<.0001:
            mode_warnings.append("Projected mouth area is under 1 square cm; no usable opening is established.")
        if approach == "vertical" and normal[2] < .45:
            mode_warnings.append("Mouth is too steep for a simple downward insertion.")
        if np.abs(centered@normal).max() > .025:
            mode_warnings.append("Mouth deviates from its fitted plane by over 25 mm; one fitted normal does not describe the folds.")
        for yaw, tilt in ((yaw,tilt) for yaw in yaws for tilt in tilts):
            target_rotation = (Rotation.from_euler("z",yaw,degrees=True)
                * Rotation.from_euler("x",tilt,degrees=True) * DOWN)
            delta = target_rotation*current_rotation.inv()
            predicted_relative = delta.apply(relative_bear)
            rotated_offset = delta.apply(offset)
            centered_prediction = predicted_relative-rotated_offset
            projected = centered_prediction@projection.T
            compression_centers=np.zeros_like(projected)
            if plan_mode=="insert_only":
                # Fixed-orientation translation is a convex combination of
                # endpoint projections. Enclose the full sweep to avoid
                # approving a centered endpoint reached through the rim.
                current_projected=(predicted_relative+measured_tcp-mouth_center)@projection.T
                current_center=(measured_tcp+rotated_offset-mouth_center)@projection.T
                projected=np.vstack((projected,current_projected))
                compression_centers=np.vstack((compression_centers,
                    np.broadcast_to(current_center,current_projected.shape)))
            inside, edge_distance = projected_inside(projected,polygon)
            hull_fits,hull_margin,maximum_outside=footprint_containment(projected,polygon)
            fit = bool(projected_crossings == 0 and hull_fits)
            needed_compression = None
            if projected_crossings == 0 and maximum_outside <= outside_limit:
                for amount in np.linspace(0.,compression_limit,9):
                    compressed=(projected-compression_centers)*(1-amount)+compression_centers
                    compressed_fits,_,_=footprint_containment(compressed,polygon)
                    if compressed_fits:
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
                staging_z = max(float(current_tcp[2]),float(above[2]),
                                float(config["table"]["height"]+.03+held_radius))
                staging = np.r_[current_tcp[:2],staging_z]
                entry = {"approach":approach,"outward_approach_axis":direction.tolist(),
                    "warnings":mode_warnings,"wrist_world_z_yaw_deg":float(yaw),
                    "mouth_projected_area_m2":projected_area,
                    "mouth_nonadjacent_edge_clearance_m":rim_clearance,
                    "wrist_world_x_tilt_deg":float(tilt),
                    "bear_clearance_above_rim_m":float(clearance),
                    "rotation_xyzw":target_rotation.as_quat().tolist(),
                    "predicted_rotated_bear_dimensions_xyz_m":np.ptp(predicted_relative,axis=0).tolist(),
                    "predicted_projected_vertices_inside_fraction":float(inside.mean()),
                    "predicted_footprint_fits_with_2mm_margin":fit,
                    "predicted_convex_footprint_boundary_separation_m":hull_margin,
                    "predicted_maximum_outside_rim_distance_m":maximum_outside,
                    "required_predicted_planar_compression":needed_compression,
                    "bounded_compression_screen_passes":needed_compression is not None,
                    "rotation_staging_tcp_m":staging.tolist(),
                    "rotation_enclosing_sphere_radius_m":held_radius,
                    "rotation_sphere_table_clearance_m":float(staging_z-held_radius-config["table"]["height"]),
                    "above_tcp_m":above.tolist(),"insert_tcp_m":insert.tolist(),
                    "robot_route_feasible":False,"robot_planning_attempted":False,"stages":[]}
                current = {s:q.copy() for s,q in qs.items()}
                try:
                    if screen_geometry_first and (mode_warnings or needed_compression is None):
                        raise ValueError("Observed mouth or bounded teddy compression screen rejects this candidate before robot planning")
                    if above[2]+predicted_relative[:,2].min() < config["table"]["height"]+.01:
                        raise ValueError("Predicted teddy would intersect the tabletop at the outside approach point")
                    if insert[2]+predicted_relative[:,2].min() < config["table"]["height"]+.03-1e-10:
                        raise ValueError("Predicted teddy would have less than 30 mm tabletop clearance at the insertion endpoint")
                    if plan_mode=="insert_only":
                        outside_distance=float(((bear-mouth_center)@direction).min()
                            -(centered@direction).max())
                        entry["observed_bear_clearance_outside_mouth_m"]=outside_distance
                        if outside_distance < .005:
                            raise ValueError("Observed teddy is not fully outside the current mouth plane with 5 mm clearance")
                    entry["robot_planning_attempted"] = True
                    stages=(("height_at_current_xy",staging,current_rotation),
                            ("rotate_held_teddy",staging,target_rotation),
                            ("translate_outside_mouth",above,target_rotation),
                            ("insert_candidate",insert,target_rotation))
                    if plan_mode=="insert_only": stages=stages[-1:]
                    for name, target, rotation in stages:
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
    Candidates include world-X tilts of 0 and +/-90 degrees (then a world-Z
    yaw), so the observed long teddy axis can become vertical before entry.
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
        "physical_success":False,"requires_physical_validation":True,
        "requires_reobservation_before_insertion":True,
        "requires_runtime_grasp_and_deformation_gates":True,
        "requires_contact_deformation_trial":viable[0]["required_predicted_planar_compression"]>0,
        "reobserve_before_every_stage":True,
        "observe_before_stage":"insert_candidate",
        "compression_is_a_prediction_not_applied_deformation":True}


def reobserve_insertion(rest, state, config, selected_route, *, output=None):
    """Recheck insertion from fresh observed state after the preparation moves.

    Keep the selected wrist orientation and approach, but recompute the mouth,
    payload offset, outside point and insertion point. Only the direct motion
    from the observed robot state to the new insertion point is preflighted.
    The actual payload remains a contact-driven soft body throughout.
    """
    route=selected_route.get("route",selected_route)
    requested_rotation=Rotation.from_quat(route["rotation_xyzw"])
    planner=Planner(config)
    qs={s:np.asarray(state["q_"+s],float).reshape(-1)[:7] for s in ("left","right")}
    _,actual_rotation=planner.fk("right",qs["right"])
    orientation_error=float((requested_rotation*actual_rotation.inv()).magnitude())
    if orientation_error > .015:
        report={"kind":"fresh_insertion_rejected","warnings":[
            "The preparation wrist orientation has not been reached within 0.015 rad"],
            "observed_orientation_error_rad":orientation_error,"geometry_screen_candidates":0}
        if output is not None: Path(output).write_text(json.dumps(report,indent=2)+"\n")
        raise NoInsertionRoute(report)
    report=candidates(rest,state,config,
        yaws=(route["wrist_world_z_yaw_deg"],),
        tilts=(route.get("wrist_world_x_tilt_deg",0.),),
        clearances=(route["bear_clearance_above_rim_m"],),
        approaches=(route["approach"],),align_yaw=False,
        screen_geometry_first=True,plan_mode="insert_only")
    report["kind"]="fresh_observed_state_insertion_geometry"
    report["observed_orientation_error_rad"]=orientation_error
    viable=[c for c in report["candidates"] if c["geometry_screen_passes"]]
    if output is not None: Path(output).write_text(json.dumps(report,indent=2)+"\n")
    if not viable: raise NoInsertionRoute(report)
    chosen=viable[0]
    return {"insert_stage":chosen["stages"][0],"outside_tcp_m":chosen["above_tcp_m"],
        "route":chosen,"report":report,"physical_success":False,
        "requires_physical_validation":True,
        "requires_runtime_grasp_and_deformation_gates":True,
        "requires_contact_deformation_trial":chosen["required_predicted_planar_compression"]>0,
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
