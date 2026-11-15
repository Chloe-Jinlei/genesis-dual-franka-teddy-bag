"""Reviewable, offline vertical *guarded contact trial*; never runs physics.

This is deliberately separate from insertion_candidates' rigid footprint fit.
It uses actual three-dimensional rim heights and horizontal tetrahedral sections,
allows bounded rim contact, and never scales a teddy or moves a simulated node.
Passing means a candidate is worth a guarded physical trial, not that the teddy
fits the bag volume or that friction/contact/deformation will remain admissible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from contact_slice_geometry import (convex_hull_metrics, horizontal_section_hulls,
    simple_polygon_triangles, convex_polygon_overlap_area)
from insertion_candidates import (projected_inside,
    minimum_nonadjacent_edge_distance, polygon_centroid)
from motion import Planner, minimum_jerk
from contact_robot_geometry import check_translation_sweep
from physical_monitor import PhysicalMonitor
from validate import mouth_structure, proper_segment_crossings, signed_volumes

HERE = Path(__file__).resolve().parent


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:_serial(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_serial(v) for v in value]
    return value


def _emit(result, output):
    result = _serial(result)
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    return result


def _validate_recenter_record(selected):
    """Reject damaged serialized budget records before any NaN comparisons."""
    if type(selected.get("attempt_index")) is not int or selected["attempt_index"] not in (1,2) \
            or selected.get("max_attempts")!=2:
        raise ValueError("Invalid prior correction counter/budget metadata")
    values=np.asarray([selected["cumulative_xy_m_before"],selected["cumulative_xy_m_after"],
                       selected["correction_distance_m"]],float)
    if values.shape!=(3,) or not np.isfinite(values).all() or np.any(values<0.) \
            or values[0]>.05 or values[1]>.05 or values[2]>.03:
        raise ValueError("Prior correction budget values must be finite and within their bounds")
    origin=np.asarray(selected["observed_tcp_xy_m"],float)
    delta=np.asarray(selected["correction_xy_m"],float)
    quaternion=np.asarray(selected["original_rotation_xyzw"],float)
    if origin.shape!=(2,) or delta.shape!=(2,) or quaternion.shape!=(4,) \
            or not all(np.isfinite(v).all() for v in (origin,delta,quaternion)) \
            or abs(np.linalg.norm(quaternion)-1.)>1e-8:
        raise ValueError("Prior correction reference position/direction/orientation is invalid")
    if abs(values[1]-values[0]-values[2])>1e-9 or abs(np.linalg.norm(delta)-values[2])>1e-9:
        raise ValueError("Prior correction budget is inconsistent with its actual XY target")


def _policy(config):
    settings = config.get("task", {}).get("contact_trial", {})
    bottom = float(settings.get("bottom_clearance_m", .03))
    if not np.isfinite(bottom) or not .01 <= bottom <= .05:
        raise ValueError("contact_trial.bottom_clearance_m must be in [0.01,0.05]")
    lateral=float(settings.get("max_lateral_excursion_fraction_of_mouth_width",.25))
    if not np.isfinite(lateral) or not 0.<=lateral<=.80:
        raise ValueError("Declared lateral trial policy must be finite and in [0,0.80] of mouth width")
    overlap=float(settings.get("minimum_section_envelope_overlap_fraction",.50))
    if not np.isfinite(overlap) or not .40<=overlap<=1.:
        raise ValueError("Declared section-overlap trial policy must be finite and in [0.40,1]")
    initial_radius=float(settings.get("maximum_initial_com_offset_m",.03))
    if not np.isfinite(initial_radius) or not .03<=initial_radius<=.06:
        raise ValueError("Declared initial COM search radius must be finite and in [0.03,0.06] m")
    strategy=settings.get("transport_xy_strategy","direct")
    if strategy not in ("direct","y_then_x"):
        raise ValueError("transport_xy_strategy must be direct or y_then_x")
    soft_gap=float(settings.get("left_robot_soft_sweep_clearance_m",.005))
    if not np.isfinite(soft_gap) or not .005<=soft_gap<=.03:
        raise ValueError("Declared teddy/left-robot sweep margin must be finite and in [0.005,0.03] m")
    return dict(bottom_clearance_m=bottom, endpoint_below_lowest_rim_m=.002,
        start_above_highest_rim_m=.04, planned_outside_clearance_m=.045,
        com_boundary_clearance_m=.002, lateral_excursion_fraction_of_mouth_width=lateral,
        minimum_section_envelope_overlap_fraction=overlap,
        maximum_initial_com_offset_m=initial_radius,
        transport_xy_strategy=strategy,
        rim_perimeter_clearance_m=.002, rigid_clearance_m=.008,
        left_robot_soft_sweep_clearance_m=soft_gap,
        minimum_insert_duration_s=6., minimum_tetra_volume_ratio=.05,
        maximum_cloth_edge_stretch_ratio=1.15, maximum_table_penetration_m=.002,
        maximum_left_designated_handle_distance_m=.018,
        maximum_right_teddy_distance_m=.045, maximum_tcp_fk_error_m=.002,
        force_and_torque_runtime_limits={"maximum_contact_force_N":8.,
            "maximum_wrist_contact_torque_Nm":.15},
        note="Lateral/overlap/perimeter are declared trial policies, not physical laws. "
             "Explicit lateral/overlap values permit a bounded contact experiment, not a material-feasibility claim. "
             "The initial COM search radius does not change the 30 mm individual or 50 mm cumulative correction budget. "
             "The 5 mm minimum teddy/left-robot clearance is an engineering preflight allowance, not a physical law. "
             "Bottom clearance and lateral trial fraction are configurable; no compression is prescribed.")


def _state_context(rest, state, config, policy):
    for name in ("bag", "bear"):
        values = np.asarray(state[name], float)
        reference = np.asarray(rest[name], float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape != reference.shape \
                or len(values) < 4 or not np.isfinite(values).all() or not np.isfinite(reference).all():
            raise ValueError(f"Invalid matching finite {name} coordinates")
    for name,width,n in (("bag_faces",3,len(rest["bag"])),("bear_tets",4,len(rest["bear"]))):
        cells=np.asarray(rest[name])
        if cells.ndim!=2 or cells.shape[1]!=width or not len(cells) or cells.dtype.kind not in "iu" \
                or cells.min()<0 or cells.max()>=n or np.any(np.diff(np.sort(cells,axis=1),axis=1)==0):
            raise ValueError(f"Invalid topology {name}")
    observed_time=float(state["time"])
    if not np.isfinite(observed_time): raise ValueError("Observed time is nonfinite")
    planner=Planner(config)
    qs, fingers, tcp_errors = {}, {}, {}
    for side in ("left", "right"):
        q=np.asarray(state["q_"+side],float).reshape(-1)
        tcp=np.asarray(state["tcp_"+side],float)
        if q.shape!=(9,) or tcp.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(tcp).all():
            raise ValueError(f"Invalid {side} robot measurement")
        if np.any(q[7:]<-.0005) or np.any(q[7:]>.0405):
            raise ValueError(f"{side} finger outside physical travel tolerance")
        bounds=planner.limits[side]
        if np.any(q[:7]<bounds[:,0]) or np.any(q[:7]>bounds[:,1]):
            raise ValueError(f"{side} measured arm joint outside physical limits")
        qs[side]=q[:7].copy()
        # Planner's paired-finger geometry API uses a single half opening.
        # Record the approximation; the real native contact guard remains mandatory.
        fingers[side]=float(np.clip(q[7:].mean(),0.,.04))
        tcp_errors[side]=float(np.linalg.norm(planner.fk(side,qs[side])[0]-tcp))
    if max(tcp_errors.values()) > policy["maximum_tcp_fk_error_m"]:
        raise ValueError("Measured joints and TCP disagree by over 2 mm")
    _,rotation=planner.fk("right",qs["right"])
    bag,bear=np.asarray(state["bag"]),np.asarray(state["bear"])
    monitor=PhysicalMonitor(rest,config)
    measurement=monitor.measure(observed_time,bag,bear)
    physical=dict(zip(monitor.columns,measurement.tolist()))
    reasons=[]
    if measurement[-1]!=1 or not np.isfinite(measurement).all(): reasons.append("nonfinite soft state")
    if measurement[1]<policy["minimum_tetra_volume_ratio"]: reasons.append("tetrahedron volume ratio below 0.05")
    if max(measurement[2:4])>policy["maximum_table_penetration_m"]: reasons.append("table penetration exceeds 2 mm")
    if measurement[4]>policy["maximum_cloth_edge_stretch_ratio"]: reasons.append("cloth edge stretch exceeds 15 percent")
    source_path=Path(config["bag"]["npz_file"])
    if not source_path.is_absolute(): source_path=HERE/"assets"/source_path
    group=config.get("task",{}).get("handle_grasps",{}).get("left",{}).get("group","grasp_front")
    with np.load(source_path,allow_pickle=False) as source:
        material=source["points"][source[group]]+np.asarray(config["bag"]["position"])
    errors,ids=cKDTree(np.asarray(rest["bag"])).query(material)
    if errors.max()>1e-6: raise ValueError("Cannot map the designated left handle material")
    left_distance=float(np.linalg.norm(bag[np.unique(ids)]-state["tcp_left"],axis=1).min())
    right_distance=float(np.linalg.norm(bear-state["tcp_right"],axis=1).min())
    if left_distance>policy["maximum_left_designated_handle_distance_m"]: reasons.append("designated held handle farther than 18 mm")
    if right_distance>policy["maximum_right_teddy_distance_m"]: reasons.append("right teddy contact farther than 45 mm")
    clearances=planner.clearance(qs,fingers)
    if min(v["distance"] for v in clearances.values())<policy["rigid_clearance_m"]:
        reasons.append("current robot rigid clearance below 8 mm")
    physical.update(left_designated_material_distance_m=left_distance,
        right_nearest_teddy_distance_m=right_distance,tcp_fk_errors_m=tcp_errors,
        actual_rigid_clearances=clearances,
        finger_pair_half_opening_approximation_error_m={s:float(np.max(np.abs(
            np.asarray(state["q_"+s])[7:]-fingers[s]))) for s in fingers},
        contact_force_and_torque_not_in_saved_state=True,
        physical_state_rejections=reasons)
    tets=np.asarray(rest["bear_tets"],int)
    weights=np.abs(signed_volumes(np.asarray(rest["bear"]),tets))
    com=np.average(bear[tets].mean(1),axis=0,weights=weights)
    return dict(planner=planner,qs=qs,fingers=fingers,rotation=rotation,com=com,
        time=observed_time,physical=physical,reasons=reasons)


def _mouth(rest,state,config):
    _,rim,topology=mouth_structure(rest["bag"],rest["bag_faces"],config)
    if len(rim)<4 or not topology["body_boundary_only_at_top"] \
            or not topology["two_panels_directly_sewn_along_u"]:
        raise ValueError("No verified two-panel U-seam body rim")
    rp=np.asarray(state["bag"])[rim]
    reference=np.asarray(rest["bag"])[rim]
    polygon=rp[:,:2].copy()
    crossings=proper_segment_crossings(polygon)
    near=minimum_nonadjacent_edge_distance(polygon)
    if crossings or near<max(.0005,2*float(config["bag"].get("thickness",.0003))):
        raise ValueError("Vertical rim projection is self-crossed, touching or closed")
    triangles=simple_polygon_triangles(polygon)
    a,b=triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]
    area=float(np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0]).sum()/2)
    if area<.0001: raise ValueError("Actual projected opening below 1 square cm")
    hull=convex_hull_metrics(polygon)["hull_xy"]
    edge=np.roll(hull,-1,axis=0)-hull
    normals=np.column_stack((-edge[:,1],edge[:,0]))/np.linalg.norm(edge,axis=1)[:,None]
    minimum_width=float(np.ptp(hull@normals.T,axis=0).min())
    rest_length=float(np.linalg.norm(reference-np.roll(reference,-1,axis=0),axis=1).sum())
    current_length=float(np.linalg.norm(rp-np.roll(rp,-1,axis=0),axis=1).sum())
    center=rp.mean(0)
    _,_,basis=np.linalg.svd(rp-center,full_matrices=False)
    return polygon,triangles,dict(topology=topology,ordered_rim_indices=rim.tolist(),
        ordered_rim_world_m=rp.tolist(),projected_area_m2=area,
        projected_crossings=crossings,nonadjacent_edge_clearance_m=near,
        projected_minimum_caliper_width_m=minimum_width,
        rim_z_min_m=float(rp[:,2].min()),rim_z_max_m=float(rp[:,2].max()),
        fitted_plane_max_nonplanarity_m=float(np.abs((rp-center)@basis[-1]).max()),
        nonplanarity_is_diagnostic_not_rejection=True,
        reference_rim_arc_length_m=rest_length,current_rim_arc_length_m=current_length,
        available_nonstretch_rim_perimeter_m=min(rest_length,current_length))


def _section_envelopes(bear,tets):
    sections=horizontal_section_hulls(bear,tets)
    # Every edge intersection within an event interval is affine in Z; the
    # endpoint union hull encloses all intermediate sections of that thin slab.
    envelopes=[]
    for a,b in zip(sections,sections[1:]):
        metrics=convex_hull_metrics(np.vstack((a["hull_xy"],b["hull_xy"])))
        envelopes.append(dict(z_min=a["z"],z_max=b["z"],**metrics))
    return sections,envelopes


def _evaluate_offset(envelopes,delta,polygon,triangles,mouth,policy):
    rows=[]
    outside_limit=policy["lateral_excursion_fraction_of_mouth_width"]*mouth["projected_minimum_caliper_width_m"]
    available=mouth["available_nonstretch_rim_perimeter_m"]
    clearance=policy["rim_perimeter_clearance_m"]
    padding=clearance*np.asarray([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    # Cover interiors as well as edges. Boundary-only distance sampling misses
    # broad concave pockets reached through a narrow neck in a simple polygon.
    # A square cell's centre is at most radius from every point in that cell.
    # Retain every centre in all hull edge halfspaces expanded by radius; this
    # includes all cells touching the hull (and harmless additional cells).
    # Distance to the closed polygon interior is 1-Lipschitz, giving max+radius
    # as a conservative bound over the COMPLETE slab envelope, including tips.
    spacing=.004
    vertices=np.vstack([e["hull_xy"] for e in envelopes])+delta
    grids={}
    def covered_distance(hull,normals,step):
        radius=step/np.sqrt(2.)
        if step not in grids:
            lo,hi=vertices.min(0)-step,vertices.max(0)+step
            count=np.maximum(1,np.ceil((hi-lo)/step).astype(int))
            gx,gy=np.meshgrid(lo[0]+(np.arange(count[0])+.5)*step,
                              lo[1]+(np.arange(count[1])+.5)*step)
            grid=np.column_stack((gx.ravel(),gy.ravel()))
            grid_inside,grid_dist=projected_inside(grid,polygon)
            grid_dist[grid_inside]=0.
            grids[step]=(grid,grid_dist)
        grid,grid_dist=grids[step]
        bbox=np.all((grid>=hull.min(0)-radius)&(grid<=hull.max(0)+radius),axis=1)
        subgrid=grid[bbox]
        keep=np.all(subgrid@normals.T-np.sum(hull*normals,axis=1)>=-radius-1e-12,axis=1)
        if not keep.any(): raise ValueError("Interior coverage grid failed to cover a section")
        return float(grid_dist[bbox][keep].max()+radius+1e-8)
    for env in envelopes:
        hull=env["hull_xy"]+delta
        edges=np.roll(hull,-1,axis=0)-hull
        normals=np.column_stack((-edges[:,1],edges[:,0]))/np.linalg.norm(edges,axis=1)[:,None]
        outside=covered_distance(hull,normals,spacing)
        used_spacing=spacing
        if outside>outside_limit:
            # Refine numerical coverage, never the actual trial threshold.
            outside=covered_distance(hull,normals,.001)
            used_spacing=.001
        # Pad only the object, not the existing mouth. Adding a global 2*pi*r
        # to an already-flat convex rim would reject even a tiny centred teddy.
        # The square buffer conservatively contains a 2 mm circular buffer.
        buffered=(hull[:,None,:]+padding[None,:,:]).reshape(-1,2)
        union=convex_hull_metrics(np.vstack((polygon,buffered)))
        overlap=convex_polygon_overlap_area(hull,pretriangles=triangles)
        ratio=min(1.,max(0.,float(overlap/env["area"])))
        rows.append(dict(z_interval_m=[env["z_min"],env["z_max"]],
            convex_section_slab_area_m2=env["area"],convex_section_slab_perimeter_m=env["perimeter"],
            opening_overlap_area_m2=overlap,opening_overlap_fraction=ratio,
            maximum_outside_excursion_bound_m=outside,
            interior_distance_grid_spacing_m=used_spacing,
            entire_slab_interior_distance_bound=True,
            mouth_and_section_envelope_perimeter_m=union["perimeter"]))
    maximum_outside=max(r["maximum_outside_excursion_bound_m"] for r in rows)
    minimum_overlap=min(r["opening_overlap_fraction"] for r in rows)
    perimeter=max(r["mouth_and_section_envelope_perimeter_m"] for r in rows)
    reasons=[]
    if maximum_outside>outside_limit: reasons.append("section lateral excursion exceeds declared fraction of actual opening minimum width")
    if minimum_overlap<policy["minimum_section_envelope_overlap_fraction"]:
        reasons.append(f"section slab overlap with actual opening below {100*policy['minimum_section_envelope_overlap_fraction']:g} percent")
    if perimeter>available: reasons.append("mouth-plus-section perimeter with clearance exceeds available nonstretch rim length")
    return dict(translation_xy_m=delta.tolist(),maximum_outside_excursion_bound_m=maximum_outside,
        allowed_outside_excursion_m=outside_limit,minimum_section_slab_overlap_fraction=minimum_overlap,
        maximum_required_enclosing_rim_perimeter_m=perimeter,
        section_perimeter_square_buffer_halfwidth_m=clearance,available_rim_perimeter_m=available,
        interior_distance_initial_grid_spacing_m=spacing,
        interior_distance_refined_grid_spacing_m=.001,
        interior_distance_refined_slab_count=sum(r["interior_distance_grid_spacing_m"]<spacing for r in rows),
        perimeter_margin_m=available-perimeter,section_guard_rejections=reasons,sections=rows)


def _plan(rest,state,config,selected=None,*,correction_offsets=None):
    policy=_policy(config)
    report=dict(kind="guarded_contact_trial",physical_success=False,
        runtime_execution_performed=False,policy=policy,geometry_guarantees=[
            "Uses the ordered actual 3D body rim, not a fitted mouth plane.",
            "The observed teddy is translated without imposed compression or rotation.",
            "Thin-slab convex envelopes enclose every horizontal section between tetra vertex-height events.",
            "Whole observed teddy begins above highest rim by at least 40 mm and ends below lowest rim by 2 mm, with configured table clearance.",
            "Every preparation and insertion segment's complete frozen-shape teddy translation must clear every left robot collision volume by the declared engineering allowance.",
            "Robot path retains analytic continuous joint-limit and sampled 8 mm rigid-clearance checks."],
        not_guaranteed=["Deformable 3D bag interior containment or passage under contact.",
            "Frictional grasp retention, force/torque, cloth strain or tetrahedron quality during execution.",
            "A nonplanar rim can contact several teddy heights simultaneously; independent section guards do not certify that coupled contact.",
            "Cartesian frozen-shape sweep certificates do not predict deformation or prove continuous FK tracking; sampled FK position/orientation errors are reported separately.",
            "Overlap fraction is a bounded trial heuristic for slab envelopes, not an exact occupied-section or continuous-contact theorem.",
            "Saved observations lack native contact force/torque; runtime guards and final containment audit are mandatory."],
        rejections=[])
    context=_state_context(rest,state,config,policy)
    report.update(observed_time_s=context["time"],physical_state=context["physical"])
    report["rejections"].extend(context["reasons"])
    polygon,triangles,mouth=_mouth(rest,state,config)
    report["mouth"]=mouth
    bear=np.asarray(state["bear"])
    tcp=np.asarray(state["tcp_right"],float)
    com=context["com"]
    rel=bear-tcp
    bottom=float(config["table"]["height"])+policy["bottom_clearance_m"]
    end_z=bottom-float(rel[:,2].min())
    top=end_z+float(rel[:,2].max())
    capacity=mouth["rim_z_min_m"]-policy["endpoint_below_lowest_rim_m"]-bottom
    height=float(np.ptp(bear[:,2]))
    report["vertical_capacity"]=dict(observed_bear_height_m=height,
        usable_height_below_actual_lowest_rim_m=capacity,height_margin_m=capacity-height,
        planned_bear_bottom_z_m=bottom,planned_bear_top_z_m=top,
        planned_table_clearance_m=policy["bottom_clearance_m"],
        whole_bear_end_below_lowest_rim=bool(capacity>=height),
        bottom_clearance_is_planning_allowance_not_physical_contact_law=True)
    if capacity<height: report["rejections"].append("observed teddy height exceeds space between required table clearance and lowest actual rim")
    sections,envelopes=_section_envelopes(bear,np.asarray(rest["bear_tets"],int))
    report["horizontal_sections"]=dict(vertex_height_event_count=len(sections),
        slab_envelope_count=len(envelopes),maximum_event_section_perimeter_m=max(s["perimeter"] for s in sections),
        maximum_event_section_area_m2=max(s["area"] for s in sections),
        maximum_z_event_spacing_m=max(s["z_max"]-s["z_min"] for s in envelopes))
    if selected is None:
        anchor=polygon_centroid(polygon)
        requested=(correction_offsets if correction_offsets is not None else
            config.get("task",{}).get("contact_trial",{}).get("center_offsets_xy_m",
            ((0.,0.),(-.01,0.),(.01,0.),(0.,-.01),(0.,.01),(.004,.02))))
        offsets=np.asarray(requested,float)
        radius=.03 if correction_offsets is not None else policy["maximum_initial_com_offset_m"]
        if offsets.ndim!=2 or offsets.shape[1]!=2 or not 1<=len(offsets)<=25 \
                or not np.isfinite(offsets).all() or np.any(np.linalg.norm(offsets,axis=1)>radius):
            raise ValueError(f"Contact trial needs 1–25 finite offsets no farther than {1000*radius:g} mm from its declared anchor")
        if correction_offsets is None:
            report["declared_com_offsets_from_mouth_area_centroid_m"]=offsets.tolist()
            positions=[anchor+d for d in offsets]
        else:
            report["declared_actual_tcp_xy_corrections_m"]=offsets.tolist()
            report["correction_anchor"]="current observed TCP/COM, not mouth area centroid"
            positions=[com[:2]+d for d in offsets]
    else:
        if "status" in selected and selected["status"]!="candidate":
            raise ValueError("Fresh check cannot reuse a rejected trial result")
        prior=selected.get("route",selected)
        if prior is None or selected.get("kind",prior.get("kind"))!="guarded_contact_trial":
            raise ValueError("Fresh contact check needs an accepted guarded_contact_trial")
        old_time=float(selected.get("observed_time_s",prior["observed_time_s"]))
        if not np.isfinite(old_time) or context["time"]<=old_time:
            raise ValueError("Fresh check needs finite, strictly newer actual state times")
        angle=float((context["rotation"]*Rotation.from_quat(prior["rotation_xyzw"]).inv()).magnitude())
        report["fresh_rotation_error_rad"]=angle
        if angle>.015: report["rejections"].append("hand has not retained selected full orientation")
        if selected.get("is_bounded_recenter",False):
            _validate_recenter_record(selected)
            original=Rotation.from_quat(selected["original_rotation_xyzw"])
            original_error=float((context["rotation"]*original.inv()).magnitude())
            report["fresh_original_rotation_error_rad"]=original_error
            if original_error>.015:
                report["rejections"].append("hand orientation differs from the original trial by over 0.015 rad")
            position_error=float(np.linalg.norm(tcp[:2]-np.asarray(prior["outside_tcp_m"])[:2]))
            report["correction_endpoint_xy_error_m"]=position_error
            if position_error>.002:
                report["rejections"].append("physical correction has not reached its selected XY target within 2 mm")
            actual_distance=float(np.linalg.norm(tcp[:2]-np.asarray(selected["observed_tcp_xy_m"])))
            actual_total=float(selected["cumulative_xy_m_before"])+actual_distance
            report["observed_correction_distance_m"]=actual_distance
            report["observed_cumulative_xy_m_after"]=actual_total
            if not np.isfinite(actual_distance) or not np.isfinite(actual_total) \
                    or actual_distance>.03 or actual_total>.05:
                report["rejections"].append("observed correction exceeds the 30 mm individual or 50 mm cumulative XY budget")
        positions=[com[:2].copy()]
        observed_above=float(bear[:,2].min()-mouth["rim_z_max_m"])
        report["observed_clearance_above_highest_rim_m"]=observed_above
        if observed_above<policy["start_above_highest_rim_m"]:
            report["rejections"].append("fresh teddy not wholly above highest current rim by 40 mm")
    candidates=[]
    for target in positions:
        inside,dist=projected_inside(np.array([target]),polygon)
        if not inside[0] or dist[0]<policy["com_boundary_clearance_m"]: continue
        evaluation=_evaluate_offset(envelopes,target-com[:2],polygon,triangles,mouth,policy)
        evaluation.update(center_of_mass_xy_target_m=target.tolist(),com_boundary_distance_m=float(dist[0]))
        candidates.append(evaluation)
    if not candidates: raise ValueError("No planned COM lies inside actual projected opening with 2 mm margin")
    candidates.sort(key=lambda c:(bool(c["section_guard_rejections"]),
        len(c["section_guard_rejections"]),c["maximum_outside_excursion_bound_m"],
        -c["minimum_section_slab_overlap_fraction"]))
    chosen=candidates[0]
    report["centering_candidates"]=[{k:v for k,v in c.items() if k!="sections"} for c in candidates]
    def targets_for(candidate):
        outside=tcp.copy()
        outside[:2]+=candidate["translation_xy_m"]
        outside[2]=max(float(tcp[2]),mouth["rim_z_max_m"]+policy["planned_outside_clearance_m"]-float(rel[:,2].min()))
        insert=outside.copy(); insert[2]=end_z
        if selected is not None: insert[:2]=tcp[:2]
        return outside,insert,np.r_[tcp[:2],outside[2]]
    outside,insert,stage_height=targets_for(chosen)
    stages=[];attempts=[];passed=False;withdraw_stage=None
    for candidate in candidates if not report["rejections"] else []:
        if candidate["section_guard_rejections"]: continue
        outside,insert,stage_height=targets_for(candidate)
        stages=[];failure=None;withdraw_stage=None
        soft_sweep=None;sweep_segments=[]
        qs={k:v.copy() for k,v in context["qs"].items()}
        targets=[("height_at_current_xy",stage_height,2.5),("translate_outside_mouth",outside,3.),
                 ("insert_candidate",insert,policy["minimum_insert_duration_s"])]
        if selected is None and correction_offsets is None and policy["transport_xy_strategy"]=="y_then_x":
            waypoint=np.array([stage_height[0],outside[1],outside[2]])
            targets.insert(1,("transport_y_at_current_x",waypoint,3.))
        if selected is not None: targets=targets[-1:]
        segment_start=tcp.copy()
        for name,target,duration in targets:
            try:
                soft_sweep=check_translation_sweep(bear,np.asarray(rest["bear_tets"],int),
                    np.asarray(state["q_left"],float),config,
                    translation_m=segment_start-tcp,displacement_m=target-segment_start,
                    required_gap_m=policy["left_robot_soft_sweep_clearance_m"],
                    planner=context["planner"])
                sweep_segments.append(dict(stage=name,**soft_sweep))
                if soft_sweep["status"]!="clear":
                    raise ValueError("frozen teddy continuous translation does not clear the left robot")
                plan=context["planner"].trajectory(qs,{"right":target},rotations={"right":context["rotation"]},
                    samples=45,validate_samples=101,duration=duration,
                    finger_start=context["fingers"],finger_end=context["fingers"],
                    clearance_margin=policy["rigid_clearance_m"])
                maximum_position_error=0.;maximum_angle_error=0.
                for check_time in np.linspace(0.,plan.duration,201):
                    sample=plan.sample(float(check_time))
                    actual_position,actual_rotation=context["planner"].fk("right",sample["q"]["right"])
                    progress=float(minimum_jerk(check_time/plan.duration)[0])
                    expected_position=segment_start+progress*(target-segment_start)
                    maximum_position_error=max(maximum_position_error,float(np.linalg.norm(actual_position-expected_position)))
                    maximum_angle_error=max(maximum_angle_error,float((actual_rotation*context["rotation"].inv()).magnitude()))
                if maximum_position_error>.0005 or maximum_angle_error>.002:
                    raise ValueError("sampled FK deviates from declared straight frozen-shape path beyond IK tolerances")
                qs=plan.end_q()
                stages.append(dict(name=name,right_tcp_m=target.tolist(),
                    rotation_xyzw=context["rotation"].as_quat().tolist(),duration_s=plan.duration,
                    right_q_end=qs["right"].tolist(),analytic_joint_extrema=plan.joint_extrema(),
                    rigid_clearance_sample_count=101,rigid_clearance_required_m=policy["rigid_clearance_m"],
                    straight_path_fk_review=dict(sample_count=201,
                        maximum_position_error_m=maximum_position_error,
                        maximum_orientation_error_rad=maximum_angle_error,
                        is_continuous_error_bound=False,
                        declared_start_tcp_m=segment_start.tolist(),declared_end_tcp_m=target.tolist())))
                segment_start=target.copy()
            except ValueError as exc:
                failure=f"{name}: {exc}"; break
        if failure is None:
            try:
                release_clearance=float("inf")
                opened=dict(context["fingers"]);opened["right"]=.04
                for fraction in np.linspace(0.,1.,11):
                    finger=dict(context["fingers"])
                    finger["right"]=(1-fraction)*finger["right"]+fraction*.04
                    margin=min(v["distance"] for v in context["planner"].clearance(qs,finger).values())
                    release_clearance=min(release_clearance,margin)
                    if margin<policy["rigid_clearance_m"]:
                        raise ValueError(f"opening right hand at insert pose leaves only {margin:.6f} m rigid clearance")
                withdrawal=context["planner"].trajectory(qs,{"right":outside},
                    rotations={"right":context["rotation"]},samples=45,validate_samples=101,duration=3.,
                    finger_start=opened,finger_end=opened,clearance_margin=policy["rigid_clearance_m"])
                withdraw_stage=dict(name="withdraw_after_release",right_tcp_m=outside.tolist(),
                    rotation_xyzw=context["rotation"].as_quat().tolist(),duration_s=withdrawal.duration,
                    right_q_end=withdrawal.end_q()["right"].tolist(),
                    analytic_joint_extrema=withdrawal.joint_extrema(),right_finger_half_opening_m=.04,
                    release_opening_clearance_sample_count=11,minimum_release_opening_rigid_clearance_m=release_clearance,
                    rigid_clearance_sample_count=101,rigid_clearance_required_m=policy["rigid_clearance_m"],
                    diagnostic_only_until_actual_release=True)
            except ValueError as exc:
                failure=f"withdraw_after_release: {exc}"
        attempts.append(dict(center_of_mass_xy_target_m=candidate["center_of_mass_xy_target_m"],
            rejection=failure,stages_completed=len(stages),left_robot_teddy_sweep=soft_sweep,
            left_robot_teddy_sweep_segments=sweep_segments))
        if failure is None:
            chosen=candidate;passed=True;break
    if not passed and not report["rejections"]:
        report["rejections"].extend(chosen["section_guard_rejections"] or
            ["All geometrically admissible centering candidates failed the robot route"])
    report["robot_centering_attempts"]=attempts
    report["selected_sections"]=chosen
    report["predicted_targets"]=dict(outside_tcp_m=outside.tolist(),insert_tcp_m=insert.tolist(),
        rotation_xyzw=context["rotation"].as_quat().tolist(),
        center_of_mass_offset_from_tcp_m=(com-tcp).tolist(),
        no_soft_points_or_material_parameters_modified=True,imposed_compression=0.)
    report["robot_stages"]=stages
    report["withdraw_after_release_preflight"]=withdraw_stage
    accepted=not report["rejections"]
    route=dict(kind="guarded_contact_trial",observed_time_s=context["time"],
        rotation_xyzw=context["rotation"].as_quat().tolist(),
        outside_tcp_m=outside.tolist(),insert_tcp_m=insert.tolist(),stages=stages,
        withdraw_stage=withdraw_stage,
        in_plane_strategy=("bounded_actual_tcp_xy_correction_then_fixed_xy_on_fresh_observation"
            if correction_offsets is not None else "current_pose_area_centroid_neighbourhood_then_fixed_xy_on_fresh_observation"),
        physical_success=False) if accepted else None
    result=dict(kind="guarded_contact_trial",status="candidate" if accepted else "rejected",route=route,
        report=report,observed_time_s=context["time"],physical_success=False,
        runtime_execution_performed=False,requires_physical_validation=True,
        requires_per_step_native_contact_force_torque_grasp_and_deformation_gates=True,
        requires_final_actual_3d_containment_audit=True,
        stages_to_execute=(stages[:-1] if selected is None else stages) if accepted else [],
        next_observation="before_insertion" if selected is None else "during_every_physics_step")
    if selected is not None and accepted:
        result.update(insert_stage=stages[-1],outside_tcp_m=outside.tolist())
    return result


def plan_guarded_contact_trial(rest,state,config,*,output=None):
    """Return status candidate/rejected, route, report; performs CPU planning only.

    The initial result withholds its provisional insertion stage from
    stages_to_execute. Reobserve after transporting above the bag. A rejected
    result has no executable stages. Input/mesh failures also fail closed.
    """
    try: result=_plan(rest,state,config)
    except (ValueError,KeyError,IndexError,OSError) as exc:
        result=dict(kind="guarded_contact_trial",status="rejected",route=None,
            report={"rejections":[str(exc)]},physical_success=False,
            runtime_execution_performed=False,stages_to_execute=[])
    return _emit(result,output)


def reobserve_guarded_contact_trial(rest,state,config,selected,*,output=None):
    """Fresh vertical-only insertion from actual pose/shape; no lateral correction.

    Fixed full hand rotation, current XY, fresh rim/COM/sections and six-second
    minimum insertion are mandatory. Dynamic force/torque remains runtime work.
    """
    try: result=_plan(rest,state,config,selected)
    except (ValueError,KeyError,IndexError,OSError) as exc:
        result=dict(kind="guarded_contact_trial",status="rejected",route=None,
            report={"rejections":[str(exc)]},physical_success=False,
            runtime_execution_performed=False,stages_to_execute=[])
    return _emit(result,output)


def recenter_guarded_contact_trial(rest,state,config,selected,*,attempt_index:int,
                                  cumulative_xy_m:float,output=None):
    """Return at most two preparation stages for a bounded physical correction.

    ``attempt_index`` is one-based (1 or 2). Distances are actual TCP XY changes,
    not offsets from a changing mouth centroid. The caller must execute the
    returned preparation stages with all native guards, take a strictly newer
    observation, and call reobserve_guarded_contact_trial before any insertion.
    No wrist orientation is introduced; the original trial's full orientation
    remains the reference throughout both possible corrections.
    """
    base=dict(kind="guarded_contact_trial",status="rejected",route=None,
        physical_success=False,runtime_execution_performed=False,stages_to_execute=[])
    report=dict(kind="bounded_guarded_contact_recenter",rejections=[],candidates=[],
        physical_success=False,maximum_correction_xy_m=.03,maximum_cumulative_xy_m=.05,
        max_attempts=2,attempt_index=attempt_index,
        keeps_original_full_orientation_within_rad=.015,
        correction_height_is_body_rim_clearance_not_all_handle_material_clearance=True,
        budget_measurement="planned straight XY corrections, reconciled with new observed endpoint displacements; not per-step actual arc length",
        native_contact_force_torque_grasp_and_deformation_gates_must_remain_active=True)
    try:
        if type(attempt_index) is not int or attempt_index not in (1,2):
            raise ValueError("Correction attempt_index is one-based and must be 1 or 2")
        cumulative=float(cumulative_xy_m)
        if not np.isfinite(cumulative) or not 0.<=cumulative<=.05:
            raise ValueError("Cumulative TCP XY correction distance must be finite and in [0,0.05] m")
        if selected.get("status")!="candidate" or selected.get("route") is None:
            raise ValueError("Correction needs the previous accepted trial candidate")
        old_time=float(selected["observed_time_s"])
        now=float(state["time"])
        if not np.isfinite(old_time) or not np.isfinite(now) or now<=old_time:
            raise ValueError("Correction needs a strictly newer finite actual observation")
        policy=_policy(config)
        if selected.get("report",{}).get("policy")!=policy:
            raise ValueError("Do not change contact/physical policy within a correction sequence")
        context=_state_context(rest,state,config,policy)
        if context["reasons"]: raise ValueError("; ".join(context["reasons"]))
        tcp=np.asarray(state["tcp_right"],float)
        original=np.asarray(selected.get("original_rotation_xyzw",selected["route"]["rotation_xyzw"]),float)
        angle=float((context["rotation"]*Rotation.from_quat(original).inv()).magnitude())
        report["original_rotation_error_rad"]=angle
        if angle>.015: raise ValueError("Current hand differs from original trial orientation by over 0.015 rad")
        if selected.get("is_bounded_recenter",False):
            _validate_recenter_record(selected)
            if attempt_index!=selected["attempt_index"]+1:
                raise ValueError("Correction attempt counter must continue from the previous candidate")
            if abs(cumulative-float(selected["cumulative_xy_m_after"]))>1e-9:
                raise ValueError("Correction cumulative budget cannot be reset or changed by the caller")
            if np.linalg.norm(tcp[:2]-np.asarray(selected["route"]["outside_tcp_m"])[:2])>.002:
                raise ValueError("Previous physical correction has not reached its selected XY target")
            observed_distance=float(np.linalg.norm(tcp[:2]-np.asarray(selected["observed_tcp_xy_m"])))
            if not np.isfinite(observed_distance) or observed_distance>.03:
                raise ValueError("Previous actual correction exceeded 30 mm or is nonfinite")
            cumulative=float(selected["cumulative_xy_m_before"])+observed_distance
            if not np.isfinite(cumulative) or cumulative>.05:
                raise ValueError("Observed cumulative correction distance exceeds 50 mm or is nonfinite")
        elif attempt_index!=1 or cumulative!=0.:
            raise ValueError("First correction requires attempt_index=1 and cumulative_xy_m=0")
        report.update(observed_time_s=now,observed_tcp_xy_m=tcp[:2].tolist(),
            cumulative_xy_m_before=cumulative)
        # This finite local search is tied to the tabletop layout; users may
        # supply reviewed alternatives. Stronger geometry margin is preferred
        # over the first near-threshold point in the full07 read-only audit.
        offsets=np.asarray(config.get("task",{}).get("contact_trial",{}).get(
            "recenter_offsets_xy_m",[[-.015,.02],[-.01,.02],[-.015,.01],[-.01,.01],
                                    [.015,-.02],[.01,-.02],[.015,-.01],[.01,-.01]]),float)
        if offsets.ndim!=2 or offsets.shape[1]!=2 or not 1<=len(offsets)<=8 \
                or not np.isfinite(offsets).all() or np.any(np.linalg.norm(offsets,axis=1)>.03) \
                or np.any(np.linalg.norm(offsets,axis=1)<1e-6):
            raise ValueError("Declare 1–8 nonzero finite corrections of at most 30 mm each")
        for delta in offsets:
            distance=float(np.linalg.norm(delta))
            if cumulative+distance>.05:
                report["candidates"].append(dict(correction_xy_m=delta.tolist(),
                    rejections=["would exceed 50 mm cumulative TCP XY correction budget"]))
                continue
            result=_plan(rest,state,config,correction_offsets=np.array([delta]))
            report["candidates"].append(dict(correction_xy_m=delta.tolist(),
                rejections=result["report"]["rejections"],status=result["status"]))
            if result["status"]!="candidate": continue
            report["selected_correction_xy_m"]=delta.tolist()
            result["report"]["recenter"]=report
            result.update(is_bounded_recenter=True,attempt_index=attempt_index,max_attempts=2,
                correction_xy_m=delta.tolist(),correction_distance_m=distance,
                cumulative_xy_m_before=cumulative,cumulative_xy_m_after=cumulative+distance,
                observed_tcp_xy_m=tcp[:2].tolist(),original_rotation_xyzw=original.tolist(),
                requires_strictly_new_observation_after_correction=True,
                requires_reobservation_before_insertion=True)
            # _plan's provisional insert/withdraw are preflight evidence only.
            # Its public executable stages remain height + translation only.
            return _emit(result,output)
        report["rejections"].append("No bounded fixed-orientation correction passed every geometry and robot gate")
    except (ValueError,KeyError,IndexError,OSError,TypeError) as exc:
        report["rejections"].append(str(exc))
    base["report"]=report
    return _emit(base,output)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path);parser.add_argument("--state",default="lift_teddy.npz")
    parser.add_argument("--config",type=Path);parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    config=json.loads((args.config or args.run/"config.json").read_text())
    with np.load(args.run/"rest.npz",allow_pickle=False) as r,np.load(args.run/args.state,allow_pickle=False) as s:
        result=plan_guarded_contact_trial(dict(r),dict(s),config,output=args.output)
    print(json.dumps({"status":result["status"],"rejections":result["report"]["rejections"]}))
