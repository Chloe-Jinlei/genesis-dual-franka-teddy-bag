"""Read-only, finite-triangle material/deformation baseline for completed full30.

No Genesis initialization, model import, material edit, or simulation. The
construction axes are explicitly proxies, not measured textile yarn axes.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_30"
OUTPUT = ROOT / "runtime_ipc/full30-cloth-material-baseline.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def weighted_quantiles(x, weights):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return {name: float(x[order[min(np.searchsorted(cdf, q), len(x)-1)]])
            for name, q in (("minimum", 0.), ("p50", .5), ("p95", .95), ("p99", .99), ("maximum", 1.))}


def triangle_frames(vertices, faces):
    tri = vertices[faces]
    return np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=-1)


def intrinsic_inverse(edges):
    e1 = edges[:, :, 0]
    e2 = edges[:, :, 1]
    a = np.linalg.norm(e1, axis=1)
    t1 = e1/a[:, None]
    b = np.einsum("ni,ni->n", t1, e2)
    c = np.linalg.norm(e2-b[:, None]*t1, axis=1)
    assert np.all(a > 0) and np.all(c > 0)
    D = np.zeros((len(a), 2, 2))
    D[:, 0, 0], D[:, 0, 1], D[:, 1, 1] = a, b, c
    return np.linalg.inv(D), a*c/2


def deformation(edges, inverse):
    F = edges @ inverse
    C = np.swapaxes(F, 1, 2) @ F
    eig = np.linalg.eigvalsh(C)
    assert np.isfinite(eig).all() and np.all(eig > 0)
    return np.sqrt(eig), F


def reconstruction_strip(asset, face_ids, name, checks):
    """Recover builder's strip exactly from its two-triangle quad ordering.

    The builder may reverse triangle winding but preserves vertex sets and
    face order. Prove every recovered cell agrees with both stored triangles.
    """
    faces = asset["faces"][face_ids]
    row = asset["handle_"+name+"_root_left"]
    nx = len(row)
    assert len(faces) % (2*(nx-1)) == 0
    grid = np.full((1+len(faces)//(2*(nx-1)), nx), -1, int)
    grid[0] = row
    for j in range(len(grid)-1):
        for i in range(nx-1):
            f0, f1 = faces[2*(j*(nx-1)+i):2*(j*(nx-1)+i)+2]
            a, b = grid[j, i:i+2]
            assert a in f0 and b in f0
            d, = set(map(int, f0)) - {int(a), int(b)}
            assert a in f1 and d in f1
            c, = set(map(int, f1)) - {int(a), int(d)}
            assert grid[j+1, i] in (-1, c)
            assert grid[j+1, i+1] in (-1, d)
            grid[j+1, i:i+2] = [c, d]
    assert np.array_equal(grid[-1], asset["handle_"+name+"_root_right"][::-1])
    assert np.array_equal(np.unique(grid), asset["handle_"+name])
    checks["handle_"+name+"_entire_strip_grid_reconstructed"] = True
    cross = asset["material_x"][abs(asset["material_x"]+.12) <= .0125+1e-9]+.12
    assert len(cross) == nx
    center = int(np.argmin(abs(cross)))
    assert abs(cross[center]) < 1e-12
    centerline = asset["points"][grid[:, center]]
    along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))]
    return grid, cross, along


def phase_at(t, events):
    starts = np.array([e["start"] for e in events])
    return events[max(0, int(np.searchsorted(starts, t+1e-8, side="right")-1))]["phase"]


def main():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite completed baseline: "+str(OUTPUT))
    cfg = json.loads((RUN/"config.json").read_text())
    asset_path = RUN/"source/assets"/cfg["bag"]["npz_file"]
    protected = [RUN/name for name in ("config.json", "rest.npz", "states.npz", "soft_metrics.npy", "events.json", "input_sha256.json")]
    protected += [asset_path, RUN/"source/two_panel_builder.py", ROOT/"config.json"]
    before = {str(p): sha(p) for p in protected}
    asset = dict(np.load(asset_path))
    rest = dict(np.load(RUN/"rest.npz"))
    raw = dict(np.load(RUN/"states.npz"))
    events = json.loads((RUN/"events.json").read_text())
    # Keep the last sample of each coincident time, as the final validator does.
    times = raw["time"]
    keep = np.r_[np.diff(times) > 1e-9, True]
    data = {k: v[keep] for k, v in raw.items()}
    times = data["time"]
    assert np.all(np.diff(times) > 0)
    vertices, faces = rest["bag"], rest["bag_faces"]
    source_points = asset["points"]+np.asarray(cfg["bag"]["position"])
    distances, mapping = cKDTree(vertices).query(source_points)
    checks = {
        "finite_native_rest_and_saved_positions": bool(np.isfinite(vertices).all() and np.isfinite(data["bag"]).all()),
        "source_to_rest_identity_bijection": bool(np.array_equal(mapping, np.arange(len(vertices))) and max(distances)<1e-8),
        "source_native_faces_identical": bool(np.array_equal(faces, asset["faces"])),
    }
    assert all(checks.values())
    groups = {name: asset[name+"_face_indices"] for name in ("panel_front", "panel_back")}
    body = set(map(int, asset["body_face_indices"]))
    for side in ("front", "back"):
        members = set(map(int, asset["handle_"+side]))
        groups["handle_"+side] = np.array([i for i, f in enumerate(faces) if i not in body and set(map(int, f)) <= members], int)
    all_ids = np.concatenate(list(groups.values()))
    checks["four_groups_disjoint_complete_all_faces"] = bool(np.array_equal(np.sort(all_ids), np.arange(len(faces))))
    assert checks["four_groups_disjoint_complete_all_faces"]

    coordinates = np.full((len(faces), 3, 2), np.nan)
    grid_info = {}
    for name, ids in groups.items():
        if name.startswith("panel_"):
            grid = asset[name+"_grid"]
            width, along = asset["material_x"], asset["material_height"]
            axes = ["construction_panel_width", "construction_panel_height"]
        else:
            grid, width, along = reconstruction_strip(asset, ids, name.removeprefix("handle_"), checks)
            axes = ["construction_strip_width", "construction_strip_centerline_length"]
        uv = np.full((len(vertices), 2), np.nan)
        uv[grid] = np.stack(np.meshgrid(width, along), axis=-1)
        coordinates[ids] = uv[faces[ids]]
        grid_info[name] = {"grid": grid, "axes": axes}
    assert np.isfinite(coordinates).all()
    material_edges = np.stack((coordinates[:,1]-coordinates[:,0], coordinates[:,2]-coordinates[:,0]), axis=-1)
    material_inverse = np.linalg.inv(material_edges)
    reference_edges = triangle_frames(vertices, faces)
    inverse, area = intrinsic_inverse(reference_edges)
    ref_vectors = reference_edges @ material_inverse
    ref_lengths = np.linalg.norm(ref_vectors, axis=1)
    ref_cos = np.sum(ref_vectors[:,:,0]*ref_vectors[:,:,1], axis=1)/np.prod(ref_lengths, axis=1)
    ref_angle = np.arccos(np.clip(ref_cos, -1., 1.))

    # Compact scalar-per-triangle fields for all observed frames, no interpolation.
    nf, nt = len(faces), len(times)
    metrics = {k: np.empty((nt, nf), float) for k in ("principal_min", "principal_max", "area_ratio", "width_ratio", "length_ratio", "shear_angle_change_deg")}
    for j, x in enumerate(data["bag"]):
        edges = triangle_frames(x, faces)
        singular, _ = deformation(edges, inverse)
        dirs = edges @ material_inverse
        lengths = np.linalg.norm(dirs, axis=1)
        angle = np.arccos(np.clip(np.sum(dirs[:,:,0]*dirs[:,:,1], axis=1)/np.prod(lengths, axis=1), -1., 1.))
        metrics["principal_min"][j], metrics["principal_max"][j] = singular[:,0], singular[:,1]
        metrics["area_ratio"][j] = np.prod(singular, axis=1)
        metrics["width_ratio"][j] = lengths[:,0]/ref_lengths[:,0]
        metrics["length_ratio"][j] = lengths[:,1]/ref_lengths[:,1]
        metrics["shear_angle_change_deg"][j] = np.degrees(angle-ref_angle)
    checks["all_triangle_metrics_finite"] = all(np.isfinite(x).all() for x in metrics.values())
    checks["time_zero_reference_metrics_identity"] = bool(all(np.max(abs(metrics[k][0]-1))<1e-10 for k in metrics if k != "shear_angle_change_deg") and np.max(abs(metrics["shear_angle_change_deg"][0]))<1e-9)

    # Small analytic identity checks for this diagnostic formula, not new gates.
    identity = np.array([[[1.,0.], [0.,1.], [0.,0.]]])
    ref_inv, _ = intrinsic_inverse(identity)
    test_R = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    checks["formula_rigid_rotation_invariant"] = bool(np.max(abs(deformation(test_R[None]@identity, ref_inv)[0]-1.))<1e-12)
    checks["formula_known_axial_stretch"] = bool(np.max(abs(deformation(identity@np.diag([1.02,.98]), ref_inv)[0]-[.98,1.02]))<1e-12)
    shear = .1
    expected = [(np.sqrt(shear*shear+4)-shear)/2, (np.sqrt(shear*shear+4)+shear)/2]
    checks["formula_known_simple_shear"] = bool(np.max(abs(deformation(identity@np.array([[1.,shear],[0.,1.]]), ref_inv)[0]-expected))<1e-12)

    def witness(matrix, ids, rows, absolute=False, minimum=False):
        part = matrix[np.ix_(rows, ids)]
        search = abs(part) if absolute else part
        flat = np.argmin(search) if minimum else np.argmax(search)
        ij = np.unravel_index(flat, search.shape)
        j, f = int(rows[ij[0]]), int(ids[ij[1]])
        return {"value":float(matrix[j,f]), "absolute_value":float(abs(matrix[j,f])), "time_s":float(times[j]), "saved_frame_index":j, "native_and_source_face_index":f, "vertex_indices":faces[f].tolist(), "phase_at_time":phase_at(times[j],events), "reference_triangle_area_m2":float(area[f])}

    def snapshot(ids, j):
        return {"time_s":float(times[j]), "area_weighted_triangle_quantiles":{k:weighted_quantiles(v[j,ids],area[ids]) for k,v in metrics.items()},
                "reference_area_fraction_principal_max_over":{str(level):float(np.dot(area[ids],metrics["principal_max"][j,ids]>level)/area[ids].sum()) for level in (1.01,1.02,1.05,1.10)}}

    def interval(ids, rows):
        peak = {k:witness(v,ids,rows,absolute=k=="shear_angle_change_deg") for k,v in metrics.items() if k != "principal_min"}
        peak["principal_min"] = witness(metrics["principal_min"],ids,rows,minimum=True)
        fractions = {}
        for level in (1.01,1.02,1.05,1.10):
            values = (metrics["principal_max"][np.ix_(rows,ids)]>level) @ area[ids]/area[ids].sum()
            j = int(np.argmax(values))
            fractions[str(level)] = {"fraction":float(values[j]),"time_s":float(times[rows[j]])}
        return {"sample_count":len(rows), "first_time_s":float(times[rows[0]]), "last_time_s":float(times[rows[-1]]), "extrema":peak, "maximum_reference_area_fraction_principal_max_over":fractions, "endpoint":snapshot(ids,int(rows[-1]))}

    periods = {"entire_run":np.arange(nt)}
    phase_periods = {}
    for i,event in enumerate(events):
        end = events[i+1]["start"] if i+1<len(events) else times[-1]
        rows = np.flatnonzero((times>=event["start"]-1e-8)&(times<=end+1e-8))
        assert len(rows)
        phase_periods[event["phase"]] = rows
    # Explicit opening period includes final spread endpoint before right release.
    periods["bag_opening_close_drape_spread"] = np.flatnonzero(times<=events[3]["start"]+1e-8)
    periods["final_settle"] = phase_periods["final_settle"]

    group_results = {}
    for name, ids in groups.items():
        g = {"face_count":len(ids),"source_face_index_range":[int(min(ids)),int(max(ids))],"reference_area_m2":float(area[ids].sum()),
             "construction_axis_labels":grid_info[name]["axes"],
             "initial_construction_axes_angle_deg":weighted_quantiles(np.degrees(ref_angle[ids]),area[ids]),
             "periods":{label:interval(ids,rows) for label,rows in periods.items()},
             "each_recorded_phase":{label:interval(ids,rows) for label,rows in phase_periods.items()},"final":snapshot(ids,nt-1)}
        # Track material COM with constant reference-area nodal weights. Current
        # area weights would spuriously change mass under deformation.
        nodal = np.zeros(len(vertices))
        np.add.at(nodal, faces[ids].ravel(), np.repeat(area[ids]/3,3))
        nodal /= np.sum(nodal)
        centroid = np.einsum("tnc,n->tc",data["bag"],nodal)
        relative = centroid-data["tcp_left"]
        def dynamic(rows):
            path = relative[rows]
            delta_t = np.diff(times[rows])
            velocity = np.diff(path,axis=0)/delta_t[:,None]
            speed = np.linalg.norm(velocity,axis=1)
            displacement = np.diff(data["bag"][rows],axis=0)/delta_t[:,None,None]
            # Internal deformation/swing about this group COM, no claim that
            # removing translation also removes rotation or flexure.
            centroid_v = np.diff(centroid[rows],axis=0)/delta_t[:,None]
            internal2 = np.sum((displacement-centroid_v[:,None,:])**2,axis=-1)@nodal
            return {"first_time_s":float(times[rows[0]]),"last_time_s":float(times[rows[-1]]),"relative_centroid_peak_to_peak_xyz_m":np.ptp(path,axis=0).tolist(),
                    "relative_centroid_net_displacement_m":(path[-1]-path[0]).tolist(),"relative_centroid_polyline_length_m":float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                    "relative_centroid_peak_interval_average_speed_m_s":float(max(speed)),"relative_centroid_time_weighted_rms_interval_average_speed_m_s":float(np.sqrt(np.dot(speed*speed,delta_t)/sum(delta_t))),
                    "peak_interval_material_rms_speed_after_own_centroid_translation_removed_m_s":float(np.sqrt(max(internal2))),
                    "maximum_observed_interval_s":float(max(delta_t))}
        sway_periods={**periods,"after_right_handle_release":np.flatnonzero(times>=events[4]["start"]-1e-8)}
        settle_start=events[-1]["start"]
        sway_periods["final_settle_first_half"]=np.flatnonzero((times>=settle_start-1e-8)&(times<=settle_start+2+1e-8))
        sway_periods["final_settle_second_half"]=np.flatnonzero(times>=settle_start+2-1e-8)
        g["saved_frame_motion"]={k:dynamic(v) for k,v in sway_periods.items()}
        grid=grid_info[name]["grid"]
        ref_rows=np.linalg.norm(np.diff(vertices[grid],axis=1),axis=-1).sum(axis=1)
        ref_cols=np.linalg.norm(np.diff(vertices[grid],axis=0),axis=-1).sum(axis=0)
        # Whole polyline lengths are a complementary aggregate, not peak yarn strain.
        length_ratios={}
        for label,j in (("opened",int(periods["bag_opening_close_drape_spread"][-1])),("final",nt-1)):
            points=data["bag"][j,grid]
            row=np.linalg.norm(np.diff(points,axis=1),axis=-1).sum(axis=1)/ref_rows
            col=np.linalg.norm(np.diff(points,axis=0),axis=-1).sum(axis=0)/ref_cols
            length_ratios[label]={"time_s":float(times[j]),"whole_width_polylines_min_max_ratio":[float(min(row)),float(max(row))],"whole_length_polylines_min_max_ratio":[float(min(col)),float(max(col))]}
        g["construction_grid_polyline_lengths_relative_rest"]=length_ratios
        group_results[name]=g

    soft=np.load(RUN/"soft_metrics.npy")
    # Independent edge statistic only for comparison; never label it yarn strain.
    edges=np.unique(np.sort(np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    edge_rest=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
    edge_peak=0.; edge_witness=None
    for j,x in enumerate(data["bag"]):
        ratios=np.linalg.norm(x[edges[:,1]]-x[edges[:,0]],axis=1)/edge_rest
        k=int(np.argmax(ratios))
        if ratios[k]>edge_peak:edge_peak=float(ratios[k]);edge_witness={"time_s":float(times[j]),"edge":edges[k].tolist()}
    checks["saved_edge_peak_not_above_per_step_edge_peak"]=bool(edge_peak<=max(soft[:,4])+1e-9)
    checks["inputs_unchanged"]=all(sha(p)==h for p,h in before.items())
    assert all(checks.values()),checks
    result={"kind":"independent_full30_cloth_material_baseline","created_utc":datetime.now(timezone.utc).isoformat(),"status":"measured_saved_evidence","all_diagnostic_checks_passed":all(checks.values()),"checks":checks,
            "physical_task_success_claim_from_this_diagnostic":False,"source_run":"two_panel_full_30","input_sha256":before,"script_sha256":sha(__file__),
            "sampling":{"raw_saved_frames":len(raw["time"]),"unique_saved_frames":nt,"duplicates_removed":int(len(raw["time"])-nt),"maximum_saved_interval_s":float(max(np.diff(times))),"integration_step_metric_rows":len(soft),"source_to_native_mapping_maximum_error_m":float(max(distances)),"all_native_cloth_faces":nf},
            "old_material_configuration":cfg["bag"],
            "definitions":{"reference":"Each actual native triangle at t=0; the opened rest geometry already has curved/oblique construction-grid axes.",
                "principal_stretches":"Singular values of the 3x2 triangle deformation gradient in an orthonormal rest tangent basis. Rotation/bending of an undeformed triangle alone leaves both equal to 1; principal maximum minus 1 is maximum local continuum line extension, not independently calibrated yarn strain.",
                "directional_ratios":"Current/reference norm of each triangle's derivative along panel material_x/height, or reconstructed strip width/centerline parameter. These are construction-direction proxies. Real warp/weft yarn orientations are not declared; no measured warp/weft split is claimed.",
                "shear":"Signed current-minus-rest angle between the two construction tangent vectors, in degrees. This subtracts the opened rest grid's initial obliquity. Reported extrema maximize absolute change while retaining sign; a shear angle is not a percent stretch.",
                "UV":"Texture uv=(projected rest X/width, rest Z/.5) is not a textile material coordinate chart and is deliberately not used for strain directions.",
                "area_weighting":"Quantiles and area fractions use actual native triangle rest area; no current-area mass redistribution.",
                "motion":"Group material centroid with constant reference-area lumped weights, minus measured left TCP translation, in world axes. Finite differences are interval-average saved-frame velocities. The internal RMS removes centroid translation, not rotation. These are forced motion/contact responses, not restitution or damping identification.",
                "phase_boundary":"Both adjacent phase summaries include their common endpoint; peak witness phase_at_time uses the next phase at a shared boundary."},
            "groups":group_results,
            "edge_comparison":{"per_step_maximum_mesh_edge_ratio":float(max(soft[:,4])),"per_step_peak_time_s":float(soft[np.argmax(soft[:,4]),0]),"saved_frame_maximum_mesh_edge_ratio":edge_peak,"saved_frame_peak":edge_witness,"is_yarn_strain":False},
            "limitations":["Full per-triangle metrics exist only at saved frames (at most about50ms apart), so group maxima may miss inter-frame peaks; per-step soft summaries only retain the existing edge maximum, not principal/directional strain.","There is no measured warp/weft orientation, yarn constitutive response, hysteresis, or damping calibration. Construction axes are useful reproducible proxies only.","Single flat triangular elements do not resolve within-element wrinkles. In-plane apparent compression/shear can coexist with unresolved buckling; do not interpret it as fiber compression without a calibrated model.","All metrics are relative to the pre-opened native rest, not an experimentally flat stress-free woven sheet.","Observed final settling, peak velocity, or stretch cannot by itself identify friction, elastic modulus, energy loss or coefficient of restitution.","This baseline leaves the successful motion, all materials, assets, source, config and acceptance gates untouched; it is not a new simulation or physical success claim."]}
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"path":str(OUTPUT),"sha256":sha(OUTPUT),"checks":len(checks),"all_passed":all(checks.values()),"groups":{name:{"principal_peak":g["periods"]["entire_run"]["extrema"]["principal_max"],"shear_peak":g["periods"]["entire_run"]["extrema"]["shear_angle_change_deg"],"final_principal":g["final"]["area_weighted_triangle_quantiles"]["principal_max"]} for name,g in group_results.items()}},indent=2))


if __name__ == "__main__":
    main()
