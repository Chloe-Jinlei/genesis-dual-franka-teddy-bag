"""Read-only frozen rigid-grasp yaw/Slerp comparison; no motion is planned or run."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_17"
sys.path.insert(0, str(RUN / "source"))
from motion import Planner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs = [Path(__file__).resolve(), RUN / "lower_teddy_to_rim.npz", RUN / "config.json",
              RUN / "guarded_contact_candidate.json", RUN / "source/motion.py",
              RUN / "source/assets/panda_tactile.xml"]
    before = {str(p): sha(p) for p in inputs}
    cfg = json.loads((RUN / "config.json").read_text())
    state = dict(np.load(RUN / "lower_teddy_to_rim.npz", allow_pickle=False))
    original = Rotation.from_quat(json.loads((RUN / "guarded_contact_candidate.json").read_text())[
        "route"]["rotation_xyzw"])
    planner = Planner(cfg)
    fk_tcp, actual = planner.fk("right", state["q_right"][:7])
    tcp = state["tcp_right"]
    offsets = state["bear"] - tcp
    radius = float(np.linalg.norm(offsets, axis=1).max())
    initial_difference = original * actual.inv()
    initial_error_angle = float(initial_difference.magnitude())
    endpoint_error = np.linalg.norm(initial_difference.apply(offsets) - offsets, axis=1)
    count = 4097
    fractions = np.linspace(0., 1., count)
    rows = []
    for degrees in (15., 30.):
        angle = float(np.deg2rad(degrees))
        yaw = Rotation.from_rotvec([0., 0., angle])
        target = yaw * original
        generator_pure = np.array([0., 0., angle])
        generator_slerp = (target * actual.inv()).as_rotvec()
        generator_difference = float(np.linalg.norm(generator_pure-generator_slerp))
        commanded = Slerp([0., 1.], Rotation.from_quat([actual.as_quat(), target.as_quat()]))
        maximum = -1.; at = None; equivalence = 0.
        for fraction in fractions:
            pure = Rotation.from_rotvec(fraction * generator_pure)
            relative = commanded(fraction) * actual.inv()
            exponential = Rotation.from_rotvec(fraction * generator_slerp)
            equivalence = max(equivalence, float((relative * exponential.inv()).magnitude()))
            errors = np.linalg.norm(pure.apply(offsets) - relative.apply(offsets), axis=1)
            index = int(np.argmax(errors))
            if errors[index] > maximum:
                maximum = float(errors[index]); at = dict(path_fraction=float(fraction), vertex_index=index)
        # ||exp(sA)-exp(sB)||_2 <= s ||A-B||_2 for skew-symmetric generators.
        analytic_bound = radius * generator_difference
        # D'= (A-B)exp(sA)+B(exp(sA)-exp(sB)); swap A/B for min angular speed.
        lipschitz = radius * generator_difference * (1. + min(np.linalg.norm(generator_pure),
                                                            np.linalg.norm(generator_slerp)))
        half_grid = 1. / (2. * (count-1))
        gap_bound = float(lipschitz * half_grid)
        sampled_plus_gap = maximum + gap_bound
        numerical_pad_m = 1e-12
        continuous_bound = min(analytic_bound, sampled_plus_gap) + numerical_pad_m
        final_diff = np.linalg.norm(yaw.apply(offsets) - (target * actual.inv()).apply(offsets), axis=1)
        rows.append(dict(world_z_yaw_degrees=degrees, physical_success=False,
            actual_to_target_shortest_slerp_angle_rad=float(np.linalg.norm(generator_slerp)),
            pure_world_z_generator_rad=generator_pure.tolist(),
            actual_to_target_world_generator_rad=generator_slerp.tolist(),
            generator_difference_norm_rad=generator_difference,
            endpoint_maximum_vertex_error_m=float(final_diff.max()),
            endpoint_rotation_norm_bound_m=float(2.*radius*np.sin(initial_error_angle/2.)),
            endpoint_equal_to_initial_orientation_mismatch_max_error_m=float(abs(final_diff.max()-endpoint_error.max())),
            sampled_path_count=count, sampled_path_maximum_vertex_error_m=maximum,
            maximum_sampled_error_location=at,
            maximum_sampled_slerp_exponential_equivalence_error_rad=equivalence,
            continuous_duhamel_outer_bound_m=float(analytic_bound),
            point_error_norm_lipschitz_bound_m_per_unit_fraction=float(lipschitz),
            maximum_distance_to_nearest_sample_fraction=half_grid,
            continuous_between_sample_addition_bound_m=gap_bound,
            continuous_sample_plus_lipschitz_outer_bound_m=float(sampled_plus_gap),
            floating_point_report_padding_m=numerical_pad_m,
            reported_continuous_outer_bound_m=float(continuous_bound),
            continuous_outer_bound_exceeds_sample_and_endpoint=bool(continuous_bound >= max(maximum,final_diff.max())),
            frozen_tetrahedral_points_covered_by_vertex_convexity=True,
            actual_joint_fk_path_or_dynamic_soft_grasp_certified=False))
    unchanged = before == {str(p): sha(p) for p in inputs}
    if not unchanged: raise RuntimeError("An input changed during read-only mathematics check")
    report = dict(kind="independent_frozen_yaw_vs_ideal_slerp_math", physical_success=False,
        runtime_or_config_modified=False, gpu_used=False, motion_planning_performed=False,
        input_sha256=before, all_input_hashes_unchanged=unchanged,
        actual_observed_time_s=float(state["time"]), node_count=len(offsets),
        measured_fixed_tcp_m=tcp.tolist(), fk_tcp_m=fk_tcp.tolist(),
        fk_vs_measured_tcp_distance_m=float(np.linalg.norm(fk_tcp-tcp)),
        actual_fk_rotation_xyzw=actual.as_quat().tolist(), original_command_rotation_xyzw=original.as_quat().tolist(),
        original_vs_actual_angle_rad=initial_error_angle,
        maximum_vertex_radius_about_measured_tcp_m=radius,
        angle_independent_endpoint_maximum_vertex_error_m=float(endpoint_error.max()),
        rows=rows,
        definitions=[
            "A is actual FK orientation, C is original full command orientation, p is the actual frozen vertex minus measured fixed TCP.",
            "Pure path is exp(s[a])p, a=theta*world_ez. Rigid-grasp ideal Slerp path is exp(s[b])p, b=Log(Rz(theta)*C*A^-1). Both start at the observed frozen bear.",
            "Endpoint difference is Rz(theta)*(I-C*A^-1)*p; its norm is independent of theta. This is not by itself a continuous path error bound.",
            "For skew-symmetric [a],[b], Duhamel and orthogonal operator norms give ||exp(s[a])-exp(s[b])|| <= s||a-b||.",
            "The error-norm Lipschitz constant is bounded by rmax*||a-b||*(1+min(||a||,||b||)); every s lies within half a grid interval of a sample.",
            "A tetrahedral point is a convex combination of vertices. At each s the displacement error is linear in p, so vertex norm maxima bound every point in every frozen tetrahedron."],
        limitations=[
            "Analytical inequalities hold in exact arithmetic. Reported bounds are double-precision evaluations with a declared 1e-12 m numerical pad, not directed-rounding interval certificates.",
            "Only two ideal fixed-TCP paths and the same rigidly held frozen teddy are compared; no actual cloth, deformation, slip, force, contact, release or settling is predicted.",
            "TCP FK-to-measurement mismatch is separately reported and is not silently added to this fixed-pivot orientation comparison.",
            "No continuous bound on an IK joint spline's realized FK or native robot motion follows. Existing sampled robot preflight remains separate.",
            "These small discrepancies may be subtracted from a pure frozen-yaw separation bound only within this exact ideal rigid-grasp, fixed-TCP comparison; they do not certify physical collision clearance."])
    output = RUN / "independent-yaw-path-math-review.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"endpoint_m": report["angle_independent_endpoint_maximum_vertex_error_m"],
        "fk_tcp_error_m": report["fk_vs_measured_tcp_distance_m"], "rows": rows},indent=2))


if __name__ == "__main__":
    main()
