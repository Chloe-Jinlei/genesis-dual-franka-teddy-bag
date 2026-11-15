"""Bounded CPU-only independent checks of the continuous soft/robot preflight."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from contact_robot_geometry import _convex, convex_clearance_certificate, check_vertical_sweep
from motion import Planner


def cube(center, half):
    return np.asarray(list(itertools.product((-1., 1.), repeat=3))) * half + center


def run():
    cases = []
    def pair(name, a, b, expected, gap=.005):
        result = convex_clearance_certificate(_convex(a), _convex(b), required_gap_m=gap)
        passed = (result['status'] == 'clear') == expected
        cases.append(dict(name=name, expected_clear=expected, passed=passed, result=result))
        assert passed, name
        return result

    # Known gap, offset axes, and threshold-side distinctions.
    r = pair('parallel_boxes_6mm_gap', cube([0,0,0], .01), cube([.026,0,0], .01), True)
    assert .0059998 <= r['conservative_gap_lower_bound_m'] <= .006
    pair('parallel_boxes_4mm_gap', cube([0,0,0], .01), cube([.024,0,0], .01), False)
    pair('nested_solid_no_surface_crossing', cube([0,0,0], .05), cube([0,0,0], .005), False)
    pair('intersecting_long_faces_all_vertices_far',
         cube([0,0,0], np.array([.10,.005,.005])),
         cube([0,0,0], np.array([.005,.10,.005])), False)

    # Endpoints clear but an intermediate continuous translation crosses a box.
    start = cube([0,0,.08], .01)
    end = start + [0,0,-.16]
    obstacle = cube([0,0,0], .015)
    pair('vertical_start_clear', start, obstacle, True)
    pair('vertical_end_clear', end, obstacle, True)
    pair('continuous_sweep_midsegment_blocked', np.r_[start,end], obstacle, False)

    p = HERE/'runs/two_panel_full_08'
    config = json.loads((p/'config.json').read_text())
    state = dict(np.load(p/'recenter_teddy_xy_1.npz'))
    rest = dict(np.load(p/'rest.npz'))
    fresh = json.loads((p/'guarded_contact_reobserved_01.json').read_text())
    tilt = json.loads((p/'hypothetical_left_tilt_sweep.json').read_text())
    q85 = next(x['q_left_hypothetical'] for x in tilt['rows'] if x['left_preferred_tilt_deg']==85)
    planner = Planner(config)
    saved_qpos = planner.d.qpos.copy()
    saved = {name:state[name].copy() for name in ('bear','q_left')}
    saved_tets = rest['bear_tets'].copy()
    dz = float(fresh['insert_stage']['right_tcp_m'][2]-state['tcp_right'][2])
    actual = check_vertical_sweep(state['bear'],rest['bear_tets'],state['q_left'],config,
        vertical_displacement_m=dz, required_gap_m=.005, planner=planner)
    assert actual['status'] != 'clear'
    hypothetical_q = np.r_[q85,state['q_left'][7:]]
    hypothetical = check_vertical_sweep(state['bear'],rest['bear_tets'],hypothetical_q,config,
        vertical_displacement_m=dz, required_gap_m=.005, planner=planner)
    assert hypothetical['status']=='clear'
    assert hypothetical['left_collision_geom_count']==22
    assert .0062 < hypothetical['conservative_gap_lower_bound_m'] < .0064
    assert np.array_equal(saved_qpos,planner.d.qpos)
    assert all(np.array_equal(state[name],saved[name]) for name in saved)
    assert np.array_equal(rest['bear_tets'],saved_tets)

    bad = rest['bear_tets'].copy();bad[0,0]=len(state['bear'])
    invalid = check_vertical_sweep(state['bear'],bad,state['q_left'],config,
        vertical_displacement_m=dz,required_gap_m=.005,planner=planner)
    assert invalid['status']=='indeterminate'
    output = dict(kind='independent_cpu_sweep_review',physical_success=False,
        module_sha256=hashlib.sha256((HERE/'contact_robot_geometry.py').read_bytes()).hexdigest(),
        analytic_cases=cases,actual35_status=actual['status'],actual35_rejections=actual['rejections'],
        hypothetical85_status=hypothetical['status'],
        hypothetical85_lower_gap_m=hypothetical['conservative_gap_lower_bound_m'],
        hypothetical85_all_collision_geometries=hypothetical['left_collision_geom_count'],
        invalid_topology_fails_closed=invalid['status'],
        input_arrays_and_planner_qpos_unchanged=True,
        static_review=dict(
            every_intermediate_translation_covered=True,
            face_edge_and_solid_interior_covered=True,
            acceptance_uses_original_vertex_projection_not_mujoco_distance=True,
            measured_fingers_preserved_separately=True,
            unsupported_geometry_fails_closed=True,
            initial_plan_recenter_and_fresh_all_invoke_sweep=True,
            candidate_rejected_before_insert_if_sweep_not_clear=True),
        limitations=['5 mm is an explicit engineering preflight allowance, not a physical or hardware safety guarantee.',
            'Hypothetical 85-degree result uses the old observed teddy; changed dynamics require fresh actual geometry.',
            'Only a fixed-left-pose frozen-shape vertical translation is certified; transport, later soft deformation, and affine proxy compliance are not.',
            'All existing real physics, hold, raw force/torque, and final containment gates remain required.'])
    out=HERE/'runtime_ipc/independent-sweep-review.json'
    out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('analytic_cases','static_review','limitations')},indent=2))


if __name__ == '__main__':
    run()
