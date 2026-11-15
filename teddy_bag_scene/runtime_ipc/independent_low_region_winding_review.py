"""Bounded synthetic audit of the existing pointwise low-region diagnostic.

No simulation or production edits. Two small open topological disks test the
geometric predicate; these are mathematical fixtures, not simulated cloth bags.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import diagnose_low_region_winding as diagnostic
from validate import containment, proper_segment_crossings, signed_volumes

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = HERE / 'low-region-winding-synthetic-fixtures'
base.mkdir(exist_ok=True)
faces = np.array([[0,1,2],[0,2,3],[0,3,4],[0,4,1]], dtype=int)
rim = np.arange(1,5)
rest_bag = np.array([[-5.,0,0],[-1,-1,3],[1,-1,3],[1,1,3],[-1,1,3]])
tets = np.array([[0,1,2,3]], dtype=int)
small_tet = .002*np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
config = {'bag':{'height_m':3.,'position':[0,0,0], 'thickness':.0003,
                 'mesh':'__synthetic_winding_open_disk.obj'}}

def run_case(name, bag, query, case_faces=faces):
    folder = base/name
    folder.mkdir(exist_ok=True)
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    bear = query + small_tet
    np.savez(folder/'rest.npz', bag=rest_bag, bag_faces=case_faces, bear=bear, bear_tets=tets)
    np.savez(folder/'state.npz', bag=bag, bear=bear, time=0.)
    result = diagnostic.diagnose(folder, 'state.npz')
    (folder/'diagnostic.json').write_text(json.dumps(result,indent=2,default=diagnostic.serialize)+'\n')
    old = containment(bag,bear,case_faces,rim,np.abs(signed_volumes(bear,tets)),tets)
    return dict(status=result['status'], all_queries_inside=result['all_query_points_inside_by_guarded_unit_winding'],
        physical_success=result['physical_task_success_certified'], reasons=result['reasons'],
        topology=result.get('actual_topology_orientation'), spatial_rim=result.get('actual_spatial_rim'),
        cap_max_difference=result.get('cap_winding',{}).get('maximum_absolute_cap_difference'),
        raw_mean_winding=result.get('cap_winding',{}).get('mean_apex_raw_winding'),
        old_validator=old, full_report=str(folder/'diagnostic.json'))

# Radial projection from the cone tip to z=1 is a simple polygon. Therefore
# distinct cone triangles meet only on their intended common edge or tip:
# each positive-z ray intersects a simple polygon edge at most as prescribed.
positive = np.array([[-5.,0,0],[-1,0,3],[1,.2,3],[.3,-.3,6],[.3,.5,6]])
positive_query = np.array([-3.8,.015,1.])
radial = (positive[1:,:2]-positive[0,:2])/positive[1:,2,None]
radial3 = np.column_stack([radial,np.zeros(4)])
radial_nonadjacent_gap = min(diagnostic.segment_distance(radial3[i],radial3[(i+1)%4],radial3[j],radial3[(j+1)%4]) for i,j in [(0,2),(1,3)])
positive_result = run_case('embedded_projected_crossing', positive, positive_query)
assert proper_segment_crossings(positive[1:,:2]) == 1
assert proper_segment_crossings(radial) == 0
assert radial_nonadjacent_gap > 1e-6
assert positive_result['all_queries_inside']
assert not positive_result['old_validator']['virtual_closed_body_valid']

# Opposite body triangles share a whole interior segment, not just their
# allowed common tip. Both have (0,0,1) strictly in their triangle interiors.
negative = np.array([[0.,0,0],[-1,-1,3],[1,1,3],[-1,1,6],[1,-1,6]])
negative_query = np.array([.1,.2,1.])
negative_result = run_case('body_self_intersection_undetected', negative, negative_query)
assert negative_result['all_queries_inside']  # deliberately exposes scope limit
crossing = np.array([0.,0,1.])
bary_a = np.array([2/3,1/6,1/6])
bary_b = np.array([5/6,1/12,1/12])
# Correct barycentric weights for z=1 on planes with rim z=3 and z=6.
assert np.allclose(bary_a @ negative[faces[0]], crossing, atol=1e-15, rtol=0)
assert np.allclose(bary_b @ negative[faces[2]], crossing, atol=1e-15, rtol=0)
assert (bary_a > 0).all() and (bary_b > 0).all()

reversed_result = run_case('embedded_global_orientation_reversed', positive, positive_query, faces[:,::-1])
assert reversed_result['all_queries_inside']  # orientation convention invariant
bad_local_faces = faces.copy(); bad_local_faces[0] = bad_local_faces[0,::-1]
local_reverse_result = run_case('inconsistent_single_face_orientation', positive, positive_query, bad_local_faces)
assert not local_reverse_result['all_queries_inside']
degenerate = positive.copy(); degenerate[0] = (positive[1]+positive[2])/2
degenerate_result = run_case('one_degenerate_body_triangle', degenerate, positive_query)
assert not degenerate_result['all_queries_inside']

frozen = ROOT/'runs/two_panel_full_21/source/validate.py'
report = dict(kind='independent_low_region_winding_math_and_synthetic_review', physical_success=False,
    production_files_modified=False, genesis_or_gpu_used=False, real_terminal_state_evaluated=False,
    inputs_sha256={str(p):sha(p) for p in [Path(__file__),HERE/'diagnose_low_region_winding.py',ROOT/'validate.py',frozen]},
    current_validate_matches_full21_frozen=sha(ROOT/'validate.py')==sha(frozen),
    cases=dict(embedded_projected_crossing=positive_result, body_self_intersection_undetected=negative_result,
               global_orientation_reversed=reversed_result, inconsistent_face_orientation=local_reverse_result,
               degenerate_body_triangle=degenerate_result),
    positive_embedding_certificate=dict(radial_projection_polygon=radial.tolist(),
        radial_projection_proper_crossings=0, nonadjacent_radial_edge_minimum_gap_m=radial_nonadjacent_gap,
        description='Four radial edges are simple, nondegenerate, and non-overlapping; at each positive z below all rim vertices, the conical body cuts are scaled copies. Cone tip is the only common point of nonadjacent triangles.'),
    negative_embedding_counterexample=dict(body_face_indices=[0,2],common_interior_point=crossing.tolist(),
        barycentric_coordinates=[bary_a.tolist(),bary_b.tolist()],
        all_barycentric_entries_strictly_positive=True,
        conclusion='An actual body self-intersection away from the queries passes all existing low-region guards with unit winding. A simple 3D rim and coherent triangle orientation do not certify embedded body geometry.'),
    math_scope=dict(cap_independence='For h=min(rim_z), both fan caps are entirely in z>=h. Their oriented difference is a closed triangle chain in that halfspace. For q_z<h its winding is zero, so the two completed body chains have identical degree, even if the caps fold or their 2D projections cross.',
        interior_condition='Unit absolute degree is physical interior only with an appropriately embedded, orientable bag boundary/body closure. For an immersed/self-intersecting body it is merely an algebraic pointwise degree. Checking finitely many unit degrees cannot prove the body is embedded.',
        orientation='Combinatorial single-face reversal fails consistency. Global reversal only negates winding and is harmless. Geometric local folding/self-crossing can keep all combinatorial orientation checks true.',
        degeneracy='Existing actual body twice-area >1e-14 rejects exactly degenerate/very tiny triangles, but is not a conditioning, aspect-ratio, or self-intersection certificate.',
        whole_tetrahedra='All vertices and COM below h implies all convex tetrahedra lie below h; it does NOT imply those tetrahedra stay in the nonconvex body interior. Boundary intersections through tet faces/interiors can occur without any queried vertex or COM being outside.'),
    implementation_findings=[
        'Existing diagnostic correctly declares that it does not certify physical body nonintersection, continuous whole-bear containment, or task success.',
        'Two caps and integer/unit guards improve numerical interpretation but are not independent evidence of physical body embedding.',
        'Rim separation checks nonadjacent rim segments only. It does not test body/body triangle intersections, adjacent-triangle overfolding beyond their shared simplex, body/rim intersections, or handle/body contact.',
        'Finite numerical margins are explicit guards, not rigorous floating-point interval certificates. In particular the near-parallel segment branch and 10nm subtraction are heuristic unless conditioned by actual edge lengths.',
        'The script imports live validate.py and resolves mouth label assets relative to the live project. A final frozen-run audit should bind to frozen validator/assets and include their hashes. Current live/frozen validator hashes agree.',
        'Old containment invalid-cap branch returns 0/false although no point classification was performed; consumers must use virtual_closed_body_valid and treat invalid as unknown, not measured outside.'
    ],
    minimal_reviewable_correction_if_needed=[
        'First diagnose actual final and required hold frames; do not infer them from frozen transport predictions.',
        'Keep existing native physics/contact/release/stability requirements unchanged and preserve original validation output.',
        'Represent unusable cap classification as unknown/null rather than fabricated zero fraction.',
        'A narrowly gated low-region fallback can replace only the SVD projection rejection when every query is >=2mm below actual rim minimum, original topology/orientation/nondegeneracy and real 3D rim guards pass, and both high caps give finite matching integer degree without multiplicities.',
        'Before promoting that pointwise fallback to a physical inside claim, independently check actual body triangle embedding, including nonadjacent pairs and adjacent pairs intersecting beyond their shared simplex; ambiguous near-contact must remain unknown rather than be silently skipped.',
        'If whole teddy volume containment is claimed, additionally rule out actual body intersections with all teddy tetrahedra/boundary and disconnected/internal pockets. Vertex fraction and discrete bag penetration samples alone do not establish it.'
    ])
referenced = ROOT/'runs/two_panel_full_21/independent-transport-reobserve-geometry/hypothetical_reobserved_insert_low_region.json'
if referenced.exists():
    prior = json.loads(referenced.read_text())
    report['referenced_frozen_hypothetical_diagnostic_observation'] = dict(path=str(referenced),sha256=sha(referenced),
        real_terminal_state=False, status=prior['status'], height_gap_m=prior['height_precondition']['minimum_query_height_gap_m'],
        rim_vertex_count=prior['actual_spatial_rim']['vertex_count'],
        projection=prior['projection_diagnostics_only'],
        all_query_points_inside=prior['all_query_points_inside_by_guarded_unit_winding'], reasons=prior['reasons'],
        scope='Existing hypothetical result inspected only; not recomputed or promoted to native outcome.')
out=HERE/'independent-low-region-winding-review.json'
out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(report=str(out),sha256=sha(out),cases={k:dict(status=v['status'],all_queries_inside=v['all_queries_inside'],old_cap_valid=v['old_validator']['virtual_closed_body_valid']) for k,v in report['cases'].items()}),indent=2))
