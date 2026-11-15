"""CPU-only independent bounded review of generic translational sweep and detour integration."""
from pathlib import Path
import importlib.util
import itertools
import json
import hashlib
import sys
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import contact_robot_geometry as new
spec=importlib.util.spec_from_file_location('old_geometry_for_review',HERE/'runs/two_panel_full_11/source/contact_robot_geometry.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def cube(center,half=.01):
    return np.asarray(list(itertools.product((-1.,1.),repeat=3)))*half+center


def main():
    tests=[]
    def check(name,ok,**details):
        tests.append(dict(name=name,passed=bool(ok),**details));assert ok,name
    obstacle=cube(np.zeros(3),.015)
    points=cube(np.array([-.08,-.08,0.]));tets=Delaunay(points).simplices
    before=points.copy();q=np.arange(9,dtype=float)/10;q_before=q.copy();before_tets=tets.copy()
    fake=object()
    def set_obstacle(vertices):
        solid=new._convex(vertices);solid.update(geom_id=1,geom_name='synthetic',body='left_hand',original_geom_type='box',surface_watertight_after_position_weld=True,surface_degenerate_faces=0,surface_boundary_edges=0,surface_nonmanifold_edges=0)
        new.left_collision_solids=lambda planner,ql:[solid]
        old.left_collision_solids=lambda planner,ql:[solid]
    set_obstacle(obstacle)
    def sweep(p,d,translation=(0.,0.,0.)):
        return new.check_translation_sweep(p,tets,q,{},displacement_m=d,translation_m=translation,required_gap_m=.005,planner=fake)
    for label,p,d,expect in (
        ('diagonal_start_safe',points,[0,0,0],True),
        ('diagonal_end_safe',points+[.16,.16,0],[0,0,0],True),
        ('diagonal_middle_blocked',points,[.16,.16,0],False),
        ('axis_detour_first_leg_clear',points,[0,.16,0],True),
        ('axis_detour_second_leg_clear',points+[0,.16,0],[.16,0,0],True)):
        r=sweep(p,d);check(label,(r['status']=='clear')==expect,status=r['status'],gap_lower_m=r['conservative_gap_lower_bound_m'])
    rot=Rotation.from_rotvec([.37,-.21,.45]).as_matrix();set_obstacle(obstacle@rot.T)
    r=sweep(points@rot.T,np.array([.16,.16,0])@rot.T)
    check('arbitrarily_rotated_midsegment_blocked',r['status']!='clear',status=r['status'])
    set_obstacle(obstacle)
    for label,p,dz in (('crossing',cube([0,0,.08]),-.16),('clear',points,.03),('no_op',points,0.)):
        a=old.check_vertical_sweep(p,tets,q,{},vertical_displacement_m=dz,required_gap_m=.005,planner=fake)
        b=new.check_vertical_sweep(p,tets,q,{},vertical_displacement_m=dz,required_gap_m=.005,planner=fake)
        check('vertical_compatibility_'+label,a['status']==b['status'] and abs(a['conservative_gap_lower_bound_m']-b['conservative_gap_lower_bound_m'])<1e-12 and a['outside_translation_m']==b['outside_translation_m'] and a['vertical_displacement_m']==b['vertical_displacement_m'],old_status=a['status'],new_status=b['status'])
    for name,d in (('nan',[np.nan,0,0]),('wrong_shape',[0,0])):
        r=sweep(points,d);check('invalid_general_displacement_'+name,r['status']=='indeterminate')
    check('inputs_unchanged',np.array_equal(points,before) and np.array_equal(q,q_before) and np.array_equal(tets,before_tets))
    root=HERE/'runs/two_panel_full_10';candidate=json.loads((root/'detour_contact_candidate.json').read_text());regression=json.loads((root/'detour_contact_regression.json').read_text());cfg=json.loads((HERE/'config_guarded_detour_trial.json').read_text());prior=json.loads((HERE/'config_guarded_soft_opening_trial.json').read_text());cf=cfg['task']['contact_trial'].pop('transport_xy_strategy')
    check('config_only_strategy_change',cf=='y_then_x' and cfg==prior)
    for name,digest in regression['source_sha256'].items():check('frozen_hash_'+name,hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest)
    stages=candidate['report']['robot_stages'];sweeps=candidate['report']['robot_centering_attempts'][-1]['left_robot_teddy_sweep_segments'];expected=['height_at_current_xy','transport_y_at_current_x','translate_outside_mouth','insert_candidate']
    check('every_candidate_segment_has_clear_all22_certificate',[s['name'] for s in stages]==expected and [s['stage'] for s in sweeps]==expected and all(s['status']=='clear' and s['complete_robot_certificate'] and len(s['per_geom'])==s['left_collision_geom_count']==22 for s in sweeps))
    check('initial_insertion_withheld',[s['name'] for s in candidate['stages_to_execute']]==expected[:-1])
    state=dict(np.load(root/'lift_teddy.npz'));start=state['tcp_right'].copy()
    for stage,sw in zip(stages,sweeps):
        end=np.asarray(stage['right_tcp_m']);check('segment_geometry_'+stage['name'],np.allclose(np.asarray(sw['translation_m']),start-state['tcp_right'],rtol=0,atol=1e-14) and np.allclose(np.asarray(sw['displacement_m']),end-start,rtol=0,atol=1e-14));start=end
    check('actual_left_fingers_preserved',all(np.array_equal(np.asarray(sw['q_left_measured']),state['q_left']) for sw in sweeps))
    check('reported_FK_errors_within_explicit_bounds',all(s['straight_path_fk_review']['maximum_position_error_m']<=.0005 and s['straight_path_fk_review']['maximum_orientation_error_rad']<=.002 and not s['straight_path_fk_review']['is_continuous_error_bound'] for s in stages))
    result=dict(kind='independent_translation_and_detour_review',all_checks_passed=all(t['passed'] for t in tests),tests=tests,source_sha256={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ('run.py','contact_insertion.py','contact_robot_geometry.py')},candidate_minimum_frozen_gap_m=min(s['conservative_gap_lower_bound_m'] for s in sweeps),static_review={'general_translation_preserves_exact_tet_prism_union':True,'every_clear_geom_requires_all_tetrahedra_or_enclosing_hull_certificate':True,'corrections_remain_two_stages_and_30_50mm_budgets':True,'fresh_insertion_still_required_after_preparation_and_each_correction':True,'raw_physics_and_contact_limits_unchanged':True,'runtime_checks_declared_height_y_x_route':True},physical_success=False,limitations=['The all-segment frozen-shape result is rechecked for saved report consistency here; the full expensive section and route candidate was computed by the planner regression, not repeated in this review.','Synthetic checks exercise the generic translation core without simulator initialization.','5 mm is an engineering preflight allowance; frozen geometry and 201 sampled FK checks do not certify dynamic deformation, continuous tracking error or retention.','Release and withdrawal still use physical guards and rigid route checks; released teddy motion is not a frozen attached sweep.','Actual execution must pass fresh observed geometry, native forces/torque, both holding requirements, soft deformation, table penetration and final containment.'])
    (HERE/'runtime_ipc/independent-translation-detour-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(checks=len(tests),all_checks_passed=result['all_checks_passed'],minimum_gap_m=result['candidate_minimum_frozen_gap_m'],source_sha256=result['source_sha256']),indent=2))

if __name__=='__main__':main()
