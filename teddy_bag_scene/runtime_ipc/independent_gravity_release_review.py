"""Bounded independent gravity-release policy/eligibility/runner review.

No Genesis import, native stepping, or production mutation. Geometry and native
motion are mocked only for control-flow boundary injection; actual-input CPU
preflight remains a separate artifact.
"""
from pathlib import Path
import ast
import copy
import hashlib
import json
import sys
import tempfile
import types
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravity_release as gr
from motion import MotionPlan
from postlift_yaw import _serial, RIGID_CATEGORIES

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
protected = ['run.py', 'gravity_release.py', 'motion.py', 'validate.py',
             'contact_insertion.py', 'contact_guard.py', 'physics_config.py',
             'scene.py', 'config.json']
before = {name: sha(ROOT/name) for name in protected}
checks = []
cases = {}


def check(name, good, detail=None):
    checks.append(dict(name=name, passed=bool(good), detail=detail))


def cfg():
    return dict(table={'height': .74}, task={'contact_trial': {
        'insertion_strategy': 'vertical', 'recenter_enabled': False,
        'gravity_release': {'mode': 'gravity_seating', 'tcp_height_above_table_m': .345,
                            'minimum_inside_vertex_fraction': .5, 'settle_seconds': 4.}}})


def denied(fn):
    try:
        value = fn()
        return isinstance(value, dict) and value.get('status') == 'rejected'
    except Exception:
        return True


check('absent option preserves legacy policy', gr.release_policy({}) is None)
check('declared policy accepted', gr.release_policy(cfg()) == cfg()['task']['contact_trial']['gravity_release'])
for key, values in {
    'mode': ['', None, 'gravity', True],
    'tcp_height_above_table_m': [.249999, .450001, float('nan'), float('inf'), True, '0.345', []],
    'minimum_inside_vertex_fraction': [.499999, 1.000001, float('nan'), float('inf'), True, '0.5'],
    'settle_seconds': [1.9999, 10.0001, float('nan'), True, '4'],
}.items():
    for i, value in enumerate(values):
        config = cfg(); config['task']['contact_trial']['gravity_release'][key] = value
        check(f'policy rejects {key} invalid {i}', denied(lambda c=config: gr.release_policy(c)))
for lo, hi, name in [(.25, .45, 'tcp_height_above_table_m'), (.5, 1., 'minimum_inside_vertex_fraction'), (2., 10., 'settle_seconds')]:
    for value in (lo, hi):
        config = cfg(); config['task']['contact_trial']['gravity_release'][name] = value
        check(f'policy accepts exact {name} bound {value}', gr.release_policy(config)[name] == value)
for name, value in [('insertion_strategy', 'lower_then_center'), ('recenter_enabled', True), ('recenter_enabled', 0)]:
    config = cfg(); config['task']['contact_trial'][name] = value
    check(f'policy rejects incompatible {name} {value!r}', denied(lambda c=config: gr.release_policy(c)))


def actual_case(fault=None):
    config = cfg()
    state = {'time': 20., 'bag': np.zeros((4, 3)), 'bear': np.ones((4, 3)),
             'tcp_right': np.array([0., 0., 1.085]), 'q_left': np.zeros(9), 'q_right': np.zeros(9)}
    rest = {'bag': np.zeros((4, 3)), 'bag_faces': np.array([[0, 1, 2], [0, 2, 3]]),
            'bear': np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
            'bear_tets': np.array([[0, 1, 2, 3]])}
    candidate = {'status': 'candidate', 'policy': copy.deepcopy(config['task']['contact_trial']['gravity_release']),
                 'observed_time_s': 10., 'outside_tcp_m': [0., 0., 1.3],
                 'insert_stage': {'right_tcp_m': [0., 0., 1.085], 'rotation_xyzw': [0., 0., 0., 1.]}}
    contact = {'time_s': 20., 'right_two_sided_contact_required': True, 'stop_reason': None,
               'right_finger_opening_projection_n': [2., 2.]}
    joints = np.zeros((2, 19)); joints[:, 0] = [19.995, 20.]
    capture = {'virtual_closed_body_valid': True, 'bear_center_of_mass_inside': True,
               'bear_vertex_inside_fraction': .5, 'bear_center_of_mass_m': [0., 0., 1.],
               'all_bear_vertices_inside': False}
    context = {'reasons': [], 'rotation': Rotation.identity(), 'qs': {'left': np.zeros(7), 'right': np.zeros(7)}}
    if fault == 'invalid_cap': capture['virtual_closed_body_valid'] = False
    if fault == 'COM_outside': capture['bear_center_of_mass_inside'] = False
    if fault == 'fraction_under': capture['bear_vertex_inside_fraction'] = .5 - 1e-10
    if fault == 'fraction_nan': capture['bear_vertex_inside_fraction'] = float('nan')
    if fault == 'physical_rejection': context['reasons'] = ['tetrahedron volume ratio below 0.05']
    if fault == 'position_error': state['tcp_right'][0] = .00201
    if fault == 'orientation_error': context['rotation'] = Rotation.from_rotvec([0., 0., .01501])
    if fault == 'old_candidate': candidate['observed_time_s'] = 20.
    if fault == 'changed_policy': candidate['policy']['settle_seconds'] = 5.
    if fault == 'contact_stale': contact['time_s'] -= .005
    if fault == 'contact_stop': contact['stop_reason'] = 'Original IPC torque gate stopped'
    if fault == 'holding_false': contact['right_two_sided_contact_required'] = False
    if fault == 'one_finger_at_limit': contact['right_finger_opening_projection_n'][1] = .1
    if fault == 'opening_nan': contact['right_finger_opening_projection_n'][1] = float('nan')
    if fault == 'opening_inf': contact['right_finger_opening_projection_n'][1] = float('inf')
    if fault == 'opening_missing': contact.pop('right_finger_opening_projection_n')
    if fault == 'trace_stale': joints[-1, 0] -= .001
    if fault == 'trace_gap': joints[0, 0] = 19.97
    if fault == 'trace_nan': joints[-1, 10] = float('nan')
    if fault == 'velocity_high': joints[0, 10] = -.000251
    if fault == 'trace_q_mismatch': joints[:, 10] = .01
    if fault == 'trace_left_q_mismatch': joints[:, 1] = .01
    if fault == 'opening_geometry_failure':
        def opening(*args, **kwargs): raise ValueError('High release opening rigid gate failed')
    else:
        empty = MotionPlan({s: CubicSpline([0., 1.], np.zeros((2, 7)), axis=0) for s in ('left', 'right')},
                           4., {'left': 0., 'right': .04}, {'left': 0., 'right': .04})
        opening = lambda *args, **kwargs: ({'scope': 'Mocked robot preflight for eligibility isolation'}, empty)
    with patch.object(gr, '_state_context', return_value=context), patch.object(gr, '_policy', return_value={}), \
         patch.object(gr, 'mouth_structure', return_value=(rest['bag_faces'], np.arange(4), {'two_panels_directly_sewn_along_u': True})), \
         patch.object(gr, 'containment', return_value=capture), patch.object(gr, '_opening_and_withdrawal', side_effect=opening):
        try:
            result = gr.review_actual_release(rest, state, config, candidate, contact, joints)
            return result.get('status') == 'admissible_release', result
        except Exception as error:
            return False, {'error_type': type(error).__name__, 'error': str(error)}


ok, data = actual_case(); cases['actual_boundary_positive'] = data
check('exact 50 percent plus COM valid slow endpoint is admissible', ok)
check('partial admission never claims physical success', not data.get('physical_success', True))
for fault in ['invalid_cap', 'COM_outside', 'fraction_under', 'fraction_nan', 'physical_rejection',
              'position_error', 'orientation_error', 'old_candidate', 'changed_policy',
              'contact_stale', 'contact_stop', 'holding_false', 'one_finger_at_limit', 'opening_nan',
              'opening_inf', 'opening_missing', 'trace_stale', 'trace_gap', 'trace_nan', 'velocity_high',
              'trace_q_mismatch', 'trace_left_q_mismatch', 'opening_geometry_failure']:
    ok, data = actual_case(fault); cases['actual_'+fault] = data
    check('actual review rejects '+fault, not ok)


class CartesianTestPlanner:
    """Exact affine FK/IK fixture; no claim about the actual robot geometry."""
    def __init__(self, fault=None):
        self.fault = fault
        self.limits = {side: np.tile([-10., 10.], (7, 1)) for side in ('left', 'right')}
    def fk(self, side, q):
        position = np.asarray(q[:3], float).copy()
        if self.fault == 'FK_deviation' and side == 'right' and .01 < position[2] < .19:
            position[0] += .00051
        rotation = Rotation.from_rotvec([0., 0., .00201]) if self.fault == 'orientation_deviation' else Rotation.identity()
        return position, rotation
    def plan_cartesian(self, commanded, targets, **kwargs):
        final = commanded['right'].copy(); final[:3] = targets['right']
        rows = []
        for fraction in np.linspace(0., 1., 45):
            q = {'left': commanded['left'].copy(), 'right': (1-fraction)*commanded['right']+fraction*final}
            if self.fault == 'changed_left': q['left'][6] += fraction*.001
            rows.append({'fraction': fraction if self.fault != 'incomplete_knots' else .1+.8*fraction, 'q': q})
        return rows
    def clearance(self, qs, fingers):
        rows = {key: {'distance': .02, 'bodies': ['fixture_a', 'fixture_b']} for key in sorted(RIGID_CATEGORIES)}
        if self.fault == 'missing_category': rows.pop('arms')
        if self.fault == 'raw_nan': rows['right_table']['distance'] = float('nan')
        if self.fault == 'hidden_inf': rows['right_table']['aux'] = [float('inf')]
        if self.fault == 'rigid_below': rows['arms']['distance'] = .007999999
        return rows


def motion_case(fault=None):
    commanded = {s: np.zeros(7) for s in ('left', 'right')}
    context = {'planner': CartesianTestPlanner(fault), 'qs': copy.deepcopy(commanded), 'fingers': {'left': .02, 'right': .025}}
    if fault == 'command_nan': commanded['right'][0] = float('nan')
    if fault == 'command_wrong_shape': commanded['right'] = np.zeros(8)
    return gr._checked_motion(context, commanded, np.array([.1, .05, .2]), Rotation.identity(), 6., context['fingers'])


plan, proof = motion_case()
check('checked motion preserves exact command starts', all(np.array_equal(plan.sample(0.)['q'][s], np.zeros(7)) for s in ('left', 'right')))
check('checked motion stationary endpoints', all(np.max(abs(plan.sample(t)[key][s])) < 1e-10 for t in (0., plan.duration) for key in ('qd', 'qdd') for s in ('left', 'right')))
check('checked motion all 401 samples and original rates', proof['rigid_sample_count'] == 401 and all(np.max(r['velocity_upper_bound_rad_s']) <= .35 and np.max(r['acceleration_upper_bound_rad_s2']) <= .5 for r in proof['continuous_rate_bounds'].values()))
samples = [plan.sample(float(t)) for t in np.linspace(0., plan.duration, 10003)]
check('independent dense sample confirms checked motion rates', max(np.max(abs(p[k][s]))/limit for p in samples for k, limit in [('qd', .35), ('qdd', .5)] for s in ('left', 'right')) <= 1.)
for fault in ['missing_category', 'raw_nan', 'hidden_inf', 'rigid_below', 'FK_deviation', 'orientation_deviation', 'changed_left', 'incomplete_knots', 'command_nan', 'command_wrong_shape']:
    check('checked motion rejects '+fault, denied(lambda f=fault: motion_case(f)))
for fault in ['missing_category', 'raw_nan', 'hidden_inf', 'rigid_below']:
    context = {'planner': CartesianTestPlanner(fault), 'qs': {s: np.zeros(7) for s in ('left', 'right')}, 'fingers': {'left': .02, 'right': .025}}
    check('high opening rejects '+fault, denied(lambda c=context: gr._opening_and_withdrawal(c, c['qs'], [.1, .05, .2], Rotation.identity())))


def plan_case(fault=None):
    config = cfg(); policy = {'left_robot_soft_sweep_clearance_m': .005}
    state = {'time': 10., 'tcp_right': np.array([0., 0., 1.3]), 'q_left': np.zeros(9),
             'tcp_left': np.zeros(3), 'q_right': np.array([0., 0., 1.3, 0., 0., 0., 0., 0., 0.]),
             'bear': np.ones((4, 3))}
    rest = {'bear_tets': np.array([[0, 1, 2, 3]])}
    qs = {'left': np.zeros(7), 'right': np.array([0., 0., 1.3, 0., 0., 0., 0.])}
    context = {'planner': CartesianTestPlanner(), 'qs': qs, 'fingers': {'left': .02, 'right': .025},
               'rotation': Rotation.identity(), 'reasons': []}
    depth = {'status': 'candidate', 'observed_time_s': 10., 'report': {'rejections': [], 'policy': policy},
             'insert_stage': {'right_tcp_m': [0., 0., .9]}, 'outside_tcp_m': [0., 0., 1.3]}
    sweep = {'status': 'clear', 'test_fixture': True, 'conservative_gap_lower_bound_m': .01}
    if fault == 'full_depth_rejected': depth['status'] = 'rejected'
    if fault == 'full_depth_reasons': depth['report']['rejections'] = ['original full-depth gate failed']
    if fault == 'full_depth_stale': depth['observed_time_s'] -= .005
    if fault == 'full_depth_policy': depth['report']['policy'] = {}
    if fault == 'full_depth_nonvertical': depth['insert_stage']['right_tcp_m'][0] = .001
    if fault == 'not_interior': depth['insert_stage']['right_tcp_m'][2] = 1.08
    if fault == 'physics': context['reasons'] = ['original physics failure']
    if fault == 'sweep_reject': sweep['status'] = 'not_certified'
    if fault == 'sweep_nan': sweep['conservative_gap_lower_bound_m'] = float('nan')
    if fault == 'command_actual_mismatch': qs = copy.deepcopy(qs); qs['right'][0] = .000501
    with patch.object(gr, '_state_context', return_value=context), patch.object(gr, '_policy', return_value=policy), patch.object(gr, 'check_translation_sweep', return_value=sweep):
        return gr.plan_gravity_release(rest, state, config, depth, qs, return_plan=True)


candidate, plan = plan_case()
check('candidate returns the checked executable plan', candidate['status'] == 'candidate' and isinstance(plan, MotionPlan) and candidate['insert_stage']['duration_s'] == plan.duration)
check('candidate cannot report task success', candidate['physical_success'] is False and candidate['requires_actual_partial_capture_before_release'] is True and candidate['requires_unchanged_final_full_containment_and_retention'] is True)
check('both actual and command left soft sweeps retained', set(candidate['left_robot_soft_sweeps']) == {'actual', 'commanded'})
for fault in ['full_depth_rejected', 'full_depth_reasons', 'full_depth_stale', 'full_depth_policy', 'full_depth_nonvertical', 'not_interior', 'physics', 'sweep_reject', 'sweep_nan', 'command_actual_mismatch']:
    check('partial preflight rejects '+fault, denied(lambda f=fault: plan_case(f)))


class GateFailure(Exception):
    pass


def runner_case(fault=None, legacy=False, old_source=False):
    source = (ROOT/('runs/two_panel_full_24/source/run.py' if old_source else 'run.py')).read_text()
    method = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == 'run_guarded_contact_trial')
    ns = dict(np=np, json=json, Rotation=Rotation, GateFailure=GateFailure)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<independent isolated runner>', 'exec'), ns)
    calls = []
    def stage(name, point):
        return {'name': name, 'right_tcp_m': np.asarray(point).tolist(), 'rotation_xyzw': [0., 0., 0., 1.], 'duration_s': 1.}
    def prepare(rest, state, config, output):
        calls.append('full_prepare')
        p = state['tcp_right']; targets = [p+[0., 0., .1], p+[.02, .01, .1]]
        return {'status': 'candidate', 'stages_to_execute': [stage(n, p) for n, p in zip(['height_at_current_xy', 'translate_outside_mouth'], targets)]}
    def fresh(rest, state, config, choice, output):
        calls.append('fresh_full_depth')
        return {'status': 'rejected' if fault == 'fresh_rejected' else 'candidate',
                'insert_stage': stage('full_depth', state['tcp_right']-[0., 0., .2]),
                'outside_tcp_m': state['tcp_right'].tolist()}
    def fixed_plan(duration):
        return MotionPlan({s: CubicSpline([0., 1.], np.zeros((2, 7)), axis=0) for s in ('left', 'right')},
                          duration, {'left': .02, 'right': .02}, {'left': .02, 'right': .02})
    insert_plan, withdraw_plan = fixed_plan(6.), fixed_plan(4.)
    def partial(rest, state, config, full_depth, command_qs, return_plan):
        calls.append('partial_preflight')
        assert return_plan is True and full_depth['status'] == 'candidate'
        if fault == 'partial_exception': raise ValueError('Frozen whole-bear left sweep rejected')
        candidate = {'status': 'rejected' if fault == 'partial_status' else 'candidate',
                     'insert_stage': stage('partial', state['tcp_right']-[0., 0., .1]),
                     'continuous_reference_zero_endpoint_velocity_acceleration': fault != 'nonzero_reference_endpoint'}
        if fault == 'changed_command_start': insert_plan.curves['right'].c[-1, 0, 0] += .001
        return candidate, None if fault == 'missing_insert_plan' else insert_plan
    def actual(rest, state, config, candidate, contact_row, joint_trace, command_qs, return_plan):
        calls.append('actual_release_review')
        assert return_plan is True
        if fault == 'actual_geometry_exception': raise ValueError('Actual high opening rigid gate rejected')
        return {'status': 'rejected' if fault == 'actual_rejected' else 'admissible_release'}, None if fault == 'missing_withdraw_plan' else withdraw_plan
    contact = types.ModuleType('contact_insertion')
    contact.plan_guarded_contact_trial = prepare; contact.reobserve_guarded_contact_trial = fresh
    contact.recenter_guarded_contact_trial = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Unexpected correction'))
    contact._serial = _serial
    gravity = types.ModuleType('gravity_release'); gravity.release_policy = gr.release_policy
    gravity.plan_gravity_release = partial; gravity.review_actual_release = actual
    class Sim:
        run_guarded_contact_trial = ns['run_guarded_contact_trial']
        def __init__(self, output):
            self.config = cfg()
            self.config['task']['contact_trial']['transport_xy_strategy'] = 'direct'
            if legacy: self.config['task']['contact_trial'].pop('gravity_release')
            self.out = output; self.rest = {}; self.time = 10.; self.report = {}; self.must_hold_teddy = True
            self.qs = {s: np.zeros(7) for s in ('left', 'right')}; self.finger = {'left': 0., 'right': 0.}
            self.rotations = {}; self.frames = [{'time': 10., 'tcp_right': np.array([0., 0., 1.3])}]
            self.contact_guard_trace = [{}]; self.joint_trace = []; self.dt = .005
        def begin_guarded_contact_monitor(self): calls.append('guard_no_reset')
        def move_guarded(self, label, step):
            calls.append(label)
            self.time += step['duration_s']; self.frames.append({'time': self.time, 'tcp_right': np.array(step['right_tcp_m'])})
        def execute(self, label, plan):
            calls.append(label)
            if label == 'insert_teddy':
                assert plan is insert_plan and self.must_hold_teddy is True
                assert plan.finger_start == plan.finger_end == self.finger == {'left': 0., 'right': 0.}
                if fault == 'native_force_failure': raise GateFailure('Native original IPC force/torque gate stopped')
                if fault == 'native_J_failure': raise GateFailure('Native original tetrahedron gate stopped')
                if fault == 'native_contact_failure': raise GateFailure('Native original two-sided contact gate stopped')
            else:
                assert plan is withdraw_plan and self.must_hold_teddy is False
                assert plan.finger_start == plan.finger_end == self.finger == {'left': 0., 'right': .04}
            self.time += np.ceil(plan.duration/self.dt)*self.dt
            self.qs = plan.end_q(); self.finger = plan.finger_end.copy()
            self.frames.append({'time': self.time, 'tcp_right': self.frames[-1]['tcp_right']})
        def hold(self, label, duration, **fingers):
            calls.append(label)
            if label == 'release_teddy':
                assert not self.must_hold_teddy and self.finger['right'] == 0. and fingers == {'right': .04}
            self.time += duration; self.finger.update(fingers)
    with tempfile.TemporaryDirectory(prefix='independent-gravity-flow-') as temporary:
        sim = Sim(Path(temporary)); error = None
        with patch.dict(sys.modules, {'contact_insertion': contact, 'gravity_release': gravity}):
            try: sim.run_guarded_contact_trial()
            except Exception as exc: error = type(exc).__name__+': '+str(exc)
        files = sorted(p.name for p in Path(temporary).iterdir())
        return {'calls': calls, 'error': error, 'must_hold_teddy': sim.must_hold_teddy,
                'finger_motor_targets': sim.finger, 'report': sim.report, 'files': files}


legacy = runner_case(legacy=True); old_legacy = runner_case(legacy=True, old_source=True)
cases['legacy_runner'] = legacy
check('legacy absent-option runner exactly preserves phase flow', legacy == old_legacy and legacy['error'] is None)
row = runner_case(); cases['gravity_runner_positive'] = row
check('gravity runner executes same insert and withdraw objects with command continuity', row['error'] is None and row['report'].get('sequence_completed') is True)
check('gravity runner fresh full-depth before partial motion and actual review before intentional release', row['calls'].index('fresh_full_depth') < row['calls'].index('partial_preflight') < row['calls'].index('insert_teddy') < row['calls'].index('actual_release_review') < row['calls'].index('release_teddy'))
check('gravity runner archives exact insert and withdraw curves', 'gravity_insert_plan.npz' in row['files'] and 'gravity_withdraw_plan.npz' in row['files'])
for fault in ['fresh_rejected', 'partial_exception', 'partial_status', 'missing_insert_plan', 'nonzero_reference_endpoint',
              'changed_command_start', 'native_force_failure', 'native_J_failure', 'native_contact_failure',
              'actual_geometry_exception', 'actual_rejected', 'missing_withdraw_plan']:
    row = runner_case(fault); cases['runner_'+fault] = row
    check('runner retains holding and never releases after '+fault, row['error'] is not None and row['must_hold_teddy'] is True and 'release_teddy' not in row['calls'] and row['finger_motor_targets']['right'] == 0. and not row['report'].get('sequence_completed', False))

check('original final validation and contact monitor unchanged from full24', all(sha(ROOT/name) == sha(ROOT/'runs/two_panel_full_24/source'/name) for name in ['validate.py', 'contact_guard.py', 'physics_config.py', 'contact_insertion.py', 'scene.py', 'motion.py']))


def finish():
    check('production and default config unchanged by independent review', all(sha(ROOT/name) == digest for name, digest in before.items()))
    result = dict(kind='independent_gravity_release_review', physical_success=False, native_execution_performed=False,
                  all_checks_passed=all(row['passed'] for row in checks), check_count=len(checks),
                  checks=checks, failed_checks=[row for row in checks if not row['passed']], cases=cases,
                  production_source_sha256=before, limitations=[
                      'Injected eligibility tests mock robot and containment computations; they certify control-flow rejection semantics only.',
                      'Partial node inclusion is not a mass or volume fraction, stable support proof, or final containment.',
                      'Actual native motion and final whole-teddy containment remain pending.'])
    output = ROOT/'runtime_ipc/independent-gravity-release-review.json'
    output.write_text(json.dumps(_serial(result), indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'sha256': sha(output), 'checks': len(checks),
                      'all_checks_passed': result['all_checks_passed'], 'failed_checks': result['failed_checks']}, indent=2))
    return result


if __name__ == '__main__':
    result = finish()
    if not result['all_checks_passed']: raise SystemExit(1)
