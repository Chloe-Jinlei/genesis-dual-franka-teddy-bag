"""Isolated real finish() regression, with no simulator import or native run.

Uses the saved full25 gravity candidate and restores its NumPy containers.
Validator/motion data are bounded fixtures; no original failed-run report is
created, reconstructed, or modified.
"""
from pathlib import Path
import ast
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import time
import types
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/two_panel_full_25'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_source = (ROOT/'runtime_ipc/report-serialization-before-run.py').read_text()
new_source = (ROOT/'run.py').read_text()
candidate_json = json.loads((RUN/'gravity_release_candidate.json').read_text())
protected = {str(p): sha(p) for p in [*sorted((RUN/'source').rglob('*.py')),
             RUN/'input_sha256.json', RUN/'config.json', RUN/'gravity_release_candidate.json',
             ROOT/'config.json', ROOT/'gravity_release.py', ROOT/'motion.py', ROOT/'scene.py',
             ROOT/'contact_insertion.py', ROOT/'validate.py', ROOT/'physics_config.py',
             ROOT/'runtime_ipc/audit_guarded_run.py', ROOT/'runtime_ipc/audit_gravity_execution.py']}
checks = []; cases = {}


def check(name, value):
    checks.append({'name': name, 'passed': bool(value)})


def fn(source, name):
    return next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.FunctionDef) and node.name == name)


serializer_node = fn((ROOT/'contact_insertion.py').read_text(), '_serial')
serializer_namespace = {'np': np}
exec(compile(ast.Module(body=[serializer_node], type_ignores=[]), '<existing serialization helper>', 'exec'), serializer_namespace)
serial = serializer_namespace['_serial']


def rehydrate(value):
    if isinstance(value, dict): return {key: rehydrate(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return np.asarray(value)
        return [rehydrate(item) for item in value]
    if isinstance(value, bool): return np.bool_(value)
    if isinstance(value, int): return np.int64(value)
    if isinstance(value, float): return np.float64(value)
    return value


def scenario(source, *, physical_error=None, validation_success=True, validation_error=False, fault=None, legacy=False):
    node = fn(source, 'finish'); namespace = dict(np=np, json=json, time=time)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<isolated real finish>', 'exec'), namespace)
    fake_contact = types.ModuleType('contact_insertion'); fake_contact._serial = serial
    fake_validate = types.ModuleType('validate')
    def audit_arrays(*args, **kwargs):
        if validation_error: raise ValueError('Synthetic validator unavailable')
        return {'task_success': validation_success, 'checks': {'synthetic_fixture': True}}
    fake_validate.audit_arrays = audit_arrays
    with tempfile.TemporaryDirectory(prefix='report-serialization-finish-') as temp:
        output = Path(temp)
        obj = types.SimpleNamespace(out=output, writer=types.SimpleNamespace(close=lambda: None),
            time=49.055, wall=time.monotonic()-1., events=[{'phase': 'insert_teddy', 'start': 38.865, 'duration': 12.9}],
            frames=[{'time': .005*i, 'bag': np.zeros((4, 3)), 'bear': np.zeros((4, 3))} for i in range(5)],
            joint_trace=[np.r_[.005*i, np.zeros(18)] for i in range(1, 5)],
            soft_metrics=[np.array([.005*i, .0436 if physical_error else 1., 0., 0., 1., 1.]) for i in range(5)],
            physical_monitor=types.SimpleNamespace(columns=['time_s', 'min_J', 'bag_table', 'bear_table', 'cloth', 'finite']),
            rest={}, config={}, args=types.SimpleNamespace(task='full'), handle_grip_trace=[],
            staged_hand_monitor=None, contact_guard=None, contact_guard_trace=[], coupling_trace=[{'time': 49.055, 'links': []}],
            report={'sequence_completed': physical_error is None, 'original_finite_report': {'x': 1.25, 'flag': True, 'unknown': None}})
        candidate = rehydrate(candidate_json)
        if not legacy:
            obj.report['gravity_release_candidate'] = candidate
            obj.report['gravity_release_actual_review'] = {
                'synthetic_test_record': True, 'status': 'admissible_release',
                'actual_containment': {'bear_vertex_inside_fraction': np.float64(.5), 'bear_center_of_mass_m': np.array([.1, .2, 1.]),
                                       'virtual_closed_body_valid': np.bool_(True)},
                'actual_right_endpoint_joint_velocity_rad_s': np.zeros(7),
                'nested_scalar_and_array': (np.int64(3), {'zero_dimensional_array': np.array(4.5)})}
        if fault is not None:
            obj.report['nested_bad_evidence'] = {'list': [{'value': fault}]}
        error = None
        with patch.dict(sys.modules, {'contact_insertion': fake_contact, 'validate': fake_validate}), contextlib.redirect_stdout(io.StringIO()):
            try: namespace['finish'](obj, physical_error)
            except Exception as exc: error = {'type': type(exc).__name__, 'message': str(exc)}
        report = None; strict_parse = False
        if (output/'report.json').exists():
            def invalid_constant(token): raise ValueError('Non-standard JSON constant '+token)
            report = json.loads((output/'report.json').read_text(), parse_constant=invalid_constant)
            strict_parse = True
        return {'error': error, 'report': report, 'strict_finite_json_parsed': strict_parse,
                'candidate_value_roundtrip': legacy or report is not None and report['gravity_release_candidate'] == candidate_json,
                'in_memory_candidate_identity_preserved': legacy or obj.report['gravity_release_candidate'] is candidate,
                'in_memory_candidate_array_preserved': legacy or isinstance(obj.report['gravity_release_candidate']['joint_extrema']['right']['minimum'], np.ndarray),
                'files': sorted(p.name for p in output.iterdir())}


for physical_error in (None, 'GateFailure: Teddy element excessively compressed or inverted: volume ratio 0.0436'):
    label = 'normal' if physical_error is None else 'physical_failure'
    before = scenario(old_source, physical_error=physical_error, validation_success=physical_error is None)
    check(label+' reproduces original ndarray serialization failure', before['error'] is not None and before['error']['type'] == 'TypeError' and 'ndarray' in before['error']['message'] and before['report'] is None)
    row = scenario(new_source, physical_error=physical_error, validation_success=physical_error is None)
    cases[label] = row
    check(label+' saves strict finite JSON and coupling evidence', row['error'] is None and row['strict_finite_json_parsed'] and 'coupling_trace.json' in row['files'])
    check(label+' exact actual gravity candidate value roundtrip', row['candidate_value_roundtrip'])
    check(label+' preserves in-memory arrays and candidate identity', row['in_memory_candidate_identity_preserved'] and row['in_memory_candidate_array_preserved'])
    check(label+' preserves physical outcome and error', row['report']['status'] == ('completed' if physical_error is None else 'failed') and row['report']['error'] == physical_error and row['report']['requested_task_passed'] is (physical_error is None))
    check(label+' converts nested actual-review NumPy values', row['report']['gravity_release_actual_review']['actual_containment']['virtual_closed_body_valid'] is True and row['report']['gravity_release_actual_review']['nested_scalar_and_array'] == [3, {'zero_dimensional_array': 4.5}])

for label, options in [('validation_failed', {'validation_success': False}), ('validation_exception', {'validation_error': True})]:
    row = scenario(new_source, **options); cases[label] = row
    check(label+' saves failure without promotion to success', row['error'] is None and row['report']['status'] == 'validation_failed' and row['report']['requested_task_passed'] is False)
    if label == 'validation_exception': check('validator error preserved verbatim', row['report']['validation_error'] == 'ValueError: Synthetic validator unavailable')

for label, value in [('python_nan', float('nan')), ('python_inf', float('inf')), ('numpy_nan', np.float32('nan')),
                     ('array_inf', np.array([1., np.inf])), ('nested_object_nan', np.array([{'value': np.nan}], dtype=object))]:
    row = scenario(new_source, fault=value); cases[label] = {k: v for k, v in row.items() if k != 'report'}
    check(label+' rejected rather than emitted as invalid JSON or null', row['error'] is not None and row['error']['type'] == 'ValueError' and row['report'] is None and 'report.json' not in row['files'])
row = scenario(new_source, fault={1, 2}); cases['unsupported_set'] = {k: v for k, v in row.items() if k != 'report'}
check('unsupported objects still rejected rather than stringified', row['error'] is not None and row['error']['type'] == 'TypeError' and row['report'] is None)
old = scenario(old_source, legacy=True); new = scenario(new_source, legacy=True)
for row in (old, new): row['report'].pop('wall_seconds')
check('legacy finite report values and outcomes unchanged', old == new)

expected = old_source.replace('        (self.out/"report.json").write_text(json.dumps(self.report,indent=2))',
    '        from contact_insertion import _serial\n        (self.out/"report.json").write_text(json.dumps(_serial(self.report),indent=2,allow_nan=False))')
check('production delta only intended report serialization', expected == new_source)
old_tree, new_tree = ast.parse(old_source), ast.parse(new_source)
for tree in (old_tree, new_tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Experiment': node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != 'finish']
check('all non-finish source AST including guards planning and snapshot actions unchanged', ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False))
check('failed full25 original report remains absent', not (RUN/'report.json').exists())
check('frozen run other production defaults and auditor hashes unchanged', all(sha(path) == digest for path, digest in protected.items()))
check('live run remained stable during regression', (ROOT/'run.py').read_text() == new_source)
result = {'kind': 'report_serialization_only_finish_regression', 'physical_success': False, 'native_execution_performed': False,
          'all_checks_passed': all(row['passed'] for row in checks), 'check_count': len(checks), 'checks': checks,
          'failed_checks': [row for row in checks if not row['passed']], 'cases': cases,
          'before_run_sha256': hashlib.sha256(old_source.encode()).hexdigest(), 'after_run_sha256': hashlib.sha256(new_source.encode()).hexdigest(),
          'actual_saved_candidate_sha256': sha(RUN/'gravity_release_candidate.json'), 'protected_input_sha256': protected,
          'scope': 'Real finish method, real saved gravity candidate with restored NumPy containers; synthetic validator and native state fixtures. No simulator or trajectory planning.',
          'limitations': ['Full25 in-memory report fields were lost and have not been reconstructed as original evidence.', 'This fixes future report writing only; full25 physical J failure remains unchanged.']}
out = ROOT/'runtime_ipc/report-serialization-regression.json'
out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'path': str(out), 'sha256': sha(out), 'all_checks_passed': result['all_checks_passed'],
                  'check_count': len(checks), 'failed_checks': result['failed_checks'], 'new_run_sha256': result['after_run_sha256']}, indent=2))
if not result['all_checks_passed']: raise SystemExit(1)
