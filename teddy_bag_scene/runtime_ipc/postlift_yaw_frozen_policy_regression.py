"""Frozen 15/30-degree policy recognition, independent of live helper imports."""
from pathlib import Path
import copy,hashlib,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime_ipc'))
import audit_guarded_run as audit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
legacy=ROOT/'runs/two_panel_full_21'
old=(legacy/'source/postlift_yaw.py').read_text();new=(ROOT/'postlift_yaw.py').read_text()
checks=[]
def check(name,ok):checks.append(dict(name=name,passed=bool(ok)))
with tempfile.TemporaryDirectory(prefix='frozen-yaw-policy-') as directory:
    run=Path(directory);(run/'source/assets').mkdir(parents=True)
    cfg=json.loads((legacy/'config.json').read_text())
    required=['source/postlift_yaw.py','source/motion.py','source/run.py','config.json','source/assets/panda_tactile.xml','source/assets/'+Path(cfg['bag']['npz_file']).name]
    for name in required:
        if name not in ('config.json','source/postlift_yaw.py'):(run/name).write_bytes((legacy/name).read_bytes())
    (run/'events.json').write_text('[]')
    def policy(source):
        (run/'source/postlift_yaw.py').write_text(source)
        return audit.frozen_postlift_yaw_policy(run)
    for label,source,limit in [('old',old,15.),('new',new,30.)]:
        p=policy(source);check(label+'_exact_static_boundary',p['maximum_absolute_world_z_degrees']==limit and p['minimum_duration_s']==6.)
        for angle in (-30.001,-30.,-15.1,-15.,0.,15.,15.1,30.,30.001):
            current=copy.deepcopy(cfg);current['task']['contact_trial']['postlift_yaw']={'world_z_degrees':angle,'duration_s':6.}
            (run/'config.json').write_text(json.dumps(current))
            (run/'input_sha256.json').write_text(json.dumps({n:sha(run/n) for n in required}))
            out=audit.audit_postlift_yaw(run,{'events':[]},{},None,current,{})
            admitted=bool(out['checks'].get('frozen_optional_config_valid'))
            check(f'{label}_{angle:g}_bound',admitted==(0<abs(angle)<=limit))
            check(f'{label}_{angle:g}_no_execution_no_pass',out['passed'] is False)
    invalid={
        'unrecognized_literal30':old.replace('abs(degrees) <= 15','abs(degrees) <= 30'),
        'constant60':new.replace('MAX_POSTLIFT_YAW_DEGREES = 30','MAX_POSTLIFT_YAW_DEGREES = 60'),
        'constant_boolean':new.replace('MAX_POSTLIFT_YAW_DEGREES = 30','MAX_POSTLIFT_YAW_DEGREES = True'),
        'constant_expression':new.replace('MAX_POSTLIFT_YAW_DEGREES = 30','MAX_POSTLIFT_YAW_DEGREES = 15+15'),
        'constant_reassigned':new+'\nMAX_POSTLIFT_YAW_DEGREES = 60\n',
        'duration_weakened':new.replace('or duration < 6:','or duration < 5:'),
        'nonzero_removed':new.replace('not 0 < abs(degrees)','not -1 < abs(degrees)'),
        'guard_raises_removed':new.replace('raise ValueError(f"postlift yaw requires','print(f"postlift yaw requires'),
        'guard_nested_dead':new.replace('        if not 0 < abs(degrees) <= MAX_POSTLIFT_YAW_DEGREES or duration < 6:\n            raise ValueError(f"postlift yaw requires nonzero |degrees|<={MAX_POSTLIFT_YAW_DEGREES} and duration>=6s")',
            '        if False:\n            if not 0 < abs(degrees) <= MAX_POSTLIFT_YAW_DEGREES or duration < 6:\n                raise ValueError("dead guard")'),
    }
    for name,source in invalid.items():
        try:policy(source);rejected=False
        except (ValueError,SyntaxError):rejected=True
        check(name+'_rejected',rejected)
    check('frozen_module_never_executed',policy(new+'\nraise RuntimeError("must never execute frozen helper")\n')['maximum_absolute_world_z_degrees']==30.)
    # Real archived runs retain their exact policy even when the live module is 30.
    for number in (20,21):
        p=audit.frozen_postlift_yaw_policy(ROOT/f'runs/two_panel_full_{number}')
        check(f'actual_full{number}_policy_remains15',p['maximum_absolute_world_z_degrees']==15. and p['source_form']=='legacy_literal_15')
    prefix=(ROOT/'runtime_ipc/postlift_yaw_prefix_review.py').read_text()
    check('prefix_uses_same_frozen_AST_reader','frozen_policy = audit.frozen_postlift_yaw_policy(run)' in prefix and 'abs(angle) <= frozen_policy["maximum_absolute_world_z_degrees"]' in prefix)
result=dict(kind='frozen_postlift_yaw_policy_boundary_regression',physical_success=False,all_checks_passed=all(c['passed'] for c in checks),
    check_count=len(checks),checks=checks,failed_checks=[c for c in checks if not c['passed']],
    source_sha256={str(p):sha(p) for p in (ROOT/'postlift_yaw.py',ROOT/'runtime_ipc/audit_guarded_run.py',ROOT/'runtime_ipc/postlift_yaw_prefix_review.py',Path(__file__))})
out=ROOT/'runtime_ipc/postlift-yaw30-frozen-policy-regression.json';out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if not result['all_checks_passed']:raise SystemExit(1)
