"""Narrow CPU contract/policy regression; never imports or executes a frozen runner."""
from pathlib import Path
import ast,copy,hashlib,json,tempfile
import numpy as np
import audit_guarded_run as audit
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
def check(name,good):
 checks[name]=bool(good)
 if not good:raise AssertionError(name)
base=ROOT/'runs/two_panel_full_20/source/run.py';live=ROOT/'run.py';source=live.read_text()
paths=[live,ROOT/'postlift_yaw.py',ROOT/'motion.py',ROOT/'config.json',ROOT/'config_guarded_upright_offset_trial.json',ROOT/'runtime_ipc/audit_guarded_run.py',ROOT/'runtime_ipc/postlift_yaw_prefix_review.py',Path(__file__)]
before={str(p):sha(p) for p in paths}
for name,path in [('full20',base),('full21',ROOT/'runs/two_panel_full_21/source/run.py'),('probe',ROOT/'runs/two_panel_handoff_upright_offset_01/source/run.py')]:
 r=audit.frozen_runner_yaw_contract(path,base);check(name+'_legacy_exact',r['passed'] and r['mode']=='legacy_exact')
r=audit.frozen_runner_yaw_contract(live,base);check('new_runner_only_exact_five_lines_normalized',r['passed'] and r['mode']=='reviewed_recenter_disabled_extension')
entry='        recenter_enabled=self.config.get("task",{}).get("contact_trial",{}).get("recenter_enabled",True)\n        if type(recenter_enabled) is not bool:\n            raise GateFailure("contact_trial.recenter_enabled must be a boolean")\n'
stop='            if not recenter_enabled:\n                raise GateFailure("Fresh contact insertion rejected and recentering is disabled; see "+filename)\n'
accepted='            if fresh["status"]=="candidate":\n                break\n'
mutations={
 'default_false':source.replace('get("recenter_enabled",True)','get("recenter_enabled",False)'),
 'weak_type_check':source.replace('type(recenter_enabled) is not bool','not isinstance(recenter_enabled,(bool,int))'),
 'missing_disabled_stop':source.replace(stop,''),
 'stop_before_fresh_acceptance':source.replace(accepted+stop,stop+accepted),
 'changed_stop_behavior':source.replace('if not recenter_enabled:','if recenter_enabled:'),
 'altered_original_monitor':source.replace('    def begin_guarded_contact_monitor(self):','    def begin_guarded_contact_monitor(self):\n        pass'),
}
with tempfile.TemporaryDirectory(prefix='recenter-audit-contract-') as temp:
 for name,text in mutations.items():
  assert text!=source
  p=Path(temp)/(name+'.py');p.write_text(text);r=audit.frozen_runner_yaw_contract(p,base);check(name+'_rejected',not r['passed'])
 p=Path(temp)/'comments.py';p.write_text(source+'\n# harmless comment\n');check('comments_do_not_change_contract',audit.frozen_runner_yaw_contract(p,base)['passed'])

cfg={'task':{'contact_trial':{'recenter_enabled':False}}};empty={'events':[],'guarded_contact_corrections':[]}
check('disabled_empty_valid',audit.audit_recenter_policy(cfg,empty)['passed'])
for status in ['candidate','rejected']:
 r=copy.deepcopy(empty);r['guarded_contact_corrections']=[{'status':status}]
 check('disabled_'+status+'_correction_rejected',not audit.audit_recenter_policy(cfg,r)['passed'])
for phase in ['recenter_height_1','recenter_teddy_xy_1']:
 r=copy.deepcopy(empty);r['events']=[{'phase':phase}]
 check('disabled_'+phase+'_rejected',not audit.audit_recenter_policy(cfg,r)['passed'])
for value in [None,0,1,'false',[],{},np.bool_(False)]:
 c=copy.deepcopy(cfg);c['task']['contact_trial']['recenter_enabled']=value
 check('invalid_type_'+type(value).__name__+'_'+str(value)+'_rejected',not audit.audit_recenter_policy(c,empty)['passed'])
legacy={'events':[{'phase':'recenter_height_1'}],'guarded_contact_corrections':[{'status':'candidate'}]}
check('default_true_preserves_existing_policy',audit.audit_recenter_policy({},legacy)['passed'])
check('explicit_true_preserves_existing_policy',audit.audit_recenter_policy({'task':{'contact_trial':{'recenter_enabled':True}}},legacy)['passed'])
for name in ['two_panel_full_20','two_panel_full_21','two_panel_handoff_upright_offset_01']:
 run=ROOT/'runs'/name;check(name+'_historical_saved_policy_passes',audit.audit_recenter_policy(json.loads((run/'config.json').read_text()),json.loads((run/'report.json').read_text()))['passed'])
# The old correction sequence function itself remains AST-identical.
def function(path,name):
 return ast.dump(next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
check('legacy_correction_sequence_algorithm_unchanged',function(ROOT/'runtime_ipc/recenter-before-audit_guarded_run.py','audit_corrections')==function(ROOT/'runtime_ipc/audit_guarded_run.py','audit_corrections'))
check('all_inputs_unchanged',all(sha(Path(p))==h for p,h in before.items()))
r={'kind':'recenter_static_contract_and_frozen_policy_regression','physical_success':False,'passed':all(checks.values()),'check_count':len(checks),'checks':checks,'input_sha256':before,'scope':'No native simulation, geometry planning, or task success assertion.'}
p=ROOT/'runtime_ipc/recenter-audit-policy-regression.json';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'checks':len(checks),'passed':r['passed']}))
