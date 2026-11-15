"""Optional early-yaw baseline integrates with the existing staged auditor.

Synthetic baseline records; hand/force evidence is covered by separate native
trace audits. No actual yaw, contact, release or task success is asserted here.
"""
from pathlib import Path
import copy,json,sys,tempfile,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime_ipc'))
import staged_sequence_audit_regression as legacy
import audit_guarded_run as audit
cfg=copy.deepcopy(legacy.config);report=copy.deepcopy(legacy.report)
cfg['task']['contact_trial']['postlift_yaw']={'world_z_degrees':15.,'duration_s':6.}
start=report['guarded_contact_candidate']['observed_time_s']-6.025
report['postlift_yaw']={'observed_time_s':start}
report['contact_guard_baseline']={'time_s':start};report['contact_guard']['baseline']={'time_s':start}
checks=[];old=audit.audit_staged_hand
audit.audit_staged_hand=lambda *a,**kw:dict(passed=True,status='synthetic_separately_tested_hand')
with tempfile.TemporaryDirectory(prefix='yaw-staged-baseline-') as path:
 run=Path(path);(run/'source/assets').mkdir(parents=True)
 for name in ['motion.py','assets/panda_tactile.xml']:(run/'source'/name).write_bytes((legacy.P/'source'/name).read_bytes())
 def case(name,r,guardtime,expected):
  (run/'contact_guard_trace.json').write_text(json.dumps([{'time_s':guardtime}]))
  out=audit.audit_staged_sequence(run,r,legacy.states,legacy.trace,cfg,{'passed':True})
  checks.append(dict(name=name,passed=out['passed']==expected,audit_passed=out['passed'],failed_checks=[k for k,v in out['checks'].items() if not v]))
 case('single_guard_from_before_yaw_preserved_through_stages',report,start,True)
 bad=copy.deepcopy(report);bad['contact_guard_baseline']={'time_s':start+6.025};bad['contact_guard']['baseline']={'time_s':start+6.025}
 case('guard_reset_at_postyaw_prep_is_rejected',bad,start+6.025,False)
 bad=copy.deepcopy(report);bad.pop('postlift_yaw')
 case('missing_actual_yaw_observation_rejected',bad,start,False)
 bad=copy.deepcopy(report);bad['contact_guard']['baseline']={'time_s':start+.005}
 case('final_baseline_changed_rejected',bad,start,False)
audit.audit_staged_hand=old
result=dict(kind='optional_yaw_staged_guard_baseline_regression',physical_success=False,all_checks_passed=all(v['passed'] for v in checks),checks=checks,auditor_sha256=hashlib.sha256((ROOT/'runtime_ipc/audit_guarded_run.py').read_bytes()).hexdigest())
(ROOT/'runtime_ipc/postlift-yaw-staged-baseline-regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if not result['all_checks_passed']:raise SystemExit(1)
