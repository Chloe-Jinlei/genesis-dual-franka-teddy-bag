"""Synthetic explicit-budget auditor cases; no simulated task success."""
from pathlib import Path
import sys,json,copy,tempfile,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
# Runs the existing 17 sequence regression cases before this added contract.
import staged_sequence_audit_regression as legacy
import audit_guarded_run as audit
from scipy.spatial.transform import Rotation
cfg=copy.deepcopy(legacy.config);cfg['task']['contact_trial']['staged'].update(maximum_individual_xy_m=.07,maximum_cumulative_xy_m=.09,center_offset_xy_m=[-.02,-.05])
r=copy.deepcopy(legacy.report);ss={k:v.copy() for k,v in legacy.states.items()};tt=legacy.trace.copy();targetxy=ss['tcp_right'][1,:2]+[-.02,-.05]
source=legacy.source;motion=audit.load_module(legacy.P/'source/motion.py','budget_fixture_frozen_motion');planner=motion.Planner(cfg);rot=Rotation.from_quat(r['guarded_contact_candidate']['route']['rotation_xyzw'])
for index in [2,3]:
 ss['tcp_right'][index,:2]=targetxy;q,pe,re=planner.solve('right',ss['tcp_right'][index],rot,seed=ss['q_right'][index,:7]);assert pe<.0005 and re<.002;ss['q_right'][index,:7]=q;tt[index,10:19]=ss['q_right'][index]
total=0.
for i,plan in enumerate(r['staged_contact_plans']):
 before=ss['tcp_right'][i];target=ss['tcp_right'][i+1];distance=float(np.linalg.norm(target[:2]-before[:2]));plan['stage']['right_tcp_m']=target.tolist();plan['stage']['right_q_end']=ss['q_right'][i+1,:7].tolist();plan['checkpoint']['observed_tcp_world_m']=before.tolist();plan['checkpoint']['staged_policy'].update(maximum_individual_xy_m=.07,maximum_cumulative_xy_m=.09);plan.update(cumulative_xy_m_before=total,cumulative_xy_m_after=total+distance,runtime_observed_endpoint_displacement_m=distance,runtime_observed_cumulative_endpoint_displacement_m=total+distance);total+=distance
results=[];old_hand=audit.audit_staged_hand;audit.audit_staged_hand=lambda *args,**kw:dict(passed=True,status='synthetic_separately_tested_native_height')
with tempfile.TemporaryDirectory(prefix='staged-budget-audit-') as path:
 run=Path(path);(run/'source/assets').mkdir(parents=True)
 for rel in ['motion.py','assets/panda_tactile.xml']:(run/'source'/rel).write_bytes((legacy.P/'source'/rel).read_bytes())
 (run/'contact_guard_trace.json').write_text(json.dumps([{'time_s':r['guarded_contact_candidate']['observed_time_s']}]))
 def case(name,config=None,report=None,expected=False):
  out=audit.audit_staged_sequence(run,r if report is None else report,ss,tt,cfg if config is None else config,{'passed':True});ok=bool(out['passed'])==expected;results.append(dict(case=name,expected_audit_passed=expected,regression_passed=ok,status=out['status'],failed_checks=[k for k,v in out['checks'].items() if not v],error=out.get('error')));print(name,ok,flush=True)
 case('explicit_70_90mm_allows_declared_53p85mm_center',expected=True)
 for name,individual,cumulative in [('individual_above_ceiling',.10001,.12),('individual_below_floor',.02999,.09),('cumulative_above_ceiling',.07,.12001),('cumulative_below_floor',.03,.04999),('cumulative_below_individual',.10,.09),('nonfinite_individual',float('nan'),.09)]:
  bad=copy.deepcopy(cfg);bad['task']['contact_trial']['staged'].update(maximum_individual_xy_m=individual,maximum_cumulative_xy_m=cumulative);case(name,config=bad)
 badcfg=copy.deepcopy(cfg);badcfg['task']['contact_trial']['staged'].pop('maximum_individual_xy_m');badcfg['task']['contact_trial']['staged'].pop('maximum_cumulative_xy_m');badrep=copy.deepcopy(r)
 for plan in badrep['staged_contact_plans']:plan['checkpoint']['staged_policy'].update(maximum_individual_xy_m=.03,maximum_cumulative_xy_m=.05)
 case('default_30_50mm_still_rejects_large_motion',config=badcfg,report=badrep)
 bad=copy.deepcopy(r);bad['staged_contact_plans'][1]['checkpoint']['staged_policy']['maximum_individual_xy_m']=.10;case('checkpoint_forges_larger_budget',report=bad)
 bad=copy.deepcopy(r);bad['staged_contact_plans'][2]['cumulative_xy_m_before']=0.;case('cumulative_budget_reset_rejected',report=bad)
 badcfg=copy.deepcopy(cfg);badcfg['task']['contact_trial']['staged']['center_offset_xy_m']=[-.02,-.04];case('undeclared_center_target_rejected',config=badcfg)
audit.audit_staged_hand=old_hand
out=dict(kind='synthetic_staged_explicit_xy_budget_audit_regression',physical_success=False,all_checks_passed=all(x['regression_passed'] for x in results),case_count=len(results),cases=results,existing_sequence_regression_all_passed=legacy.result['all_checks_passed'],auditor_sha256=hashlib.sha256((HERE/'audit_guarded_run.py').read_bytes()).hexdigest(),scope='Synthetic sequence data exercise budget-only contract; hand-height and raw native reactions are separately tested and mocked here. No physical success is established.')
(HERE/'staged-budget-audit-regression.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['all_checks_passed'] and out['existing_sequence_regression_all_passed']
