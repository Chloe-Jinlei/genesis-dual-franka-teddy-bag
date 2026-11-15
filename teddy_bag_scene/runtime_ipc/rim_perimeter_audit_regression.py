"""Frozen-source/report criterion regression; no simulation or source execution."""
from pathlib import Path
import copy,hashlib,json,tempfile,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime_ipc'));import audit_guarded_run as audit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks={};summaries={}
def check(name,good):
 checks[name]=bool(good)
 if not good:raise AssertionError(name)
for runname in ['two_panel_full_17','two_panel_full_20','two_panel_full_21','two_panel_full_22']:
 r=ROOT/'runs'/runname;result=audit.audit_rim_perimeter_policy(r,json.loads((r/'report.json').read_text()),json.loads((r/'config.json').read_text()));summaries[runname]=result
 check(runname+'_historical_fixed_union',result['passed'] and result['mode']=='fixed_mouth_union' and result['source_form']=='legacy_union_only')
cfg=json.loads((ROOT/'config_guarded_deformable_rim_trial.json').read_text());candidate=json.loads((ROOT/'runs/two_panel_full_22/independent-counterfactual-deformable-rim/counterfactual-reobserved-candidate.json').read_text());source=(ROOT/'contact_insertion.py').read_text()
paths=[ROOT/'contact_insertion.py',ROOT/'run.py',ROOT/'motion.py',ROOT/'config_guarded_deformable_rim_trial.json',ROOT/'runtime_ipc/audit_guarded_run.py',Path(__file__)];before={str(p):sha(p) for p in paths}
with tempfile.TemporaryDirectory(prefix='rim-policy-audit-') as tmp:
 run=Path(tmp);(run/'source').mkdir();helper=run/'source/contact_insertion.py';helper.write_text(source)
 baseline=audit.audit_rim_perimeter_policy(run,{'guarded_contact_reobserved':candidate},cfg);summaries['actual_state_counterfactual']=baseline
 check('counterfactual_opt_with_negative_legacy_margin_passes',baseline['passed'] and baseline['mode']=='deformable_rim_contact' and candidate['report']['selected_sections']['perimeter_margin_m']<0)
 mutations={}
 x=copy.deepcopy(candidate);x['report']['policy']['rim_perimeter_policy']='fixed_mouth_union';mutations['wrong_report_mode']=x
 x=copy.deepcopy(candidate);x['report']['policy']['rim_perimeter_criterion']='mouth_and_buffered_section_union';mutations['wrong_report_criterion']=x
 x=copy.deepcopy(candidate);x['report']['policy']['rigid_clearance_m']=.007;mutations['changed_other_original_policy']=x
 x=copy.deepcopy(candidate);del x['report']['policy'];mutations['missing_policy']=x
 x=copy.deepcopy(candidate);x['report']['selected_sections']['perimeter_margin_m']=0.;mutations['erase_negative_union']=x
 x=copy.deepcopy(candidate);x['report']['selected_sections']['mouth_union_perimeter_rejections']=[];mutations['erase_union_diagnostic']=x
 x=copy.deepcopy(candidate);x['report']['selected_sections']['active_required_enclosing_rim_perimeter_m']+=.01;mutations['forged_active_value']=x
 x=copy.deepcopy(candidate);del x['report']['selected_sections']['active_rim_perimeter_policy'];mutations['missing_active_policy']=x
 x=copy.deepcopy(candidate);x['report']['selected_sections']['sections'][0]['buffered_section_envelope_perimeter_m']+=.001;mutations['forged_slab_buffer']=x
 x=copy.deepcopy(candidate);x['report']['selected_sections']['active_perimeter_margin_m']=float('nan');mutations['nonfinite_active_margin']=x
 x=copy.deepcopy(candidate);x['report']['rejections']=['geometry refused'];mutations['candidate_with_rejections']=x
 for name,c in mutations.items():
  result=audit.audit_rim_perimeter_policy(run,{'guarded_contact_reobserved':c},cfg);check(name+'_rejected',not result['passed'])
 c=copy.deepcopy(cfg);c['task']['contact_trial']['rim_perimeter_policy']='fixed_mouth_union';check('frozen_config_fixed_cannot_accept_opt_report',not audit.audit_rim_perimeter_policy(run,{'guarded_contact_reobserved':candidate},c)['passed'])
 c=copy.deepcopy(cfg);c['task']['contact_trial']['insertion_strategy']='lower_then_center';check('opt_nonvertical_rejected',not audit.audit_rim_perimeter_policy(run,{},c)['passed'])
 for val in [None,True,'invalid']:
  c=copy.deepcopy(cfg);c['task']['contact_trial']['rim_perimeter_policy']=val;check('invalid_config_'+str(val),not audit.audit_rim_perimeter_policy(run,{},c)['passed'])
 helper.write_text((ROOT/'runs/two_panel_full_22/source/contact_insertion.py').read_text());check('legacy_source_cannot_accept_opt',not audit.audit_rim_perimeter_policy(run,{'guarded_contact_reobserved':candidate},cfg)['passed'])
 helper.write_text(source.replace('else section_perimeter','else perimeter'));check('unreviewed_source_arithmetic_rejected',not audit.audit_rim_perimeter_policy(run,{'guarded_contact_reobserved':candidate},cfg)['passed'])
check('all_inputs_unchanged',all(sha(Path(p))==h for p,h in before.items()))
out={'kind':'rim_perimeter_frozen_policy_auditor_regression','physical_success':False,'passed':all(checks.values()),'checks':checks,'check_count':len(checks),'input_sha256':before,'baseline_results':summaries,'scope':'Reports/AST/arithmetic only; counterfactual full22 evaluation is not a native executed record.'};p=ROOT/'runtime_ipc/rim-perimeter-audit-regression.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'checks':len(checks),'passed':out['passed']}))
