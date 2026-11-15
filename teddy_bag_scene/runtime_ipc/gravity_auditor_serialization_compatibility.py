"""Narrow tests of exactly the reviewed output-only runner change."""
from pathlib import Path
import hashlib,json,sys,tempfile,copy
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE));import audit_gravity_execution as a
checks={}
def ck(k,v):
 checks[k]=bool(v)
 if not v:raise AssertionError(k)
base=ROOT/'runs/two_panel_full_20/source/run.py';old=ROOT/'runs/two_panel_full_25/source/run.py';new=ROOT/'run.py';helper=ROOT/'gravity_release.py'
ck('actual_old_source_hash_pinned',a.sha(old)==a.APPROVED['run.py']);ck('actual_new_source_hash_pinned',a.sha(new)==a.SERIALIZATION_ONLY_RUN_SHA256);ck('helper_unchanged',a.sha(helper)==a.APPROVED['gravity_release.py'])
ck('exact_text_only_serialization_delta',new.read_text().count(a.SERIALIZATION_NEW)==1 and new.read_text().replace(a.SERIALIZATION_NEW,a.SERIALIZATION_OLD)==old.read_text())
old_c=a.runner_contract(old,base);new_c=a.runner_contract(new,base)
ck('original_full25_contract_still_accepted',old_c['passed'] and old_c['mode']=='exact_reviewed_gravity_release_extension')
ck('new_contract_requires_serialization_equivalence',new_c['passed'] and new_c['checks']['exact_serialization_only_whole_source_equivalence'])
with tempfile.TemporaryDirectory(prefix='gravity-auditor-serialization-') as temp:
 r=Path(temp);(r/'source').mkdir();(r/'source/gravity_release.py').write_bytes(helper.read_bytes());manifest={'source/gravity_release.py':a.sha(helper)}
 for label,src in [('original',old),('serialization_only',new)]:
  (r/'source/run.py').write_bytes(src.read_bytes());manifest['source/run.py']=a.sha(src);ck(label+'_manifest_matches',a.approved_manifest_sources(r,manifest));bad=copy.deepcopy(manifest);bad['source/run.py']='0'*64;ck(label+'_wrong_manifest_rejected',not a.approved_manifest_sources(r,bad))
 (r/'source/run.py').write_text(new.read_text().replace('self.must_hold_teddy=False','self.must_hold_teddy=True'))
 ck('changed_control_not_accepted',not a.runner_contract(r/'source/run.py',base)['passed'])
 (r/'source/run.py').write_bytes(new.read_bytes());(r/'source/gravity_release.py').write_text(helper.read_text()+'\n# unreviewed\n');ck('changed_helper_not_accepted',not a.runner_contract(r/'source/run.py',base)['passed'])
 (r/'config.json').write_text((ROOT/'config_gravity_seating_trial.json').read_text());missing=a.audit(r);ck('missing_actual_final_evidence_still_incomplete',missing['status']=='incomplete' and missing['passed'] is False and missing['physical_success'] is False)
actual_incomplete=a.audit(ROOT/'runs/two_panel_full_25');ck('original_failed_full25_still_missing_report_incomplete',actual_incomplete['passed'] is False and 'report.json' in actual_incomplete['incomplete_evidence'])
ck('full25_report_not_reconstructed',not (ROOT/'runs/two_panel_full_25/report.json').exists());ck('production_exact_hashes_unchanged_during_audit',a.sha(new)==a.SERIALIZATION_ONLY_RUN_SHA256 and a.sha(helper)==a.APPROVED['gravity_release.py'])
r={'kind':'gravity_auditor_serialization_only_compatibility_review','passed':all(checks.values()),'count':len(checks),'checks':checks,'physical_success':False,'native_execution':False,'old_contract':old_c,'new_contract':new_c,'old_run_sha256':a.sha(old),'new_run_sha256':a.sha(new),'unchanged_helper_sha256':a.sha(helper),'new_auditor_sha256':a.sha(HERE/'audit_gravity_execution.py'),'new_prefix_script_sha256':a.sha(HERE/'gravity_insert_prefix_review.py'),'script_sha256':a.sha(Path(__file__)),'scope':'Only final JSON output normalization changed in the explicitly reviewed second runner; no arbitrary AST normalization or control edit is accepted. Original full25 frozen input and missing report remain untouched.','changes':['Retain original frozen full25 runner hash.','Allow exact serialization-only runner hash only after whole-source reversal equality.','Both candidate/final audit and prefix manifest checks recognize either exact runner plus unchanged gravity helper.']}
out=HERE/'gravity-auditor-serialization-compatibility-review.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'path':str(out),'sha256':a.sha(out),'count':len(checks),'passed':r['passed'],'new_auditor_sha256':r['new_auditor_sha256']},indent=2))
