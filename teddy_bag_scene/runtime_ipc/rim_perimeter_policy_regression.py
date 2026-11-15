"""Bounded CPU geometry/policy regression; no dynamics or production config edits."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import contact_insertion as new
oldpath=ROOT/'runtime_ipc/rim-policy-before-contact_insertion.py';spec=importlib.util.spec_from_file_location('old_contact_rim_policy',oldpath);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();checks={}
def check(name,good):
 checks[name]=bool(good)
 if not good:raise AssertionError(name)
paths=[ROOT/n for n in ['contact_insertion.py','run.py','motion.py','physics_config.py','config.json','config_guarded_upright_offset_trial.json']]+[Path(__file__)];hashes={str(p):sha(p) for p in paths}
base={};fixed={'task':{'contact_trial':{'rim_perimeter_policy':'fixed_mouth_union'}}};opt={'task':{'contact_trial':{'rim_perimeter_policy':'deformable_rim_contact'}}}
p=new._policy(base);po=new._policy(opt);oldp=old._policy(base)
check('default_exact_fixed_mode',p==new._policy(fixed))
check('old_other_policy_values_identical',{k:v for k,v in p.items() if k not in ['rim_perimeter_policy','rim_perimeter_criterion']}==oldp)
check('new_mode_declares_only_buffered_section_criterion',po['rim_perimeter_policy']=='deformable_rim_contact' and po['rim_perimeter_criterion']=='buffered_section_only')
for x in [None,False,0,[],{},'','fixed','DEFORMABLE_RIM_CONTACT',np.str_('deformable_rim_contact')]:
 cfg={'task':{'contact_trial':{'rim_perimeter_policy':x}}}
 try:new._policy(cfg);bad=False
 except ValueError:bad=True
 check('invalid_'+type(x).__name__+'_'+repr(x),bad)
for strategy in ['lower_then_center','unknown',None]:
 cfg=copy.deepcopy(opt);cfg['task']['contact_trial']['insertion_strategy']=strategy
 check('opt_rejects_nonvertical_'+str(strategy),new.plan_guarded_contact_trial({}, {}, cfg)['status']=='rejected')
# Do not quietly reinterpret historical selected metadata under the new schema.
selected={'status':'candidate','report':{'policy':oldp}}
check('historical_missing_policy_tags_rejected',new.reobserve_guarded_contact_trial({}, {}, {}, selected)['status']=='rejected')
selected={'status':'candidate','report':{'policy':p}}
check('fresh_cross_policy_rejected','Do not change contact/physical policy' in new.reobserve_guarded_contact_trial({}, {}, opt, selected)['report']['rejections'][0])
check('raw_route_not_silently_accepted',new.reobserve_guarded_contact_trial({}, {}, {}, {'kind':'guarded_contact_trial'})['status']=='rejected')
for value in [None,[],False]:
 check('malformed_selected_report_'+str(value),new.reobserve_guarded_contact_trial({}, {}, {}, {'report':value})['status']=='rejected')
poly=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]]);tri=new.simple_polygon_triangles(poly);h=np.array([[-.025,-.025],[.025,-.025],[.025,.025],[-.025,.025]])
env=[dict(z_min=0.,z_max=.01,**new.convex_hull_metrics(h))];mouth={'projected_minimum_caliper_width_m':.2,'available_nonstretch_rim_perimeter_m':.70}
oldrow=old._evaluate_offset(env,np.zeros(2),poly,tri,mouth,oldp);default=new._evaluate_offset(env,np.zeros(2),poly,tri,mouth,p);active=new._evaluate_offset(env,np.zeros(2),poly,tri,mouth,po)
def legacy_equal(a,b):
 return all((all(all(row[k]==a['sections'][i][k] for k in a['sections'][i]) for i,row in enumerate(b['sections'])) if key=='sections' else a[key]==b[key]) for key in a)
check('synthetic_default_all_old_values_exact',legacy_equal(oldrow,default))
check('opt_retains_negative_old_union',active['maximum_required_enclosing_rim_perimeter_m']==default['maximum_required_enclosing_rim_perimeter_m'] and active['perimeter_margin_m']==default['perimeter_margin_m']<0 and active['mouth_union_perimeter_rejections']==default['section_guard_rejections'])
check('opt_only_active_section_gate_passes',not active['section_guard_rejections'] and active['active_perimeter_margin_m']>0 and active['mouth_union_perimeter_is_diagnostic_only'])
check('analytic_square_buffer_plus8r',abs(active['maximum_buffered_section_envelope_perimeter_m']-(env[0]['perimeter']+.016))<1e-14)
for delta,expect in [(-1e-10,True),(0.,False),(1e-10,False)]:
 m=dict(mouth,available_nonstretch_rim_perimeter_m=active['active_required_enclosing_rim_perimeter_m']+delta);r=new._evaluate_offset(env,np.zeros(2),poly,tri,m,po)
 check('section_boundary_'+str(delta),bool(r['section_guard_rejections'])==expect)
r=new._evaluate_offset(env,np.array([.2,0]),poly,tri,mouth,po)
check('opt_preserves_lateral_and_overlap_rejections',len(r['section_guard_rejections'])==2 and any('lateral' in x for x in r['section_guard_rejections']) and any('overlap' in x for x in r['section_guard_rejections']))
check('section_only_perimeter_translation_invariant',abs(r['active_required_enclosing_rim_perimeter_m']-active['active_required_enclosing_rim_perimeter_m'])<1e-14)
# All implementation outside the three authorized functions remains AST-identical.
def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef)}
a=funcs(oldpath);b=funcs(ROOT/'contact_insertion.py');check('other_functions_AST_identical',all(a[k]==b[k] for k in a if k not in ['_policy','_evaluate_offset','_plan']))
newplan=next(n for n in ast.walk(ast.parse((ROOT/'contact_insertion.py').read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_plan');del newplan.body[1]
check('plan_changes_only_early_policy_equality',ast.dump(newplan,include_attributes=False)==a['_plan'])
# One actual geometry observation: preserve every old field against its frozen saved report.
RUN=ROOT/'runs/two_panel_full_22';rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'transport_teddy.npz'));cfg=json.loads((RUN/'config.json').read_text());oldactual=json.loads((RUN/'guarded_contact_reobserved.json').read_text())['report']['selected_sections']
polygon,triangles,actualmouth=new._mouth(rest,state,cfg);_,envelopes=new._section_envelopes(state['bear'],rest['bear_tets']);actual=new._evaluate_offset(envelopes,np.zeros(2),polygon,triangles,actualmouth,new._policy(cfg))
check('actual_legacy_all_evaluation_values_exact',legacy_equal({k:v for k,v in oldactual.items() if k not in ['center_of_mass_xy_target_m','com_boundary_distance_m']},actual))
changed=copy.deepcopy(cfg);changed['task']['contact_trial']['rim_perimeter_policy']='deformable_rim_contact';actualopt=new._evaluate_offset(envelopes,np.zeros(2),polygon,triangles,actualmouth,new._policy(changed))
check('actual_opt_union_failure_remains_exact',actualopt['perimeter_margin_m']==oldactual['perimeter_margin_m']<0 and actualopt['mouth_union_perimeter_rejections']==oldactual['section_guard_rejections'])
check('actual_opt_only_perimeter_reason_removed',not actualopt['section_guard_rejections'] and actualopt['active_perimeter_margin_m']>0)
check('every_actual_slab_square_buffer_perimeter',all(abs(x['buffered_section_envelope_perimeter_m']-x['convex_section_slab_perimeter_m']-.016)<1e-12 for x in actualopt['sections']))
check('inputs_unchanged',all(sha(Path(k))==v for k,v in hashes.items()))
out={'kind':'rim_perimeter_policy_targeted_regression','physical_success':False,'passed':all(checks.values()),'check_count':len(checks),'checks':checks,'input_sha256':hashes,'actual_fixed_summary':{k:v for k,v in actual.items() if k!='sections'},'actual_opt_summary':{k:v for k,v in actualopt.items() if k!='sections'},'scope':'Geometry/policy tests only. No robot route, cloth feasibility or actual insertion demonstrated. Historical old-schema selected records are intentionally rejected by the new helper; replay them with frozen source or regenerate preparations, never silently rewrite history.'}
p=ROOT/'runtime_ipc/rim-perimeter-policy-regression.json';p.write_text(json.dumps(out,indent=2,default=new._serial,allow_nan=False)+'\n');print(json.dumps({'path':str(p),'sha256':sha(p),'checks':len(checks),'passed':out['passed'],'legacy_margin_m':actualopt['perimeter_margin_m'],'active_margin_m':actualopt['active_perimeter_margin_m']}))
