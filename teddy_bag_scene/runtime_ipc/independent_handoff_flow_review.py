"""Isolated AST-only handoff branch/finish review; no Genesis import or execution."""
import ast, copy, hashlib, json, tempfile, time, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'run.py';before=sha(source)
tree=ast.parse(source.read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Experiment')
methods={n.name:n for n in klass.body if isinstance(n,ast.FunctionDef)}
ns=dict(np=np,json=json,time=time,GateFailure=RuntimeError,DOWN='DOWN')
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Subject',bases=[],keywords=[],body=[methods['run'],methods['finish']],decorator_list=[])],type_ignores=[])),str(source),'exec'),ns)
Subject=ns['Subject'];checks=[]
def check(name,v):checks.append(dict(name=name,passed=bool(v)))
def obj(task='handoff'):
 o=Subject();o.args=SimpleNamespace(task=task);o.config={'task':{}};o.report={};o.events=[];o.robots={'left':None,'right':None};o.bag=object();o.must_hold_bag=False;o.must_hold_right_handle=False
 o.hold=lambda label,duration,**f:o.events.append(dict(phase=label,duration=duration,fingers=f))
 o.move=lambda label,targets,duration,**kw:o.events.append(dict(phase=label,duration=duration))
 o.near_material=lambda *args:{'near_vertices':2,'nearest_m':.001}
 o.orient_held_bag=lambda target:o.events.append(dict(phase='optional_orient'))
 return o
o=obj();o.run();phases=[e['phase'] for e in o.events]
check('exact handoff sequence and early return',phases==['close_both_handles','drape_under_gravity','spread_bag','optional_orient','release_right_handle','right_clear_handle','settle_handoff'])
check('ten second hold',o.events[-1]['duration']==10.)
check('right release command opens both native finger targets',o.events[-3]['fingers']=={'right':.04})
check('left hold monitor stays active and right prehandoff contact requirement disabled',o.must_hold_bag and not o.must_hold_right_handle)
check('no full task completion claim',o.report.get('handoff_probe_completed') is True and not o.report.get('sequence_completed',False))
o=obj('open');o.run();check('open still returns before handoff',not any(e['phase']=='release_right_handle' for e in o.events))
o=obj();oldhold=o.hold
def failhold(label,duration,**f):
 if label=='settle_handoff':raise RuntimeError('synthetic native gate')
 oldhold(label,duration,**f)
o.hold=failhold
try:o.run()
except RuntimeError:pass
check('aborted hold never marks completed',not o.report.get('handoff_probe_completed',False))
fn=methods['finish'];gate=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='physical_keys' for t in n.targets))
keys=ast.literal_eval(gate.value)
# Exercise exactly production finish method with a synthetic audit dependency.
synthetic=types.ModuleType('validate');sys.modules['validate']=synthetic
results=[]
def finish_case(name,key=None,right=(.04,.04),completed=True,error=None,audit_error=False,task='handoff'):
 with tempfile.TemporaryDirectory() as tmp:
  o=obj(task);o.out=Path(tmp);o.writer=SimpleNamespace(close=lambda:None);o.time=12.;o.wall=time.monotonic();o.frames=[{'time':0.}]
  o.joint_trace=np.zeros((4,19));o.joint_trace[:,0]=[.005,.010,.015,.020];o.joint_trace[-1,17:19]=right
  o.soft_metrics=[];o.physical_monitor=SimpleNamespace(columns=[]);o.staged_hand_monitor=None;o.contact_guard=None;o.handle_grip_trace=[];o.coupling_trace=[];o.rest={}
  o.report={'task_passed':False,'handoff_probe_completed':completed};a={k:True for k in keys}
  if key:a[key]=False
  def audit(*args,**kwargs):
   if audit_error:raise RuntimeError('synthetic audit failure')
   return {'task_success':False,'checks':a}
  synthetic.audit_arrays=audit;o.finish(error)
  results.append(dict(case=name,requested_passed=o.report['requested_task_passed'],handoff_passed=o.report.get('handoff_probe_passed',False),status=o.report['status'],full_task_passed=o.report['task_passed']))
  return o.report
r=finish_case('valid isolated contract');check('handoff success scoped without full task success',r['requested_task_passed'] and not r['task_passed'])
for key in keys:
 r=finish_case('missing gate '+key,key=key);check('gate fails closed '+key,not r['requested_task_passed'])
for name,kwargs in [('incomplete',{'completed':False}),('native failure',{'error':'synthetic native failure'}),('audit failure',{'audit_error':True}),('right finger1 closed',{'right':(.037,.04)}),('right finger2 closed',{'right':(.04,.037)}),('right nonfinite',{'right':(np.nan,.04)}),('full does not reuse probe success',{'task':'full'})]:
 r=finish_case(name,**kwargs);check(name,not r['requested_task_passed'])
check('right threshold inclusive',finish_case('exact threshold',right=(.038,.038))['requested_task_passed'])
# Production diff is narrowly task-guarded; normalize removal of handoff-only AST.
old=ast.parse((ROOT/'runs/two_panel_full_21/source/run.py').read_text())
class StripHandoff(ast.NodeTransformer):
 def visit_If(self,node):
  if ast.unparse(node.test) in ("self.args.task == 'handoff'",):
   return self.visit(node.orelse[0]) if node.orelse else None
  return self.generic_visit(node)
 def visit_Tuple(self,node):
  node.elts=[e for e in node.elts if not (isinstance(e,ast.Constant) and e.value=='handoff')];return self.generic_visit(node)
normalized=StripHandoff().visit(copy.deepcopy(tree));check('removing only handoff guards gives exact prior full21 AST',ast.dump(normalized,include_attributes=False)==ast.dump(old,include_attributes=False))
oldcfg=json.loads((ROOT/'config_guarded_small_teddy_vertical_trial.json').read_text());cfg=json.loads((ROOT/'config_upright_handle_probe.json').read_text())
def diffs(a,b,p=''):
 if isinstance(a,dict) and isinstance(b,dict):return sum((diffs(a.get(k,'<missing>'),b.get(k,'<missing>'),p+'/'+k) for k in sorted(set(a)|set(b))),[])
 return [] if a==b else [dict(path=p,before=a,after=b)]
changes=diffs(oldcfg,cfg)
check('config only initial approach and optional held orientation changes',set(c['path'] for c in changes)<= {'/task/handle_grasps/left/preferred_approach','/task/handle_grasps/right/preferred_approach','/task/left_hold_preferred_approach','/task/left_hold_rotation_duration_s'})
check('production source stable during audit',sha(source)==before)
report=dict(kind='independent_handoff_AST_only_flow_review',native_execution=False,physical_success=False,source_sha256=before,config_sha256=sha(ROOT/'config_upright_handle_probe.json'),checks=checks,passed=all(c['passed'] for c in checks),count=len(checks),failures=[c['name'] for c in checks if not c['passed']],finish_cases=results,config_diff_from_full21=changes,scope=['Production run/finish AST only, synthetic audit output; no physics claim.','Actual left named-material and PD-clamp checks retain existing saved-frame scope. Physical metrics remain per-step in unchanged observe_physics.','Right actual fingers are trace columns 17 and 18, tested at final sample.'],existing_final_contact_gate_in_probe_scope='no_final_sampled_bag_bear_penetration_over_1mm' in keys)
out=ROOT/'runtime_ipc/independent-handoff-flow-review.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['passed','count','failures','source_sha256','existing_final_contact_gate_in_probe_scope']},indent=2));print(out)
