"""Independent CPU checks of rim policy; no robot planning or physics execution."""
from pathlib import Path
import importlib.util,sys,json,hashlib,copy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import contact_insertion as new
from contact_slice_geometry import convex_hull_metrics,simple_polygon_triangles
spec=importlib.util.spec_from_file_location('frozen_full22_contact_review',ROOT/'runs/two_panel_full_22/source/contact_insertion.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source_hash=sha(ROOT/'contact_insertion.py');checks=[]
def check(name,condition,detail=None):
    checks.append({'name':name,'passed':bool(condition),'detail':detail})
def equal(a,b): return json.dumps(new._serial(a),sort_keys=True,allow_nan=False)==json.dumps(new._serial(b),sort_keys=True,allow_nan=False)
def cfg(mode='default',**kw):
    c={'task':{'contact_trial':{'max_lateral_excursion_fraction_of_mouth_width':.8,'minimum_section_envelope_overlap_fraction':.4,**kw}}}
    if mode!='default':c['task']['contact_trial']['rim_perimeter_policy']=mode
    return c
fixed=new._policy(cfg());deform=new._policy(cfg('deformable_rim_contact'));legacy=old._policy(cfg())
check('default_is_fixed',fixed['rim_perimeter_policy']=='fixed_mouth_union')
check('all_old_policy_fields_exact',all(equal(v,fixed[k]) for k,v in legacy.items()))
for value in [None,True,False,0,1,[],{},'','Deformable_rim_contact','deformable_rim_contact ']:
    try:new._policy(cfg(value));ok=False
    except (ValueError,TypeError):ok=True
    check('invalid_policy_'+repr(value),ok)
for strategy in ['staged',None,True,{}]:
    try:new._policy(cfg('deformable_rim_contact',insertion_strategy=strategy));ok=False
    except (ValueError,TypeError):ok=True
    check('nonvertical_optin_'+repr(strategy),ok)

def rect(w,h): return np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
def env(points,z=0):return {'z_min':float(z),'z_max':float(z+.001),**convex_hull_metrics(points)}
def evaluate(envelopes,policy,poly=None,length=.9152328087150292,delta=(0,0),module=new):
    if poly is None:poly=rect(.32,.12)
    mouth={'available_nonstretch_rim_perimeter_m':length,'projected_minimum_caliper_width_m':float(np.ptp(poly[:,1]))}
    return module._evaluate_offset(envelopes,np.array(delta,dtype=float),poly,simple_polygon_triangles(poly),mouth,policy)
examples=[]
for label,points,expected in [
 ('rectangle',rect(.08,.04),.24+.016),
 ('tiny_rectangle',rect(2e-6,1e-6),6e-6+.016),
 ('very_thin_rectangle',rect(.4,2e-6),.800004+.016),
 ('long_thin_rectangle',rect(.5,.0002),1.0004+.016),
 ('right_triangle',np.array([[0.,0.],[.06,0.],[0.,.08]]),.24+.016)]:
    ee=[env(points)]
    rr=evaluate(ee,deform);ff=evaluate(ee,fixed);oo=evaluate(ee,legacy,module=old)
    actual=rr['maximum_buffered_section_envelope_perimeter_m']
    check(label+'_analytic_square_buffer',abs(actual-expected)<2e-12,{'expected':expected,'actual':actual})
    check(label+'_legacy_aggregate_exact',all(equal(v,ff[k]) for k,v in oo.items() if k!='sections'))
    check(label+'_legacy_rows_exact',all(all(equal(v,n[k]) for k,v in o.items()) for o,n in zip(oo['sections'],ff['sections'])))
    check(label+'_union_dominates_section',ff['maximum_required_enclosing_rim_perimeter_m']+1e-12>=actual)
    check(label+'_other_gates_identical', [s for s in rr['section_guard_rejections'] if 'perimeter' not in s]==[s for s in ff['section_guard_rejections'] if 'perimeter' not in s])
    examples.append({'name':label,'new_section_m':actual,'old_union_m':ff['maximum_required_enclosing_rim_perimeter_m'],'new_rejections':rr['section_guard_rejections']})
    if label=='long_thin_rectangle':check('long_thin_too_long_is_rejected',any('buffered-section perimeter' in s for s in rr['section_guard_rejections']))
    if label=='tiny_rectangle':check('tiny_positive_area_is_not_ignored',actual>.016 and rr['minimum_section_slab_overlap_fraction']>.999)
# Analytic square Minkowski buffer adds 8r to every positive-area convex polygon,
# including arbitrary rotations, instead of a mistaken 2*pi*r circle perimeter.
tri=np.array([[0.,0.],[.06,0.],[0.,.08]])
for degrees in [17.,45.,89.]:
    angle=np.deg2rad(degrees);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    rr=evaluate([env(tri@rotation.T)],deform)
    check('rotated_triangle_buffer_'+str(degrees),abs(rr['maximum_buffered_section_envelope_perimeter_m']-.256)<2e-12)
# Borderline +/- 1nm tests ensure active comparison uses buffered, not unbuffered, metric.
ee=[env(rect(.08,.04))];base=evaluate(ee,deform)['maximum_buffered_section_envelope_perimeter_m']
for change,expected in [(-1e-9,True),(0.,False),(1e-9,False)]:
    rr=evaluate(ee,deform,length=base+change)
    check('strict_perimeter_boundary_'+str(change),any('buffered-section perimeter' in s for s in rr['section_guard_rejections'])==expected)
# Every slab matters; an extremal long-thin middle slab cannot be replaced by max area.
ee=[env(rect(.2,.1),0),env(rect(.5,.0002),1),env(rect(.06,.06),2)]
rr=evaluate(ee,deform)
check('all_slabs_max_not_area_or_first_last',abs(rr['maximum_buffered_section_envelope_perimeter_m']-1.0164)<2e-12)
# Translation changes overlap/union but MUST NOT change section-only perimeter.
ee=[env(rect(.2,.2))];center=evaluate(ee,deform,poly=rect(.4,.12),length=1.04)
shift=evaluate(ee,deform,poly=rect(.4,.12),length=1.04,delta=(.08,.015))
check('new_perimeter_translation_invariant',abs(center['active_required_enclosing_rim_perimeter_m']-shift['active_required_enclosing_rim_perimeter_m'])<1e-12)
check('legacy_negative_margin_retained',center['perimeter_margin_m']<0 and center['mouth_union_perimeter_rejections'] and center['mouth_union_perimeter_is_diagnostic_only'])
check('new_active_margin_separate',center['active_perimeter_margin_m']>0 and not center['section_guard_rejections'])
far=evaluate([env(rect(.02,.02))],deform,delta=(.35,0))
check('small_perimeter_does_not_bypass_overlap_or_lateral',far['active_perimeter_margin_m']>0 and len(far['section_guard_rejections'])>=2)
# Public fresh contract rejects changed and malformed policy before geometry work.
for label,selected in [
 ('missing_policy',{'report':{}}),('wrong_policy',{'report':{'policy':deform}}),
 ('report_null',{'report':None}),('report_list',{'report':[]}),('nonmapping_record',[])]:
    try:
        result=new.reobserve_guarded_contact_trial({}, {}, cfg(), selected)
        ok=result['status']=='rejected' and result.get('route') is None and result.get('stages_to_execute')==[]
        detail=result.get('report')
    except Exception as exc:
        ok=False;detail={'uncaught':type(exc).__name__,'message':str(exc)}
    check('fresh_'+label,ok,detail)
# Invalid tetra-volume cases are rejected upstream, not assigned zero-size perimeter.
for label,points in [('coplanar',np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)),('nonfinite',np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,np.nan]],float))]:
    try:new._section_envelopes(points,np.array([[0,1,2,3]]));ok=False
    except ValueError:ok=True
    check('invalid_tet_'+label,ok)
check('source_stable_during_review',sha(ROOT/'contact_insertion.py')==source_hash)
report={'scope':'Independent analytic and synthetic geometry/contract tests only; no native or robot route planning. No physical-fit assertion.',
 'production_contact_insertion_sha256':source_hash,'frozen_legacy_sha256':sha(ROOT/'runs/two_panel_full_22/source/contact_insertion.py'),
 'passed':all(c['passed'] for c in checks),'checks_passed':sum(c['passed'] for c in checks),'checks_total':len(checks),'checks':checks,'examples':examples,
 'important_limits':['The new perimeter is invariant to XY translation: it constrains section size, not alignment. Existing COM/overlap/lateral/height/rigid/soft-sweep/force gates remain necessary.',
 'Thin-slab envelopes cover intermediate horizontal sections, but independent sections do not prove coupled passage through a nonplanar rim, clearance from cloth, 3D containment or dynamically attainable reshaping.',
 'Continue selected-route continuous held-front and pre-rim handle checks; keep permitted later free-back contact explicit. Native force, J, cloth stretch, table and final released containment determine success.'],
 'physical_success':False}
out=ROOT/'runtime_ipc/independent-deformable-rim-policy-review.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'path':str(out),'source_sha':source_hash,'passed':report['passed'],'count':[report['checks_passed'],len(checks)],'failures':[c for c in checks if not c['passed']]},indent=2))
