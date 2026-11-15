"""Targeted CPU regression for continuous minimum-jerk cubic retiming."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion import MotionPlan, Planner, _real_polynomial_roots_unit

checks=[];details=[]
def check(name, condition):
    checks.append(dict(name=name,passed=bool(condition)))
    if not condition: print("FAIL",name,flush=True)

def plan(x=None,y=None,T=1.):
    x=np.asarray([0.,1.] if x is None else x)
    y=np.zeros((len(x),7)) if y is None else np.asarray(y)
    return MotionPlan({s:CubicSpline(x,y.copy(),axis=0) for s in ("left","right")},T)

def dense(m,n=50001):
    """Independent chain-rule check, including all inverse-clock knot times."""
    inv=lambda s:0. if s<=0 else 1. if s>=1 else brentq(lambda u:10*u**3-15*u**4+6*u**5-s,0.,1.,xtol=5e-15)
    u=np.unique(np.r_[np.linspace(0.,1.,n),*[np.array([inv(x) for x in c.x]) for c in m.curves.values()]])
    s=10*u**3-15*u**4+6*u**5;su=30*u**2-60*u**3+30*u**4;suu=60*u-180*u**2+120*u**3
    v=a=0.
    for c in m.curves.values():
        v=max(v,float(np.abs(c(s,1)*su[:,None]/m.duration).max()))
        a=max(a,float(np.abs(c(s,2)*(su[:,None]/m.duration)**2+c(s,1)*suu[:,None]/m.duration**2).max()))
    return v,a

def maxima(m):
    rows=m.continuous_rate_extrema()
    return (max(float(v['maximum_velocity_rad_s'].max()) for v in rows.values()),
            max(float(v['maximum_acceleration_rad_s2'].max()) for v in rows.values()))

def rejects(name, callback, instance=None):
    before=instance.duration if instance is not None else None
    try:callback()
    except ValueError:check(name,True)
    except Exception as e:check(name+" wrong exception "+type(e).__name__,False)
    else:check(name,False)
    if instance is not None:check(name+" duration remains unchanged",instance.duration==before)

start=time.perf_counter()
for duration in (.125,1.,3.7,128.):
    y=np.zeros((2,7));y[1,0]=.23
    m=plan(y=y,T=duration);v,a=maxima(m)
    check(f"linear analytic speed T={duration}",math.isclose(v,.23*15/8/duration,rel_tol=2e-12,abs_tol=1e-13))
    check(f"linear analytic accel T={duration}",math.isclose(a,.23*10/math.sqrt(3)/duration**2,rel_tol=2e-12,abs_tol=1e-13))
    old=[c.c.copy() for c in m.curves.values()];returned=m.limit_rate(.35,.5)
    check(f"linear retiming API T={duration}",returned==m.duration and m.duration>=duration*1.002)
    check(f"linear dense limits T={duration}",dense(m)[0]<=.35 and dense(m)[1]<=.5)
    check(f"linear path unchanged T={duration}",all(np.array_equal(before,c.c) for before,c in zip(old,m.curves.values())))

zero=plan(T=2.);check("zero extrema exactly zero",maxima(zero)==(0.,0.))
check("zero original 1.002 margin retained",zero.limit_rate(.35,.5)==2.*1.002)
check("zero all direct samples finite",all(np.isfinite(v).all() for sample in (zero.sample(0.),zero.sample(zero.duration)) for group in sample.values() for v in group.values()))
y=np.zeros((2,7));y[1,0]=1e-12
check("near-zero path is not dropped",maxima(plan(y=y))[0]>1e-12)

for x in ([0.,.07,.2,.49,.67,.94,1.], [0.,.2,.5000001,.5000002,.83,1.]):
    x=np.array(x);y=np.stack([.08*np.sin((j+1)*x)+.01*x*x for j in range(7)],axis=1)
    m=plan(x,y,T=1.7);before=maxima(m);sampled=dense(m,100001)
    check("nonuniform/narrow knots extrema bound dense "+str(len(x)),all(a+1e-7>=b for a,b in zip(before,sampled)))
    m.limit_rate(.35,.5);after=dense(m,100001)
    check("nonuniform/narrow knots retimed dense "+str(len(x)),after[0]<=.35 and after[1]<=.5)

repeated=np.polynomial.polynomial.polymul(np.polynomial.polynomial.polypow([-.25,1.],2),np.polynomial.polynomial.polypow([-.75,1.],3))
roots=_real_polynomial_roots_unit(repeated)
check("repeated stationary roots preserved",all(any(abs(r-target)<2e-6 for r in roots) for target in (.25,.75)))
check("constant root polynomial has no arbitrary root",_real_polynomial_roots_unit([2.])==[] and _real_polynomial_roots_unit([0.,0.])==[])
check("endpoint and interior stationary roots retained",all(any(abs(r-t)<1e-12 for r in _real_polynomial_roots_unit([0.,-.5,1.5,-1.])) for t in (0.,.5,1.)))
rejects("nonfinite root coefficients",lambda:_real_polynomial_roots_unit([1.,float('nan')]))
rejects("complex root coefficients",lambda:_real_polynomial_roots_unit([1.,2j]))

archive_dir=ROOT/'runs/two_panel_full_21/independent-positive-y20-route-review'
reference=json.loads((archive_dir/'independent-continuous-rates.json').read_text())
reference_by_name={Path(r['archive']).name:r for r in reference['curves']}
for path in sorted(archive_dir.glob('*.npz')):
    a=np.load(path);m=MotionPlan({s:PPoly(a[s+'_c'],a[s+'_x']) for s in ('left','right')},float(a['duration']))
    original=m.duration;reference_row=reference_by_name[path.name];before=maxima(m)
    expected=tuple(reference_row['explicit_continuous_candidate_peak'][k]['absolute_value'] for k in ('velocity','acceleration'))
    check(path.stem+" independently archived extrema",np.allclose(before,expected,rtol=2e-9,atol=2e-10))
    payload={s:(c.x.copy(),c.c.copy()) for s,c in m.curves.items()}
    fingers=(m.finger_start.copy(),m.finger_end.copy())
    sample501=max(float(np.abs(m.sample(float(t))['qdd'][s]).max()) for t in np.linspace(0.,m.duration,501) for s in ('left','right'))
    m.limit_rate(.35,.5);after=maxima(m);observed=dense(m,100001)
    check(path.stem+" retimed continuous and independent dense limits",max(after[0],observed[0])<=.35 and max(after[1],observed[1])<=.5)
    check(path.stem+" only duration mutated",all(np.array_equal(c.x,payload[s][0]) and np.array_equal(c.c,payload[s][1]) for s,c in m.curves.items()) and fingers==(m.finger_start,m.finger_end))
    if '_3_runner_proxy' in path.name:
        check(path.stem+" old501 missed known knot peak",sample501<.5 and before[1]>.524 and m.duration>original)
    details.append(dict(archive=str(path),archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),before_duration_s=original,
        old_501_peak_acceleration_rad_s2=sample501,before_continuous_peaks=before,
        after_duration_s=m.duration,after_continuous_peaks=after,independent_100001_plus_knots_peaks=observed))

cfg=json.loads((archive_dir.parent/'config.json').read_text())
native_state=np.load(archive_dir.parent/'orient_teddy_postlift.npz')
planner=Planner(cfg)
qs={s:native_state['q_'+s][:7] for s in ('left','right')}
fingers={s:float(native_state['q_'+s][7:].mean()) for s in ('left','right')}
position,rotation=planner.fk('right',qs['right'])
ordinary=planner.trajectory(qs,{'right':position+[0.,0.,.005]},rotations={'right':rotation},
    duration=1.,finger_start=fingers,finger_end=fingers)
ordinary_rates=dense(ordinary,100001)
check("ordinary default Planner.trajectory API and rates",ordinary.duration>=1.002 and ordinary_rates[0]<=.55 and ordinary_rates[1]<=1.)
check("ordinary trajectory still reaches original FK target",np.linalg.norm(planner.fk('right',ordinary.end_q()['right'])[0]-(position+[0.,0.,.005]))<.0005)

for value in (0.,-1.,float('nan'),float('inf'),True,None,'1',[]):
    rejects("invalid constructor duration "+repr(value),lambda value=value:plan(T=value))
for value in (0.,-1.,float('nan'),float('inf'),True,None,'1',[]):
    for which in ('velocity','acceleration'):
        m=plan()
        rejects("invalid "+which+" "+repr(value),lambda m=m,value=value,which=which:m.limit_rate(value,.5) if which=='velocity' else m.limit_rate(.35,value),m)
for field,value in [('duration',float('nan')),('duration',float('inf')),('duration',0.)]:
    m=plan();setattr(m,field,value);rejects("mutated invalid "+field+repr(value),m.continuous_rate_extrema)
for kind in ('missing_side','wrong_shape','nan_coefficient','complex_coefficient','inf_knot','nonincreasing','partial_domain','c0_jump','c1_jump','c2_jump','overflow'):
    m=plan([0.,.5,1.]);c=m.curves['right'].c.copy();x=m.curves['right'].x.copy()
    if kind=='missing_side':del m.curves['left']
    elif kind=='wrong_shape':c=c[:,:,:6]
    elif kind=='nan_coefficient':c[0,0,0]=np.nan
    elif kind=='complex_coefficient':c=c.astype(complex);c[0,0,0]=1j
    elif kind=='inf_knot':x[1]=np.inf
    elif kind=='nonincreasing':x[1]=0.
    elif kind=='partial_domain':x[0]=.1
    elif kind.startswith('c'):c[3-int(kind[1]),1,0]=.001
    elif kind=='overflow':
        x=np.array([0.,1.]);c=np.zeros((4,1,7));c[0,0,0]=1e308
    if kind!='missing_side':m.curves['right']=PPoly.construct_fast(c,x)
    rejects("invalid path "+kind,m.limit_rate,m)
m=plan(y=np.array([[0.]*7,[1.]*7]));rejects("retimed duration overflow",lambda:m.limit_rate(1e-320,1e-320),m)

out=ROOT/'runtime_ipc/motion-continuous-rate-regression.json'
report=dict(kind='targeted_continuous_rate_retiming_regression',native_simulation=False,
    checks=checks,passed=all(c['passed'] for c in checks),count=len(checks),failed=[c['name'] for c in checks if not c['passed']],
    archived_curve_results=details,elapsed_seconds=time.perf_counter()-start,
    ordinary_trajectory_smoke=dict(duration_s=ordinary.duration,dense_peaks=ordinary_rates,native_simulation=False),
    motion_sha256=hashlib.sha256((ROOT/'motion.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Analytic simple cases, archived independent roots, explicit missed-knot witnesses, 100001 time samples plus all knots, C2 and malformed/nonfinite inputs. No native or geometric replanning.')
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('passed','count','failed','elapsed_seconds','motion_sha256')},indent=2));print(out)
if not report['passed']:raise SystemExit(1)
