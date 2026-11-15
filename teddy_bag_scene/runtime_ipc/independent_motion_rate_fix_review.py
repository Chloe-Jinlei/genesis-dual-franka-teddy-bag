"""Independent CPU-only continuous-rate review using Bernstein enclosure algebra."""
import hashlib, json, math, sys
from pathlib import Path
import numpy as np
from numpy.polynomial import polynomial as pol
from scipy.interpolate import CubicSpline, PPoly
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motion import MotionPlan
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'motion.py';expected='544d6bed1aedfe1603db860af0cd4d230641714827ff23ef27c93d0a12e32a31'
assert sha(source)==expected
checks=[];rows=[]
def check(name,v):checks.append(dict(name=name,passed=bool(v)))
cache={}
def outer(c):
 n=len(c)-1
 if n not in cache:
  B=np.array([[math.comb(k,i)/math.comb(n,i) if i<=k else 0. for i in range(n+1)] for k in range(n+1)])
  matrices=[]
  for a,b in zip(np.linspace(0.,1.,65)[:-1],np.linspace(0.,1.,65)[1:]):
   R=np.array([[math.comb(i,k)*a**(i-k)*(b-a)**k if i>=k else 0. for i in range(n+1)] for k in range(n+1)])
   matrices.append(B@R)
  cache[n]=np.asarray(matrices)
 return np.max(np.abs(np.einsum('sij,jk->sik',cache[n],c)),axis=(0,1))
def bounds(plan):
 vv=np.zeros(7);aa=np.zeros(7)
 for curve in plan.curves.values():
  clocks=[0. if s==0. else 1. if s==1. else brentq(lambda u:10*u**3-15*u**4+6*u**5-s,0.,1.,xtol=5e-15) for s in curve.x]
  for k,(a,b) in enumerate(zip(clocks[:-1],clocks[1:])):
   h=b-a;u=np.array([a,h]);p=pol.polyadd(pol.polyadd(10*pol.polypow(u,3),-15*pol.polypow(u,4)),6*pol.polypow(u,5));p[0]-=curve.x[k]
   q=np.zeros((16,7))
   for d in range(4):
    basis=pol.polypow(p,d);q[:len(basis)]+=basis[:,None]*curve.c[3-d,k]
   qd=q[1:]*np.arange(1,len(q))[:,None]/(h*plan.duration)
   qdd=qd[1:]*np.arange(1,len(qd))[:,None]/(h*plan.duration)
   vv=np.maximum(vv,outer(qd));aa=np.maximum(aa,outer(qdd))
 return float(vv.max()+1e-9),float(aa.max()+1e-9)
def peak(plan):
 r=plan.continuous_rate_extrema()
 return max(float(x['maximum_velocity_rad_s'].max()) for x in r.values()),max(float(x['maximum_acceleration_rad_s2'].max()) for x in r.values())
def run_case(name,plan):
 old=plan.duration;coeff={s:(c.x.copy(),c.c.copy()) for s,c in plan.curves.items()};fingers=(plan.finger_start.copy(),plan.finger_end.copy());pv,pa=peak(plan)
 before_b=bounds(plan);new=plan.limit_rate(.35,.5);bv,ba=bounds(plan)
 check(name+' root peaks below independent Bernstein bounds',pv<=before_b[0] and pa<=before_b[1])
 check(name+' independent retimed continuous velocity bound',bv<=.35)
 check(name+' independent retimed continuous acceleration bound',ba<=.5)
 check(name+' exact duration scaling formula',abs(new-old*max(1.,pv/.35,math.sqrt(pa/.5))*1.002)<1e-12)
 check(name+' no path or finger mutation',all(np.array_equal(x,plan.curves[s].x) and np.array_equal(c,plan.curves[s].c) for s,(x,c) in coeff.items()) and fingers==(plan.finger_start,plan.finger_end))
 rows.append(dict(name=name,before_duration_s=old,after_duration_s=new,before_production_extrema=[pv,pa],before_independent_Bernstein_bounds=before_b,after_independent_Bernstein_bounds=[bv,ba]))
 return plan
DIR=ROOT/'runs/two_panel_full_21/independent-positive-y20-route-review'
for name in ('zero','positive_y20'):
 a=np.load(DIR/(name+'_3_runner_proxy.npz'));curves={s:PPoly(a[s+'_c'],a[s+'_x']) for s in ('left','right')};plan=MotionPlan(curves,float(a['duration']),{'left':0.,'right':0.},{'left':0.,'right':.04})
 oldT=plan.duration;witness_clock=brentq(lambda u:10*u**3-15*u**4+6*u**5-19/44,0.,1.);oldw=abs(plan.sample(oldT*witness_clock)['qdd']['right'][2])
 run_case(name+' archived missed knot',plan);neww=abs(plan.sample(plan.duration*witness_clock)['qdd']['right'][2]);check(name+' direct original witness corrected',oldw>.524 and neww<=.5)
 rows[-1].update(before_witness=oldw,after_witness=neww,archive_sha256=sha(DIR/(name+'_3_runner_proxy.npz')))
# Independent polynomial families with sign changes and nonzero endpoint path derivatives.
for idx,knots in enumerate(([0.,1.],[0.,.13,.49,.86,1.],[0.,.18,.23,.79,.999,1.])):
 x=np.array(knots);y=np.column_stack([.12*np.sin((j+1)*x)+.025*(-1)**j*x**3 for j in range(7)])
 plan=MotionPlan({s:CubicSpline(x,y if s=='right' else y*.37,axis=0) for s in ('left','right')},[.7,2.3,5.1][idx])
 run_case('nonuniform polynomial family '+str(idx),plan)
# Symmetry: reverse path and translate angles cannot change continuous extrema.
x=np.array([0.,.07,.31,.68,.93,1.]);y=np.column_stack([.08*np.cos((j+1)*x)+.03*x for j in range(7)])
base=MotionPlan({s:CubicSpline(x,y,axis=0) for s in ('left','right')},1.3);reverse=MotionPlan({s:CubicSpline(1-x[::-1],y[::-1]+.7,axis=0) for s in ('left','right')},1.3)
check('time reversal and angle offset invariant extrema',np.allclose(peak(base),peak(reverse),rtol=3e-10,atol=1e-10));run_case('reversed and angle translated',reverse)
# No hidden finite sample dependency: a stationary root is required away from archived grid.
z=np.zeros((2,7));z[1,0]=.8;linear=MotionPlan({s:CubicSpline([0.,1.],z,axis=0) for s in ('left','right')},.87)
v,a=peak(linear);check('closed form minjerk velocity',abs(v-.8*15/8/.87)<1e-12);check('closed form minjerk acceleration',abs(a-.8*10/math.sqrt(3)/.87**2)<1e-11)
# A rate failure must leave duration and paths untouched.
for label,mutation in [('nonfinite coefficient',lambda m:m.curves['left'].c.__setitem__((0,0,0),np.nan)),('C1 discontinuity',lambda m:m.curves['left'].c.__setitem__((2,1,0),1.))]:
 m=MotionPlan({s:CubicSpline([0.,.5,1.],np.zeros((3,7)),axis=0) for s in ('left','right')},2.);mutation(m)
 try:m.limit_rate(.35,.5)
 except ValueError:check(label+' fail closed without duration mutation',m.duration==2.)
 else:check(label+' fail closed without duration mutation',False)
check('frozen production source unchanged',sha(source)==expected)
report=dict(kind='independent_motion_continuous_rate_fix_review',native_execution=False,physical_success=False,source_sha256=expected,checks=checks,count=len(checks),passed=all(x['passed'] for x in checks),failures=[x['name'] for x in checks if not x['passed']],cases=rows,method='Independent expanded-power composition and 64-subinterval Bernstein convex-hull bounds for each derivative polynomial; 1e-9 numerical pad. This is floating-point algebra, not directed interval arithmetic.',mathematical_review=['On each inverse-knot clock interval, q composed with the quintic clock has degree at most15. Time derivatives divide by interval width and duration once per order.','Endpoint plus all real derivative roots gives extrema of absolute rates; recursive derivative-root isolation covers monotone intervals and retains repeated stationary roots as extra candidates.','Both sides of all cubic junctions are evaluated, and malformed/nonfinite/visible C0-C2 discontinuities reject.','Duration rescaling changes velocity by 1/T and acceleration by 1/T^2; the original1.002 factor remains.','The test validates command-reference rates. Physical measured rates, soft contacts, rigid clearance and future reobserved insertion still require existing separate gates.'],limitations=['Double-precision root isolation and1.002 margin are a numerical algorithm, not a proof for arbitrary ill-conditioned adversarial polynomials.','Permitted C0/C1/C2 numerical junction tolerances are not exact continuity proofs; source production paths are scipy CubicSpline.','Changing duration alters the future native deformable state; previous frozen route success is not transferable to a new run.'])
out=ROOT/'runtime_ipc/independent-motion-rate-fix-review.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['passed','count','failures','source_sha256']},indent=2));print(out,sha(out))
