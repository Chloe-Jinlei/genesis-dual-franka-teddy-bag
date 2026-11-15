"""Independent polynomial-rate and native retiming audit of 16 saved curves only."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.interpolate import PPoly
from scipy.optimize import brentq
from numpy.polynomial import polynomial as pol

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/two_panel_full_21"
DIR = RUN / "independent-positive-y20-route-review"
review = json.loads((DIR / "review.json").read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs = [DIR / "review.json", RUN / "source/motion.py", RUN / "source/run.py", RUN / "postlift_yaw_plan.npz", RUN / "orient_teddy_postlift.npz", *sorted(DIR.glob("*.npz"))]
hashes = {str(p): sha(p) for p in inputs}

def jerk(u):
    return 10*u**3-15*u**4+6*u**5, 30*u**2-60*u**3+30*u**4, 60*u-180*u**2+120*u**3

def inverse(s):
    return 0. if s <= 0 else 1. if s >= 1 else brentq(lambda u:jerk(u)[0]-s,0.,1.,xtol=5e-15)

transform_cache = {}
def bernstein_outer(coefficients):
    """Eight [0,1] subintervals, exact power-to-Bernstein algebra in doubles."""
    n = len(coefficients)-1
    if n not in transform_cache:
        power_to_bernstein = np.array([[math.comb(k,i)/math.comb(n,i) if i<=k else 0. for i in range(n+1)] for k in range(n+1)])
        transforms=[]
        for a,b in zip(np.linspace(0.,1.,9)[:-1],np.linspace(0.,1.,9)[1:]):
            restriction=np.array([[math.comb(i,k)*a**(i-k)*(b-a)**k if i>=k else 0. for i in range(n+1)] for k in range(n+1)])
            transforms.append(power_to_bernstein@restriction)
        transform_cache[n] = np.asarray(transforms)
    return np.max(np.abs(np.einsum("sij,jk->sik",transform_cache[n],coefficients)),axis=(0,1))

def roots_in_unit(coefficients):
    if not np.any(coefficients): return []
    roots=pol.polyroots(np.trim_zeros(coefficients,"b"))
    return [float(np.clip(r.real,0.,1.)) for r in roots if abs(r.imag)<=1e-8 and -1e-10<=r.real<=1.+1e-10]

def derivative(coefficients):
    return coefficients[1:]*np.arange(1,len(coefficients))[:,None]

def at_times(curves,T,times):
    u=np.asarray(times)/T; s,su,suu=jerk(u)
    result={}
    for side,curve in curves.items():
        result[side]=(curve(s,1)*su[:,None]/T,
            curve(s,2)*(su[:,None]/T)**2+curve(s,1)*suu[:,None]/T**2)
    return result

rows=[]
yaw=np.load(RUN/"postlift_yaw_plan.npz")
actual=np.load(RUN/"orient_teddy_postlift.npz")
for case in review["cases"]:
    previous={"runner_proxy":{s:PPoly(yaw[s+"_c"],yaw[s+"_x"])(1.) for s in ("left","right")},
        "planner_report":{s:actual["q_"+s][:7] for s in ("left","right")}}
    for stage in case["stages"]:
        for kind in ("planner_report","runner_proxy"):
            file=DIR/f"{case['name']}_{stage['index']}_{kind}.npz"
            archive=np.load(file);T=float(archive["duration"])
            curves={s:PPoly(archive[s+"_c"],archive[s+"_x"]) for s in ("left","right")}
            limits=(.35,.5) if kind=="runner_proxy" else (.55,1.)
            checks=dict(archive_sha_matches_geometric_review=sha(file)==stage[kind]["archived_plan_sha256"],
                all_raw_archive_values_finite=all(np.isfinite(archive[k]).all() for k in archive.files),
                duration_positive_finite=bool(np.isfinite(T) and T>0),
                original_knots_cover_0_1_with_45_strictly_increasing_samples=all(len(c.x)==45 and c.x[0]==0 and c.x[-1]==1 and np.all(np.diff(c.x)>0) and c.c.shape==(4,44,7) for c in curves.values()),
                starts_from_actual_or_prior_exact_command_endpoint=all(np.max(np.abs(c(0.)-previous[kind][s]))<1e-12 for s,c in curves.items()))
            peaks={"velocity":dict(absolute_value=0.),"acceleration":dict(absolute_value=0.)}
            bounds={};continuity={}
            for side,curve in curves.items():
                outer_v=np.zeros(7);outer_a=np.zeros(7);errors=np.zeros(3)
                for segment,width in enumerate(np.diff(curve.x)):
                    if segment+1<len(curve.x)-1:
                        for order in range(3):
                            c=curve.c[:,segment,:].copy()
                            for _ in range(order): c=c[:-1]*np.arange(len(c)-1,0,-1)[:,None]
                            value=np.zeros(7)
                            for cc in c:value=value*width+cc
                            expected=curve.c[3-order,segment+1,:]*math.factorial(order)
                            errors[order]=max(errors[order],float(np.max(np.abs(value-expected))))
                    ua,ub=inverse(curve.x[segment]),inverse(curve.x[segment+1]);du=ub-ua
                    u=np.array([ua,du])
                    progress=pol.polyadd(pol.polyadd(10*pol.polypow(u,3),-15*pol.polypow(u,4)),6*pol.polypow(u,5))
                    progress[0]-=curve.x[segment]
                    powers=[np.array([1.]),progress,pol.polypow(progress,2),pol.polypow(progress,3)]
                    q=np.zeros((16,7))
                    for degree,basis in enumerate(powers):q[:len(basis)]+=basis[:,None]*curve.c[3-degree,segment,:]
                    qd=derivative(q)/(du*T);qdd=derivative(qd)/(du*T);qddd=derivative(qdd)/(du*T)
                    outer_v=np.maximum(outer_v,bernstein_outer(qd));outer_a=np.maximum(outer_a,bernstein_outer(qdd))
                    for label,critical in (("velocity",qdd),("acceleration",qddd)):
                        for joint in range(7):
                            candidates=[0.,1.,*roots_in_unit(critical[:,joint])]
                            for r in candidates:
                                clock=ua+du*r;s,su,suu=jerk(clock)
                                value=float(curve(s,1)[joint]*su/T if label=="velocity" else curve(s,2)[joint]*(su/T)**2+curve(s,1)[joint]*suu/T**2)
                                if abs(value)>peaks[label]["absolute_value"]:
                                    peaks[label]=dict(absolute_value=abs(value),signed_value=value,side=side,joint_index_zero_based=joint,
                                        time_s=clock*T,clock_fraction=clock,path_fraction=s,segment_index=segment,segment_local_clock=r,
                                        source="Direct archived PPoly derivative at a polynomial critical point or spline-knot boundary")
                bounds[side]=dict(continuous_velocity_upper_bound_rad_s=(outer_v+1e-9).tolist(),continuous_acceleration_upper_bound_rad_s2=(outer_a+1e-9).tolist())
                continuity[side]=dict(maximum_C0_C1_C2_junction_error=errors.tolist())
            checks["C0_C1_C2_junctions_continuous_with_numerical_tolerance"]=all(np.max(v["maximum_C0_C1_C2_junction_error"])<1e-7 for v in continuity.values())
            for side,curve in curves.items():previous[kind][side]=curve(1.)
            sampled=at_times(curves,T,np.linspace(0.,T,501))
            sv=max(float(np.abs(v[0]).max()) for v in sampled.values());sa=max(float(np.abs(v[1]).max()) for v in sampled.values())
            bound_v=max(max(v["continuous_velocity_upper_bound_rad_s"]) for v in bounds.values())
            bound_a=max(max(v["continuous_acceleration_upper_bound_rad_s2"]) for v in bounds.values())
            retiming=None
            if kind=="runner_proxy":
                start_duration=float(case["reported_route"]["stages"][stage["index"]]["duration_s"])
                before=at_times(curves,start_duration,np.linspace(0.,start_duration,501))
                before_v=max(float(np.abs(v[0]).max()) for v in before.values());before_a=max(float(np.abs(v[1]).max()) for v in before.values())
                expected=start_duration*max(1.,before_v/.35,math.sqrt(before_a/.5))*1.002
                retiming=dict(supplied_stage_duration_s=start_duration,peak_501_before_velocity_rad_s=before_v,peak_501_before_acceleration_rad_s2=before_a,
                    expected_duration_s=expected,archived_duration_s=T,absolute_duration_error_s=abs(expected-T),
                    native_formula="T *= max(1, sampled501_v/.35, sqrt(sampled501_a/.5))*1.002")
                checks["archived_duration_matches_original_501_runner_retiming"]=abs(expected-T)<1e-10
            row=dict(case=case["name"],stage_index=stage["index"],stage=stage["name"],kind=kind,provisional_only=stage["provisional_only"],duration_s=T,
                archive=str(file),archive_sha256=sha(file),checks=checks,evidence_consistent=all(checks.values()),
                limits=dict(velocity_rad_s=limits[0],acceleration_rad_s2=limits[1]),peak_501_velocity_rad_s=sv,peak_501_acceleration_rad_s2=sa,
                explicit_continuous_candidate_peak=peaks,continuous_outer_bound_by_arm=bounds,
                maximum_continuous_velocity_upper_bound_rad_s=bound_v,maximum_continuous_acceleration_upper_bound_rad_s2=bound_a,
                continuous_rate_bounds_pass=bound_v<=limits[0] and bound_a<=limits[1],
                witnessed_rate_violation=peaks["velocity"]["absolute_value"]>limits[0] or peaks["acceleration"]["absolute_value"]>limits[1],
                continuity=continuity,runner_retiming=retiming)
            rows.append(row)
            print(case["name"],stage["index"],kind,"upper",bound_v,bound_a,"witness",peaks["acceleration"]["absolute_value"],flush=True)

summary=dict(kind="independent_saved_route_continuous_rate_and_retiming_review",physical_success=False,native_execution=False,
    input_sha256=hashes,inputs_unchanged=all(sha(Path(p))==h for p,h in hashes.items()),
    all_16_archive_evidence_checks_pass=all(r["evidence_consistent"] for r in rows),
    all_6_runner_initial_preparation_curves_within_continuous_0p35_0p5=all(r["continuous_rate_bounds_pass"] for r in rows if r["kind"]=="runner_proxy" and not r["provisional_only"]),
    provisional_runner_insertion_violations=[dict(case=r["case"],witness=r["explicit_continuous_candidate_peak"]["acceleration"],peak_501_acceleration_rad_s2=r["peak_501_acceleration_rad_s2"]) for r in rows if r["kind"]=="runner_proxy" and r["witnessed_rate_violation"]],
    method="Each original cubic q(s) is composed with minimum-jerk s(u), yielding degree <=15 on each inverse-knot clock interval. Derivatives are bounded by Bernstein convex hulls on 8 equal local-clock subdivisions, with 1e-9 numerical pad. Polynomial derivative roots and interval endpoints yield direct PPoly witnesses. Double precision, not directed-rounding interval arithmetic.",
    scope=["These are archived planner-report curves and a frozen runner reconstruction, not an observed command stream or future native run.",
        "The first three runner stages match the runner's saved command start, duration formula and .35/.5 inputs. Actual fingers and post-transport soft state will be reobserved in a native run.",
        "The fourth insert is provisional only: fresh actual reobservation may alter or reject it. This witness proves existing 501-point retiming does not guarantee continuous .5 acceleration, not that the same future native insert will execute.",
        "No duration, motion source, configuration, asset, or physical gate was changed by this audit."],
    curves=rows,source_sha256=sha(Path(__file__).resolve()))
out=DIR/"independent-continuous-rates.json"
out.write_text(json.dumps(summary,indent=2)+"\n")
print("REPORT",out,"SHA",sha(out),flush=True)
