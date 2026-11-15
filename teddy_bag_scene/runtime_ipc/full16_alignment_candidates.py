"""Bounded frozen-state alignment exploration, CPU only, no live mutations."""
from full16_alignment_investigation import *
geo=json.loads((RUN/'independent-alignment-geometry.json').read_text())['states']
results=[]
for name,state in states.items():
 start=np.asarray(state['tcp_right']);qs={s:state['q_'+s][:7].copy() for s in ['left','right']};fingers={s:float(np.mean(state['q_'+s][7:])) for s in ['left','right']}
 for dx in [-.02,-.04,-.06,-.08]:
  for dy in [-.02,-.04,-.06,-.08]:
   target=start+np.array([dx,dy,0.]);row=dict(state=name,observed_time_s=float(state['time']),delta_xy_m=[dx,dy],target_tcp_m=target.tolist(),physical_success=False,failed_observation_not_executable=name=='failed_end',fixed_original_rotation_xyzw=original.as_quat().tolist())
   try:
    q,err,ang=planner.solve('right',target,original,seed=qs['right']);row.update(endpoint_position_error_m=err,endpoint_rotation_error_rad=ang)
    if err>.0005 or ang>.002:row.update(status='endpoint_ik_rejected');results.append(row);continue
    endqs={**qs,'right':q};checks=planner.clearance(endqs,fingers);cat,worst=min(checks.items(),key=lambda x:x[1]['distance']);row.update(endpoint_minimum_rigid_clearance_m=worst['distance'],endpoint_limiting_pair=worst,endpoint_limiting_category=cat)
    if worst['distance']<.008:row.update(status='endpoint_rigid_rejected');results.append(row);continue
    paths=planner.plan_cartesian(qs,{'right':target},rotations={'right':original},samples=13);mn=float('inf');lim=None;limfrac=None
    for pp in paths:
     kk={**qs,'right':pp['q']['right']};cc=planner.clearance(kk,fingers);ccat,v=min(cc.items(),key=lambda x:x[1]['distance'])
     if v['distance']<mn:mn=v['distance'];lim=dict(category=ccat,**v);limfrac=pp['fraction']
    row.update(sampled_straight_ik_minimum_rigid_clearance_m=mn,limiting_pair=lim,limiting_fraction=limfrac,rigid_sample_count=13,status='coarse_rigid_clear' if mn>=.008 else 'sampled_rigid_rejected',continuous_rigid_certificate=False)
   except ValueError as exc:row.update(status='planning_rejected',error=str(exc))
   results.append(row)
 out={'kind':'full16_bounded_xy_alignment_candidates','physical_success':False,'scope':'Each candidate is a fresh frozen-state hypothetical with unchanged original full rotation, measured left joints, actual mean finger openings, 8mm rigid threshold. Larger XY limits are exploration only and not runtime authorization. Failed-end candidates are not executable due to prior cloth strain failure. Coarse 13-sample IK route screening is not a validated spline or continuous clearance proof.','candidates':results}
 (RUN/'independent-alignment-candidates.json').write_text(json.dumps(out,indent=2,default=scalar)+'\n')
 print(name,[(x['delta_xy_m'],x['status'],round(x.get('sampled_straight_ik_minimum_rigid_clearance_m',x.get('endpoint_minimum_rigid_clearance_m',-9)),5),x.get('limiting_pair',x.get('endpoint_limiting_pair',{})).get('bodies')) for x in results if x['state']==name],flush=True)
