"""CPU-only full geometric remainder for two frozen replacement centers."""
from full16_alignment_investigation import *
from motion import MotionPlan,CubicSpline,minimum_jerk
st=states['entry_end'];start=np.asarray(st['tcp_right']);fingers={s:float(np.mean(st['q_'+s][7:])) for s in ['left','right']};initial_q={s:st['q_'+s][:7].copy() for s in ['left','right']}

def checked_path(qs,target,fingers,duration):
 paths=planner.plan_cartesian(qs,{'right':target},rotations={'right':original},samples=45)
 fractions=np.asarray([e['fraction'] for e in paths]);curves={s:CubicSpline(fractions,np.asarray([e['q'].get(s,qs[s]) for e in paths]),axis=0) for s in ['left','right']}
 plan=MotionPlan(curves,duration,fingers,fingers);extrema=plan.validate_joint_limits(planner.limits);plan.limit_rate(.35,.5)
 start_tcp=planner.fk('right',qs['right'])[0];minimum=float('inf');worst=None;first=None;max_pe=0.;max_re=0.;peaks=[0.,0.]
 for time in np.linspace(0.,plan.duration,401):
  sample=plan.sample(float(time));checks=planner.clearance(sample['q'],sample['finger']);cat,value=min(checks.items(),key=lambda item:item[1]['distance']);tcp,rot=planner.fk('right',sample['q']['right']);u=float(minimum_jerk(time/plan.duration)[0]);data=dict(time_s=float(time),time_fraction=float(time/plan.duration),path_fraction=u,tcp_m=tcp.tolist(),category=cat,**value)
  if value['distance']<minimum:minimum=value['distance'];worst=data
  if first is None and value['distance']<.008:first=data
  max_pe=max(max_pe,float(np.linalg.norm(tcp-(start_tcp+u*(target-start_tcp)))));max_re=max(max_re,float((rot*original.inv()).magnitude()))
 for time in np.linspace(0.,plan.duration,501):
  p=plan.sample(float(time));peaks[0]=max(peaks[0],*[float(np.abs(p['qd'][s]).max()) for s in ['left','right']]);peaks[1]=max(peaks[1],*[float(np.abs(p['qdd'][s]).max()) for s in ['left','right']])
 row=dict(status='geometric_candidate' if minimum>=.008 and max_pe<=.0005 and max_re<=.002 else 'rejected',duration_s=plan.duration,target_tcp_m=target.tolist(),original_rotation_xyzw=original.as_quat().tolist(),minimum_sampled_rigid_clearance_m=minimum,first_8mm_violation=first,worst_rigid_sample=worst,rigid_sample_count=401,maximum_sampled_tcp_straight_path_error_m=max_pe,maximum_sampled_original_rotation_error_rad=max_re,command_rate_limits=dict(velocity_rad_s=.35,acceleration_rad_s2=.5,sampled_peak_velocity_rad_s=peaks[0],sampled_peak_acceleration_rad_s2=peaks[1],rate_sample_count=501),continuous_joint_limits_pass=True,analytic_joint_extrema=extrema,physical_success=False)
 return row,plan
results=[]
for dy in [-.06,-.05]:
 delta=np.array([-.02,dy,0.]);center_target=start+delta;row=dict(replacement_center_delta_m=delta.tolist(),observed_state='full16 actual entry endpoint',observed_time_s=float(st['time']),physical_success=False,runtime_execution_performed=False,rest_of_route_is_frozen_hypothesis=True,stiffer_handles_not_simulated=True)
 try:
  row['center'],cp=checked_path(initial_q,center_target,fingers,4.)
  row['center_bear_left_5mm_sweep']=check_translation_sweep(st['bear'],rest['bear_tets'],st['q_left'],cfg,displacement_m=delta,required_gap_m=.005,planner=planner)
  dz=cfg['table']['height']+.01-float(st['bear'][:,2].min());insert_target=center_target+[0,0,dz];row['insert'],ip=checked_path(cp.end_q(),insert_target,fingers,6.)
  row['insert_bear_left_5mm_sweep']=check_translation_sweep(st['bear'],rest['bear_tets'],st['q_left'],cfg,translation_m=delta,displacement_m=[0,0,dz],required_gap_m=.005,planner=planner)
  row['frozen_final_bear_bounds_m']=[(st['bear']+delta+[0,0,dz]).min(0).tolist(),(st['bear']+delta+[0,0,dz]).max(0).tolist()];row['frozen_actual_rim_minimum_z_m']=float(st['bag'][rim,2].min());row['whole_frozen_bear_below_lowest_rim_by_2mm']=bool(row['frozen_final_bear_bounds_m'][1][2]<=row['frozen_actual_rim_minimum_z_m']-.002)
  opened={**fingers,'right':.04};release=[]
  for fraction in np.linspace(0.,1.,21):
   ff={**fingers,'right':float((1-fraction)*fingers['right']+fraction*.04)};checks=planner.clearance(ip.end_q(),ff);cat,v=min(checks.items(),key=lambda item:item[1]['distance']);release.append(dict(fraction=float(fraction),right_half_opening_m=ff['right'],category=cat,**v))
  worst=min(release,key=lambda x:x['distance']);row['release']=dict(status='geometric_candidate' if worst['distance']>=.008 else 'rejected',minimum_sampled_rigid_clearance_m=worst['distance'],worst_sample=worst,sample_count=21,final_right_half_opening_m=.04,physical_release_not_executed=True)
  row['withdraw'],wp=checked_path(ip.end_q(),center_target,opened,4.)
  row['all_frozen_geometric_gates_pass']=all(row[k]['status']=='geometric_candidate' for k in ['center','insert','release','withdraw']) and all(row[k]['status']=='clear' for k in ['center_bear_left_5mm_sweep','insert_bear_left_5mm_sweep']) and row['whole_frozen_bear_below_lowest_rim_by_2mm']
 except (ValueError,KeyError) as e:row.update(error=str(e),all_frozen_geometric_gates_pass=False)
 results.append(row)
 print('candidate',delta.tolist(),[(k,row.get(k,{}).get('status'),row.get(k,{}).get('minimum_sampled_rigid_clearance_m'),row.get(k,{}).get('first_8mm_violation')) for k in ['center','insert','release','withdraw']],flush=True)
 out=dict(kind='full16_frozen_new_center_remaining_path_review',physical_success=False,runtime_execution_performed=False,results=results,limits=['Each center uses the actual full16 entry-end shape and measured joints. Later shapes and joint starts are frozen or kinematic hypotheses.','A changed handle material alters the native cloth dynamics; all phase gates must be recomputed from fresh actual observations.','No force, strain, grip retention, real final containment, release, or settled success is demonstrated.','All 8mm rigid failures are kept as failures. Diagnostic continuation through a blocked hypothetical segment only localizes remaining obstacles and does not authorize execution.','Frozen bear-left clearance is continuous for the declared translation; robot clearance and Cartesian tracking are sampled at 401 points.'])
 (RUN/'independent-new-center-remainder.json').write_text(json.dumps(out,indent=2,default=scalar)+'\n')
