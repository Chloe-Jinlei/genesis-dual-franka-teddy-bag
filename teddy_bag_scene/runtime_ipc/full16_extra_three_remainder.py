"""Exactly three additional frozen route candidates, then refine one."""
from pathlib import Path
common=(Path(__file__).with_name('full16_new_center_remainder.py')).read_text().split('results=[]')[0]
exec(compile(common,str(Path(__file__).with_name('full16_new_center_remainder.py')),'exec'),globals())
results=[]
for dx,dy in [(-.010,-.070),(0.,-.080),(-.010,-.060)]:
 delta=np.array([dx,dy,0.]);ct=start+delta;dz=cfg['table']['height']+.01-float(st['bear'][:,2].min());it=ct+[0,0,dz];row=dict(center_delta_m=delta.tolist(),physical_success=False,after_center_is_frozen_hypothesis=True)
 try:
  row['center'],cp=checked_path(initial_q,ct,fingers,4.);row['insert'],ip=checked_path(cp.end_q(),it,fingers,6.);opened={**fingers,'right':.04};release=[]
  for fraction in np.linspace(0.,1.,21):
   ff={**fingers,'right':float((1-fraction)*fingers['right']+fraction*.04)};checks=planner.clearance(ip.end_q(),ff);cat,v=min(checks.items(),key=lambda x:x[1]['distance']);release.append(dict(fraction=float(fraction),category=cat,**v))
  worst=min(release,key=lambda x:x['distance']);row['release']=dict(status='geometric_candidate' if worst['distance']>=.008 else 'rejected',minimum_sampled_rigid_clearance_m=worst['distance'],worst_sample=worst,sample_count=21)
  row['withdraw'],wp=checked_path(ip.end_q(),ct,opened,4.);row['rigid_route_pass']=all(row[k]['status']=='geometric_candidate' for k in ['center','insert','release','withdraw'])
 except ValueError as e:row.update(error=str(e),rigid_route_pass=False)
 results.append(row);print(delta.tolist(),[(k,row.get(k,{}).get('status'),row.get(k,{}).get('minimum_sampled_rigid_clearance_m'),row.get(k,{}).get('worst_rigid_sample',{}).get('bodies')) for k in ['center','insert','release','withdraw']],flush=True)
passed=[x for x in results if x['rigid_route_pass']]
if passed:
 # Larger -Y best addresses the observed body displacement, without inventing success.
 best=min(passed,key=lambda x:x['center_delta_m'][1]);delta=np.array(best['center_delta_m']);best['selected_for_frozen_teddy_sweep']=True
 best['center_bear_left_5mm_sweep']=check_translation_sweep(st['bear'],rest['bear_tets'],st['q_left'],cfg,displacement_m=delta,required_gap_m=.005,planner=planner)
 best['insert_bear_left_5mm_sweep']=check_translation_sweep(st['bear'],rest['bear_tets'],st['q_left'],cfg,translation_m=delta,displacement_m=[0,0,dz],required_gap_m=.005,planner=planner)
 best['all_frozen_geometric_gates_pass']=all(best[k]['status']=='clear' for k in ['center_bear_left_5mm_sweep','insert_bear_left_5mm_sweep'])
 print('REFINED',best['center_delta_m'],best['all_frozen_geometric_gates_pass'],best['center_bear_left_5mm_sweep']['conservative_gap_lower_bound_m'],flush=True)
(RUN/'independent-extra-three-remainder.json').write_text(json.dumps(dict(kind='three_bounded_extra_frozen_route_checks',physical_success=False,runtime_execution_performed=False,candidates=results,scope='Exact three candidates requested. Actual full16 entry shape frozen; no changed handle material simulated. 8mm rigid, original full R, .35/.5 command rates maintained. Only selected candidate receives full 5mm frozen bear sweeps. No actual contact, deformation or containment success is established.'),indent=2,default=scalar)+'\n')
