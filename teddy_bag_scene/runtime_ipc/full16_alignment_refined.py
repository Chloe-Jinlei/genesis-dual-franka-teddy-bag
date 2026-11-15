"""Limited CPU refinement; frozen hypotheses only, never native contact success."""
from full16_alignment_investigation import *
from staged_contact_insertion import _robot_segment
from validate import containment
from contact_insertion import _policy,_state_context
selected=[]
for delta in [np.array([-.02,-.04,0.]),np.array([-.02,-.06,0.])]:
 st=states['entry_end'];ctx=_state_context(rest,st,cfg,_policy(cfg));row=dict(state='entry_end',delta_m=delta.tolist(),physical_success=False)
 row['full_left_frozen_sweep']=check_translation_sweep(st['bear'],rest['bear_tets'],st['q_left'],cfg,displacement_m=delta,required_gap_m=.005,planner=planner)
 try:
  stage,evidence,plan=_robot_segment(ctx,st,cfg,st['tcp_right']+delta,'center',original);row.update(status='frozen_geometric_candidate',stage=stage,robot_trajectory=evidence)
 except ValueError as e:row.update(status='rejected',error=str(e))
 lower_delta=delta.copy();lower_delta[2]=cfg['table']['height']+.01-st['bear'][:,2].min()
 row['hypothetical_final_depth_containment_in_frozen_bag']=containment(st['bag'],st['bear']+lower_delta,body,rim,weights,rest['bear_tets'])
 row['hypothetical_final_translation_m']=lower_delta.tolist();selected.append(row)
 print('full entry',delta.tolist(),row['status'],'sweep',row['full_left_frozen_sweep']['status'],'gap',row['full_left_frozen_sweep'].get('conservative_gap_lower_bound_m'),'containment',row['hypothetical_final_depth_containment_in_frozen_bag'],flush=True)
extra=[];st=states['center_end'];start=np.asarray(st['tcp_right']);qs={s:st['q_'+s][:7].copy() for s in ['left','right']};fingers={s:float(np.mean(st['q_'+s][7:])) for s in ['left','right']}
for dz in [0.,-.02,-.04,-.06]:
 for dy in [-.02,-.04,-.06,-.08]:
  row=dict(state='center_end',first_down_m=dz,second_y_m=dy,physical_success=False,scope='Two frozen-geometry segments only; second segment must be replanned from actual new shape in any future native trial.')
  try:
   current={k:v.copy() for k,v in qs.items()};lowest=float('inf');lim=None;stage_min=[]
   for target in [start+[0,0,dz],start+[0,dy,dz]]:
    pts=planner.plan_cartesian(current,{'right':target},rotations={'right':original},samples=13);segment_min=float('inf')
    for pp in pts:
     ppqs={**current,'right':pp['q']['right']};checks=planner.clearance(ppqs,fingers);cat,c=min(checks.items(),key=lambda x:x[1]['distance']);segment_min=min(segment_min,c['distance'])
     if c['distance']<lowest:lowest=c['distance'];lim={'category':cat,**c,'fraction_in_segment':pp['fraction']}
    current={**current,'right':pts[-1]['q']['right']};stage_min.append(segment_min)
   row.update(status='coarse_rigid_clear' if lowest>=.008 else 'sampled_rigid_rejected',minimum_sampled_rigid_clearance_m=lowest,limiting_pair=lim,segment_minima_m=stage_min,actual_hand_clearance_preserved_for_frozen_translation=left_hand_minimum_z(st['q_left'],cfg,planner=planner)-float(st['bear'][:,2].max()+dz)>=.005)
  except ValueError as e:row.update(status='ik_rejected',error=str(e))
  extra.append(row)
print('center variants',[(x['first_down_m'],x['second_y_m'],x['status'],x.get('minimum_sampled_rigid_clearance_m'),x.get('limiting_pair',{}).get('bodies')) for x in extra],flush=True)
out=dict(kind='full16_refined_frozen_alignment_diagnostic',physical_success=False,runtime_execution_performed=False,entry_replacement_candidates=selected,center_hypothetical_pure_y_or_lower_then_y=extra,limitations=['No cloth nodes are constrained or moved in any live simulation. Frozen translations only support bounded geometric planning.','All contact forces, deformation, grab retention and final real containment remain untested for these candidates.','The full entry candidate test uses 5mm continuous frozen bear-left-robot sweep and 8mm sampled right-robot trajectory with complete original rotation, but exceeds old engineered XY budget.','Body section and containment diagnostics refer to frozen cloth; contact can move the mouth and body substantially.'])
(RUN/'independent-alignment-refined.json').write_text(json.dumps(out,indent=2,default=scalar)+'\n')
