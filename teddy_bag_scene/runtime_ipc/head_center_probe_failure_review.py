"""Bounded post-run review of the declared 3 N head-grasp probe; CPU only."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_head_center_probe_01'
sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from validate import signed_volumes,mesh_edges
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((RUN/n).read_text())
cfg,report,manifest=read('config.json'),read('report.json'),read('input_sha256.json')
rest=dict(np.load(RUN/'rest.npz'));states=dict(np.load(RUN/'states.npz'));failed=dict(np.load(RUN/'failed_soft_state.npz'))
joint=np.load(RUN/'joint_trace.npy');soft=np.load(RUN/'soft_metrics.npy');coupling=read('coupling_trace.json')
files=[RUN/n for n in manifest]+[RUN/n for n in ('report.json','states.npz','rest.npz','failed_soft_state.npz','joint_trace.npy','soft_metrics.npy','coupling_trace.json','events.json')]
before={str(p):sha(p) for p in files}
cfgp=json.loads(json.dumps(cfg));cfgp['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml');planner=Planner(cfgp)
tet=rest['bear_tets'];v0=signed_volumes(rest['bear'],tet);weights=abs(v0)
origin=np.array(cfg['bear']['position']);ref=origin+cfg['task']['bear_grasp_local']
roi=np.flatnonzero(np.linalg.norm(rest['bear']-ref,axis=1)<cfg['task']['bear_grasp_tracking_radius_m'])
dt=float(cfg['simulation']['dt']);table=float(cfg['table']['height'])
def observe(t,bear,q,tcp_saved=None):
    tcp,R=planner.fk('right',q[:7]);relative=(bear[roi].mean(0)-tcp)@R.as_matrix()
    js=signed_volumes(bear,tet)/v0
    return dict(time_s=float(t),tcp_world_m=tcp,TCP_matches_saved_error_m=None if tcp_saved is None else float(np.linalg.norm(tcp-tcp_saved)),finger_half_openings_m=q[7:],bear_mass_COM_m=np.average(bear[tet].mean(1),weights=weights,axis=0),material_ROI_centroid_world_m=bear[roi].mean(0),material_ROI_centroid_hand_local_m=relative,bear_minimum_z_m=float(bear[:,2].min()),clearance_above_table_m=float(bear[:,2].min()-table),min_J=float(js.min()),min_J_tetrahedron=int(js.argmin()))
timeline=[]
for t in [0,.5,1.66,2.01,2.66,3.01,3.31,3.61]:
    k=int(np.argmin(abs(states['time']-t)));r=observe(states['time'][k],states['bear'][k],states['q_right'][k],states['tcp_right'][k]);r['saved_PD_finger_force_n']=states['force_right'][k,7:];timeline.append(r)
last=observe(failed['time'],failed['bear'],joint[-1,10:19]);last['robot_source']='Exact failure-time actual joint trace; TCP/FK recomputed with frozen model, not an archived complete failure frame.';timeline.append(last)
close=next(x for x in timeline if abs(x['time_s']-2.66)<1e-7)
ratios=signed_volumes(failed['bear'],tet)/v0
failure_rows=[]
for k in np.argsort(ratios)[:10]:
    vertices=failed['bear'][tet[k]];tcp,R=planner.fk('right',joint[-1,10:17])
    failure_rows.append(dict(tetrahedron=int(k),vertex_indices=tet[k],J=float(ratios[k]),rest_volume_m3=float(v0[k]),rest_centroid_local_m=rest['bear'][tet[k]].mean(0)-origin,failed_vertices_world_m=vertices,failed_centroid_world_m=vertices.mean(0),failed_centroid_hand_local_m=(vertices.mean(0)-tcp)@R.as_matrix()))
reaction_rows=[];unavailable_reaction_times=[]
for row in coupling:
    fingers=[x for x in row['links'] if x['entity']=='right_franka' and x['link'].endswith('_finger')]
    if not all(x.get('last_feedback_force_world_n') is not None for x in fingers):
        unavailable_reaction_times.append(row['time']);continue
    f=np.array([x['last_feedback_force_world_n'] for x in fingers]);axes=np.array([x['aim_opening_axis_world'] for x in fingers])
    reaction_rows.append(dict(time_s=row['time'],names=[x['link'] for x in fingers],forces_world_n=f,force_norms_n=np.linalg.norm(f,axis=1),opening_projections_n=np.einsum('ij,ij->i',f,axes),summed_force_on_robot_fingers_world_n=f.sum(0),equal_opposite_force_on_coupled_soft_world_n=-f.sum(0),reconstruction_max_error_n=float(max(np.max(abs(f[i]-x['com_translation_gradient_formula_n'])) for i,x in enumerate(fingers)))))
forcearray=np.array([r['forces_world_n'] for r in reaction_rows]);reaction_times=np.array([r['time_s'] for r in reaction_rows])
reaction_timeline=[min(reaction_rows,key=lambda r:abs(r['time_s']-t)) for t in [1.66,2.01,2.66,3.01,3.31,3.61]]
edges=mesh_edges(rest['bag_faces'])[0];l0=np.linalg.norm(rest['bag'][edges[:,1]]-rest['bag'][edges[:,0]],axis=1)
failed_stretch=float((np.linalg.norm(failed['bag'][edges[:,1]]-failed['bag'][edges[:,0]],axis=1)/l0).max())
checks=dict(
    frozen_manifest_matches=all(sha(RUN/n)==h for n,h in manifest.items()),
    live_production_still_equals_frozen=all(sha(ROOT/n.removeprefix('source/'))==h for n,h in manifest.items() if n.startswith('source/')),
    all_native_saved_arrays_finite=all(np.isfinite(x).all() for x in [*states.values(),*rest.values(),*failed.values(),joint,soft]),
    original_grasp_only_task=report['requested_task']=='grasp' and cfg['task']['bear_finger_force_limit']==3 and cfg['task']['bear_finger_target']==0,
    original_four_phase_order= [x['phase'] for x in report['events']]==['grasp_probe_settle','teddy_grasp_approach','close_teddy_gripper','lift_teddy'],
    only_partial_lift_no_success=not (RUN/'lift_teddy.npz').exists() and not report['requested_task_passed'] and report['status']=='failed',
    exact_failure_joint_soft_time_match=abs(joint[-1,0]-float(failed['time']))<1e-9 and np.array_equal(failed['metrics'],soft[-1]),
    every_step_joint_soft_coverage=len(joint)==round(report['simulated_seconds']/dt) and len(soft)==len(joint)+1 and abs(soft[0,0])<1e-9 and np.allclose(joint[:,0],soft[1:,0],atol=1e-9,rtol=0) and np.allclose(np.diff(soft[:,0]),dt,atol=1e-9,rtol=0),
    first_under_5percent_J_is_final_step=np.flatnonzero(soft[:,1]<.05).tolist()==[len(soft)-1],
    failed_J_and_stretch_independently_recompute=abs(ratios.min()-soft[-1,1])<1e-12 and abs(failed_stretch-soft[-1,4])<1e-12,
    all_other_soft_summary_gates_pass=soft[:,4].max()<=1.15 and soft[:,2:4].max()<=.002 and soft[:,5].min()==1,
    actual_complete_frame_TCP_matches_frozen_FK=all(r['TCP_matches_saved_error_m']<1e-9 for r in timeline[:-1]),
    sampled_contact_reaction_values_finite=np.isfinite(forcearray).all() and all(np.isfinite(r['opening_projections_n']).all() for r in reaction_rows),
    archived_exact_feedback_matches_COM_gradient=all(r['reconstruction_max_error_n']<1e-7 for r in reaction_rows),
    three_newton_PD_clamp_preserved_after_close=bool(np.max(abs(states['force_right'][states['time']>=1.66-1e-9,7:]))<=3.+1e-9),
)
checks['all_original_inputs_unchanged']=all(sha(p)==h for p,h in before.items())
out=dict(kind='independent_head_center_3N_probe_failure_review',status='failure_evidence_consistent' if all(checks.values()) else 'review_failed',physical_grasp_success=False,full_task_success=False,checks=checks,failed_checks=[k for k,v in checks.items() if not v],native_error=report['error'],failure_time_s=float(failed['time']),last_complete_robot_and_soft_frame_s=float(states['time'][-1]),lift=dict(start_s=2.66,planned_duration_s=report['events'][-1]['duration'],executed_s=float(failed['time'])-report['events'][-1]['start'],completed=False),failure_tetrahedra=failure_rows,under_5percent_tetrahedron_count=int(sum(ratios<.05)),inverted_tetrahedron_count=int(sum(ratios<=0)),material_ROI=dict(indices=roi,count=len(roi),reference_world_m=ref,radius_m=cfg['task']['bear_grasp_tracking_radius_m']),motion_and_deformation_timeline=timeline,
    from_close_end_to_failure=dict(tcp_world_displacement_m=last['tcp_world_m']-close['tcp_world_m'],ROI_world_displacement_m=last['material_ROI_centroid_world_m']-close['material_ROI_centroid_world_m'],ROI_hand_local_displacement_m=last['material_ROI_centroid_hand_local_m']-close['material_ROI_centroid_hand_local_m'],bear_COM_displacement_m=last['bear_mass_COM_m']-close['bear_mass_COM_m'],half_finger_opening_change_m=last['finger_half_openings_m']-close['finger_half_openings_m']),
    contact_reaction=dict(scope='Saved IPC STC right-finger reactions only; sparse complete-frame samples, not every-step reaction guard. No contact_guard_trace exists for this grasp-only branch.',rows=len(reaction_rows),unavailable_reaction_times_s=unavailable_reaction_times,first_time_s=float(reaction_times[0]),last_time_s=float(reaction_times[-1]),maximum_interval_s=float(np.diff(reaction_times).max()),per_finger_sampled_force_norm_peaks_n=np.linalg.norm(forcearray,axis=2).max(0),timeline=reaction_timeline,last_sample=reaction_rows[-1],no_claim_of_exact_failure_time_force=True),
    soft_metrics=dict(min_J=float(soft[:,1].min()),max_cloth_stretch=float(soft[:,4].max()),max_table_penetration_m=soft[:,2:4].max(0),final_ten_per_step_rows=soft[-10:]),
    interpretation=['The fingers kept closing during the early lift while the teddy remained supported near the table. No completed lift or stable suspended grasp was achieved.','The tracked head region moves downward relative to the upward-moving hand, with increasing local compression. The last sampled finger reaction has a net upward world-Z component on the robot, hence an equal opposite downward component on the coupled soft material. This supports sloped-cap squeezing and relative downward motion before the J gate, not a certified complete ejection or a friction-only sliding decomposition.','The last complete reaction sample is 40ms before the failure, so its force is not labelled the exact failure-time peak.','The failed tet centroid is on the positive-X upper-head region in rest coordinates, lateral to the central target. Material-specific contact identity is not encoded in the force trace.'],
    two_newton_followup=dict(production_change_performed=False,additional_evidence_requiring_cancellation=False,scope='No additional input/initialization defect was found that requires canceling the explicitly bounded 3N-to-2N probe. Retain the 5% J stop and all original monitors. This is not evidence that 2N will hold or avoid compression.',simple_coulomb_2muF_N=2*float(cfg['bear']['friction_mu'])*2,rest_mass_kg=float(weights.sum()*cfg['bear']['rho']),weight_N=float(weights.sum()*cfg['bear']['rho']*9.81),why_force_budget_is_not_a_grasp_proof='The simple 2 mu F estimate assumes appropriate opposing contact normals and available tangential friction. Curved sloping surfaces, evolving deformation and actual normal loads invalidate interpreting that scalar inequality as a holding guarantee.'),
    limitations=['Tracking a finite material ROI distinguishes relative motion from a rigid attachment but does not separate elastic deformation from true contact slip.','Rigid-FK reconstruction of the failure TCP uses actual failure-time joints. Native affine finger geometry at that exact time was not archived.','Tetrahedral J is a local volume ratio; minJ does not mean the entire head or teddy has that compression.','Full grasp-only success, full-task insertion, continuous contact-force peaks, and hardware safety are not certified.'],input_sha256=before)
path=RUN/'independent-grasp-failure-review.json';path.write_text(json.dumps(out,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
print(json.dumps(dict(path=str(path),sha256=sha(path),checks=len(checks),all_passed=all(checks.values()),failed=out['failed_checks'],failure_tet=int(ratios.argmin()),min_J=float(ratios.min()),sampled_force_peaks=out['contact_reaction']['per_finger_sampled_force_norm_peaks_n'].tolist(),relative_head_down_m=float(last['material_ROI_centroid_hand_local_m'][2]-close['material_ROI_centroid_hand_local_m'][2])),indent=2))
