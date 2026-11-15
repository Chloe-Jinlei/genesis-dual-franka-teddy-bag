"""Resolve exact linear-translation parameters of already recorded convex sweep witnesses."""
from pathlib import Path
import json,hashlib,numpy as np
from scipy.optimize import linprog
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/two_panel_handoff_upright_offset_01/independent-hybrid-postlift-route';OLD=ROOT/'runs/two_panel_full_21'
a=json.loads((OUT/'guarded_candidate.json').read_text());r=a['report'];state=np.load(OUT/'hybrid_hypothetical_observation.npz');rest=np.load(OUT/'rest.npz');cfg=json.loads((OUT/'config.json').read_text());anchor=np.array(r['observed_mouth_area_centroid_xy_m'])
rows=[]
for attempt in r['robot_centering_attempts']:
 s=attempt['left_robot_teddy_sweep_segments'][-1];geom=next(g for g in s['per_geom'] if g['sweep_certificate']['status']!='clear');cert=geom['sweep_certificate'];w=cert['first_uncertified_tetrahedron'];tet=int(w['tetrahedron_index']);ids=rest['bear_tets'][tet]
 assert w['solid_envelope_intersection_proven'] and cert['exact_tetra_prism_union']
 C=np.array(w['common_interior_ball']['center_world_m']);d=np.array(s['displacement_m']);start=state['bear'][ids]+s['translation_m']
 eq=np.zeros((4,6));eq[:3,:4]=start.T;eq[:3,4]=d;eq[3,:4]=1.;rhs=np.r_[C,1.];ub=np.zeros((4,6));ub[:,:4]=-np.eye(4);ub[:,5]=1
 result=linprog([0,0,0,0,0,-1],A_eq=eq,b_eq=rhs,A_ub=ub,b_ub=np.zeros(4),bounds=[(0.,1.)]*5+[(0.,1.)],method='highs')
 assert result.success;weights=result.x[:4];fraction=float(result.x[4]);res=float(np.linalg.norm(weights@start+fraction*d-C));assert res<1e-10 and weights.min()>0 and 0<=fraction<=1
 local=(rest['bear'][ids].mean(0)-cfg['bear']['position'])/.85;where='foot/leg' if local[1]<-.10 else 'head/ear' if local[1]>.09 else 'torso/arm'
 rows.append(dict(offset_from_mouth_centroid_m=(np.array(attempt['center_of_mass_xy_target_m'])-anchor).tolist(),first_blocking_stage=s['stage'],first_blocking_left_body=geom['body'],original_geom_id=geom['geom_id'],original_geom_type=geom['original_geom_type'],tetrahedron_index=tet,tetrahedron_vertices=ids.tolist(),rest_material_centroid_original100percent_local_m=local.tolist(),coarse_region=where,
 continuous_translation_fraction=fraction,translation_start_m=s['translation_m'],translation_displacement_m=d.tolist(),witness_world_m=C.tolist(),witness_tetra_barycentric=weights.tolist(),minimum_barycentric=float(weights.min()),reconstruction_residual_m=res,
 common_swept_tetra_prism_and_left_convex_solid_interior_ball_radius_m=w['common_interior_ball']['radius_m'],mujoco_convex_signed_distance_diagnostic_m=w['mujoco_convex_signed_distance_diagnostic_m'],wholebear_left_required_gap_m=s['required_gap_m'],lower_gap_certificate_m=s['conservative_gap_lower_bound_m']))
report=dict(kind='hybrid_existing_four_route_left_hand_witness_resolution',physical_success=False,native_execution=False,rows=rows,scope=['Recorded common-interior point is reconstructed as a strictly positive tetra barycentric point at one explicit fraction of the declared straight X translation.','The tetra sweep is exactly tet+[0,d]; convex-hull approximation of the whole teddy is not used for this witness.','The left obstacle is the planner MuJoCo convex collision solid; original nonconvex rendered/native mesh surface distance was not computed.','Coarse region is based solely on original100percent rest local Y: foot/leg<-0.10, head/ear>0.09, otherwise torso/arm.','No actual simultaneous native hybrid state, collision impulse or native penetration is inferred.'])
path=OUT/'blocking-witnesses.json';path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows,indent=2));print(path,hashlib.sha256(path.read_bytes()).hexdigest())
