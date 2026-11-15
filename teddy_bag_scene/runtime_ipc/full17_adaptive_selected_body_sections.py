"""Bounded frozen-body comparison, no runtime policy edits or native simulator."""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_17'
sys.path.insert(0,str(ROOT))
from validate import mouth_structure,tetra_boundary,signed_volumes,proper_segment_crossings,mesh_edges,winding
from insertion_candidates import projected_inside,polygon_centroid
from runtime_ipc.diagnose_low_region_winding import spatial_rim,capped_surface

# The previously reviewed actual-body cut method is reused verbatim. The prefix
# defines functions/imports only; its old analysis/output block is not executed.
method_path=Path('/tmp/full17_body_sections.py')
method_source=method_path.read_text().split('\nout={\'scope\':')[0]
namespace={};exec(compile(method_source,str(method_path),'exec'),namespace)
section=namespace['section'];insidedist=namespace['insidedist'];area=namespace['area']
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'lower_teddy_to_rim.npz'))
old=json.loads((RUN/'independent-entry-three-balanced-centers.json').read_text())
actual_review=json.loads((RUN/'independent-adaptive-center-actual-input-review.json').read_text())
yaw_review=json.loads((RUN/'independent-yaw-contingency-review.json').read_text())
body,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg)
bear_faces=tetra_boundary(state['bear'],rest['bear_tets'])
weights=np.abs(signed_volumes(rest['bear'],rest['bear_tets']))
com=np.average(state['bear'][rest['bear_tets']].mean(1),axis=0,weights=weights)
heights=np.array(old['height_grid_m']);dz=float(cfg['table']['height']+.01-state['bear'][:,2].min())
bagcuts=[section(state['bag'],body,float(z)) for z in heights]
bearcuts=[section(state['bear']+np.array([0.,0.,dz]),bear_faces,float(z)) for z in heights]
assert all(q['closed_valid'] and q['loop_count']==1 for q in bagcuts)
assert all(q['closed_valid'] for q in bearcuts)
assert np.all(heights<=state['bag'][rim,2].min()-.002)
assert all(np.allclose(q['loops'][0],old['body_cuts'][i]['loops'][0],rtol=0,atol=1e-12) for i,q in enumerate(bagcuts))
bodymesh=trimesh.Trimesh(state['bag'],body,process=False)
clothmesh=trimesh.Trimesh(state['bag'],rest['bag_faces'],process=False)
_,_,boundary=mesh_edges(body)
spatial=spatial_rim(state['bag'][rim],rest['bag'][rim],cfg['bag']['thickness'])
apex=state['bag'][rim].mean(0)
caps=[capped_surface(state['bag'],body,boundary,a) for a in [apex,apex+np.array([.031,-.023,.15])]]
cap_meshes=[trimesh.Trimesh(v,f,process=False) for v,f in caps]
orientation_pass=bool(bodymesh.is_winding_consistent and clothmesh.is_winding_consistent and all(m.is_watertight and m.is_winding_consistent for m in cap_meshes))

def frozen_containment(final_bear,final_com):
 queries=np.vstack([final_bear,final_com]);heightgap=float(state['bag'][rim,2].min()-queries[:,2].max())
 result=dict(scope='Frozen hypothetical final points against actual full17 bag, NOT an actual executed final state',
  height_precondition_met=heightgap>=.002,minimum_query_below_rim_margin_m=heightgap,
  all_whole_bear_volume_containment_certified=False,physical_success=False)
 if heightgap<.002:return result
 a,b=[winding(queries,v,f) for v,f in caps]
 unit=(np.abs(np.abs(a)-1)<=1e-6)&(np.abs(np.abs(b)-1)<=1e-6)
 residual=float(max(np.max(abs(a-np.rint(a))),np.max(abs(b-np.rint(b)))))
 difference=float(np.max(abs(a-b)));multi=bool(np.any(abs(np.rint(a))>1)|np.any(abs(np.rint(b))>1))
 closest,distance,triangles=trimesh.proximity.closest_point(bodymesh,queries)
 guards=bool(orientation_pass and spatial['passed'] and residual<=1e-6 and difference<=1e-8 and not multi and distance.min()>1e-6)
 result.update(cap_agreement_max_difference=difference,max_integer_residual=residual,multiple_winding_detected=multi,
  numerical_topology_spatial_rim_and_near_surface_guards_pass=guards,
  unit_winding_bear_vertex_count=int(unit[:-1].sum()),total_bear_vertices=len(final_bear),com_unit_winding=bool(unit[-1]),
  all_vertices_and_com_have_guarded_unit_winding=bool(guards and unit.all()),outside_bear_vertex_count=int((~unit[:-1]).sum()),
  minimum_body_vertex_distance_m=float(distance[:-1].min()),com_actual_body_distance_m=float(distance[-1]),
  maximum_outside_bear_vertex_body_distance_m=float(distance[:-1][~unit[:-1]].max()) if (~unit[:-1]).any() else 0.,
  near_body_1um_query_indices=np.flatnonzero(distance<=1e-6).tolist(),
  limitation='Pointwise winding only; positive vertex results would not establish entire tetrahedral volume containment, physical cloth nonintersection, release or stability. Any clearly exterior vertex rules out all-point containment of this frozen hypothesis.')
 if (~unit[:-1]).any():
  outside=np.flatnonzero(~unit[:-1]);worst=int(outside[np.argmax(distance[outside])])
  result['farthest_outside_vertex']=dict(index=worst,frozen_world_m=final_bear[worst].tolist(),
   reference_bear_local_m=(rest['bear'][worst]-cfg['bear']['position']).tolist(),
   closest_body_point_m=closest[worst].tolist(),closest_body_face_index=int(triangles[worst]),body_distance_m=float(distance[worst]))
 return result

candidates=[(0.,[-.015,-.045]),(0.,[-.0165,-.0535]),(0.,[-.015,-.055]),(0.,[-.02,-.05]),(45.,[-.025,-.05])]
rows=[]
for yaw_deg,raw_delta in candidates:
 delta=np.asarray(raw_delta);rot=Rotation.from_euler('z',yaw_deg,degrees=True)
 frozen_bear=rot.apply(state['bear']-state['tcp_right'])+state['tcp_right']
 frozen_com=rot.apply(com-state['tcp_right'])+state['tcp_right']
 if yaw_deg:
  use_bearcuts=[section(frozen_bear+np.array([0.,0.,dz]),bear_faces,float(z)) for z in heights]
  assert all(c['closed_valid'] for c in use_bearcuts)
 else:use_bearcuts=bearcuts
 q=frozen_com[:2]+delta;env=np.asarray(old['contact_envelope_xy_m']);rimxy=state['bag'][rim,:2]
 ein,ed=projected_inside(q[None],env);rin,rd=projected_inside(q[None],rimxy)
 parts=[]
 for z,bc,tc in zip(heights,bagcuts,use_bearcuts):
  loops=[np.asarray(p) for p in bc['loops']];cen=polygon_centroid(loops[0]);inside,dist=insidedist(q[None],loops)
  seg=np.asarray(tc['segments']);samples=np.concatenate([seg[:,0],seg.mean(1)])+delta
  ii,dd=insidedist(samples,loops)
  parts.append(dict(height_m=float(z),com_axis_inside=bool(inside[0]),com_axis_signed_clearance_m=float(dist[0]*(1 if inside[0] else -1)),
   com_axis_centroid_distance_m=float(np.linalg.norm(q-cen)),frozen_teddy_section_sample_count=len(samples),
   frozen_teddy_section_samples_inside_fraction=float(ii.mean()),maximum_frozen_sample_outside_distance_m=float(dd[~ii].max()) if (~ii).any() else 0.,
   actual_body_section_centroid_xy_m=cen.tolist()))
 summary=dict(section_count=len(parts),com_axis_inside_count=sum(p['com_axis_inside'] for p in parts),
  mean_com_axis_signed_clearance_m=float(np.mean([p['com_axis_signed_clearance_m'] for p in parts])),
  minimum_com_axis_signed_clearance_m=min(p['com_axis_signed_clearance_m'] for p in parts),
  mean_com_axis_centroid_distance_m=float(np.mean([p['com_axis_centroid_distance_m'] for p in parts])),
  mean_frozen_cut_sample_inside_fraction=float(np.mean([p['frozen_teddy_section_samples_inside_fraction'] for p in parts])),
  worst_frozen_cut_sample_outside_distance_m=max(p['maximum_frozen_sample_outside_distance_m'] for p in parts),
  worst_com_axis_height_m=parts[int(np.argmin([p['com_axis_signed_clearance_m'] for p in parts]))]['height_m'])
 if yaw_deg:
  candidate=next(x for x in next(y for y in yaw_review['results'] if y['world_z_yaw_degrees']==45.)['center_remainders'] if np.array_equal(x['center_delta_m'],np.r_[delta,0.]))
  rigidmin=min(candidate[k]['minimum_sampled_rigid_clearance_m'] for k in ['center','insert','release','withdraw']);score=None
 else:
  candidate=next(x for x in actual_review['candidate_summaries'] if np.array_equal(x['offset_xy_m'],delta))
  rigidmin=candidate['minimum_center_insert_release_rigid_clearance_m'];score=candidate['score_m']
 final_bear=frozen_bear+np.r_[delta,dz];final_com=frozen_com+np.r_[delta,dz]
 rows.append(dict(world_z_yaw_degrees=yaw_deg,offset_xy_m=delta.tolist(),hypothetical_center_tcp_world_m=(state['tcp_right']+np.r_[delta,0.]).tolist(),
  hypothetical_center_com_world_m=(frozen_com+np.r_[delta,0.]).tolist(),hypothetical_final_com_world_m=final_com.tolist(),
  contact_envelope_com_signed_margin_m=float(ed[0]*(1 if ein[0] else -1)),ordered_rim_projection_signed_margin_m=float(rd[0]*(1 if rin[0] else -1)),
  runtime_selection_rigid_minimum_m=rigidmin,runtime_selection_score_m=score,
  body_section_summary=summary,sections=parts,frozen_pointwise_containment=frozen_containment(final_bear,final_com)))

# Confirm the two repeated legacy candidates reproduce their prior diagnostics.
legacy=[]
for row in rows:
 matches=[x for x in old['candidates'] if np.array_equal(x['offset_xy_m'],row['offset_xy_m'])]
 if matches:
  expected=matches[0]['body_section_summary'];now=row['body_section_summary']
  legacy.append(dict(offset_xy_m=row['offset_xy_m'],maximum_matching_summary_absolute_difference=max(abs(now[k]-v) for k,v in expected.items())))

selected=rows[0];comparisons=[]
for other in rows[1:4]:
 a=selected['body_section_summary'];b=other['body_section_summary']
 per=[dict(height_m=x['height_m'],COM_signed_margin_change_m=x['com_axis_signed_clearance_m']-y['com_axis_signed_clearance_m'],
  contour_sample_inside_fraction_change=x['frozen_teddy_section_samples_inside_fraction']-y['frozen_teddy_section_samples_inside_fraction']) for x,y in zip(selected['sections'],other['sections'])]
 comparisons.append(dict(selected_offset_xy_m=selected['offset_xy_m'],reference_offset_xy_m=other['offset_xy_m'],
  selected_minus_reference_summary={k:a[k]-b[k] for k in a if k not in ['section_count','worst_com_axis_height_m']},
  selected_minus_reference_runtime_score_m=selected['runtime_selection_score_m']-other['runtime_selection_score_m'],per_height=per))

report=dict(kind='full17_adaptive_selected_frozen_true_body_section_comparison',physical_success=False,runtime_execution_performed=False,
 source_config_or_policy_modified=False,actual_time_s=float(state['time']),actual_com_world_m=com.tolist(),actual_tcp_world_m=state['tcp_right'].tolist(),
 actual_rim_min_z_m=float(state['bag'][rim,2].min()),actual_rim_max_z_m=float(state['bag'][rim,2].max()),
 height_grid_m=heights.tolist(),frozen_final_z_translation_m=dz,frozen_final_bottom_m=.75,
 body_cut_method='Physical body triangle-plane intersections, endpoint join tolerance 1e-9 m, original ordered closed loops; no bag convex hull or rim-shadow substitution.',
 body_sections_all_single_closed_loop=True,recomputed_body_loops_match_previous_method=True,legacy_summary_comparison=legacy,
 actual_body_winding_orientation_and_cap_closure_pass=orientation_pass,actual_rim_3d_diagnostic=spatial,
 yaw_contingency_scope='Fifth candidate uses intended +45-degree WORLD-Z pure frozen yaw about actual TCP, then XY and final Z translation. Existing actual-FK vs commanded-baseline discrepancy can move a rigidly grasped endpoint by up to 0.108737 mm. No assumption of real soft-grasp rotation retention, or current-policy approval, is made.',
 outline_method='Original contour endpoint/midpoint discrete metric retained at the same 12 heights; these are unweighted sampled frozen sections, not area/volume fractions or dynamic passage clearance.',
 candidates=rows,selected_vs_references=comparisons,
 input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'rest.npz',RUN/'lower_teddy_to_rim.npz',RUN/'config.json',RUN/'independent-entry-three-balanced-centers.json',RUN/'independent-adaptive-center-actual-input-review.json']},
 prior_section_method_sha256=hashlib.sha256(method_path.read_bytes()).hexdigest(),
 bounded_conclusion=dict(
  translation_selection='On the same frozen full17 body, selected [-15,-45] mm loses COM-axis clearance at all 12 sampled heights versus the three alternatives, increases mean distance to section centroids, slightly lowers mean discrete contour coverage, and increases the worst sampled exterior distance. It buys robot clearance, while current contact-envelope scoring does not measure this actual body alignment.',
  next_iteration='If a later physical failure motivates revision, use newly observed body alignment as an additional diagnostic when candidates have similar current scores; do not change the ongoing full19 policy or turn frozen section metrics into unvalidated physical acceptance thresholds.',
  yaw='The +45-degree frozen yaw improves upper-section coverage and reduces the number of exterior vertices, but creates a substantially larger localized exterior protrusion at middle heights. It is not a uniformly better body fit and is not evidence of actual yaw/grasp feasibility.',
  containment='All five hypotheses meet the low-region precondition and all numerical/topology guards, but all have clearly exterior teddy vertices. None is whole-bear-contained in the unchanged captured bag; future real cloth deformation is not predicted by this test.'),
 limitations=['Frozen full17 actual-entry cloth shape is held fixed. It is neither a rigid aperture nor a prediction of full19, whose native cloth may differ.',
 'Center COM vertical-axis clearance is not full teddy containment. All four frozen teddy shapes exceed some captured body sections.',
 'Discrete outline samples are not area/volume fractions or a continuous outline-containment guarantee.',
 'The highest cut is 0.98 m, below the actual lowest rim 1.003277 m. No free opening or upper-rim passage is inferred.',
 'No soft strain, force, grasp, physical contact, release, settled containment, or route safety is decided by this diagnostic. Runtime gates and ongoing full19 stay unchanged.'])
out=RUN/'independent-adaptive-selected-body-sections.json'
out.write_text(json.dumps(report,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
fields=[('com_axis_signed_clearance_m',1000,'COM vertical-axis signed margin (mm)'),('frozen_teddy_section_samples_inside_fraction',100,'Contour samples inside (%)'),
 ('com_axis_centroid_distance_m',1000,'COM axis distance from cut centroid (mm)'),('maximum_frozen_sample_outside_distance_m',1000,'Worst sampled exterior distance (mm)')]
for row in rows:
 label=str([round(v*1000,1) for v in row['offset_xy_m']])+' mm'+(' SELECTED' if row is selected else '')+(' yaw +45' if row['world_z_yaw_degrees'] else '')
 for ax,(key,scale,title) in zip(axs.flat,fields):
  ax.plot(heights,[p[key]*scale for p in row['sections']],marker='o',markersize=3,label=label,linewidth=2 if row is selected else 1)
  ax.set_ylabel(title);ax.grid(alpha=.25)
for ax in axs[-1]:ax.set_xlabel('World cut height (m)')
axs[0,0].axhline(0,color='black',linewidth=.5);axs[0,0].legend(fontsize=8)
fig.suptitle('Full17 actual entry at 51.520 s — frozen final teddy vs true closed cloth cuts\nGeometric comparison only; cloth can deform and full19 can differ',fontsize=12)
fig.tight_layout();fig.savefig(RUN/'independent-adaptive-selected-body-sections.png',dpi=150);plt.close(fig)
for row in rows:print(json.dumps(dict(offset_mm=(np.array(row['offset_xy_m'])*1000).tolist(),summary=row['body_section_summary'])))
print(out)
