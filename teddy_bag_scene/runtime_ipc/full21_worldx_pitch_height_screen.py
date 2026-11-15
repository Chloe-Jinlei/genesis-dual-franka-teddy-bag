"""Two signed world-X orientations; stop at the unchanged height gate."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21'
sys.path.insert(0,str(RUN/'source'))
from validate import mouth_structure,signed_volumes
from contact_insertion import _policy,_state_context,_mouth
from insertion_candidates import polygon_centroid,projected_inside
from postlift_yaw import _serial
cfg=json.loads((RUN/'config.json').read_text());cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
rest=dict(np.load(RUN/'rest.npz'));state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
policy=_policy(cfg);ctx=_state_context(rest,state,cfg,policy);assert not ctx['reasons']
polygon,triangles,mouth=_mouth(rest,state,cfg);anchor=polygon_centroid(polygon)
bear=state['bear'];tcp=state['tcp_right'];com=ctx['com'];actual=ctx['rotation'];tets=rest['bear_tets']
bottom=float(cfg['table']['height'])+policy['bottom_clearance_m'];top=mouth['rim_z_min_m']-policy['endpoint_below_lowest_rim_m'];capacity=top-bottom
report=dict(kind='full21_two_world_x_negative_pitch_frozen_height_screen',physical_success=False,native_executed=False,gpu_used=False,
    production_modified=False,source_actual_state=str(RUN/'orient_teddy_postlift.npz'),source_time_s=float(state['time']),
    source_actual_full_rotation_xyzw=actual.as_quat(),source_actual_physical=ctx['physical'],
    original_base_and_scale_and_material_unchanged=True,source_3d_dimensions_m=np.ptp(bear,axis=0),
    actual_body_rim_vertex_count=len(mouth['ordered_rim_indices']),actual_rim_z_min_m=mouth['rim_z_min_m'],
    actual_rim_z_max_m=mouth['rim_z_max_m'],actual_projected_mouth_area_centroid_xy_m=anchor,
    unchanged_height_rule=dict(table_height_m=cfg['table']['height'],bottom_clearance_m=policy['bottom_clearance_m'],
        minimum_frozen_bear_bottom_z_m=bottom,required_margin_below_actual_minimum_rim_m=policy['endpoint_below_lowest_rim_m'],
        maximum_frozen_bear_top_z_m=top,available_height_m=capacity),cases=[],
    scope=['Diagnostic point copies undergo a proper rigid rotation about the measured TCP; no native particle,robot joint,asset,or config is changed.',
        'Target rotation is G_worldX * actual_full_hand_rotation, not a right multiplication or an Euler replacement.',
        'Height rejection is conditional on the actual observed teddy remaining this frozen shape. It is not proof that a deformable teddy could never change shape naturally.',
        'The existing plan does not assume compression to evade the lowest-rim/table-clearance rule. No gate is lowered.',
        'COM centering translation is an unexecuted geometric diagnostic, not opening/body containment.',
        'Both heights fail, so body/handle sweeps and IK are intentionally not attempted; no search expansion.'],
    inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'config.json',RUN/'rest.npz',RUN/'orient_teddy_postlift.npz',RUN/'source/motion.py',RUN/'source/contact_insertion.py',RUN/'source/validate.py',Path(__file__)]})
for degrees in [-60.,-90.]:
    G=Rotation.from_rotvec([np.deg2rad(degrees),0.,0.]);target=G*actual
    points=G.apply(bear-tcp)+tcp;rotated_com=G.apply(com-tcp)+tcp
    dims=np.ptp(points,axis=0);inside,dist=projected_inside(np.asarray([rotated_com[:2]]),polygon)
    center_translation=anchor-rotated_com[:2]
    floor_translation=bottom-float(points[:,2].min())
    maximum_top=float(points[:,2].max()+floor_translation)
    case=dict(world_x_rotation_degrees=degrees,world_axis=[1.,0.,0.],full_target_rotation_xyzw=target.as_quat(),
        target_times_actual_inverse_rotvec_rad=(target*actual.inv()).as_rotvec(),rotation_about_tcp_m=tcp,
        frozen_3d_dimensions_m=dims,frozen_bounds_world_m=[points.min(0),points.max(0)],frozen_mass_com_world_m=rotated_com,
        frozen_untranslated_com_in_actual_rim_projection=bool(inside[0]),frozen_untranslated_com_projection_boundary_distance_m=float(dist[0]),
        xy_translation_to_actual_mouth_area_centroid_m=center_translation,centered_com_world_xy_m=anchor,
        original_frozen_height_gate_pass=bool(dims[2]<=capacity),height_margin_m=float(capacity-dims[2]),
        minimum_possible_top_at_original_floor_clearance_m=maximum_top,
        excess_over_allowed_top_m=float(maximum_top-top),status='rejected_original_frozen_height_gate',
        ik_assessed=False,body_and_front_handle_sweep_assessed=False,
        rigid_rotation_preserves_tetra_signed_volumes_max_abs_error_m3=float(np.max(np.abs(signed_volumes(points,tets)-signed_volumes(bear,tets)))))
    assert not case['original_frozen_height_gate_pass']
    report['cases'].append(case)
path=RUN/'independent-worldx-pitch-height-screen.json'
path.write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),available_height_m=capacity,
    cases=[dict(degrees=c['world_x_rotation_degrees'],dimensions_m=c['frozen_3d_dimensions_m'].tolist(),height_margin_m=c['height_margin_m'],status=c['status']) for c in report['cases']]),indent=2))
