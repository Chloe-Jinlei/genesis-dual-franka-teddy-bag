"""One CPU-only long-handle candidate; never overwrite an existing asset/config.

The 0.46 x 0.40 m two-panel body is unchanged. Each handle's two straight
legs grow from 0.04 to 0.18 m; its original top arc and attachment roots remain.
"""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
import two_panel_builder as builder
from contact_grasp import cloth_grasp_frame
from contact_insertion import _serial
from motion import Planner,SIDES,hold_trajectory
from postlift_yaw import _clearance_rows
HERE=Path(__file__).resolve().parent
STEM='two_panel_bag_wide_tall_long_handles_coarse'
METADATA='metadata_two_panel_wide_tall_long_handles_coarse.json'
CONFIG=HERE/'config_long_handles_gravity_trial.json'
REPORT=HERE/'runtime_ipc/long-handles-cpu-validation.json'
EXTRA=.18
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    outputs=[builder.ASSETS/(STEM+s) for s in ['.obj','.mtl','_collision.obj','.npz']]+[builder.ASSETS/METADATA,CONFIG,REPORT]
    if any(p.exists() for p in outputs):raise FileExistsError('Candidate outputs already exist; refusing to overwrite any asset/config/report')
    protected=[p for p in builder.ASSETS.iterdir() if p.is_file()]+[p for p in HERE.glob('config*.json')]+[HERE/p for p in ['run.py','scene.py','motion.py','contact_insertion.py','postlift_yaw.py','gravity_release.py']]
    before={str(p):sha(p) for p in protected}
    base=json.loads((HERE/'config_gravity_seating_trial.json').read_text())
    builder.W,builder.H=.46,.40
    point,geometry=builder.opened_panel_geometry()
    bag=builder.build_bag(.02,STEM,point,geometry,handle_extra=EXTRA)
    oldmeta=json.loads((builder.ASSETS/base['bag']['metadata']).read_text())
    sources={p:sha(HERE/p) for p in ['build_long_handle_bag.py','two_panel_builder.py','assets_builder.py']}
    metadata={'units':'metres','generator':'build_long_handle_bag.py','construction_source':'two_panel_builder.py','generator_source_sha256':sources,'candidate_status':'CPU geometry only; native cloth/hold/contact validation pending','bag':bag,'bear':oldmeta['bear'],'bear_metadata_scope':'Legacy inherited reference only; actual configured bear mesh/material and native rest are authoritative. No bear asset changed.','actual_configured_bear_mesh':base['bear']['mesh'],'handle_change':{'old_vertical_leg_extra_m':.04,'new_vertical_leg_extra_m':EXTRA,'added_length_per_leg_m':.14,'added_length_per_handle_m':.28,'body_unchanged':True,'mass_preservation_rescaling_applied':False}}
    (builder.ASSETS/METADATA).write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    config=copy.deepcopy(base);config['bag'].update(mesh=bag['file'],npz_file=bag['npz_file'],metadata=METADATA)
    for target in config['task']['open_targets'].values():target[2]=1.40
    config['task']['contact_trial']['postlift_yaw']['world_z_degrees']=15.
    CONFIG.write_text(json.dumps(config,indent=2,allow_nan=False)+'\n')
    new=dict(np.load(builder.ASSETS/bag['npz_file'],allow_pickle=False));old=dict(np.load(builder.ASSETS/base['bag']['npz_file'],allow_pickle=False))
    body=old['body'];bf=old['body_face_indices'];checks={
      'exact_body_vertex_positions':np.array_equal(new['points'][new['body']],old['points'][body]),
      'exact_body_face_indices_and_winding':np.array_equal(new['faces'][new['body_face_indices']],old['faces'][bf]),
      'exact_body_node_indices':np.array_equal(new['body'],body),
      'exact_U_seam_and_rim_and_panels':all(np.array_equal(new[k],old[k]) for k in ['u_seam','bottom_seam','seam_left','seam_right','rim','panel_front_grid','panel_back_grid','panel_front_face_indices','panel_back_face_indices','material_x','material_height']),
      'roots_exact_positions':all(np.array_equal(new['points'][new['handle_'+side+'_root_'+end]],old['points'][old['handle_'+side+'_root_'+end]]) for side in ['front','back'] for end in ['left','right']),
      'zero_width_bottom':bool(np.ptp(new['points'][new['bottom_seam'],1])==0),
      'one_manifold_three_loops':bag['mesh']['connected_components']==1 and bag['mesh']['nonmanifold_edges']==0 and bag['mesh']['boundary_loops']==3 and bag['mesh']['consistent_winding'],
      'body_one_opening_no_gusset':bag['body_mesh']['boundary_loops']==1 and not bag['body_mesh']['watertight'],
      'no_degenerate_triangles':bag['mesh']['degenerate_triangles']==0,
      'no_mass_rescaling': 'production_rho_multiplier_for_equal_mass' not in bag,
      'each_handle_centerline_grows_028':all(abs(bag['handles'][s]['centerline_length_m']-oldmeta['bag']['handles'][s]['centerline_length_m']-.28)<1e-12 for s in ['front','back'])}
    oldarea=oldmeta['bag']['mesh']['triangle_area_m2'];newarea=bag['mesh']['triangle_area_m2'];area_density=config['bag']['rho']*2*config['bag']['thickness']
    report=dict(status='initial_IK_pending',physical_success=False,native_execution_performed=False,configuration=str(CONFIG),source_config=str(HERE/'config_gravity_seating_trial.json'),source_sha256=sources,topology_checks=checks,mesh=bag['mesh'],handle_geometry=bag['handles'],cloth_mass={'old_mesh_area_m2':oldarea,'new_mesh_area_m2':newarea,'added_area_m2':newarea-oldarea,'config_rho_kg_m3':config['bag']['rho'],'config_thickness_radius_m':config['bag']['thickness'],'physical_full_thickness_m':2*config['bag']['thickness'],'config_areal_density_kg_m2':area_density,'old_area_times_rho_full_thickness_kg':oldarea*area_density,'new_area_times_rho_full_thickness_kg':newarea*area_density,'added_area_times_rho_full_thickness_kg':(newarea-oldarea)*area_density,'metadata_suggested_areal_density_kg_m2':bag['suggested_material']['surface_density_kg_m2'],'metadata_estimated_mass_kg':bag['estimated_mass_kg'],'scope':'Area*density*(2*IPC thickness radius) is derived mass, not native readback; metadata estimate uses its separately declared suggested 0.24 kg/m2. Neither is forced to old mass.'},initial={},limitations=['No native/GPU execution; no grasp retention or usable deformed mouth/insert clearance claim.','Only initial IK and existing spread path tested; later right retreat, teddy pickup and insertion remain untested.','401 samples cover original seven rigid categories; continuous joint/rate extrema are numerical polynomial checks, not a continuous collision proof.','No soft-cloth/robot intersection or IPC initialization proof follows from rigid robot separation.'])
    def save():
      report['protected_existing_files_unchanged']=all(sha(p)==h for p,h in before.items());report['protected_existing_file_count']=len(before);report['asset_sha256']={str(p.relative_to(HERE)):sha(p) for p in outputs if p.exists() and p!=REPORT};REPORT.write_text(json.dumps(_serial(report),indent=2,allow_nan=False)+'\n')
    save();assert all(checks.values()),str(checks)
    planner=Planner(config);qs={};fingers={};rotations={}
    for side in SIDES:
      spec=config['task']['handle_grasps'][side];frame=cloth_grasp_frame(new['points'],new['faces'],new[spec['group']],shell_radius=config['bag']['thickness'],preferred_approach=spec['preferred_approach']);target=np.asarray(config['bag']['position'])+frame.position
      q,e,a=planner.solve(side,target,frame.rotation);qs[side]=q;fingers[side]=max(.001,frame.minimum_half_gap);rotations[side]=frame.rotation
      report['initial'][side]=dict(target_tcp_m=target,q_arm_rad=q,finger_q_m=fingers[side],position_error_m=e,rotation_error_rad=a,grasp_frame=frame.as_dict(),reachable=bool(np.isfinite([e,a]).all() and e<=.0005 and a<=.002))
    save()
    try:
      if not all(v['reachable'] for v in report['initial'].values()):raise ValueError('Initial exact declared grasp IK failed 0.5mm/2mrad; no target search allowed')
      initial=planner.clearance(qs,fingers);_clearance_rows(initial,'initial');report['initial_clearance']=initial
      minima={};closed=dict.fromkeys(SIDES,0.)
      # Runtime original spread speed retained; no production speed change.
      speed=float(config['task']['motion_speed_scale']);plan=planner.trajectory(qs,config['task']['open_targets'],rotations=rotations,duration=2.5/speed,finger_start=closed,finger_end=closed,max_joint_velocity=.55*speed,max_joint_acceleration=speed**2,clearance_margin=.008,validate_samples=201)
      for t in np.linspace(0.,plan.duration,401):
        sample=plan.sample(float(t));raw=planner.clearance(sample['q'],sample['finger']);_clearance_rows(raw,'spread')
        for key,val in raw.items():
          if key not in minima or val['distance']<minima[key]['distance']:minima[key]=dict(val,time_s=float(t))
      rates=plan.continuous_rate_extrema();vmax=max(float(np.max(v['maximum_velocity_rad_s'])) for v in rates.values());amax=max(float(np.max(v['maximum_acceleration_rad_s2'])) for v in rates.values());assert np.isfinite([vmax,amax]).all() and vmax<=.55*speed and amax<=speed**2
      report['open_route']=dict(passed_cpu_geometry=True,duration_s=plan.duration,ik_samples=45,raw_rigid_sample_count=401,minimum_rigid_clearance_m=min(v['distance'] for v in minima.values()),clearance_minima=minima,reference_velocity_limit_rad_s=.55*speed,reference_acceleration_limit_rad_s2=speed**2,continuous_rate_extrema=rates,commanded_finger_q_m=closed,endpoint_tcp_error_m={s:float(np.linalg.norm(planner.fk(s,plan.end_q()[s])[0]-config['task']['open_targets'][s])) for s in SIDES},end_q=plan.end_q())
      np.savez_compressed(HERE/'runtime_ipc/long-handles-spread-plan.npz',duration=plan.duration,**{s+'_x':plan.curves[s].x for s in SIDES},**{s+'_c':plan.curves[s].c for s in SIDES});report['status']='CPU_initial_IK_and_spread_passed_native_pending'
    except ValueError as exc:report.update(status='rejected_initial_or_spread',rejection=str(exc))
    save();print(json.dumps(_serial({k:report[k] for k in ['status','initial','cloth_mass','protected_existing_files_unchanged']}),indent=2));print('REPORT',REPORT,sha(REPORT),flush=True)
if __name__=='__main__':main()
