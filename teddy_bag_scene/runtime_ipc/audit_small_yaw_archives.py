"""Read-only finite/sample reconstruction audit of six frozen yaw contingency archives."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_17'
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,MotionPlan

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=RUN/'independent-small-yaw-contingencies.json';report=json.loads(p.read_text())
mathfile=RUN/'independent-yaw-path-math-review.json';mathreport=json.loads(mathfile.read_text())
cfgfile=RUN/'config.json';planner=Planner(json.loads(cfgfile.read_text()))
protected={str(x.relative_to(ROOT)):sha(x) for x in [p,mathfile,cfgfile,RUN/'lower_teddy_to_rim.npz',RUN/'source/motion.py',Path(__file__).resolve()]}
rows=[];releases=[];summary=[];max_difference=0.;raw_count=0
for yaw in report['results']:
 phases=[('rotation',yaw['rotation'])]
 for candidate in yaw['center_remainders']:
  phases += [(f"candidate{candidate['index']}_{phase}",candidate[phase]) for phase in ['center','insert','withdraw']]
  metrics=candidate['body_metrics']
  summary.append(dict(yaw_deg=yaw['world_z_yaw_degrees'],offset_mm=(np.asarray(candidate['center_offset_xy_m'])*1000).tolist(),
   minimum_sampled_rigid_clearance_mm=candidate['minimum_complete_route_sampled_rigid_clearance_m']*1000,
   inside_nodes=metrics['inside_node_count'],total_nodes=metrics['node_count'],signed_com_body_distance_mm=metrics['signed_com_body_distance_m']*1000,
   maximum_outside_node_body_distance_mm=metrics['maximum_outside_node_body_distance_m']*1000))
  pose={k:np.asarray(v) for k,v in candidate['release']['fixed_joint_pose_left_right'].items()}
  fingers=report['finger_geometry']['rigid_path_half_opening_approximations_m'];minimum=float('inf');count=0
  for fraction in np.linspace(0,1,21):
   ff={**fingers,'right':float((1-fraction)*fingers['right']+fraction*.04)}
   checks=planner.clearance(pose,ff);values=np.asarray([v['distance'] for v in checks.values()])
   assert values.size and np.isfinite(values).all();count+=len(values);minimum=min(minimum,float(values.min()))
  expected=candidate['release']['minimum_sampled_rigid_clearance_m'];assert abs(minimum-expected)<1e-12
  assert minimum>=.008
  releases.append(dict(yaw_deg=yaw['world_z_yaw_degrees'],index=candidate['index'],raw_distance_count=count,all_raw_finite=True,minimum_m=minimum))
 for label,evidence in phases:
  archive=evidence['trajectory_archive'];path=ROOT/archive['path'];assert sha(path)==archive['sha256']
  arrays=np.load(path,allow_pickle=False);assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
  curves={s:CubicSpline.construct_fast(arrays[s+'_spline_coefficients'],arrays[s+'_spline_knots_fraction'],extrapolate=True,axis=0) for s in ['left','right']}
  plan=MotionPlan(curves,float(arrays['duration_s']),dict(zip(['left','right'],arrays['finger_start_left_right_m'])),dict(zip(['left','right'],arrays['finger_end_left_right_m'])))
  plan.validate_joint_limits(planner.limits)
  minimum=float('inf');count=0
  for t in np.linspace(0,plan.duration,401):
   s=plan.sample(float(t));checks=planner.clearance(s['q'],s['finger']);values=np.asarray([v['distance'] for v in checks.values()])
   assert values.size and np.isfinite(values).all();count+=len(values);minimum=min(minimum,float(values.min()))
  difference=abs(minimum-evidence['minimum_sampled_rigid_clearance_m']);assert difference<1e-12;assert minimum>=.008
  rows.append(dict(yaw_deg=yaw['world_z_yaw_degrees'],phase=label,archive=archive['path'],sha256=archive['sha256'],raw_distance_count=count,
   all_archive_arrays_and_sample_distances_finite=True,analytic_joint_limits_pass=True,minimum_reconstructed_sampled_rigid_clearance_m=minimum,reported_minimum_difference_m=difference))
  max_difference=max(max_difference,difference);raw_count+=count
assert len(rows)==20 and len(releases)==6
unchanged=protected=={k:sha(ROOT/k) for k in protected};assert unchanged
out=dict(kind='six_frozen_small_yaw_contingency_archive_and_math_summary',physical_success=False,runtime_config_modified=False,gpu_used=False,
 input_sha256=protected,inputs_unchanged=unchanged,archive_count=20,release_cases=6,path_raw_clearance_count=raw_count,
 all_reconstructed_raw_clearance_values_finite=True,maximum_reported_minimum_difference_m=max_difference,archives=rows,releases=releases,candidate_summary=summary,
 math_companion=dict(path=str(mathfile.relative_to(ROOT)),sha256=sha(mathfile),continuous_ideal_path_vertex_error_bounds_m={str(r['world_z_yaw_degrees']):r['reported_continuous_outer_bound_m'] for r in mathreport['rows']},
  scope='Only pure fixed-TCP rigid yaw versus ideal actual-FK-to-commanded Slerp. No continuous joint-FK or deformable/grasp bound.'),
 assessment='Among exactly these six frozen candidates, +15 degrees with [-20,-60] mm offers the highest inside-node count and 13.380 mm sampled rigid clearance. This is a possible next-run diagnostic, not a physical or probabilistic success claim.',
 limitations=['All six still have exterior frozen teddy nodes; none meets whole-point containment.',
  'The existing runtime original-orientation gate rejects these nonzero yaws; no change to that gate is proposed or performed here.',
  'Archived joint splines and minimum-jerk timing are exact hypothetical commands; release is a fixed-pose geometric opening path without a specified executed time law.',
  'Rigid route clearance uses 401 samples and is not a continuous collision certificate. Rotated frozen teddy versus fixed left robot uses a separate solid arc certificate.',
  'No dynamic contact, cloth expansion, grasp retention, release or settled final containment has been simulated.'])
output=RUN/'independent-small-yaw-archive-audit.json';output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:out[k] for k in ['archive_count','release_cases','path_raw_clearance_count','maximum_reported_minimum_difference_m','inputs_unchanged']}));print('output',output,'sha256',sha(output))
