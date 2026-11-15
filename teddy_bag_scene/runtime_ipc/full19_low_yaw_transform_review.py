"""One-candidate frozen transform comparison; no native simulation or runtime edits."""
from pathlib import Path
import json,hashlib,sys,tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_19'
sys.path.insert(0,str(RUN/'source'));sys.path.insert(0,str(ROOT/'runtime_ipc'))
from motion import Planner
from diagnose_low_region_winding import diagnose
from body_alignment_ranking import metrics_from_diagnostic
report=json.loads((RUN/'independent-single-low-yaw-contingency.json').read_text());y=report['results'][0];c=y['center_remainders'][0]
state=dict(np.load(RUN/'lower_teddy_to_rim.npz',allow_pickle=False));planner=Planner(json.loads((RUN/'config.json').read_text()))
initial_tcp,A=planner.fk('right',state['q_right'][:7]);C=Rotation.from_quat(report['baseline_commanded_full_rotation_xyzw']);yaw=Rotation.from_euler('z',15,degrees=True);target=yaw*C
p=state['bear']-initial_tcp;radius=np.linalg.norm(p,axis=1).max();a=np.array([0,0,np.deg2rad(15)]);b=(target*A.inv()).as_rotvec();duhamel=float(radius*np.linalg.norm(a-b))
archive=ROOT/c['insert']['trajectory_archive']['path'];z=np.load(archive,allow_pickle=False);curve=CubicSpline.construct_fast(z['right_spline_coefficients'],z['right_spline_knots_fraction'],extrapolate=True,axis=0);q=curve(1.)
end_tcp,end_R=planner.fk('right',q);exact=end_R.apply(A.inv().apply(p))+end_tcp
pure=yaw.apply(state['bear']-state['tcp_right'])+state['tcp_right']+np.r_[c['center_offset_xy_m'],0]+np.array([0,0,c['insert']['target_tcp_m'][2]-state['tcp_right'][2]])
with tempfile.TemporaryDirectory(prefix='full19-one-low-yaw-transform-') as tmp:
 path=Path(tmp)/'explicit_frozen_final_fk_rigid_grasp_hypothesis.npz';np.savez(path,time=state['time'],bag=state['bag'],bear=exact);diagnostic=diagnose(RUN,path)
metrics=metrics_from_diagnostic(diagnostic,len(exact))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(kind='full19_single_yaw_rigid_transform_scope_review',physical_success=False,runtime_execution_performed=False,active_runtime_or_configuration_modified=False,
 source_time_s=float(state['time']),source_state_sha256=sha(RUN/'lower_teddy_to_rim.npz'),parent_report_sha256=sha(RUN/'independent-single-low-yaw-contingency.json'),insert_archive_sha256=sha(archive),
 pure_yaw_versus_ideal_actual_fk_to_command_slerp_continuous_duhamel_bound_m=duhamel,
 maximum_final_pure_vs_planned_fk_rigid_grasp_vertex_difference_m=float(np.linalg.norm(pure-exact,axis=1).max()),
 planned_insert_end_tcp_m=end_tcp.tolist(),planned_insert_end_rotation_xyzw=end_R.as_quat().tolist(),
 pure_yaw_frozen_body_metrics=c['body_metrics'],planned_final_fk_rigid_grasp_body_metrics=metrics,
 scope=['The second body query uses exactly the archived hypothetical insert endpoint FK and the actual initial FK frame, leaving actual bag vertices unchanged.',
 'This is still a frozen rigid-grasp assumption, not a prediction of deformation or retention.',
 'Duhamel bound compares two ideal fixed-TCP rotations only, and is not a bound for the realized joint spline or native soft body.',
 'No new yaw angle, offset, trajectory or gate was introduced.'])
p=RUN/'independent-single-low-yaw-transform-review.json';p.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out));print('sha256',sha(p))
