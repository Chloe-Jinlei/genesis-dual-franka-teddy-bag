"""One frozen +10 degree local direction diagnostic; no route/simulator creation."""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1];P=B/'runs/two_panel_full_23'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
files=[P/x for x in ['transport_teddy.npz','states.npz','rest.npz','config.json','independent-first-body-near-contact-review.json']]
s=dict(np.load(files[0]));allstates=dict(np.load(files[1]));rest=dict(np.load(files[2]));cfg=json.loads(files[3].read_text());sourcepath=P/'source/assets'/cfg['bag']['npz_file'];src=dict(np.load(sourcepath));files += [sourcepath,Path(__file__).resolve()];hashes={str(p.relative_to(B)):sha(p) for p in files}
e,mp=cKDTree(rest['bag']).query(src['points']+cfg['bag']['position']);assert e.max()<1e-6 and len(np.unique(mp))==len(mp)
i=int(np.argmin(abs(allstates['time']-43.03)));tri=allstates['bag'][i,mp[src['faces'][1407]]];n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n);inward=np.r_[-n[:2]/np.linalg.norm(n[:2]),0.]
# Source face orientation is the verified true front body face, not a fitted rim plane.
tcp=s['tcp_right'];bear=s['bear'];v=Rotation.from_euler('z',10,degrees=True).apply(bear-tcp)+tcp;delta=v[66]-bear[66]
local=(rest['bear']-cfg['bear']['position'])/.85;feet=np.flatnonzero(local[:,1]<-.08);head=np.flatnonzero(local[:,1]>.07);axis=bear[head].mean(0)-bear[feet].mean(0);axis/=np.linalg.norm(axis)
report=dict(scope='One hypothetical rigid +10 degree world-Z rotation of actual full23 transport shape about measured TCP; local direction only',native_execution=False,production_edited=False,physical_success=False,route_candidate=False,input_sha256=hashes,actual_transport_time_s=float(s['time']),fixed_actual_tcp_m=tcp,
 source_to_native_max_mapping_error_m=float(e.max()),world_z_increment_deg=10.,nominal_total_world_z_from_original_grasp_deg=-20.,
 actual43p030_contact=dict(time_s=float(allstates['time'][i]),bear_node=66,source_body_face=1407,actual_face_vertices_m=tri,outward_normal=n,horizontal_inward_unit_vector=inward),
 node66=dict(original100percent_material_local_m=local[66],description='Left foot/leg exterior according to original implicit teddy components',transport_point_m=bear[66],transport_lever_from_TCP_m=bear[66]-tcp,hypothetical_point_m=v[66],displacement_m=delta,inward_horizontal_projection_m=float(delta@inward)),
 transport_bear_material_axis=dict(definition='Mean of nodes rest-local y>70mm minus mean of nodes y<-80mm; coarse material head/feet regions, not PCA',head_node_count=len(head),foot_node_count=len(feet),feet_mean_m=bear[feet].mean(0),head_mean_m=bear[head].mean(0),foot_to_head_unit_world=axis,tilt_above_world_XY_deg=float(np.degrees(np.arcsin(axis[2])))),
 bbox=dict(actual_span_m=np.ptp(bear,axis=0),hypothetical_span_m=np.ptp(v,axis=0),actual_min_z_m=float(bear[:,2].min()),hypothetical_min_z_m=float(v[:,2].min()),yaw_preserves_all_vertex_heights=True),
 conclusion='Reducing the magnitude of the current -30 degree yaw moves the demonstrated initial contact foot mainly +X and -Y, toward the later actual front-panel local inward direction. The same rotation changes all other body points and cannot establish passage.',
 limits=['The inward direction is measured on actual43.030 deformed front cloth; applying it at40.280 is a direction diagnostic, not a future contact normal prediction.','The stated -20 degree total is nominal relative to the commanded -30 degree pose; actual native robot orientation and bear grasp response are not replaced.','No robot IK, 8mm rigid gate, 5mm full-bear left clearance, handle sweep, section route, retention, or dynamics was tested for this alternative.','No standing-teddy, further yaw or XY candidate was searched. Parent has stopped this diagnostic at its local directional result.'])
report['inputs_unchanged']=all(sha(B/p)==h for p,h in hashes.items());out=P/'independent-local-positive10-yaw-direction.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\n');print(out);print(json.dumps({'inward_m':report['node66']['inward_horizontal_projection_m'],'delta_m':delta.tolist(),'foot_head_tilt_deg':report['transport_bear_material_axis']['tilt_above_world_XY_deg'],'sha256':sha(out)}))
