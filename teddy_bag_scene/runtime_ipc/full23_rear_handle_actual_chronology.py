"""Finite saved-frame actual rear-handle proximity diagnostics, CPU only."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
BASE=Path(__file__).resolve().parents[1];P=BASE/'runs/two_panel_full_23';D=P/'independent-actual-rear-handle-chronology'
sys.path.insert(0,str(P/'source'))
from validate import tetra_boundary,mesh_edges,signed_volumes

def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 D.mkdir(exist_ok=True)
 files=[P/n for n in ['states.npz','failed_soft_state.npz','rest.npz','config.json','transport_teddy.npz','contact_guard_trace.json','guarded_contact_reobserved.json']]+[Path(__file__)]
 hashes={str(p):sha(p) for p in files};states=dict(np.load(files[0]));failed=dict(np.load(files[1]));rest=dict(np.load(files[2]));cfg=json.loads(files[3].read_text());ref=dict(np.load(files[4]));trace=json.loads(files[5].read_text());route=json.loads(files[6].read_text());srcpath=P/'source/assets'/cfg['bag']['npz_file'];src=dict(np.load(srcpath));hashes[str(srcpath)]=sha(srcpath)
 error,mp=cKDTree(rest['bag']).query(src['points']+cfg['bag']['position']);assert error.max()<1e-6 and len(np.unique(mp))==len(mp)
 bf=tetra_boundary(rest['bear'],rest['bear_tets']);local=(rest['bear']-cfg['bear']['position'])/.85
 bary7=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3]])
 groups={};sourceids={};bodyset=set(src['body_face_indices'].tolist())
 for name in ['front','back']:
  ids=np.array([i for i,f in enumerate(src['faces']) if i not in bodyset and np.isin(f,src['handle_'+name]).all()]);assert len(ids)==352;sourceids[name]=ids;groups[name]=mp[src['faces'][ids]]
 # Explicit strip topology supplies rest arclength/material side evidence.
 rows={};arcs={}
 for name in ['front','back']:
  nc=len(src['handle_'+name+'_root_left']);inner=np.setdiff1d(src['handle_'+name],src['body']);strip=np.vstack([src['handle_'+name+'_root_left'],inner.reshape(-1,nc),src['handle_'+name+'_root_right'][::-1]])
  expected=set()
  for j in range(len(strip)-1):
   for i in range(nc-1):
    a,b,c,d=strip[j,i],strip[j,i+1],strip[j+1,i],strip[j+1,i+1];expected.update([tuple(sorted([a,b,d])),tuple(sorted([a,d,c]))])
  assert expected==set(tuple(sorted(f)) for f in src['faces'][sourceids[name]])
  centers=src['points'][strip].mean(1);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))];arcs[name]=arc;rows[name]={int(v):j for j,row in enumerate(strip) for v in row}
 shapes=[('torso',(0,-.025,.045),(.065,.075,.040)),('head',(0,.080,.051),(.058,.055,.047)),('left_ear',(-.047,.126,.057),(.023,.023,.022)),('right_ear',(.047,.126,.057),(.023,.023,.022)),('left_arm',(-.076,-.010,.040),(.040,.035,.028)),('right_arm',(.076,-.010,.040),(.040,.035,.028)),('left_foot',(-.040,-.102,.034),(.036,.041,.030)),('right_foot',(.040,-.102,.034),(.036,.041,.030)),('muzzle',(0,.068,.091),(.031,.024,.017)),('nose',(0,.083,.106),(.012,.008,.006))]
 times=states['time'];trace_times=np.array([v['time_s'] for v in trace]);vol=signed_volumes(rest['bear'],rest['bear_tets']);mass=np.abs(vol);ref_com=np.average(ref['bear'][rest['bear_tets']].mean(1),axis=0,weights=mass);X0=ref['bear']-ref['bear'].mean(0)
 def evaluate(state,label):
  t=float(state['time']);bear=state['bear'];bag=state['bag'];bm=trimesh.Trimesh(bear,bf,process=False);com=np.average(bear[rest['bear_tets']].mean(1),axis=0,weights=mass);X=bear-bear.mean(0);u,sv,vt=np.linalg.svd(X0.T@X);R=u@vt
  if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
  fit=X0@R;res=np.linalg.norm(X-fit,axis=1);J=signed_volumes(bear,rest['bear_tets'])/vol;g=trace[int(np.argmin(abs(trace_times-t)))];rim=bag[mp[src['rim']]]
  out={'label':label,'actual_geometry_time_s':t,'native_guard_time_s':g['time_s'],'native_guard_time_difference_s':float(abs(g['time_s']-t)),
       'native_tcp_torque_norm_nm':g['tcp_torque_norm_nm'],'native_net_force_norm_n':g['net_force_norm_n'],'native_link7_force_norm_n':float(np.linalg.norm(g['right_link_force_world_n'][0])),
       'native_right_finger_forces_n':g['right_finger_force_norm_n'],'native_left_material_distance_m':g['left_designated_material_distance_m'],
       'guard_aim_tcp_world_m':g['aim_tcp_world_m'],'measured_tcp_world_m':state.get('tcp_right'),'measured_tcp_available':'tcp_right' in state,
       'bear_rest_volume_weighted_COM_m':com,'bear_COM_minus_initial_COM_m':com-ref_com,'bear_bbox_min_m':bear.min(0),'bear_bbox_max_m':bear.max(0),
       'bear_bestfit_rotation_from_transport_degrees':float(np.degrees(Rotation.from_matrix(R.T).magnitude())),
       'bear_bestfit_rotation_vector_degrees':np.degrees(Rotation.from_matrix(R.T).as_rotvec()),'bear_rigid_fit_rms_residual_m':float(np.sqrt(np.mean(res**2))),'bear_rigid_fit_max_residual_m':float(res.max()),
       'minimumJ':float(J.min()),'minimumJ_tet':int(J.argmin()),'rim_min_z_m':float(rim[:,2].min()),'rim_max_z_m':float(rim[:,2].max()),'rim_mean_world_m':rim.mean(0),
       'rim_mean_displacement_from_transport_m':(bag-ref['bag'])[mp[src['rim']]].mean(0),'rim_max_displacement_from_transport_m':float(np.linalg.norm((bag-ref['bag'])[mp[src['rim']]],axis=1).max()),
       'body_mean_displacement_from_transport_m':(bag-ref['bag'])[mp[src['body']]].mean(0),'bear_minZ_minus_highest_bodyZ_m':float(bear[:,2].min()-bag[mp[src['body']],2].max()),'handles':{}}
  if 'tcp_right'in state:out['COM_relative_actualTCP_change_m']=(com-state['tcp_right'])-(ref_com-ref['tcp_right'])
  for name,faces in groups.items():
   samples=np.einsum('kc,fcj->fkj',bary7,bag[faces]).reshape(-1,3)
   near,dist,idx=trimesh.proximity.closest_point(bm,samples);k=int(dist.argmin());fi,bi=divmod(k,7);bfi=int(idx[k]);wb=trimesh.triangles.points_to_barycentric(bear[bf[[bfi]]],near[[k]])[0];material=wb@local[bf[bfi]]
   scores={n:float((np.linalg.norm((material-np.array(c))/np.array(r))-1)*min(r)) for n,c,r in shapes};sf=int(sourceids[name][fi]);sourceface=src['faces'][sf];rs=np.array([rows[name][int(v)] for v in sourceface]);arc=float(bary7[bi]@arcs[name][rs]);normal=bm.face_normals[bfi];vec=samples[k]-near[k]
   out['handles'][name]={'sampled_minimum_handle_to_bear_surface_m':float(dist[k]),'sample_count':len(samples),'bear_point_world_m':near[k],'handle_point_world_m':samples[k],
     'sample_unit_bear_to_handle_direction_world':vec/max(float(np.linalg.norm(vec)),1e-30),'bear_outward_face_normal_world':normal,'handle_side_signed_distance_along_bear_normal_m':float(vec@normal),
     'source_handle_face_index':sf,'native_handle_vertices':faces[fi],'source_handle_vertices':sourceface,'handle_triangle_barycentric':bary7[bi],
     'handle_strip_rows':rs,'handle_material_arclength_from_left_root_m':arc,'handle_material_arc_fraction_from_left_root':arc/arcs[name][-1],
     'bear_surface_face_index':bfi,'bear_surface_face_vertices':bf[bfi],'bear_face_barycentric':wb,'bear_original100percent_material_point_m':material,
     'nearest_original_bear_ellipsoid_component':min(scores,key=scores.get),'component_scores_m':scores,
     'max_handle_vertex_displacement_from_transport_m':float(np.linalg.norm((bag-ref['bag'])[mp[src['handle_'+name]]],axis=1).max())}
  print('FRAME',round(t,3),'front/back mm',*[round(out['handles'][n]['sampled_minimum_handle_to_bear_surface_m']*1000,3)for n in ['front','back']],'backface',out['handles']['back']['source_handle_face_index'],'part',out['handles']['back']['nearest_original_bear_ellipsoid_component'],'tau',round(g['tcp_torque_norm_nm'],5),'angle',round(out['bear_bestfit_rotation_from_transport_degrees'],3),flush=True)
  return out
 requested=[40.28,41.28,42.28,42.78,43.08,43.28,43.43,43.58,43.73,43.88,44.03,44.18,44.33,44.48]
 indices=sorted(set(int(np.argmin(abs(times-t))) for t in requested));results={i:evaluate({k:v[i] for k,v in states.items()},'saved_frame') for i in indices}
 # Refine only the first sampled <2mm back proximity bracket, at saved50ms frames.
 near=[i for i in indices if results[i]['handles']['back']['sampled_minimum_handle_to_bear_surface_m']<=.002]
 if near:
  first=min(near);previous=max((i for i in indices if i<first),default=first)
  for i in range(previous+1,first):results[i]=evaluate({k:v[i] for k,v in states.items()},'refined_first_back_near_bracket')
 rowsout=[results[i] for i in sorted(results)];rowsout.append(evaluate(failed,'failed_soft_state_no_exact_measured_TCP'))
 report={'kind':'actual_full23_rear_handle_contact_proximity_chronology','native_simulation_launched':False,'production_edited':False,'source_to_native_error_m':float(error.max()),'input_sha256':hashes,
         'native_insert_start_s':float(ref['time']),'planned_insert_tcp_m':route['insert_stage']['right_tcp_m'],'rows':rowsout,'sampled_frame_count':len(rowsout),
         'first_saved_sample_with_back_proximity_le2mm_s':next((r['actual_geometry_time_s'] for r in rowsout if r['handles']['back']['sampled_minimum_handle_to_bear_surface_m']<=.002),None),
         'limitations':['Actual saved states only. No fabricated soft translation, no native resumption.','Triangle vertices,edge midpoints,centroids are queried against exact bear surface triangles. Minimum is an upper bound on true surface distance; positive values do not certify separation or no intersection.',
             'Finite saved-frame proximity event is not exact earliest continuous physical contact. Contact forces are whole link/finger reactions and do not identify a unique material pair.',
             'Nearest original component is an overlapping implicit-ellipsoid material classification, not an independent anatomical segmentation.',
             'The failed_soft_state has exact soft positions but no measured robot q/TCP. Only its same-time guard AIM TCP is reported, never relabeled measured TCP.',
             'Bestfit whole-bear rotation and residual describe observed shape change, not rigid motion or causal attribution.','No bag-body containment conclusion or success follows; body analysis is independently handled.']}
 report['inputs_unchanged']=all(sha(p)==v for p,v in hashes.items());out=D/'geometry-chronology.json';assert not out.exists();out.write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\n');print('REPORT',out,flush=True)

if __name__=='__main__':main()
