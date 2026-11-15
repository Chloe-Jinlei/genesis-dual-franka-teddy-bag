"""Read-only CPU investigation of frozen full16 alignment; never executes physics."""
from pathlib import Path
import sys,json,numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/two_panel_full_16'
sys.path.insert(0,str(RUN/'source'))
from motion import Planner
from validate import mouth_structure
from physical_monitor import tetra_volumes
from staged_contact_insertion import _spatial_loop_geometry,left_hand_minimum_z
from contact_robot_geometry import check_translation_sweep
from contact_slice_geometry import convex_hull_metrics
from insertion_candidates import projected_inside
cfg=json.loads((RUN/'config.json').read_text());rest=dict(np.load(RUN/'rest.npz'));planner=Planner(cfg)
body,rim,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg)
weights=np.abs(tetra_volumes(rest['bear'],rest['bear_tets']))
original=Rotation.from_quat(json.loads((RUN/'guarded_contact_candidate.json').read_text())['route']['rotation_xyzw'])
states={name:dict(np.load(RUN/file)) for name,file in [('entry_end','lower_teddy_to_rim.npz'),('center_end','center_teddy_at_rim.npz'),('failed_end','failed_soft_state.npz')]}
trace=np.load(RUN/'joint_trace.npy');last=trace[-1]
for side,lo in [('left',1),('right',10)]:
 states['failed_end']['q_'+side]=last[lo:lo+9];states['failed_end']['tcp_'+side]=planner.fk(side,last[lo:lo+7])[0]
assert abs(float(states['failed_end']['time'])-last[0])<1e-8

def planar_sections(vertices,z):
 tris=vertices[body];out=[]
 for tri in tris:
  d=tri[:,2]-z
  if d.min()>=0 or d.max()<=0:continue
  pts=[]
  for a,b in [(0,1),(1,2),(2,0)]:
   if d[a]*d[b]<0:pts.append(tri[a]+(-d[a]/(d[b]-d[a]))*(tri[b]-tri[a]))
  if len(pts)==2:out.append(np.asarray(pts)[:,:2])
 if not out:return {'z_m':z,'closed_loops':[],'segment_count':0}
 arr=np.asarray(out);lookup={};nodes=[];adj={}
 def node(x):
  k=tuple(np.round(x,8));i=lookup.get(k)
  if i is None:i=len(nodes);lookup[k]=i;nodes.append(x);adj[i]=[]
  return i
 for seg in arr:
  a,b=map(node,seg);adj[a].append(b);adj[b].append(a)
 visited=set();loops=[];open_nodes=[i for i,n in adj.items() if len(n)!=2]
 for start in adj:
  if start in visited or len(adj[start])!=2:continue
  path=[];cur=start;prev=None
  while cur not in visited:
   visited.add(cur);path.append(cur);nxt=[v for v in adj[cur] if v!=prev]
   if not nxt:break
   prev,cur=cur,nxt[0]
  if cur!=start:continue
  poly=np.asarray(nodes)[path];q=np.roll(poly,-1,axis=0);cross=poly[:,0]*q[:,1]-q[:,0]*poly[:,1];signed=cross.sum()/2
  if abs(signed)<1e-12:continue
  centroid=((poly+q)*cross[:,None]).sum(0)/(6*signed)
  loops.append(dict(area_m2=float(abs(signed)),centroid_xy_m=centroid.tolist(),bounds_xy_m=[poly.min(0).tolist(),poly.max(0).tolist()],polygon_xy_m=poly.tolist()))
 return dict(z_m=z,segment_count=len(out),non_degree_two_nodes=len(open_nodes),closed_loops=sorted(loops,key=lambda x:-x['area_m2']))

def geometry(name,state):
 bear=state['bear'];bag=state['bag'];com=np.average(bear[rest['bear_tets']].mean(1),axis=0,weights=weights)
 poly,_,spatial=_spatial_loop_geometry(bag[rim],rest['bag'][rim],cfg['bag']['thickness'])
 p=poly;q=np.roll(p,-1,axis=0);cr=p[:,0]*q[:,1]-q[:,0]*p[:,1];area=cr.sum()/2;cent=((p+q)*cr[:,None]).sum(0)/(6*area)
 minrim=float(bag[rim,2].min());maxrim=float(bag[rim,2].max());low=float(bag[body].reshape(-1,3)[:,2].min())
 levels=sorted(set([z for z in [.80,.85,.90,.95,1.0] if low+.005<z<minrim-.001]+[float(minrim-.01)]))
 sections=[planar_sections(bag,z) for z in levels]
 return dict(state=name,time_s=float(state['time']),bear_com_m=com.tolist(),bear_bounds_m=[bear.min(0).tolist(),bear.max(0).tolist()],tcp_right_m=state['tcp_right'].tolist(),left_hand_gap_m=left_hand_minimum_z(state['q_left'],cfg,planner=planner)-float(bear[:,2].max()),rim_bounds_m=[bag[rim].min(0).tolist(),bag[rim].max(0).tolist()],rim_vertex_mean_m=bag[rim].mean(0).tolist(),rim_convex_contact_envelope_centroid_xy_m=cent.tolist(),rim_envelope_alignment_delta_xy_m=(cent-com[:2]).tolist(),rim_geometry=spatial,horizontal_body_sections_below_entire_rim=sections)

def scalar(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,np.ndarray):return v.tolist()
 raise TypeError(type(v).__name__)
if __name__=='__main__':
 geo={name:geometry(name,state) for name,state in states.items()}
 output=dict(kind='full16_actual_geometry_alignment_diagnostic',physical_success=False,runtime_execution_performed=False,states=geo,scope='Actual saved shapes and measured joints; failed_end already violates stretch and is diagnostic only. Contact envelope is not an actual free aperture. Closed horizontal sections below all rim vertices identify the frozen bag body interior.')
 (RUN/'independent-alignment-geometry.json').write_text(json.dumps(output,indent=2,default=scalar)+'\n')
 for name,g in geo.items():
  print(name,'COM',g['bear_com_m'],'handgap',g['left_hand_gap_m'],'rim centroid',g['rim_convex_contact_envelope_centroid_xy_m'],'rim align',g['rim_envelope_alignment_delta_xy_m'])
  for sl in g['horizontal_body_sections_below_entire_rim']:
   print('section',sl['z_m'],'nondegree2',sl.get('non_degree_two_nodes'),'main', [{k:v for k,v in t.items() if k!='polygon_xy_m'} for t in sl['closed_loops'][:1]])
