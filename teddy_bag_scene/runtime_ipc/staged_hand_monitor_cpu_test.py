"""Native-height API arithmetic and fail-closed tests with CPU fakes only."""
from pathlib import Path
import sys,types,itertools,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import staged_hand_monitor as sm
import contact_guard
sm.version=lambda name:{'genesis-world':'1.4.3','pyuipc':'0.0.29'}[name]
contact_guard._model_mapping=lambda path:({'path':path,'sha256':'fake'},None)
sm._palm_vertices_in_native_link=lambda entity,target,points:(points[points[:,2]>-.10],{'collision_body':'hand','fake_mapping':True})
class Link:pass
class Geometry:
 def __init__(self,v):self.v=v
 def dim(self):return 2
 def positions(self):return self
 def view(self):return self.v
class Slot:
 def __init__(self,v):self.g=Geometry(v)
 def geometry(self):return self.g
IPCCoupler=type('IPCCoupler',(),{})
def scene():
 entity=types.SimpleNamespace(name='left_franka',material=types.SimpleNamespace(coup_type='two_way_soft_constraint'),morph=types.SimpleNamespace(file='fake.xml'))
 links=[]
 for name in sm.NativeHandHeightMonitor.names:
  link=Link();link.name=name;link.entity=entity;links.append(link)
 matrices=np.tile(np.eye(4),(1,3,1,1));matrices[0,:,2,3]=[1.25,1.26,1.27]
 vertices=np.asarray(list(itertools.product((-.02,.02),(-.01,.01),(-.03,.03))))
 c=IPCCoupler();c._coupling_data=types.SimpleNamespace(links=links,ipc_transforms=matrices);c._abd_slots_by_link={link:[Slot(np.vstack((vertices,[[0.,0.,-.20]])) if link.name=='link7' else vertices.copy())] for link in links}
 sim=types.SimpleNamespace(coupler=c,substeps=1,_B=1,cur_step_global=1,fem_solver=types.SimpleNamespace(entities=[types.SimpleNamespace(name='free_soft_teddy',n_vertices=5)]))
 return types.SimpleNamespace(sim=sim)
checks=[]
def check(name,good):checks.append({'name':name,'passed':bool(good)});assert good,name
def raises(name,fn):
 try:fn()
 except (ValueError,RuntimeError):check(name,True);return
 check(name,False)
s=scene();m=sm.NativeHandHeightMonitor(s);bear=np.array([[0,0,1.1],[0,0,1.2],[0,0,1.19],[0,0,1.18],[0,0,1.214]])
mat0=s.sim.coupler._coupling_data.ipc_transforms.copy();b0=bear.copy();ref=m.reference_geometry();row,error=m.measure(.005,bear)
check('all_complete_vertices_not_only_centroid',row['highest_bear_vertex_index']==4 and abs(row['vertical_gap_m']-.006)<1e-12 and error is None)
check('arm_link7_low_vertex_excluded_from_palm',len(m.vertices[0])==8 and abs(row['left_link_minimum_z_m'][0]-1.22)<1e-12)
check('no_state_or_force_mutation',np.array_equal(mat0,s.sim.coupler._coupling_data.ipc_transforms) and np.array_equal(b0,bear))
ref['left_link7'][0,2]=9;check('reference_geometry_returns_copy',m.vertices[0][0,2]!=9)
raises('partial_bear_rejected',lambda:m.measure(.005,bear[:-1]))
raises('same_step_new_time_rejected',lambda:m.measure(.010,bear))
s.sim.cur_step_global=2;s.sim.coupler._coupling_data.ipc_transforms[0,0,2,0]=.5
row,error=m.measure(.010,bear);check('full_affine_shear_preserved',abs(row['left_distal_minimum_z_m']-1.210)<1e-12 and error is not None)
s.sim.cur_step_global=3;s.sim.coupler._coupling_data.ipc_transforms[0,0]=mat0[0,0];s.sim.coupler._coupling_data.ipc_transforms[0,2,2,3]=1.23
row,error=m.measure(.015,bear);check('both_fingers_included',abs(row['left_distal_minimum_z_m']-1.2)<1e-12 and error is not None)
s.sim.cur_step_global=4;s.sim.coupler._coupling_data.ipc_transforms[0,0,0,0]=np.nan
raises('nonfinite_native_transform_rejected',lambda:m.measure(.020,bear))
s2=scene();s2.sim.coupler._coupling_data.links[2].name='missing';raises('missing_finger_mapping_rejected',lambda:sm.NativeHandHeightMonitor(s2))
s3=scene();link=Link();link.entity=s3.sim.coupler._coupling_data.links[0].entity;link.name='hand';s3.sim.coupler._coupling_data.links.append(link);raises('unexpected_unwelded_hand_rejected',lambda:sm.NativeHandHeightMonitor(s3))
s4=scene();s4.sim._B=2;raises('multi_environment_rejected',lambda:sm.NativeHandHeightMonitor(s4))
result={'kind':'native_hand_height_monitor_independent_cpu_checks','checks':checks,'all_checks_passed':all(v['passed'] for v in checks),'source_sha256':hashlib.sha256((HERE/'staged_hand_monitor.py').read_bytes()).hexdigest(),'native_solver_initialized':False,'limitations':['CPU fakes exercise mapping, coordinate arithmetic and failure handling; a built-scene constructor/geometry read still needs the next authorized native run smoke.']}
(HERE/'runtime_ipc/staged-hand-monitor-cpu-tests.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
