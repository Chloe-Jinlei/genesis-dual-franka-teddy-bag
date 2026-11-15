"""Read-only frozen geometry decomposition; not a native state or route acceptance."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_22'
sys.path.insert(0,str(RUN/'source'))
from contact_insertion import _section_envelopes
from contact_slice_geometry import convex_hull_metrics
from insertion_candidates import polygon_centroid
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['rest.npz','orient_teddy_postlift.npz','transport_teddy.npz','guarded_contact_candidate.json','guarded_contact_reobserved.json','config.json','source/contact_insertion.py','source/contact_slice_geometry.py']
before={f:sha(RUN/f) for f in files}
load=lambda f: dict(np.load(RUN/f,allow_pickle=False))
r=load('rest.npz');old=load('orient_teddy_postlift.npz');new=load('transport_teddy.npz')
a=json.loads((RUN/'guarded_contact_candidate.json').read_text());b=json.loads((RUN/'guarded_contact_reobserved.json').read_text())
ma=a['report']['mouth'];mb=b['report']['mouth'];poly0=np.asarray(ma['ordered_rim_world_m'])[:,:2];poly1=np.asarray(mb['ordered_rim_world_m'])[:,:2]
planned=np.asarray(a['route']['outside_tcp_m']);old_bear=old['bear']+planned-old['tcp_right'];new_bear=new['bear']
_,env0=_section_envelopes(old_bear,r['bear_tets']);_,env1=_section_envelopes(new_bear,r['bear_tets'])
padding=.002*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
def evaluate(poly,env,delta=(0,0),buffer=True):
    values=[]
    for e in env:
        hull=e['hull_xy']+np.asarray(delta)
        if buffer:hull=(hull[:,None,:]+padding).reshape(-1,2)
        values.append(convex_hull_metrics(np.vstack([poly,hull]))['perimeter'])
    i=int(np.argmax(values));return {'perimeter_m':float(values[i]),'margin_m':mb['available_nonstretch_rim_perimeter_m']-float(values[i]),'max_slab_index':i,'z_interval_m':[env[i]['z_min'],env[i]['z_max']]}
combos={}
for label,p,e in [('old_mouth_old_planned_bear',poly0,env0),('new_mouth_old_planned_bear',poly1,env0),('old_mouth_actual_bear',poly0,env1),('new_mouth_actual_bear',poly1,env1)]:combos[label]=evaluate(p,e)
bare=evaluate(poly1,env1,buffer=False)
object_only=max(convex_hull_metrics((e['hull_xy'][:,None,:]+padding).reshape(-1,2))['perimeter'] for e in env1)
# Bounded local sensitivity, not candidate search or motion planning.
local={str(d):evaluate(poly1,env1,d) for d in [(-.001,0),(.001,0),(0,-.001),(0,.001)]}
grad=np.array([(local[str((.001,0))]['perimeter_m']-local[str((-.001,0))]['perimeter_m'])/.002,(local[str((0,.001))]['perimeter_m']-local[str((0,-.001))]['perimeter_m'])/.002])
# Decompose observed shape/relative changes without attributing causation to force.
bear_change=new_bear-old_bear;rim0=np.asarray(ma['ordered_rim_world_m']);rim1=np.asarray(mb['ordered_rim_world_m']);bear_rel=(new['bear']-new['tcp_right'])-(old['bear']-old['tcp_right'])
w=env1[combos['new_mouth_actual_bear']['max_slab_index']];h=(w['hull_xy'][:,None,:]+padding).reshape(-1,2);allp=np.vstack([poly1,h]);cv=ConvexHull(allp);hv=allp[cv.vertices]
source=['rim' if x<len(poly1) else 'padded_teddy' for x in cv.vertices]
# Geometry-only witness hull IDs/material heights locate which support dominates.
result={'scope':'Frozen CPU geometry at actual native yaw and transport snapshots. No simulation, planning, config/gate change, or physical-success claim.',
 'actual_times_s':{'yaw':float(old['time']),'transport':float(new['time'])},
 'status':'actual pre-insertion planner rejection, not observed teddy/rim collision',
 'rejections':b['report']['rejections'],'available_nonstretch_rim_length_m':mb['available_nonstretch_rim_perimeter_m'],
 'two_by_two_geometry_decomposition':combos,'new_union_without_buffer':bare,'new_teddy_only_buffered_max_slab_perimeter_m':float(object_only),
 'old_mouth_hull_perimeter_m':convex_hull_metrics(poly0)['perimeter'],'new_mouth_hull_perimeter_m':convex_hull_metrics(poly1)['perimeter'],
 'old_mouth_projected_area_m2':ma['projected_area_m2'],'new_mouth_projected_area_m2':mb['projected_area_m2'],
 'mouth_area_centroid_delta_xy_m':(polygon_centroid(poly1)-polygon_centroid(poly0)).tolist(),
 'actual_tcp_minus_selected_target_m':(new['tcp_right']-planned).tolist(),
 'actual_bear_minus_frozen_translation':{'max_node_displacement_m':float(np.linalg.norm(bear_change,axis=1).max()),'mean_node_displacement_m':np.mean(bear_change,axis=0).tolist(),'argmax_node':int(np.linalg.norm(bear_change,axis=1).argmax())},
 'bear_shape_relative_to_tcp_change':{'max_node_displacement_m':float(np.linalg.norm(bear_rel,axis=1).max()),'argmax_node':int(np.linalg.norm(bear_rel,axis=1).argmax())},
 'rim_actual_change':{'max_node_displacement_m':float(np.linalg.norm(rim1-rim0,axis=1).max()),'mean_node_displacement_m':np.mean(rim1-rim0,axis=0).tolist()},
 'one_mm_local_xy_sensitivity':local,'central_perimeter_gradient_m_per_m':grad.tolist(),
 'local_gradient_scope':'Only geometric finite-difference direction. No proposed displacement is certified for robots, straps, full 30mm correction route, physics, or final containment.',
 'worst_union_hull_xy_m':hv.tolist(),'worst_union_hull_vertex_sources':source,
 'key_actual_guards':{k:b['report'][k] for k in ['vertical_capacity','fresh_rotation_error_rad','observed_clearance_above_highest_rim_m','centering_candidates']},
 'mathematical_scope':[
 'The buffered frozen teddy section hull perimeter is an obstruction test only for ideal planar enclosing curves; it is not a sufficient bag-fit, self-contact or nonplanar-mouth theorem.',
 'The union-hull guard additionally requires the enclosing projected rim to retain every current rim location while accommodating each buffered section. A freely deforming cloth rim need not retain those locations. Therefore failure of this union condition alone does not prove physical impossibility.',
 'The currently held handle, panel geometry, rim height, forces, bag self-contact and tetrahedral interiors may still obstruct a real route. This decomposition does not waive any of those gates or certify insertion.'
 ],'physical_success':False,'input_hashes_before':before}
result['input_hashes_after']={f:sha(RUN/f) for f in files};assert result['input_hashes_after']==before
out=RUN/'independent-perimeter-guard-review.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(out),'sha256':sha(out),'decomposition':combos,'without_buffer':bare,'object_only_buffered_m':object_only,'local_gradient':grad.tolist(),'local_results':local,'shape':result['bear_shape_relative_to_tcp_change'],'rim':result['rim_actual_change']},indent=2))
