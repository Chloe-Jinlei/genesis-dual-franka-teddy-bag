"""Read-only complete frozen tetra translation vs physical handle triangles.

No Genesis import, no soft vertex mutation, no production/config changes.
Each tetra moves by x=sum(lambda_i*v_i)+t*d, lambda>=0,sumlambda=1,0<=t<=1.
A triangle point is sum(mu_j*f_j),mu>=0,summu=1. LP finds minimum L-infinity
residual; separation axes are independently recomputed on all original eight
swept vertices and all three triangle vertices. Only that projection certificate,
minus a numerical guard, can authorize a clearance result. Strict-intersection
LP witnesses are independently checked before reporting intersection.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from scipy.optimize import linprog,minimize,Bounds,LinearConstraint
from scipy.spatial import cKDTree

GUARD=1e-7
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def projection(v,f,d,axes):
 axes=np.asarray(axes,float).reshape(-1,3)
 lengths=np.linalg.norm(axes,axis=1);axes=axes[lengths>1e-12]/lengths[lengths>1e-12,None]
 if not len(axes):return 0.,None
 hull=np.concatenate([v,v+d]);p=hull@axes.T;q=f@axes.T
 gaps=np.maximum(p.min(0)-q.max(0),q.min(0)-p.max(0))
 k=int(gaps.argmax())
 return max(0.,float(gaps[k])-GUARD),axes[k]
def matrix(v,f,d):
 origin=f.mean(0);a=np.column_stack([(v-origin).T,-(f-origin).T,d])
 e=np.zeros((2,8));e[0,:4]=1;e[1,4:7]=1
 return a,e
def initial_axes(v,f):
 out=[*np.eye(3),np.cross(f[1]-f[0],f[2]-f[0])]
 for k in range(4):
  q=np.delete(v,k,axis=0);out.append(np.cross(q[1]-q[0],q[2]-q[0]))
 return np.asarray(out)
def strict_witness(v,f,d):
 a,e=matrix(v,f,d)
 alt=[]
 for k in range(4):
  q=np.delete(v,k,axis=0);normal=np.cross(q[1]-q[0],q[2]-q[0]);normal/=np.linalg.norm(normal)
  alt.append(abs(float((v[k]-q[0])@normal)))
 tri_area2=float(np.linalg.norm(np.cross(f[1]-f[0],f[2]-f[0])))
 for k in range(3):
  q=np.delete(f,k,axis=0);alt.append(tri_area2/np.linalg.norm(q[1]-q[0]))
 ub=np.zeros((7,9))
 for k,h in enumerate(alt):ub[k,k]=-h;ub[k,8]=1.
 res=linprog([0.]*8+[-1.],A_ub=ub,b_ub=np.zeros(7),A_eq=np.column_stack([np.vstack([a,e]),np.zeros(5)]),b_eq=[0.,0.,0.,1.,1.],bounds=[(0,None)]*7+[(0,1),(None,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if not res.success:return {'strict_intersection_proven':False,'LP_status':int(res.status)}
 x=res.x[:8];bear=x[:4]@v+x[7]*d;cloth=x[4:7]@f;residual=float(np.linalg.norm(bear-cloth));depth=float(min(np.array(alt)*x[:7]))
 valid=bool(residual<=1e-8 and abs(x[:4].sum()-1)<=1e-8 and abs(x[4:7].sum()-1)<=1e-8 and x[:7].min()>=-1e-9 and -1e-9<=x[7]<=1+1e-9 and depth>GUARD)
 return {'strict_intersection_proven':valid,'minimum_interior_face_or_edge_margin_m':depth,'recomputed_coordinate_residual_m':residual,'path_fraction':float(x[7]),'tet_barycentric':x[:4].tolist(),'triangle_barycentric':x[4:7].tolist(),'bear_point_world_m':bear.tolist(),'cloth_point_world_m':cloth.tolist()}
def pair(v,f,d,required):
 gap,axis=projection(v,f,d,initial_axes(v,f))
 result={'conservative_euclidean_projection_gap_lower_m':gap,'axis_world':None if axis is None else axis.tolist(),'method':'direct_face_or_coordinate_separating_axis'}
 if gap>=required:result['status']='clear';return result
 a,e=matrix(v,f,d)
 res=linprog([0.]*8+[1.],A_ub=np.vstack([np.column_stack([a,-np.ones(3)]),np.column_stack([-a,-np.ones(3)])]),b_ub=np.zeros(6),A_eq=np.column_stack([e,np.zeros(2)]),b_eq=np.ones(2),bounds=[(0,None)]*7+[(0,1),(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
 if not res.success:result.update(status='indeterminate',LP_status=int(res.status),reason='Minimum residual LP failed; no infeasibility-only acceptance');return result
 x=res.x[:8];resid=a@x
 dual=np.asarray(res.ineqlin.marginals[:3])-np.asarray(res.ineqlin.marginals[3:])
 g2,a2=projection(v,f,d,[dual,resid])
 if g2>gap:gap,axis=g2,a2
 def witness(x):
  bp=x[:4]@v+x[7]*d;fp=x[4:7]@f;rr=bp-fp
  feasible=bool(x[:7].min()>=-1e-9 and abs(x[:4].sum()-1)<=1e-8 and abs(x[4:7].sum()-1)<=1e-8 and -1e-9<=x[7]<=1+1e-9)
  return dict(barycentric_feasible=feasible,tet_barycentric=x[:4].tolist(),triangle_barycentric=x[4:7].tolist(),path_fraction=float(x[7]),bear_point_world_m=bp.tolist(),cloth_point_world_m=fp.tolist(),euclidean_distance_m=float(np.linalg.norm(rr)),linfinity_distance_m=float(np.max(np.abs(rr))))
 w=witness(x)
 result.update(method='LP_dual_axis_independently_recomputed',LP_minimum_linfinity_residual_m=float(res.fun),closest_Linfinity_pair=w)
 if gap<required and w['euclidean_distance_m']>=required:
  # One bounded Euclidean refinement. It may improve an axis, but optimizer
  # success alone never authorizes acceptance.
  qp=minimize(lambda z:.5*float((a@z)@(a@z)),x,jac=lambda z:a.T@(a@z),method='SLSQP',bounds=Bounds([0.]*8,[np.inf]*7+[1.]),constraints=[LinearConstraint(e,[1.,1.],[1.,1.])],options={'ftol':1e-18,'maxiter':300})
  wq=witness(qp.x);gq,aq=projection(v,f,d,[a@qp.x])
  result['euclidean_refinement']={'optimizer_success':bool(qp.success),'witness':wq}
  if gq>gap:gap,axis=gq,aq
  if wq['barycentric_feasible'] and wq['euclidean_distance_m']<w['euclidean_distance_m']:w=wq
 result.update(conservative_euclidean_projection_gap_lower_m=gap,axis_world=None if axis is None else axis.tolist())
 if gap>=required:
  result['status']='clear';return result
 strict=strict_witness(v,f,d);result['strict_intersection']=strict
 if strict['strict_intersection_proven']:result['status']='strict_intersection';return result
 result['status']='within_required_band' if w['barycentric_feasible'] and w['euclidean_distance_m']<required-GUARD else 'indeterminate'
 result['witness_under_euclidean_threshold']=w
 if axis is not None:
  unit=np.asarray(axis);raw=gap+GUARD
  result['coordinate_cube_separation_diagnostic']={str(delta):float(raw-delta*np.abs(unit).sum()-GUARD) for delta in [.001,.002]}
 return result
def selftest():
 v=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
 f=np.array([[1.2,.1,.1],[1.2,.2,.1],[1.2,.1,.2]])
 a=pair(v,f,np.array([2.,0,0]),.002)
 assert a['status']=='strict_intersection'
 assert 0<a['strict_intersection']['path_fraction']<1
 f2=f+[0,2.,0];b=pair(v,f2,np.array([2.,0,0]),.002);assert b['status']=='clear'
 f3=np.array([[.2,.2,.6005],[.2,.6,.2005],[.6,.2,.2005]])
 c=pair(v,f3,np.zeros(3),.002);assert c['status']=='within_required_band'
 return {'middle_only_strict_collision':a,'whole_path_separated':b,'separated_but_under2mm':c}
def run(args):
 D=Path(args.output).parent;D.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path(args.config).read_text());state=dict(np.load(args.state));rest=dict(np.load(args.rest));source=np.load(args.source_bag);candidate=json.loads(Path(args.stages).read_text())
 assert candidate['status']=='candidate','Do not certify a rejected robot route'
 route=candidate['route'];stages=route['stages'];required=float(args.clearance)
 assert required>=.002 and np.isfinite(required)
 initial_tcp=np.asarray(state['tcp_right']);bear=np.asarray(state['bear']);bag=np.asarray(state['bag']);tets=np.asarray(rest['bear_tets'],int)
 assert bear.shape==rest['bear'].shape and bag.shape==rest['bag'].shape and len(tets)==5486
 assert np.isfinite(bear).all() and np.isfinite(bag).all() and np.isfinite(initial_tcp).all()
 assert args.scope in ('frozen_hybrid','actual_joint_observation')
 if args.scope=='actual_joint_observation':
  # Scope is explicit, never inferred from a filename. The actual source file
  # still needs native provenance; matching times prevent mixing observations.
  assert np.isfinite(float(state['time']))
  assert abs(float(candidate['observed_time_s'])-float(state['time']))<1e-9
  assert abs(float(route['observed_time_s'])-float(state['time']))<1e-9
 rots=np.asarray([stage['rotation_xyzw'] for stage in stages],float)
 assert rots.shape==(len(stages),4) and np.isfinite(rots).all()
 assert np.max(np.linalg.norm(rots-rots[0],axis=1))<1e-12, 'Frozen translation certificate does not cover changing orientation'
 err,mapping=cKDTree(rest['bag']).query(source['points']+cfg['bag']['position']);assert err.max()<1e-6 and len(np.unique(mapping))==len(mapping)
 face_lookup={tuple(sorted(f)):i for i,f in enumerate(rest['bag_faces'])};assert len(face_lookup)==len(rest['bag_faces'])
 region=[];nativefaces=[]
 bodyset=set(source['body_face_indices'].tolist())
 chosen_handles=['front','back'] if args.handles=='both' else [args.handles]
 for name in chosen_handles:
  fi=[i for i,f in enumerate(source['faces'])if i not in bodyset and np.isin(f,source['handle_'+name]).all()]
  assert len(fi)==352
  for i in fi:
   f=mapping[source['faces'][i]];native_index=face_lookup[tuple(sorted(f))]
   nativefaces.append(f);region.append({'handle':name,'source_face_index':i,'native_face_index':native_index,'native_vertices':f.tolist()})
 assert len({r['native_face_index']for r in region})==352*len(chosen_handles)
 triangles=bag[np.array(nativefaces)]
 prefix=None
 if args.descent_prefix_to_rim:
  assert stages[-1]['name']=='insert_candidate' and len(stages)>=1
  start_tcp=np.asarray(stages[-2]['right_tcp_m'],float) if len(stages)>1 else initial_tcp.copy()
  end_tcp=np.asarray(stages[-1]['right_tcp_m'],float);displacement=end_tcp-start_tcp
  assert np.max(np.abs(displacement[:2]))<1e-12 and displacement[2]<0
  start_bear=bear+start_tcp-initial_tcp;rim=bag[mapping[source['rim']]]
  fraction=(float(rim[:,2].max())-float(start_bear[:,2].min()))/float(displacement[2])
  assert 0<=fraction<=1, 'Prefix boundary must be on the declared descent'
  prefix={'definition':'whole frozen bear minZ first reaches highest actual body-rim vertex Z; height condition only, not actual open-mouth crossing or containment','full_descent_fraction':fraction,
          'original_end_tcp_m':end_tcp.tolist(),'prefix_end_tcp_m':(start_tcp+fraction*displacement).tolist(),'rim_min_z_m':float(rim[:,2].min()),'rim_max_z_m':float(rim[:,2].max())}
  stages=[dict(s) for s in stages];stages[-1]['right_tcp_m']=prefix['prefix_end_tcp_m'];stages[-1]['name']='insert_prefix_to_highest_rim_height'
 j={'kind':'complete_frozen_tetra_straight_sweep_vs_selected_handles','status':'running','physical_success':False,'native_execution':False,'source_scope':args.scope,'not_a_joint_native_snapshot':args.scope!='actual_joint_observation','production_edited':False,
    'selected_handles':chosen_handles,'descent_prefix_boundary':prefix,'source_observed_time_s':float(state['time']),
    'required_euclidean_clearance_m':required,'numerical_guard_m':GUARD,'tetrahedra':len(tets),'handle_triangles':len(triangles),'per_stage_pair_count':int(len(tets)*len(triangles)),
    'source_to_native_max_error_m':float(err.max()),'input_sha256':{str(Path(p).resolve()):sha(p)for p in [args.rest,args.state,args.stages,args.config,args.source_bag,Path(__file__)]},'analytic_fixture_tests':selftest(),'stages':[]}
 def save():Path(args.output).write_text(json.dumps(j,indent=2,default=serial,allow_nan=False)+'\n')
 save();start=initial_tcp.copy();overall=True;stopped=False
 for stage in stages:
  end=np.asarray(stage['right_tcp_m'],float);delta=end-start;v=bear[tets]+(start-initial_tcp)
  lo=np.minimum(v.min(1),v.min(1)+delta);hi=np.maximum(v.max(1),v.max(1)+delta)
  rec={'name':stage['name'],'start_tcp_m':start.tolist(),'end_tcp_m':end.tolist(),'displacement_m':delta.tolist(),'aabb_certified_pair_count':0,'fine_pair_count':0,'fine_pairs':[],
       'minimum_aabb_gap_lower_m':None,'minimum_fine_gap_lower_m':None,'minimum_complete_gap_lower_m':None,'status':'running'}
  j['stages'].append(rec);mins=[];t0=time.monotonic()
  for face_index,f in enumerate(triangles):
   lower=f.min(0);upper=f.max(0)
   gaps=np.maximum(np.maximum(lo-upper,lower-hi),0.);dist=np.linalg.norm(gaps,axis=1)-GUARD
   far=dist>=required
   rec['aabb_certified_pair_count']+=int(far.sum())
   if far.any():
    m=float(dist[far].min());rec['minimum_aabb_gap_lower_m']=m if rec['minimum_aabb_gap_lower_m'] is None else min(m,rec['minimum_aabb_gap_lower_m'])
   for tet_index in np.flatnonzero(~far):
    proof=pair(v[tet_index],f,delta,required)
    item={'tetrahedron_index':int(tet_index),'tet_vertices':tets[tet_index].tolist(),**region[face_index],**proof}
    rec['fine_pairs'].append(item);rec['fine_pair_count']+=1
    g=proof['conservative_euclidean_projection_gap_lower_m']
    rec['minimum_fine_gap_lower_m']=g if rec['minimum_fine_gap_lower_m'] is None else min(g,rec['minimum_fine_gap_lower_m'])
    if proof['status']!='clear':
     overall=False;rec['status']='rejected';rec['first_failure']=item;stopped=True;break
   if stopped:break
  allbounds=[x for x in [rec['minimum_aabb_gap_lower_m'],rec['minimum_fine_gap_lower_m']]if x is not None]
  rec['minimum_complete_gap_lower_m']=min(allbounds) if not stopped and allbounds else None
  rec['processed_pair_count']=rec['aabb_certified_pair_count']+rec['fine_pair_count'];rec['complete_pair_coverage']=rec['processed_pair_count']==j['per_stage_pair_count']
  rec['wall_s']=time.monotonic()-t0
  if not stopped:rec['status']='clear';assert rec['complete_pair_coverage'] and rec['minimum_complete_gap_lower_m']>=required
  save();print(stage['name'],rec['status'],'minlower',rec['minimum_complete_gap_lower_m'],'fine',rec['fine_pair_count'],'wall',rec['wall_s'],flush=True)
  if stopped:break
  start=end
 j['status']='all_declared_stages_frozen_handles_clear' if overall else 'rejected'
 j['unassessed_remaining_stages']=[x['name']for x in stages[len(j['stages']):]]
 scope_note='Source is explicitly hybrid, not a joint native observation.' if args.scope=='frozen_hybrid' else 'Source is explicitly designated one actual joint native observation with matching planner observation times; its future translation is still hypothetical.'
 j['limitations']=['Frozen tetrahedral bear and fixed true handle triangles only. '+scope_note,'The continuous certificate covers the declared straight TCP translations of an unchanging bear; not joint-spline FK errors, rotation, soft deformation, compliance, slipping or future cloth motion.','2mm is this new offline engineering clearance criterion; native material and contact parameters are unchanged.','LP/QP suggestions never authorize separation: all accepted fine pairs have an independently recomputed separating projection bound. AABB pruning also supplies a recomputed geometric lower bound.','Reported per-stage minimum is the smallest certified lower bound across all pairs, not an exact global Euclidean closest-distance calculation.','Cube diagnostics have L-infinity meaning, distinct from Euclidean2mm engineering gate.','Release and withdrawal robot geometry are covered elsewhere; no new physical task success or containment claim follows.','A descent-prefix boundary, if selected, is a height condition and does not establish crossing the actual open mouth or containment.','Numerical guards use double precision, not directed-rounding interval arithmetic.']
 j['inputs_unchanged']=all(sha(p)==h for p,h in j['input_sha256'].items());save();print('REPORT',args.output,j['status'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rest');parser.add_argument('--state');parser.add_argument('--stages');parser.add_argument('--config');parser.add_argument('--source-bag');parser.add_argument('--output');parser.add_argument('--clearance',type=float,default=.002);parser.add_argument('--scope',choices=['frozen_hybrid','actual_joint_observation'],default='frozen_hybrid');parser.add_argument('--handles',choices=['both','front','back'],default='both');parser.add_argument('--descent-prefix-to-rim',action='store_true');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
 if args.self_test:print(json.dumps(selftest(),indent=2,default=serial))
 else:run(args)
