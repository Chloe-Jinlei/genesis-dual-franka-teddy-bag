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
from collections import Counter, defaultdict
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
def source_topology(source):
 """Derive the complete sewn-bag partition, independently of native ordering.

 The expected topology is one connected cloth sheet with three boundary loops:
 the common outer mouth boundary and one inner hole for each real handle.
 This validates this two-panel family; it is not a general garment classifier.
 """
 points=np.asarray(source['points'],float);faces=np.asarray(source['faces'])
 assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
 assert faces.ndim==2 and faces.shape[1]==3 and np.issubdtype(faces.dtype,np.integer)
 assert faces.min()>=0 and faces.max()<len(points)
 assert all(len(set(map(int,f)))==3 for f in faces)
 assert len({tuple(sorted(map(int,f)))for f in faces})==len(faces)
 assert np.all(np.linalg.norm(np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]]),axis=1)>1e-14)
 def ids(key,n):
  a=np.asarray(source[key]);assert a.ndim==1 and np.issubdtype(a.dtype,np.integer)
  assert len(a)==len(np.unique(a)) and a.min()>=0 and a.max()<n
  return set(map(int,a))
 body_faces=ids('body_face_indices',len(faces));body=ids('body',len(points))
 assert set(map(int,faces[sorted(body_faces)].ravel()))==body
 all_faces=set(range(len(faces)));nonbody=all_faces-body_faces;regions={};handle_nodes={}
 for name in ('front','back'):
  nodes=ids('handle_'+name,len(points));handle_nodes[name]=nodes
  fi={i for i in nonbody if set(map(int,faces[i])).issubset(nodes)}
  assert fi and set(map(int,faces[sorted(fi)].ravel()))==nodes
  regions[name]=sorted(fi)
 assert not(set(regions['front'])&set(regions['back']))
 assert set(regions['front'])|set(regions['back'])==nonbody,'Every non-body face must belong to exactly one real handle'
 assert body|handle_nodes['front']|handle_nodes['back']==set(range(len(points)))
 counts=Counter(tuple(sorted((int(a),int(b))))for f in faces for a,b in zip(f,np.roll(f,-1)))
 assert all(n in (1,2)for n in counts.values()),'Non-manifold edge'
 adjacency=defaultdict(list);boundary=defaultdict(list)
 for (a,b),n in counts.items():
  adjacency[a].append(b);adjacency[b].append(a)
  if n==1:boundary[a].append(b);boundary[b].append(a)
 assert boundary and all(len(v)==2 for v in boundary.values())
 def components(adj):
  remaining=set(adj);out=[]
  while remaining:
   todo=[min(remaining)];seen=set()
   while todo:
    q=todo.pop()
    if q in seen:continue
    seen.add(q);todo.extend(adj[q])
   remaining-=seen;out.append(seen)
  return out
 assert len(components(adjacency))==1 and len(adjacency)==len(points)
 loops=components(boundary);assert len(loops)==3
 loop_reports=[];types=[]
 for loop in loops:
  members=[name for name in ('front','back')if loop&(handle_nodes[name]-body)]
  assert members and loop&body
  kind='outer_mouth_boundary' if len(members)==2 else members[0]+'_handle_inner_boundary'
  types.append(kind);loop_reports.append({'kind':kind,'vertex_count':len(loop),'body_vertex_count':len(loop&body),'handle_interior_vertex_counts':{name:len(loop&(handle_nodes[name]-body))for name in ('front','back')}})
 assert sorted(types)==sorted(['outer_mouth_boundary','front_handle_inner_boundary','back_handle_inner_boundary'])
 legacy=len(points)==1896 and len(faces)==3664
 if legacy:assert len(body_faces)==2960 and all(len(regions[n])==352 for n in ('front','back'))
 report={'source_vertices':len(points),'source_triangles':len(faces),'body_triangles':len(body_faces),'handle_triangles':{n:len(regions[n])for n in regions},'handle_vertices':{n:len(handle_nodes[n])for n in handle_nodes},'complete_disjoint_face_partition':True,'unclassified_faces':0,'connected_components':1,'boundary_loops':loop_reports,'legacy_352_per_handle_asserted':legacy}
 return regions,report

def complete_native_mapping(source,rest,cfg,chosen_handles):
 regions,topology=source_topology(source)
 native_points=np.asarray(rest['bag'],float);native_faces=np.asarray(rest['bag_faces'])
 assert native_points.shape==source['points'].shape and np.isfinite(native_points).all()
 assert native_faces.shape==source['faces'].shape and np.issubdtype(native_faces.dtype,np.integer)
 assert native_faces.min()>=0 and native_faces.max()<len(native_points)
 err,mapping=cKDTree(native_points).query(source['points']+np.asarray(cfg['bag']['position'],float))
 assert err.max()<1e-6 and len(np.unique(mapping))==len(native_points)
 lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(native_faces)}
 assert len(lookup)==len(native_faces)
 source_to_native_faces=[]
 for f in source['faces']:
  key=tuple(sorted(map(int,mapping[f])));assert key in lookup
  source_to_native_faces.append(lookup[key])
 assert len(set(source_to_native_faces))==len(native_faces),'Complete all-face native correspondence required'
 region=[];selected=[]
 for name in chosen_handles:
  for i in regions[name]:
   f=mapping[source['faces'][i]];native_index=source_to_native_faces[i]
   selected.append(f);region.append({'handle':name,'source_face_index':i,'native_face_index':native_index,'native_vertices':f.tolist()})
 assert len({r['native_face_index']for r in region})==sum(len(regions[n])for n in chosen_handles)
 topology.update(native_vertices_bijective=True,native_faces_bijective=True,all_native_faces_mapped=len(source_to_native_faces),source_to_native_max_error_m=float(err.max()))
 return mapping,err,region,np.asarray(selected,int),topology

def run(args):
 D=Path(args.output).parent;D.mkdir(parents=True,exist_ok=True)
 cfg=json.loads(Path(args.config).read_text());state=dict(np.load(args.state));rest=dict(np.load(args.rest));source=np.load(args.source_bag);candidate=json.loads(Path(args.stages).read_text())
 assert candidate['status']=='candidate','Do not certify a rejected robot route'
 route=candidate['route'];stages=route['stages'];required=float(args.clearance)
 assert required==.001 and np.isfinite(required), 'Only the single explicitly authorized weaker offline1mm diagnostic'
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
 chosen_handles=['front','back'] if args.handles=='both' else [args.handles]
 mapping,err,region,nativefaces,topology=complete_native_mapping(source,rest,cfg,chosen_handles)
 triangles=bag[nativefaces]
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
 j={'kind':'explicit_weaker_1mm_hypothetical_contact_screen','original_2mm_result_remains':'rejected','not_an_original_2mm_pass':True,'status':'running','physical_success':False,'native_execution':False,'source_scope':args.scope,'not_a_joint_native_snapshot':args.scope!='actual_joint_observation','production_edited':False,
    'selected_handles':chosen_handles,'descent_prefix_boundary':prefix,'source_observed_time_s':float(state['time']),
    'required_euclidean_clearance_m':required,'numerical_guard_m':GUARD,'tetrahedra':len(tets),'handle_triangles':len(triangles),'per_stage_pair_count':int(len(tets)*len(triangles)),
    'complete_source_topology_and_native_mapping':topology,'source_to_native_max_error_m':float(err.max()),'input_sha256':{str(Path(p).resolve()):sha(p)for p in [args.rest,args.state,args.stages,args.config,args.source_bag,Path(__file__)]},'analytic_fixture_tests':selftest(),'stages':[]}
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
 j['limitations']=['Frozen tetrahedral bear and fixed true handle triangles only. '+scope_note,'The continuous certificate covers the declared straight TCP translations of an unchanging bear; not joint-spline FK errors, rotation, soft deformation, compliance, slipping or future cloth motion.','1mm is an explicitly authorized weaker offline diagnostic, not an original2mm pass; native material/contact parameters and physical guards are unchanged.','LP/QP suggestions never authorize separation: all accepted fine pairs have an independently recomputed separating projection bound. AABB pruning also supplies a recomputed geometric lower bound.','Reported per-stage minimum is the smallest certified lower bound across all pairs, not an exact global Euclidean closest-distance calculation.','Cube diagnostics have L-infinity meaning, distinct from Euclidean2mm engineering gate.','Release and withdrawal robot geometry are covered elsewhere; no new physical task success or containment claim follows.','A descent-prefix boundary, if selected, is a height condition and does not establish crossing the actual open mouth or containment.','Numerical guards use double precision, not directed-rounding interval arithmetic.']
 j['inputs_unchanged']=all(sha(p)==h for p,h in j['input_sha256'].items());save();print('REPORT',args.output,j['status'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rest');parser.add_argument('--state');parser.add_argument('--stages');parser.add_argument('--config');parser.add_argument('--source-bag');parser.add_argument('--output');parser.add_argument('--clearance',type=float,default=.001);parser.add_argument('--scope',choices=['frozen_hybrid','actual_joint_observation'],default='frozen_hybrid');parser.add_argument('--handles',choices=['both','front','back'],default='both');parser.add_argument('--descent-prefix-to-rim',action='store_true');parser.add_argument('--self-test',action='store_true');parser.add_argument('--topology-only',action='store_true');args=parser.parse_args()
 if args.topology_only:
  assert args.source_bag and args.output
  _,report=source_topology(np.load(args.source_bag))
  report.update(kind='source_only_topology_no_native_state',native_execution=False,physical_success=False,input_sha256={str(Path(p).resolve()):sha(p)for p in [args.source_bag,Path(__file__)]})
  Path(args.output).parent.mkdir(parents=True,exist_ok=True)
  Path(args.output).write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\n')
  print('REPORT',args.output,flush=True)
 elif args.self_test:print(json.dumps(selftest(),indent=2,default=serial))
 else:run(args)
