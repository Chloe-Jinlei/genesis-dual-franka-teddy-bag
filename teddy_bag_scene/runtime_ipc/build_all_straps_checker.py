from pathlib import Path
old=Path('runtime_ipc/continuous_frozen_handles_review.py')
new=Path('runtime_ipc/continuous_frozen_handles_review_all_straps.py')
assert not new.exists()
s=old.read_text()
s=s.replace('from pathlib import Path\n','from pathlib import Path\nfrom collections import Counter, defaultdict\n',1)
helper='''def source_topology(source):
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

'''
s=s.replace('def run(args):\n',helper+'def run(args):\n',1)
start=s.index(" err,mapping=cKDTree(rest['bag'])")
end=s.index(' prefix=None\n',start)
s=s[:start]+''' chosen_handles=['front','back'] if args.handles=='both' else [args.handles]
 mapping,err,region,nativefaces,topology=complete_native_mapping(source,rest,cfg,chosen_handles)
 triangles=bag[nativefaces]
'''+s[end:]
s=s.replace("'source_to_native_max_error_m':float(err.max()),'input_sha256'", "'complete_source_topology_and_native_mapping':topology,'source_to_native_max_error_m':float(err.max()),'input_sha256'",1)
s=s.replace("parser.add_argument('--self-test',action='store_true');args=parser.parse_args()", "parser.add_argument('--self-test',action='store_true');parser.add_argument('--topology-only',action='store_true');args=parser.parse_args()",1)
s=s.replace(" if args.self_test:print(json.dumps(selftest(),indent=2,default=serial))",''' if args.topology_only:
  assert args.source_bag and args.output
  _,report=source_topology(np.load(args.source_bag))
  report.update(kind='source_only_topology_no_native_state',native_execution=False,physical_success=False,input_sha256={str(Path(p).resolve()):sha(p)for p in [args.source_bag,Path(__file__)]})
  Path(args.output).parent.mkdir(parents=True,exist_ok=True)
  Path(args.output).write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\\n')
  print('REPORT',args.output,flush=True)
 elif args.self_test:print(json.dumps(selftest(),indent=2,default=serial))''',1)
new.write_text(s)
print(new)
