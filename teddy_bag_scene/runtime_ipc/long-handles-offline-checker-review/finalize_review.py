from pathlib import Path
import ast,hashlib,json,difflib
import numpy as np
from scipy.spatial import cKDTree
P=Path.cwd();D=P/'runtime_ipc/long-handles-offline-checker-review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def ser(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
oldtool=P/'runtime_ipc/continuous_frozen_handles_review.py';newtool=P/'runtime_ipc/continuous_frozen_handles_review_all_straps.py'
oldreport=P/'runs/two_panel_full_25/independent-actual-short-descent-handles/review.json';newcontrol=D/'old-actual-control.json'
oldresult=read(oldreport)['checks']['held_front_full_short_descent'];control=read(newcontrol)
assert sha(oldtool)==oldresult['input_sha256'][str(oldtool)]
oldstage={k:v for k,v in oldresult['stages'][0].items()if k!='wall_s'};newstage={k:v for k,v in control['stages'][0].items()if k!='wall_s'}
assert oldstage==newstage
olds=oldtool.read_text();news=newtool.read_text()
def funs(s):return {f.name:ast.dump(f,include_attributes=False)for f in ast.parse(s).body if isinstance(f,ast.FunctionDef)}
a=funs(olds);b=funs(news);protected=['sha','serial','projection','matrix','initial_axes','strict_witness','pair','selftest']
assert all(a[f]==b[f]for f in protected)
assert olds[olds.index(' for stage in stages:'):olds.index(" j['status']='all_declared")]==news[news.index(' for stage in stages:'):news.index(" j['status']='all_declared")]
assert 'assert required>=.002 and np.isfinite(required)'in news
oldsource=P/'runs/two_panel_full_25/source/assets/two_panel_bag_wide_tall_coarse.npz';newsource=P/'assets/two_panel_bag_wide_tall_long_handles_coarse.npz'
o=np.load(oldsource);n=np.load(newsource)
assert np.array_equal(o['body'],n['body']) and np.array_equal(o['body_face_indices'],n['body_face_indices'])
assert np.array_equal(o['points'][o['body']],n['points'][n['body']])
assert np.array_equal(o['faces'][o['body_face_indices']],n['faces'][n['body_face_indices']])
for key in ['rim','u_seam','bottom_seam','seam_left','seam_right']:
 assert np.array_equal(o[key],n[key])
oldface=3168;oldvs=o['faces'][oldface];offset=np.array([0,0,.14]);err,newvs=cKDTree(n['points']).query(o['points'][oldvs]+offset)
assert err.max()<1e-12 and len(np.unique(newvs))==3
lookup={tuple(sorted(f)):i for i,f in enumerate(n['faces'])};newface=lookup[tuple(sorted(newvs))]
assert np.isin(newvs,n['handle_front']).all() and not np.isin(newvs,n['body']).any()
grips={}
for label in ['grasp_front_x_minus','grasp_front_x_plus','grasp_back_x_minus','grasp_back_x_plus']:
 oldp=o['points'][o[label]];newp=n['points'][n[label]]
 assert oldp.shape==newp.shape and np.max(np.abs(newp-oldp-offset))<1e-12
 grips[label]={'old_mean_local_m':oldp.mean(0),'new_mean_local_m':newp.mean(0),'displacement_m':newp.mean(0)-oldp.mean(0)}
root_unchanged={}
for name in ['front','back']:
 for side in ['left','right']:
  k=f'handle_{name}_root_{side}';root_unchanged[k]=bool(np.array_equal(o['points'][o[k]],n['points'][n[k]]));assert root_unchanged[k]
wpath=P/'runtime_ipc/full26-plus15-1mm-hypothetical-handles/review.json';w=read(wpath)['checks']['held_front_full_short_descent']['first_failure']
assert w['source_face_index']==oldface and w['strict_intersection']['strict_intersection_proven']
patch=D/'checker-vs-legacy.patch';patch.write_text(''.join(difflib.unified_diff(olds.splitlines(True),news.splitlines(True),fromfile=str(oldtool),tofile=str(newtool))))
files=[oldtool,newtool,oldreport,newcontrol,oldsource,newsource,D/'new-source-topology.json',wpath,P/'config_long_handles_gravity_trial.json',P/'assets/metadata_two_panel_wide_tall_long_handles_coarse.json',patch]
j={'scope':'Offline checker adaptation plus source material geometry comparison only; no new dynamic bag shape or native trial','status':'new_checker_complete_partition_validated_old_control_identical_new_source_topology_validated','native_execution':False,'production_edited':False,'physical_success':False,
'checker':{'path':str(newtool),'legacy_checker_unchanged':True,'protected_geometry_functions_AST_identical':protected,'per_stage_proof_loop_text_identical':True,'minimum_offline_clearance_unchanged_m':.002,'numerical_guard_unchanged_m':1e-7,'source_native_mapping':'Bijective coordinate mapping <=1e-6m plus unique sorted vertex triplets for ALL native faces; selected complete derived handle faces thereafter','old_control':{'source':'actual full25 transport39.265 snapshot; hypothetical short vertical sweep to TCP1.085','stage_geometry_and_results_exactly_equal_excluding_elapsed_wall_time':True,'full_pair_coverage':control['stages'][0]['complete_pair_coverage'],'all_tets':control['tetrahedra'],'all_front_faces':control['handle_triangles'],'pairs':control['per_stage_pair_count'],'minimum_certified_gap_lower_m':control['stages'][0]['minimum_complete_gap_lower_m'],'native_mapping_max_error_m':control['source_to_native_max_error_m']},'new_source_topology':read(D/'new-source-topology.json')},
'source_geometry':{'body_vertices_faces_rim_and_U_seam_exactly_unchanged':True,'root_material_positions_unchanged':root_unchanged,'grasp_groups':grips,'former_strict_witness_patch':{'old_source_face':oldface,'old_source_vertices':oldvs,'old_source_points_m':o['points'][oldvs],'new_source_face_matched_by_coordinates_not_reused_ID':newface,'new_source_vertices':newvs,'new_source_points_m':n['points'][newvs],'correspondence_error_m':float(err.max()),'patch_source_translation_m':offset,'location':'front handle opposite upper arch near x-plus material region, while left hand holds x-minus; not a sewn root','historical_hypothetical_world_witness_m':w['strict_intersection']['cloth_point_world_m'],'historical_short_descent_fraction':w['strict_intersection']['path_fraction'],'historical_source':'Old +15-degree hypothetical teddy with short-handle actual full25 bag; not a current long-handle observation'}},
'geometric_assessment':{'plausible_benefit':'The exact old blocking upper-arch material patch and grasp groups are 140mm higher in source geometry while all body/root points stay fixed. Matching open-target height increase preserves initial hand-to-grip alignment. This directly targets the former held-arch obstruction and motivates a fresh handoff trial.','remaining_obstructions':['Longer vertical legs still connect fixed body roots to the raised arch and can occupy the teddy transport corridor.','The opposite side of the held loop can sag/rotate despite the held grasp; source translation is not a deformed-state translation.','The released back handle has 280mm more centerline and can drape into the mouth or contact the head/feet.','Left hand and robot configuration also change at higher targets; a new actual joint route must be checked.'],'next_evidence':'Use a new same-observation native rest+bag+bear+candidate with this all-straps checker after real handoff/yaw. Complete every 576-face handle; front full intended descent, both handles transport/pre-rim. No old source face ID or rigid +140mm bag hypothesis may stand in for that evidence.'},
'limitations':['Source geometry and topology alone do not prove reachability, retained cloth grasp, usable opening, dynamic clearance, containment or successful insertion.','Only one historical geometry control was run; existing unchanged embedded analytic proof fixtures execute inside that control. New long-handle source has no native mapping/route claim yet.','All physical materials, force/torque/J/robot/contact thresholds and production configs remain unmodified by this review.'],
'input_sha256':{str(p):sha(p)for p in files}}
(D/'review.json').write_text(json.dumps(j,indent=2,default=ser,allow_nan=False)+'\n')
print('REVIEW',sha(D/'review.json'));print('CHECKER',sha(newtool));print('NEW_PATCH',newface,newvs.tolist());print('OLD_CONTROL',control['stages'][0]['minimum_complete_gap_lower_m'])
