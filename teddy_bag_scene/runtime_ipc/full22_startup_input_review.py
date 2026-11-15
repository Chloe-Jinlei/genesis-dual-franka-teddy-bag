"""Read-only full22 startup audit. Run only after final source lock and actual initial/rest exist.
No Genesis import, native stepping, meshing, planning, or production mutation.
"""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json,os,re,sys
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_full_22'
PROBE=ROOT/'runs/two_panel_handoff_upright_offset_01'
CONFIG=ROOT/'config_guarded_upright_offset_trial.json'
CONFIG_SHA='34d29f270144496818569b260185f98ba7f75d9d5c8964b63fbd073d826ec52b'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def diff(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((diff(a.get(k,'<missing>'),b.get(k,'<missing>'),(p+'.'+k).strip('.')) for k in sorted(set(a)|set(b))),[])
    return [] if a==b else [dict(path=p,before=a,after=b)]
def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def dump(n):return ast.dump(n,include_attributes=False)
def functions(path):
    return {n.name:n for n in ast.walk(ast.parse(Path(path).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def strip_doc(n):
    n=copy.deepcopy(n)
    if n.body and isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body=n.body[1:]
    return n

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-run-sha',required=True,help='Final run.py hash supplied after the authorized strict-boolean edit is frozen.')
    parser.add_argument('--launch',type=Path,default=ROOT/'runtime_ipc/full22-launch.json')
    args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{64}',args.expected_run_sha):raise ValueError('Expected final runner SHA must be 64 lowercase hex characters')
    required=[RUN/'initial.npz',RUN/'rest.npz',RUN/'config.json',RUN/'input_sha256.json',args.launch,PROBE/'initial.npz',PROBE/'rest.npz']
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise SystemExit('Not ready: actual saved startup inputs are required: '+json.dumps(missing))
    # Imports happen only after readiness and resolve into this run's frozen source.
    sys.path[:0]=[str(RUN/'source'),str(ROOT/'runtime_ipc')]
    from audit_input_manifest import audit
    from physics_config import _map_shell_material_regions
    from validate import mouth_structure,signed_volumes
    manifest=audit(RUN);current=read(RUN/'input_sha256.json');prior=read(PROBE/'input_sha256.json');cfg=read(RUN/'config.json');old=read(PROBE/'config.json');launch=read(args.launch)
    checks={}
    def check(k,v):checks[k]=bool(v)
    inputpaths=[*required,CONFIG,Path(__file__),*[RUN/k for k in current],PROBE/'config.json',PROBE/'input_sha256.json']
    inputs={str(p):sha(p) for p in inputpaths}
    check('all_declared_frozen_input_hashes_match',manifest['all_declared_hashes_match'])
    check('all_frozen_snapshot_files_manifested',manifest['all_snapshot_files_in_manifest'])
    check('all_external_robot_meshes_available',not manifest['missing_external_model_files'])
    check('supplied_config_is_reviewed_candidate',sha(CONFIG)==CONFIG_SHA and cfg==read(CONFIG))
    lc=Path(launch['config']);lc=lc if lc.is_absolute() else ROOT/lc
    check('launch_config_path_and_hash_match',lc.resolve()==CONFIG.resolve() and launch['config_sha256']==CONFIG_SHA)
    check('final_runner_hash_matches_explicit_lock',sha(RUN/'source/run.py')==args.expected_run_sha==sha(ROOT/'run.py'))
    check('all_live_source_assets_match_frozen_at_review',all(sha(ROOT/k.removeprefix('source/'))==v for k,v in current.items() if k.startswith('source/')))
    launch_sources=launch.get('source_frozen',{})
    check('all_declared_launch_source_hashes_match',all(sha(RUN/'source'/k)==v for k,v in launch_sources.items()))
    changed=sorted(k for k in set(current)&set(prior) if current[k]!=prior[k]);added=sorted(set(current)-set(prior));removed=sorted(set(prior)-set(current))
    check('only_authorized_config_runner_yaw_helper_changed_since_probe',changed==['config.json','source/postlift_yaw.py','source/run.py'] and not added and not removed)
    expected=copy.deepcopy(old);ct=expected['task']['contact_trial'];ct['postlift_yaw']['world_z_degrees']=-30.;ct['center_offsets_xy_m']=[[0.,.0345]];ct['recenter_enabled']=False;ct.pop('recenter_offsets_xy_m',None)
    check('config_exactly_four_authorized_semantic_changes',cfg==expected)
    check('bag_bear_robots_table_simulation_unchanged',all(cfg[k]==old[k] for k in ('bag','bear','robots','table','simulation')))
    task=cfg['task'];ct=task['contact_trial']
    check('single_offset_strictly_inside_original35mm_domain',ct['center_offsets_xy_m']==[[0.,.0345]] and np.linalg.norm(ct['center_offsets_xy_m'][0])<ct['maximum_initial_com_offset_m']==.035)
    check('recenter_disabled_is_literal_boolean_and_no_fallback_offsets',ct.get('recenter_enabled') is False and 'recenter_offsets_xy_m' not in ct)
    check('unchanged_vertical_y_then_x_and_original_geometry_gates',ct['insertion_strategy']=='vertical' and ct['transport_xy_strategy']=='y_then_x' and ct['left_robot_soft_sweep_clearance_m']==.005 and ct['bottom_clearance_m']==.01 and ct['minimum_section_envelope_overlap_fraction']==.4)
    check('same_initial_offcenter_named_grasps_and_open_targets',task['handle_grasps']==old['task']['handle_grasps'] and task['open_targets']==old['task']['open_targets'])
    # Compare physical observer/execution/finish and all other runner methods by AST.
    nf,of=functions(RUN/'source/run.py'),functions(PROBE/'source/run.py')
    check('runner_all_other_functions_AST_identical',set(nf)==set(of) and all(dump(nf[k])==dump(of[k]) for k in of if k!='run_guarded_contact_trial'))
    n=copy.deepcopy(nf['run_guarded_contact_trial']);removed_nodes=[]
    class StripAuthorizedSwitch(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='recenter_enabled' for t in node.targets):removed_nodes.append(node);return None
            return self.generic_visit(node)
        def visit_If(self,node):
            t=node.test
            if isinstance(t,ast.Compare) and isinstance(t.left,ast.Call) and isinstance(t.left.func,ast.Name) and t.left.func.id=='type' and any(isinstance(x,ast.Name) and x.id=='recenter_enabled' for x in t.left.args):removed_nodes.append(node);return None
            if isinstance(t,ast.UnaryOp) and isinstance(t.op,ast.Not) and isinstance(t.operand,ast.Name) and t.operand.id=='recenter_enabled':removed_nodes.append(node);return None
            return self.generic_visit(node)
    normalized=StripAuthorizedSwitch().visit(n)
    check('contact_flow_only_three_authorized_switch_statements_added',len(removed_nodes)==3 and dump(normalized)==dump(of['run_guarded_contact_trial']))
    expected_switch=[ast.parse('recenter_enabled=self.config.get("task",{}).get("contact_trial",{}).get("recenter_enabled",True)').body[0],ast.parse('if type(recenter_enabled) is not bool:\n raise GateFailure("contact_trial.recenter_enabled must be a boolean")').body[0],ast.parse('if not recenter_enabled:\n raise GateFailure("Fresh contact insertion rejected and recentering is disabled; see "+filename)').body[0]]
    check('switch_exact_default_true_strict_bool_and_fail_closed',len(removed_nodes)==3 and all(dump(a)==dump(b) for a,b in zip(removed_nodes,expected_switch)))
    # Helper's only executable delta must be the declared 15->30 magnitude boundary.
    hn,ho=functions(RUN/'source/postlift_yaw.py'),functions(PROBE/'source/postlift_yaw.py');ht=ast.parse((RUN/'source/postlift_yaw.py').read_text())
    constants=[x.value.value for x in ht.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_POSTLIFT_YAW_DEGREES' for t in x.targets) and isinstance(x.value,ast.Constant)]
    check('new_yaw_constant_is30',constants==[30])
    old_guard=next(x for x in ast.walk(ho['plan_postlift_yaw']) if isinstance(x,ast.If) and 'abs(degrees)' in ast.unparse(x.test) and 'duration < 6' in ast.unparse(x.test))
    new_guards=[x for x in ast.walk(hn['plan_postlift_yaw']) if isinstance(x,ast.If) and 'MAX_POSTLIFT_YAW_DEGREES' in ast.unparse(x.test)]
    expected_new_guard=ast.parse('if not 0 < abs(degrees) <= MAX_POSTLIFT_YAW_DEGREES or duration < 6:\n raise ValueError(f"postlift yaw requires nonzero |degrees|<={MAX_POSTLIFT_YAW_DEGREES} and duration>=6s")').body[0]
    check('new_yaw_guard_changes_only_declared_magnitude_keeps6s',len(new_guards)==1 and dump(new_guards[0])==dump(expected_new_guard))
    hplan=copy.deepcopy(hn['plan_postlift_yaw'])
    class NormalizeYaw(ast.NodeTransformer):
        def visit_If(self,node):
            if 'MAX_POSTLIFT_YAW_DEGREES' in ast.unparse(node.test):
                node.test=copy.deepcopy(old_guard.test);node.body=copy.deepcopy(old_guard.body)
            return self.generic_visit(node)
    normalized_h=NormalizeYaw().visit(hplan)
    check('yaw_all_original_gates_transforms_and_same_plan_preserved',set(hn)==set(ho) and all(dump(strip_doc(normalized_h if k=='plan_postlift_yaw' else hn[k]))==dump(strip_doc(ho[k])) for k in ho))
    # Actual saved startup geometry; no assumption about global mesh IDs in new runs.
    rest=dict(np.load(RUN/'rest.npz'));initial=dict(np.load(RUN/'initial.npz'));pr=dict(np.load(PROBE/'rest.npz'));pi=dict(np.load(PROBE/'initial.npz'))
    check('actual_initial_time_zero_and_arrays_finite',float(initial['time'])==0. and all(np.isfinite(v).all() for v in initial.values()) and all(np.isfinite(v).all() for v in rest.values()))
    check('both_soft_initial_shapes_equal_saved_rest',all(np.array_equal(initial[k],rest[k]) for k in ('bag','bear')))
    check('soft_native_rest_arrays_and_topologies_match_same_probe_assets',set(rest)==set(pr) and all(np.array_equal(rest[k],pr[k]) for k in rest))
    check('native_initial_robot_q_and_TCP_match_same_probe_initial_pose',all(np.array_equal(initial[k],pi[k]) for k in ('q_left','q_right','tcp_left','tcp_right')))
    check('actual_robot_initial_q9_and_tcp3_shapes',all(initial['q_'+s].shape==(9,) and initial['tcp_'+s].shape==(3,) for s in ('left','right')))
    vol=signed_volumes(rest['bear'],rest['bear_tets']);check('all_native_rest_tets_positive_and_finite',np.isfinite(vol).all() and np.all(vol>0))
    edges=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]);tv=rest['bear'][rest['bear_tets']];den=np.square(tv[:,edges[:,0]]-tv[:,edges[:,1]]).sum((1,2));quality=12*np.power(3*vol,2/3)/den
    check('tet_shape_quality_finite_positive',np.isfinite(quality).all() and np.all(quality>0))
    roi=np.linalg.norm(rest['bear']-(np.array(cfg['bear']['position'])+task['bear_grasp_local']),axis=1)<task['bear_grasp_tracking_radius_m'];check('initial_tracking_roi_has_at_least3_material_points',roi.sum()>=3)
    obj=RUN/'source/assets'/cfg['bear']['mesh'];objpoints=np.array([[float(x) for x in line.split()[1:4]] for line in obj.read_text().splitlines() if line.startswith('v ')])+cfg['bear']['position'];bearerr,_=cKDTree(rest['bear']).query(objpoints);check('every_physical_bear_OBJ_vertex_present_in_native_rest',bearerr.max()<2e-7)
    asset=RUN/'source/assets'/cfg['bag']['npz_file'];a=dict(np.load(asset));meta=read(RUN/'source/assets'/cfg['bag']['metadata'])['bag'];front,back,seam=(set(map(int,a[k])) for k in ('panel_front','panel_back','u_seam'));pf,pb,body=(set(map(int,a[k])) for k in ('panel_front_face_indices','panel_back_face_indices','body_face_indices'));fg,bg=a['panel_front_grid'],a['panel_back_grid']
    check('two_panels_share_exact_U_seam',front&back==seam and np.array_equal(fg[0],bg[0]) and np.array_equal(fg[:,0],bg[:,0]) and np.array_equal(fg[:,-1],bg[:,-1]))
    check('exactly_two_disjoint_panel_face_sets_cover_body',not pf&pb and pf|pb==body)
    check('bottom_zero_width_no_area_no_hidden_support_no_fixed_vertices',np.ptp(a['points'][a['bottom_seam'],1])==0 and meta['depth_m']==0 and meta['bottom_seam_transverse_depth_m']==0 and not meta['bottom_seam_has_area'] and not meta['hidden_rigid_support'] and meta['fixed_particles']==0)
    _,_,topology=mouth_structure(rest['bag'],rest['bag_faces'],cfg)
    check('native_body_open_direct_U_seam_without_bottom_or_side_panels',topology['actual_open_body_boundary'] and topology['two_panels_directly_sewn_along_u'] and topology['independent_flat_bottom_triangles']==0 and topology['independent_flat_side_triangles']==0)
    regions,mapping=_map_shell_material_regions(asset,cfg['bag']['position'],rest['bag'],rest['bag_faces']);check('native_body_and_two_handles_complete_disjoint_partition',mapping['body_and_handles_disjoint_complete_face_partition'] and {k:len(v) for k,v in regions.items()}=={'body':len(body),'handle_front':352,'handle_back':352})
    check('same50_250_material_initialization_and_native_scene_source',cfg['bag']['shear_stiffness']==50 and cfg['bag']['handle_shear_stiffness']==250 and all(current['source/'+k]==prior['source/'+k] for k in ('scene.py','physics_config.py')))
    # A startup report records code/config evidence; material attributes are not claimed before final report.
    material=read(RUN/'report.json').get('shell_conversion') if (RUN/'report.json').is_file() else None
    process=dict(pid=launch.get('pid'),observed=False)
    if process['pid'] is not None:
        proc=Path('/proc')/str(process['pid'])
        if (proc/'cmdline').exists():
            command=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x];cwd=os.readlink(proc/'cwd');process.update(observed=True,command=command,cwd=cwd)
            check('active_process_cwd_and_full_task_match',cwd==str(ROOT) and '--task' in command and command[command.index('--task')+1]=='full')
            if '--config' in command:
                cp=Path(command[command.index('--config')+1]);cp=cp if cp.is_absolute() else Path(cwd)/cp;check('active_process_config_matches_reviewed_input',cp.resolve()==CONFIG.resolve())
    logpath=Path(launch.get('log',ROOT/'runs/two_panel_full_22.log'));logpath=logpath if logpath.is_absolute() else ROOT/logpath;log=logpath.read_text() if logpath.exists() else ''
    check('native_build_completed', 'BUILD_DONE' in log)
    check('review_inputs_unchanged',all(sha(Path(p))==v for p,v in inputs.items()))
    result=dict(kind='independent_full22_startup_input_review',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physical_success=False,native_execution_by_review=False,startup_input_checks_passed=all(checks.values()),checks=checks,failed_checks=[k for k,v in checks.items() if not v],manifest=manifest,launch=launch,process=process,config_diff_from_probe=diff(old,cfg),changed_common_files=changed,added_files=added,removed_files=removed,explicit_final_runner_sha256=args.expected_run_sha,native_initial=dict(bag_vertices=len(rest['bag']),bag_faces=len(rest['bag_faces']),bear_vertices=len(rest['bear']),bear_tetrahedra=len(rest['bear_tets']),bear_dimensions_m=np.ptp(rest['bear'],axis=0),minimum_signed_volume_m3=vol.min(),total_rest_volume_m3=vol.sum(),declared_density_kg_m3=cfg['bear']['rho'],volume_times_density_mass_kg=vol.sum()*cfg['bear']['rho'],native_mass_readback=False,mean_ratio_quality_quantiles=np.quantile(quality,[0,.01,.1,.5,.9,1.]),grasp_roi_radius_m=task['bear_grasp_tracking_radius_m'],grasp_roi_node_count=roi.sum(),maximum_bear_OBJ_to_native_node_error_m=bearerr.max()),native_rest_topology=topology,native_face_partition={k:len(v) for k,v in regions.items()},material_mapping=mapping,actual_material_readback=dict(status='pending_final_report' if material is None else 'available_not_audited_by_startup_check',verified_now=False,evidence=material),saved_input_sha256=inputs,limitations=['Startup consistency is not grasp, insertion, release, containment or task success.','Frozen manifest does not authenticate all installed Genesis/libuipc code and binaries; this review imports no Genesis and writes no production files.','External model hashes are measured now, not proved equal to process startup.','rho times rest volume is an independent calculation, not native mass readback.','Material coefficients and prior native validity do not calibrate real fabric or prove dynamics.','Future native observations and every-step force/deformation/rigid guards and final retained whole-teddy containment remain required.'])
    (RUN/'independent-input-manifest-review.json').write_text(json.dumps(manifest,indent=2)+'\n');out=RUN/'independent-startup-input-review.json';out.write_text(json.dumps(result,indent=2,default=serial,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(out),sha256=sha(out),passed=result['startup_input_checks_passed'],checks=len(checks),failed_checks=result['failed_checks'],native_points=len(rest['bear']),native_tets=len(rest['bear_tets']),native_material_status=result['actual_material_readback']['status']),indent=2))
    if not result['startup_input_checks_passed']:raise SystemExit(1)
if __name__=='__main__':main()
