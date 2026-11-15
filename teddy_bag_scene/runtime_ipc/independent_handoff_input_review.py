"""Bounded independent geometry-tool and current handoff input review; no simulation."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/'runs/two_panel_full_21'
RUN=ROOT/'runs/two_panel_handoff_upright_offset_01'
sys.path[:0]=[str(FULL/'source'),str(ROOT/'runtime_ipc')]
from validate import signed_volumes
from audit_input_manifest import audit

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def serialize(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    raise TypeError(type(v).__name__)
def save(p,d):p.write_text(json.dumps(d,indent=2,default=serialize)+'\n');return sha(p)
def diff(a,b,p=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((diff(a.get(k,'<missing>'),b.get(k,'<missing>'),(p+'.'+k).strip('.')) for k in sorted(set(a)|set(b))),[])
    return [] if a==b else [dict(path=p,before=a,after=b)]

cfg=read(FULL/'config.json');rest=dict(np.load(FULL/'rest.npz'));initial=dict(np.load(FULL/'initial.npz'))
source_path=FULL/'source/assets'/cfg['bag']['npz_file'];source=dict(np.load(source_path))
errors,mapping=cKDTree(rest['bag']).query(source['points']+cfg['bag']['position'])
geometry=read(FULL/'handoff-material-geometry.json')
checks=dict(source_hash_matches=sha(source_path)==geometry['source_sha256'],
    complete_vertex_bijection=len(source['points'])==len(rest['bag'])==len(np.unique(mapping)) and errors.max()<=1e-7,
    source_faces_exactly_match_native_faces=np.array_equal(mapping[source['faces']],rest['bag_faces']),
    left_seam_matches_front_grid_material_column=np.array_equal(source['seam_left'],source['panel_front_grid'][:,0]),
    right_seam_matches_front_grid_material_column=np.array_equal(source['seam_right'],source['panel_front_grid'][:,-1]),
    both_panels_share_actual_seam_columns=np.array_equal(source['panel_front_grid'][:,[0,-1]],source['panel_back_grid'][:,[0,-1]]),
    seam_rows_match_ordered_material_height=len(source['seam_left'])==len(source['seam_right'])==len(source['material_height']) and np.all(np.diff(source['material_height'])>0),
    initial_rest_substitution_exact_for_full21=np.array_equal(initial['bag'],rest['bag']) and float(initial['time'])==0.,
    live_validate_matches_full21_frozen=sha(ROOT/'validate.py')==sha(FULL/'source/validate.py'))
phase_rows=[]
for row in geometry['phases']:
    phase=row['phase'];state=initial if phase=='initial' else dict(np.load(FULL/(phase+'.npz')))
    bag=state['bag'];matches=[]
    for key,index in [('bottom',0),('middle',len(source['seam_left'])//2),('top',-1)]:
        a=bag[mapping[source['seam_left'][index]]];b=bag[mapping[source['seam_right'][index]]];v=b-a;line=row['material_width_lines'][key]
        matches.append(np.array_equal(a,line['from_m']) and np.array_equal(b,line['to_m']) and np.array_equal(v,line['vector_m'])
            and np.isclose(np.linalg.norm(v),line['length_m'],rtol=0,atol=1e-14)
            and np.isclose(np.degrees(np.arctan2(v[1],v[0])),line['xy_yaw_deg'],rtol=0,atol=1e-12))
    for panel in ('front','back'):
        a=bag[mapping[source['handle_'+panel+'_root_left']]].mean(0)
        b=bag[mapping[source['handle_'+panel+'_root_right']]].mean(0)
        matches.append(np.array_equal(a,row['handle_roots'][panel]['from_m']) and np.array_equal(b,row['handle_roots'][panel]['to_m']))
    rim=bag[mapping[source['rim']]]
    matches.append(np.allclose(np.ptp(rim,axis=0),row['rim_span_xyz_m'],atol=1e-12,rtol=0))
    if 'left_named_material_distance_m' in row:
        distance=np.linalg.norm(bag[mapping[source[cfg['task']['handle_grasps']['left']['group']]]]-state['tcp_left'],axis=1)
        matches.append(abs(distance.min()-row['left_named_material_distance_m']['min'])<1e-12 and abs(distance.max()-row['left_named_material_distance_m']['max'])<1e-12)
    phase_rows.append(dict(phase=phase,time_s=float(state['time']),reported_clock_exact=float(state['time'])==row['time_s'],material_lines_and_groups_exact=bool(all(matches)),top_material_chord_m=row['material_width_lines']['top']['length_m'],top_material_direction_m=row['material_width_lines']['top']['vector_m']))
checks['every_reported_phase_uses_its_actual_saved_endpoint']=all(r['reported_clock_exact'] and r['material_lines_and_groups_exact'] for r in phase_rows)
tree=ast.parse((ROOT/'runtime_ipc/analyze_handoff_geometry.py').read_text());width_node=next(n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='material_width_lines')
checks['material_width_expression_uses_seam_landmarks_without_PCA']=not({'basis','projected','singular','centered'}&{n.id for n in ast.walk(width_node.value) if isinstance(n,ast.Name)})
tool_report=dict(kind='independent_handoff_material_geometry_analyzer_review',physical_success=False,source_or_native_state_modified=False,
    current_full21_result_consistent=all(checks.values()),checks=checks,phases=phase_rows,
    definition='The reported material width line is the current 3D chord joining corresponding left/right sewn material landmarks at equal reference material height. It is neither fabric arc length nor free aperture width. PCA quantities remain separate projection diagnostics.',
    generic_reporting_and_validation_gaps=[
        'It imports live validate and records only source-asset hash, so a standalone report does not fully freeze helper/state/config provenance.',
        'Zero XY chord length would make yaw undefined even though atan2(0,0) returns zero; current full21 reported chords have nonzero XY projection.'],
    current_full21_incorrect_material_axis_or_phase_value_found=False,
    input_sha256={str(p):sha(p) for p in [ROOT/'runtime_ipc/analyze_handoff_geometry.py',FULL/'handoff-material-geometry.json',FULL/'config.json',FULL/'rest.npz',FULL/'initial.npz',source_path,ROOT/'validate.py']})
tool_out=ROOT/'runtime_ipc/independent-handoff-material-geometry-review.json';tool_hash=save(tool_out,tool_report)

cfg=read(RUN/'config.json');manifest=read(RUN/'input_sha256.json');launch=read(ROOT/'runtime_ipc/two_panel_handoff_upright_offset_01-launch.json')
input_review=audit(RUN);save(RUN/'independent-input-manifest-review.json',input_review)
rest=dict(np.load(RUN/'rest.npz'));source_path=RUN/'source/assets'/cfg['bag']['npz_file'];source=dict(np.load(source_path))
errors,mapping=cKDTree(rest['bag']).query(source['points']+cfg['bag']['position'])
vol=signed_volumes(rest['bear'],rest['bear_tets']);actual_rows=[]
for phase in ('initial','close_both_handles','drape_under_gravity'):
    path=RUN/(phase+'.npz')
    if not path.exists():continue
    state=dict(np.load(path));sides={}
    for side in ('left','right'):
        group=cfg['task']['handle_grasps'][side]['group'];ids=mapping[source[group]]
        dist=np.linalg.norm(state['bag'][ids]-state['tcp_'+side],axis=1)
        sides[side]=dict(named_material_group=group,named_material_count=len(ids),native_indices=ids,
            tcp_m=state['tcp_'+side],nearest_material_distance_m=float(dist.min()),farthest_material_distance_m=float(dist.max()),
            material_centroid_m=state['bag'][ids].mean(0),actual_finger_positions_m=state['q_'+side][7:],
            saved_finger_PD_control_force_n=state['force_'+side][7:])
    actual_rows.append(dict(phase=phase,time_s=float(state['time']),all_values_finite=all(np.isfinite(v).all() for v in state.values()),
        sides=sides,minimum_J=float(np.min(signed_volumes(state['bear'],rest['bear_tets'])/vol)),sha256=sha(path)))
pid=launch['pid'];proc=Path('/proc')/str(pid)
command=(proc/'cmdline').read_bytes().split(b'\0') if proc.exists() else []
command=[x.decode() for x in command if x];cwd=os.readlink(proc/'cwd') if proc.exists() else None
checks=dict(frozen_manifest_all_entries_valid=input_review['all_declared_hashes_match'] and input_review['all_snapshot_files_in_manifest'],
    supplied_config_sha_matches_launch=sha(ROOT/launch['config'])==launch['config_sha256']=='7e7107757322ebfdb0431de2561262ef05cd44643568c690b1a3ea378974047c',
    frozen_config_semantics_match_supplied=cfg==read(ROOT/launch['config']),
    all_live_source_and_assets_match_frozen=all(sha(ROOT/k.removeprefix('source/'))==v for k,v in manifest.items() if k.startswith('source/')),
    continuous_rate_fix_frozen=manifest['source/motion.py']=='544d6bed1aedfe1603db860af0cd4d230641714827ff23ef27c93d0a12e32a31',
    handoff_scope_runner_frozen=manifest['source/run.py']=='0ed42957296a0ad41b4691856042a77ccef9ce3472a0b77f0d78d653f09c38f1',
    actual_process_uses_handoff_task='--task' in command and command[command.index('--task')+1]=='handoff' and '--config' in command and Path(command[command.index('--config')+1]).resolve()==(ROOT/launch['config']).resolve(),
    actual_process_cwd_matches_scene=cwd==str(ROOT),
    material_mapping_complete=len(mapping)==len(rest['bag'])==len(np.unique(mapping)) and errors.max()<=1e-7,
    mapped_full_native_faces_match=np.array_equal(mapping[source['faces']],rest['bag_faces']),
    same_bag_rest_and_topology_as_full21=np.array_equal(rest['bag'],np.load(FULL/'rest.npz')['bag']) and np.array_equal(rest['bag_faces'],np.load(FULL/'rest.npz')['bag_faces']),
    all_saved_initial_tetrahedra_positive=bool(np.isfinite(vol).all() and np.all(vol>0)),
    bag_effective_50_250_unchanged=cfg['bag']['shear_stiffness']==50 and cfg['bag']['handle_shear_stiffness']==250,
    unmodified_material_simulation_and_robot_configs=all(cfg[k]==read(FULL/'config.json')[k] for k in ('bag','bear','simulation','robots','table')),
    initial_phase_data_finite=all(r['all_values_finite'] for r in actual_rows))
summary=dict(kind='independent_handoff_startup_input_and_actual_prefix_review',physical_task_success=False,handoff_probe_success=False,
    status='startup_consistent_full_native_probe_pending',available_checks_passed=all(checks.values()),checks=checks,
    failed_checks=[k for k,v in checks.items() if not v],process=dict(pid=pid,command=command,cwd=cwd),
    config_semantic_diff_from_full21=diff(read(FULL/'config.json'),cfg),
    actual_saved_phase_measurements=actual_rows,
    native_material_readback_status='pending final report; configuration/source coefficients verified only',
    native_grip_evidence_scope='Actual named material proximity, finger q and saved PD control forces at saved endpoints. Complete native soft opening reaction trace remains pending; these values alone do not prove force closure.',
    final_probe_audit_required=['All handoff-scope validation checks, including discrete final bag-bear penetration, must pass.',
        'Both handles during spread and left named material/negative clamping after right release, with required trace coverage.',
        'Every-step finite/J/stretch/table and actual joint traces through the 10-second hold.',
        'Actual release, right_clear_handle, settle_handoff endpoints and 10-second timing.',
        'Initialization 50/250 native material attribute readback and unchanged protected attributes.',
        'No claim of teddy lift/insertion/release/containment or full requested task success.'],
    no_contact_guard_trace_expected_before_teddy_contact_trial=True,
    input_sha256={str(RUN/k):v for k,v in manifest.items()},
    limitations=input_review['limitations'],geometry_tool_review=dict(path=str(tool_out),sha256=tool_hash))
out=RUN/'independent-startup-input-review.json';digest=save(out,summary)
print(json.dumps(dict(geometry_tool=dict(path=str(tool_out),sha256=tool_hash,passed=tool_report['current_full21_result_consistent']),
    startup=dict(path=str(out),sha256=digest,passed=summary['available_checks_passed'],failed=summary['failed_checks'],phases=[r['phase'] for r in actual_rows])),indent=2))
