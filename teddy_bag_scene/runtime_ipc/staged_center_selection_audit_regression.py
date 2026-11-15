"""Synthetic adaptive-center evidence/tamper cases; no physical task success."""
from pathlib import Path
import copy, hashlib, json, sys, tempfile
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
# Preserve the existing 17 sequence and 11 explicit-budget contract cases.
import staged_budget_audit_regression as budget
import audit_guarded_run as audit

cfg=copy.deepcopy(budget.cfg); r=copy.deepcopy(budget.r)
states={k:v.copy() for k,v in budget.ss.items()}; trace=budget.tt.copy()
allowed=[[-.015,-.055],[-.02,-.05]]
cfg['task']['contact_trial']['staged']['center_offsets_xy_m']=copy.deepcopy(allowed)
# Build serialized policy independently from saved legacy policy, not audit output.
policy=copy.deepcopy(r['staged_contact_plans'][0]['checkpoint']['staged_policy'])
policy.update(center_offset_xy_m=[-.02,-.05],low_rim_geometry='ordered_vertical_projection',
    low_phase_section_gate='strict',center_offsets_xy_m=copy.deepcopy(allowed),
    center_selection_policy='fresh_actual_center_then_frozen_remainder_preflight',
    center_selection_score='minimum center/insert rigid surplus above 8mm and COM envelope surplus above 2mm',
    frozen_remainder_does_not_authorize_later_execution=True)
for plan in r['staged_contact_plans']:
    plan['checkpoint']['staged_policy']=copy.deepcopy(policy)
    plan['report']['staged_policy']=copy.deepcopy(policy)
    plan['checkpoint']['stage_target_tcp_world_m']=copy.deepcopy(plan['stage']['right_tcp_m'])
center=r['staged_contact_plans'][1]
center['selected_center_offset_xy_m']=copy.deepcopy(allowed[1])
center['report']['robot_trajectory']['minimum_sampled_rigid_clearance_m']=.020
center['report']['planned_com_boundary_distance_m']=.010
center['report'].pop('planned_com_contact_envelope_boundary_distance_m',None)
hypotime=center['observed_time_s']+np.ceil(center['stage']['duration_s']/cfg['simulation']['dt'])*cfg['simulation']['dt']
def remainder_report(rigid):
    return dict(physical_success=False,rejections=[],staged_policy=copy.deepcopy(policy),
        robot_trajectory={'minimum_sampled_rigid_clearance_m':rigid},
        release_withdraw_preflight={'minimum_release_opening_clearance_m':.022})
def attempt(index,rigid,com):
    return dict(index=index,center_offset_xy_m=copy.deepcopy(allowed[index]),center_status='candidate',
        center_rejections=[],physical_success=False,frozen_remainder_status='candidate',
        frozen_remainder_rejections=[],frozen_remainder_is_hypothesis_not_actual_observation=True,
        hypothetical_time_s=float(hypotime),score_m=min(rigid-.008,com-.002),
        minimum_center_insert_release_rigid_clearance_m=rigid,center_com_boundary_margin_m=com,
        frozen_remainder_report=remainder_report(rigid))
center['center_selection']=dict(kind='predeclared_finite_center_search',
    actual_observed_time_s=center['observed_time_s'],attempts=[attempt(0,.009,.006),attempt(1,.014,.010)],
    physical_success=False,later_insert_requires_fresh_actual_observation=True,
    selected_index=1,selected_offset_xy_m=copy.deepcopy(allowed[1]))
cases=[]; old_hand=audit.audit_staged_hand
audit.audit_staged_hand=lambda *args,**kw:dict(passed=True,status='synthetic_separately_tested_native_height')
with tempfile.TemporaryDirectory(prefix='staged-adaptive-audit-') as path:
    run=Path(path);(run/'source/assets').mkdir(parents=True)
    for rel in ['motion.py','assets/panda_tactile.xml']:
        (run/'source'/rel).write_bytes((budget.legacy.P/'source'/rel).read_bytes())
    (run/'contact_guard_trace.json').write_text(json.dumps([{'time_s':r['guarded_contact_candidate']['observed_time_s']}]))
    def case(name,config=None,report=None,ss=None,tt=None,expected=False):
        out=audit.audit_staged_sequence(run,r if report is None else report,
            states if ss is None else ss,trace if tt is None else tt,cfg if config is None else config,{'passed':True})
        ok=bool(out['passed'])==expected
        cases.append(dict(case=name,expected_audit_passed=expected,regression_passed=ok,
            audit_status=out['status'],failed_checks=[k for k,v in out['checks'].items() if not v],error=out.get('error')))
        print(name,ok,flush=True)
    case('declared_second_candidate_selected_and_executed',expected=True)
    # List shape, uniqueness and bounds are checked from the frozen config itself.
    invalid=[('empty',[]),('too_many',[[.001*i,-.01] for i in range(9)]),
        ('duplicate',[allowed[0],allowed[0]]),('nonfinite',[[float('nan'),.02]]),
        ('zero',[[0.,0.]]),('over_budget',[[.071,0.]]),('wrong_shape',[[.01,.02,.03]])]
    for name,offsets in invalid:
        bad=copy.deepcopy(cfg);bad['task']['contact_trial']['staged']['center_offsets_xy_m']=offsets
        case('invalid_frozen_list_'+name,config=bad)
    mutations={
        'selected_offset_not_in_list':lambda c:c.update(selected_center_offset_xy_m=[-.019,-.05]),
        'forged_selected_index':lambda c:c['center_selection'].update(selected_index=0),
        'boolean_index_is_not_integer':lambda c:c['center_selection'].update(selected_index=True),
        'negative_index':lambda c:c['center_selection'].update(selected_index=-1),
        'selection_offset_mismatch':lambda c:c['center_selection'].update(selected_offset_xy_m=allowed[0]),
        'missing_attempt':lambda c:c['center_selection']['attempts'].pop(),
        'forged_attempt_index':lambda c:c['center_selection']['attempts'][0].update(index=1),
        'undeclared_attempt_offset':lambda c:c['center_selection']['attempts'][0].update(center_offset_xy_m=[-.017,-.05]),
        'reordered_attempt_list':lambda c:c['center_selection']['attempts'].reverse(),
        'changed_checkpoint_policy_list':lambda c:c['checkpoint']['staged_policy'].update(center_offsets_xy_m=[allowed[1]]),
        'changed_report_rigid_policy':lambda c:c['report']['staged_policy'].update(rigid_clearance_m=.007),
        'changed_checkpoint_target':lambda c:c['checkpoint']['stage_target_tcp_world_m'].__setitem__(0,c['stage']['right_tcp_m'][0]+.003),
        'stale_selection_observation':lambda c:c['center_selection'].update(actual_observed_time_s=c['observed_time_s']-.005),
        'forged_score':lambda c:c['center_selection']['attempts'][1].update(score_m=.1),
        'frozen_remainder_rejected':lambda c:c['center_selection']['attempts'][1].update(frozen_remainder_status='rejected'),
        'frozen_remainder_physical_success_forged':lambda c:c['center_selection']['attempts'][1]['frozen_remainder_report'].update(physical_success=True),
        'frozen_remainder_policy_tamper':lambda c:c['center_selection']['attempts'][1]['frozen_remainder_report']['staged_policy'].update(maximum_original_rotation_error_rad=.02),
        'hypothetical_time_not_fixed_step_endpoint':lambda c:c['center_selection']['attempts'][1].update(hypothetical_time_s=hypotime-.002),
        'future_insert_authorized_by_selection':lambda c:c['center_selection'].update(later_insert_requires_fresh_actual_observation=False),
        'planning_labeled_physical_success':lambda c:c['center_selection'].update(physical_success=True),
        'nonfinite_selected_center_clearance':lambda c:c['report']['robot_trajectory'].update(minimum_sampled_rigid_clearance_m=float('inf')),
        'nonfinite_selected_com_margin':lambda c:c['report'].update(planned_com_boundary_distance_m=float('nan')),
        'executable_frozen_stage_smuggled_in_attempt':lambda c:c['center_selection']['attempts'][1].update(stage={'name':'insert_teddy'}),
        'contradictory_candidate_rejection':lambda c:c['center_selection']['attempts'][1].update(center_rejections=['Clearance failed']),
        'malformed_attempt_record':lambda c:c['center_selection']['attempts'].__setitem__(0,None),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(r);mutate(bad['staged_contact_plans'][1]);case(name,report=bad)
    bad=copy.deepcopy(r)
    bad['staged_contact_plans'][2]['checkpoint']['staged_policy']['center_offsets_xy_m'].reverse()
    case('insert_checkpoint_changes_prior_declared_policy',report=bad)
    badcfg=copy.deepcopy(cfg);badcfg['task']['contact_trial']['staged']['center_offsets_xy_m'].reverse()
    case('frozen_config_reordered_but_records_unchanged',config=badcfg)
    bad=copy.deepcopy(r);a=bad['staged_contact_plans'][1]['center_selection']['attempts'][0]
    a.update(minimum_center_insert_release_rigid_clearance_m=.02,center_com_boundary_margin_m=.02,score_m=.012)
    a['frozen_remainder_report']['robot_trajectory']['minimum_sampled_rigid_clearance_m']=.02
    case('lower_ranked_candidate_selected',report=bad)
    bad=copy.deepcopy(r);a=bad['staged_contact_plans'][1]['center_selection']['attempts'][0]
    a.update(minimum_center_insert_release_rigid_clearance_m=.014,center_com_boundary_margin_m=.010,score_m=min(.014-.008,.010-.002))
    a['frozen_remainder_report']['robot_trajectory']['minimum_sampled_rigid_clearance_m']=.014
    case('tie_must_select_first_declared_index',report=bad)
    # Keep the scalar distance and budget intact while rotating actual XY displacement.
    badstates={k:v.copy() for k,v in states.items()};actual_start=badstates['tcp_right'][1].copy()
    angle=.1;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    badstates['tcp_right'][2,:2]=actual_start[:2]+rot@np.asarray(allowed[1])
    case('actual_executed_offset_wrong_direction_same_norm',ss=badstates)
    # Config and plan targets agree on a different allowed offset, but recorded selection does not.
    bad=copy.deepcopy(r);bad['staged_contact_plans'][1]['stage']['right_tcp_m'][:2]=(actual_start[:2]+allowed[0]).tolist()
    case('actual_target_uses_other_declared_candidate',report=bad)
    # A valid rejected alternative remains a planning failure, never a global physics failure.
    bad=copy.deepcopy(r);a=bad['staged_contact_plans'][1]['center_selection']['attempts'][0]
    a.clear();a.update(index=0,center_offset_xy_m=allowed[0],center_status='rejected',
        center_rejections=['COM margin insufficient'],physical_success=False)
    case('rejected_alternative_and_selected_valid_candidate',report=bad,expected=True)
    bad=copy.deepcopy(r);bad['staged_contact_plans'][1]['center_selection']['attempts'][0]['frozen_remainder_status']='rejected'
    case('rejected_frozen_alternative_and_selected_valid_candidate',report=bad,expected=True)
    bad=copy.deepcopy(r);a=bad['staged_contact_plans'][1]['center_selection']['attempts'][0]
    for key in ['score_m','minimum_center_insert_release_rigid_clearance_m','center_com_boundary_margin_m','frozen_remainder_report']:
        a.pop(key)
    a['frozen_remainder_rejections']=['Nonfinite raw center/insert/release clearance']
    case('wrapper_margin_rejected_alternative_is_not_physical_failure',report=bad,expected=True)
    # Runtime supports an omitted legacy key while freezing its default in policy.
    badcfg=copy.deepcopy(cfg);badcfg['task']['contact_trial']['staged'].pop('center_offset_xy_m')
    bad=copy.deepcopy(r)
    for p in bad['staged_contact_plans']:
        for container in [p['checkpoint'],p['report']]:
            container['staged_policy']['center_offset_xy_m']=[-.027,-.010]
    for a in bad['staged_contact_plans'][1]['center_selection']['attempts']:
        a['frozen_remainder_report']['staged_policy']['center_offset_xy_m']=[-.027,-.010]
    case('list_only_config_uses_frozen_default_fallback',config=badcfg,report=bad,expected=True)
    # Exercise both inclusive cardinality limits through the full sequence auditor.
    for count in [1,8]:
        badcfg=copy.deepcopy(cfg);bad=copy.deepcopy(r)
        offsets=[allowed[1]]+[[.001*i,-.01] for i in range(1,count)]
        badcfg['task']['contact_trial']['staged']['center_offsets_xy_m']=offsets
        for p in bad['staged_contact_plans']:
            for container in [p['checkpoint'],p['report']]:
                container['staged_policy']['center_offsets_xy_m']=copy.deepcopy(offsets)
        sel=bad['staged_contact_plans'][1]['center_selection'];chosen=copy.deepcopy(sel['attempts'][1])
        chosen.update(index=0,center_offset_xy_m=offsets[0])
        chosen['frozen_remainder_report']['staged_policy']['center_offsets_xy_m']=copy.deepcopy(offsets)
        sel.update(selected_index=0,attempts=[chosen]+[dict(index=i,center_offset_xy_m=offsets[i],
            center_status='rejected',center_rejections=['Synthetic geometry rejection'],physical_success=False) for i in range(1,count)])
        case('valid_'+str(count)+'_candidate_boundary',config=badcfg,report=bad,expected=True)
audit.audit_staged_hand=old_hand
out=dict(kind='synthetic_staged_adaptive_center_audit_regression',physical_success=False,
    all_checks_passed=all(x['regression_passed'] for x in cases),case_count=len(cases),cases=cases,
    legacy_sequence_and_budget_regressions_passed=budget.out['all_checks_passed'] and budget.out['existing_sequence_regression_all_passed'],
    auditor_sha256=hashlib.sha256((HERE/'audit_guarded_run.py').read_bytes()).hexdigest(),
    scope='Synthetic saved-evidence tests. Native hand and force auditors are separately tested and mocked here; no physical success or route safety is inferred.')
(HERE/'staged-center-selection-audit-regression.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2));assert out['all_checks_passed'] and out['legacy_sequence_and_budget_regressions_passed']
