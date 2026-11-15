"""CPU-only counterfactual policy review of full22's actual transport state.

Never resumes simulation or overwrites historical selected/config/evidence.
Invoke only after the production helper's owner signals a tested freeze.
"""
from pathlib import Path
import argparse,copy,hashlib,json,time,sys
import numpy as np
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import contact_insertion as planner
import continuous_frozen_handles_review as handles

P=BASE/'runs/two_panel_full_22'
D=P/'independent-counterfactual-deformable-rim'
CONFIG=BASE/'config_guarded_deformable_rim_trial.json'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def slim_sections(x):return {k:v for k,v in x.items() if k!='sections'}
def main(expected_helper_sha):
    helper=BASE/'contact_insertion.py'
    assert digest(helper)==expected_helper_sha,'Helper freeze mismatch'
    D.mkdir(exist_ok=True)
    paths=[helper,CONFIG,P/'config.json',P/'rest.npz',P/'transport_teddy.npz',P/'guarded_contact_candidate.json',P/'guarded_contact_reobserved.json',BASE/'run.py',BASE/'motion.py',BASE/'contact_guard.py',BASE/'physical_monitor.py',Path(__file__),Path(handles.__file__)]
    hashes={str(p):digest(p) for p in paths}
    cfg=json.loads(CONFIG.read_text());oldcfg=json.loads((P/'config.json').read_text());selected=json.loads((P/'guarded_contact_candidate.json').read_text())
    state=dict(np.load(P/'transport_teddy.npz'));rest=dict(np.load(P/'rest.npz'))
    def differences(a,b,prefix=''):
        result=[]
        if isinstance(a,dict) and isinstance(b,dict):
            for k in sorted(set(a)|set(b)):
                key=f'{prefix}.{k}' if prefix else k
                if k not in a:result.append({'path':key,'old':'<absent>','new':b[k]})
                elif k not in b:result.append({'path':key,'old':a[k],'new':'<absent>'})
                else:result.extend(differences(a[k],b[k],key))
        elif a!=b:result.append({'path':prefix,'old':a,'new':b})
        return result
    cfgdiff=differences(oldcfg,cfg)
    newpolicy=planner._policy(cfg);oldpolicy=selected['report']['policy']
    # The selected record remains historical orientation/target provenance.
    # Its only altered original field is policy metadata, explicitly derived.
    derived=copy.deepcopy(selected);derived['report']['policy']=newpolicy
    assert differences(selected,derived)==differences({'report':{'policy':oldpolicy}},{'report':{'policy':newpolicy}})
    derivative_diff=differences(selected,derived)
    derived['offline_counterfactual_derivation']={'source_file':str(P/'guarded_contact_candidate.json'),'source_sha256':hashes[str(P/'guarded_contact_candidate.json')],
        'changes_to_original_fields':derivative_diff,'purpose':'Reuse historical original fullR and actual observation time as reference, with reviewed opt-in policy metadata. Historical route was not executed under this new policy.',
        'not_historical_runtime_evidence':True}
    derived_path=D/'derived-selected-policy-metadata.json';derived_path.write_text(json.dumps(derived,indent=2)+'\n')
    outpath=D/'counterfactual-reobserved-candidate.json';started=time.monotonic()
    x=planner.reobserve_guarded_contact_trial(rest,state,cfg,derived,output=outpath)
    summary={'kind':'counterfactual_opt_in_policy_reobserve_of_actual_full22_transport','source_observation_scope':'one actual joint native observation at40.735s; new-policy route is offline counterfactual',
        'not_full22_runtime_policy_evidence':True,'native_execution':False,'production_edited':False,'physical_success':False,
        'source_time_s':float(state['time']),'input_sha256':hashes,'config_differences_vs_original_full22':cfgdiff,
        'derived_selected_changes':derivative_diff,'original_selected_preserved':digest(P/'guarded_contact_candidate.json')==hashes[str(P/'guarded_contact_candidate.json')],
        'status':x['status'],'rejections':x['report']['rejections'],'elapsed_planner_s':time.monotonic()-started,
        'new_policy':newpolicy,'original_fullR_fresh_error_rad':x['report'].get('fresh_rotation_error_rad'),
        'centering_candidates':x['report'].get('centering_candidates'),
        'robot_attempts':x['report'].get('robot_centering_attempts'),
        'robot_stages':x['report'].get('robot_stages'),'withdraw_after_release_preflight':x['report'].get('withdraw_after_release_preflight'),
        'selected_sections':slim_sections(x['report'].get('selected_sections',{}))}
    if x['status']=='candidate':
        summary['route_targets']={k:x['route'][k] for k in ['outside_tcp_m','insert_tcp_m','rotation_xyzw']}
        summary['handle_reviews']={}
        a=argparse.Namespace(rest=str(P/'rest.npz'),state=str(P/'transport_teddy.npz'),stages=str(outpath),config=str(CONFIG),source_bag=str(P/'source/assets'/cfg['bag']['npz_file']),clearance=.002,scope='actual_joint_observation',handles='front',descent_prefix_to_rim=False,output='')
        for label,which,prefix in [('front-entire-descent','front',False),('both-pre-rim','both',True),('both-entire-descent','both',False)]:
            a.handles=which;a.descent_prefix_to_rim=prefix;a.output=str(D/(label+'-continuous.json'));handles.run(a)
            h=json.loads(Path(a.output).read_text());summary['handle_reviews'][label]={'report':a.output,'status':h['status'],'scope':h['source_scope'],'boundary':h['descent_prefix_boundary'],
               'stages':[{'name':s['name'],'status':s['status'],'minimum_gap_lower_m':s['minimum_complete_gap_lower_m'],'complete_pair_coverage':s['complete_pair_coverage'],'first_failure':s.get('first_failure')} for s in h['stages']]}
    summary['protected_unchanged_at_completion']=all(digest(k)==v for k,v in hashes.items())
    summary['limitations']=['No native simulation or corrected observation. Full22 actually rejected this state using its original fixed-union policy.',
       'Counterfactual candidate is permission for assessment only, not evidence of material passage, dynamic stability or containment.',
       'A negative fixed-mouth union margin remains diagnostic evidence; only the explicitly opted-in perimeter planning policy differs.',
       'All other original geometric, robot, motion, grasp, deformation, force and torque gates must remain active during any future native attempt.',
       'Frozen handle certificates do not predict shape changes, slip, future contact or joint-FK tracking between samples.']
    (D/'review.json').write_text(json.dumps(summary,indent=2,default=handles.serial,allow_nan=False)+'\n')
    print(json.dumps({'status':summary['status'],'rejections':summary['rejections'],'protected_unchanged':summary['protected_unchanged_at_completion'],'report':str(D/'review.json')},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--expected-helper-sha',required=True);args=a.parse_args();main(args.expected_helper_sha)
