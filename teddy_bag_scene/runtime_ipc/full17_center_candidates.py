"""Actual full17 endpoint, offline replacement centers; never executes physics."""
from pathlib import Path
import sys, json, copy, hashlib, argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/two_panel_full_17'
sys.path.insert(0,str(RUN/'source'))
from staged_contact_insertion import plan_staged_contact
from motion import Planner

rest=dict(np.load(RUN/'rest.npz'))
transport=dict(np.load(RUN/'transport_teddy.npz'))
actual=dict(np.load(RUN/'lower_teddy_to_rim.npz'))
base=json.loads((RUN/'config.json').read_text())
prep=json.loads((RUN/'guarded_contact_candidate.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('--centers', default='-.02,-.06;-.025,-.06;-.03,-.06;-.015,-.06;-.02,-.065;-.025,-.065;-.02,-.05')
parser.add_argument('--output',default='replacement_center_candidates')
args=parser.parse_args()
results=[]
out=RUN/args.output
out.mkdir(exist_ok=True)
for dx,dy in [tuple(map(float,entry.split(','))) for entry in args.centers.split(';')]:
    cfg=copy.deepcopy(base)
    cfg['task']['contact_trial']['staged']['center_offset_xy_m']=[dx,dy]
    name=f'x{round(dx*1000):d}_y{round(dy*1000):d}'
    entry=plan_staged_contact(rest,transport,cfg,phase='entry',previous=prep,output=out/f'{name}_entry.json')
    row=dict(center_delta_xy_m=[dx,dy],entry_status=entry['status'],physical_success=False,
             actual_source_time_s=float(actual['time']),center_is_planning_only=True,
             after_center_is_frozen_hypothesis=True)
    if entry['status']=='candidate':
        center=plan_staged_contact(rest,actual,cfg,phase='center',previous=entry,output=out/f'{name}_center.json')
        row.update(center_status=center['status'],center_rejections=center['report']['rejections'])
        if center['status']=='candidate':
            hypothetical={k:v.copy() for k,v in actual.items()}
            stage=center['stage']; delta=np.array(stage['right_tcp_m'])-actual['tcp_right']
            hypothetical['bear']=actual['bear']+delta
            hypothetical['tcp_right']=np.array(stage['right_tcp_m'])
            hypothetical['q_right'][:7]=np.array(stage['right_q_end'])
            hypothetical['time']=np.asarray(float(actual['time'])+stage['duration_s'])
            insert=plan_staged_contact(rest,hypothetical,cfg,phase='insert',previous=center,output=out/f'{name}_insert_hypothesis.json')
            row.update(insert_status=insert['status'],insert_rejections=insert['report']['rejections'])
            for phase,record in [('center',center),('insert',insert)]:
                report=record['report']
                row[phase+'_robot_trajectory']=report.get('robot_trajectory')
                row[phase+'_bear_left_sweep']=report.get('continuous_frozen_teddy_left_robot_sweep')
                row[phase+'_com_margin_m']=report.get('planned_com_contact_envelope_boundary_distance_m')
            row['release_withdraw']=insert['report'].get('release_withdraw_preflight')
            row['all_frozen_gates_pass']=insert['status']=='candidate'
    results.append(row)
    record=dict(kind='full17_actual_entry_replacement_center_search',physical_success=False,
        runtime_execution_performed=False,results=results,
        scope='Actual full17 transport and entry states. Entry is replanned with each predeclared candidate policy; no runtime policy mutation or resumed physics. Later center endpoint is a rigidly translated hypothetical bear with frozen cloth, not native deformation. Original full R and all original physical and clearance limits retained. Every real next stage must reobserve actual geometry.',
        source_sha256={n:hashlib.sha256((RUN/n).read_bytes()).hexdigest() for n in ['rest.npz','transport_teddy.npz','lower_teddy_to_rim.npz','config.json','source/staged_contact_insertion.py']})
    (out/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print(name, row.get('entry_status'),row.get('center_status'),row.get('center_rejections'),row.get('insert_status'),row.get('insert_rejections'),flush=True)
