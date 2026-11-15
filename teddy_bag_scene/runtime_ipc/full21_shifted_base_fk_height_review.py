"""Read archived candidate splines, checking the frozen bear height at FK samples."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_21'
OUT=RUN/'independent-shifted-right-base-high-route'
sys.path.insert(0,str(RUN/'source'))
from motion import Planner,MotionPlan
cfg=json.loads((RUN/'config.json').read_text());cfg['robots']['right_base']=[.28,.24,.76]
cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
p=Planner(cfg);state=dict(np.load(RUN/'orient_teddy_postlift.npz'))
review=json.loads((OUT/'review.json').read_text());rotation=Rotation.from_quat(review['new_base_same_pose_ik']['target_rotation_xyzw'])
relative=state['bear']-state['tcp_right'];cloth_max=float(state['bag'][:,2].max());rows=[]
for case in review['cases']:
    for short in ['high_y','high_x']:
        name=f"offset{case['index']}_{short}";path=OUT/(name+'_plan.npz');a=np.load(path)
        plan=MotionPlan({s:PPoly(a[s+'_c'],a[s+'_x']) for s in ['left','right']},float(a['duration']),{'left':0.,'right':0.},{'left':0.,'right':0.})
        minimum=float('inf');worst=None
        for time in np.linspace(0.,plan.duration,401):
            position,orient=p.fk('right',plan.sample(float(time))['q']['right'])
            bear=(orient*rotation.inv()).apply(relative)+position
            height=float(bear[:,2].min()-cloth_max)
            if height<minimum:minimum=height;worst=float(time)
        rows.append(dict(case_index=case['index'],stage=name,minimum_sampled_fk_frozen_bear_height_over_all_cloth_m=minimum,
            minimum_time_s=worst,sampled_10mm_height_pass=bool(minimum>=.010),sample_count=401,
            continuous_height_certified=False,plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
result=dict(kind='shifted_base_archived_fk_frozen_bear_high_height_review',physical_success=False,
    native_execution_performed=False,source_report_sha256=hashlib.sha256((OUT/'review.json').read_bytes()).hexdigest(),
    rows=rows,limits=['The original route uses exactly10mm nominal ideal Cartesian vertical allowance.',
        'Small FK translation/full-rotation errors reduce that allowance at some samples. These are not native deformation or tracking measurements.',
        'Consequently no candidate is certified to retain a strict10mm gap at all continuous FK times; do not present ideal-route acceptance as that stronger claim.',
        'A future implementation should reserve an explicit FK/tracking allowance above10mm and rerun all unchanged collision gates; this audit does not choose or test another height.'])
(OUT/'sampled-fk-bear-height-review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
