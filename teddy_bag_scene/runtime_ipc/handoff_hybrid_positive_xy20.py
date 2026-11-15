"""Exactly one new authorized diagnostic offset; no production policy/config edit."""
from pathlib import Path
import copy,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_handoff_upright_offset_01';OUT=RUN/'independent-hybrid-postlift-route';sys.path.insert(0,str(RUN/'source'))
from contact_insertion import plan_guarded_contact_trial,_serial
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[OUT/'hybrid_hypothetical_observation.npz',OUT/'rest.npz',RUN/'config.json',RUN/'source/contact_insertion.py',RUN/'source/motion.py',Path(__file__)];before={str(p):sha(p) for p in paths}
cfg=json.loads((RUN/'config.json').read_text());old_offsets=copy.deepcopy(cfg['task']['contact_trial']['center_offsets_xy_m']);cfg['task']['contact_trial']['center_offsets_xy_m']=[[.020,.020]]
(OUT/'positive_xy20_diagnostic_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
cfg['robots']['mjcf_path']=str(RUN/'source/assets/panda_tactile.xml')
for key in ['npz_file','metadata']:cfg['bag'][key]=str(RUN/'source/assets'/cfg['bag'][key])
state=dict(np.load(OUT/'hybrid_hypothetical_observation.npz'));rest=dict(np.load(OUT/'rest.npz'))
t=time.monotonic();a=plan_guarded_contact_trial(rest,state,cfg);a.update(not_a_joint_native_snapshot=True,hybrid_provenance_path=str(OUT/'provenance.json'),diagnostic_offset_override=dict(old_declared_offsets=old_offsets,new_only_offset_m=[.020,.020],within_unchanged_35mm_radius=bool(np.linalg.norm([.02,.02])<=.035),production_config_modified=False),input_sha256=before,inputs_unchanged=all(sha(Path(p))==h for p,h in before.items()),elapsed_s=time.monotonic()-t)
p=OUT/'positive_xy20_guarded_candidate.json';p.write_text(json.dumps(_serial(a),indent=2,allow_nan=False)+'\n');print(json.dumps(dict(path=str(p),sha256=sha(p),status=a['status'],rejections=a['report'].get('rejections'),attempts=[{k:r[k] for k in ('rejection','stages_completed')} for r in a['report'].get('robot_centering_attempts',[])],elapsed_s=a['elapsed_s']),indent=2))
