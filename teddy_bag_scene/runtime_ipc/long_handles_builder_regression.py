"""Bounded CPU regression: default exact assets, new body identity, invalid option."""
from pathlib import Path
import hashlib,json,sys,tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import two_panel_builder as b
results=[]
def check(name,value):
 results.append({'name':name,'passed':bool(value)})
 if not value:raise AssertionError(name)
assets=b.ASSETS;b.W,b.H=.46,.40
for i,bad in enumerate([0.,-.1,float('nan'),float('inf'),-float('inf'),True,False,None,'.04',[],np.array(.04)]):
 try:b.build_bag(.02,'must_not_write',None,None,handle_extra=bad)
 except ValueError:check('invalid_option_'+str(i)+'_structured_ValueError',True)
 else:check('invalid_option_'+str(i)+'_structured_ValueError',False)
check('invalid_options_no_asset_written',not (assets/'must_not_write.npz').exists())
point,geometry=b.opened_panel_geometry();old=dict(np.load(assets/'two_panel_bag_wide_tall_coarse.npz',allow_pickle=False))
with tempfile.TemporaryDirectory(prefix='long-handle-default-',dir=ROOT/'runtime_ipc') as temp:
 b.ASSETS=Path(temp)
 default=b.build_bag(.02,'default',point,geometry)
 explicit=b.build_bag(.02,'explicit',point,geometry,handle_extra=.04)
 data=dict(np.load(Path(temp)/'default.npz',allow_pickle=False));other=dict(np.load(Path(temp)/'explicit.npz',allow_pickle=False))
 check('default_all_npz_array_keys_unchanged',set(data)==set(old))
 check('default_every_existing_npz_array_bit_identical',all(np.array_equal(data[k],old[k]) for k in old))
 check('explicit_default_equal_implicit_default',all(np.array_equal(data[k],other[k]) for k in data))
 for suffix in ['.obj','_collision.obj']:
  # Filenames in textured OBJ headers legitimately differ; compare v/vt/f only.
  physical=lambda p:[s for s in p.read_text().splitlines() if s.startswith(('v ','vt ','f '))]
  check('default_OBJ_records_unchanged_'+suffix,physical(Path(temp)/('default'+suffix))==physical(assets/('two_panel_bag_wide_tall_coarse'+suffix)))
 b.ASSETS=assets
new=dict(np.load(assets/'two_panel_bag_wide_tall_long_handles_coarse.npz',allow_pickle=False));meta=json.loads((assets/'metadata_two_panel_wide_tall_long_handles_coarse.json').read_text())['bag'];oldmeta=json.loads((assets/'metadata_two_panel_wide_tall_coarse.json').read_text())['bag']
check('new_body_points_bit_exact',np.array_equal(new['points'][new['body']],old['points'][old['body']]))
check('new_body_faces_bit_exact',np.array_equal(new['faces'][new['body_face_indices']],old['faces'][old['body_face_indices']]))
check('new_U_seam_bit_exact',np.array_equal(new['u_seam'],old['u_seam']))
check('new_body_opening_and_manifold',meta['body_mesh']['boundary_loops']==1 and meta['mesh']['boundary_loops']==3 and meta['mesh']['nonmanifold_edges']==0)
check('added_area_positive',meta['mesh']['triangle_area_m2']>oldmeta['mesh']['triangle_area_m2'])
check('metadata_mass_tracks_area_not_old_mass',abs(meta['estimated_mass_kg']-meta['mesh']['triangle_area_m2']*.24)<1e-14 and meta['estimated_mass_kg']>oldmeta['estimated_mass_kg'])
# Same top-arc cloth samples translated +14cm, with identical geometry and grasp patches.
for side in ['front','back']:
 for label in ['center','x_minus','x_plus']:
  key='grasp_'+side+'_'+label
  op=old['points'][old[key]];np_=new['points'][new[key]]
  check('same_arc_grasp_'+side+'_'+label,np_.shape==op.shape and np.allclose(np_-op,[0.,0.,.14],rtol=0,atol=2e-15))
 cfg=json.loads((ROOT/'config_long_handles_gravity_trial.json').read_text());base=json.loads((ROOT/'config_gravity_seating_trial.json').read_text());mut=json.loads(json.dumps(cfg));mut['bag']['mesh']=base['bag']['mesh'];mut['bag']['npz_file']=base['bag']['npz_file'];mut['bag']['metadata']=base['bag']['metadata'];mut['task']['open_targets']=base['task']['open_targets'];mut['task']['contact_trial']['postlift_yaw']=base['task']['contact_trial']['postlift_yaw'];check('only_authorized_config_changes_'+side,mut==base)
validation=json.loads((ROOT/'runtime_ipc/long-handles-cpu-validation.json').read_text());mass=validation['cloth_mass'];check('IPC_radius_mass_convention',mass['physical_full_thickness_m']==2*cfg['bag']['thickness'] and abs(mass['new_area_times_rho_full_thickness_kg']-meta['mesh']['triangle_area_m2']*cfg['bag']['rho']*2*cfg['bag']['thickness'])<1e-14)
report={'status':'passed','physical_success':False,'checks':results,'passed':sum(x['passed'] for x in results),'total':len(results),'builder_sha256':hashlib.sha256((ROOT/'two_panel_builder.py').read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256((ROOT/'build_long_handle_bag.py').read_bytes()).hexdigest(),'scope':'CPU asset generation and input contract only; no native dynamics or cloth robot clearance proof.'}
path=ROOT/'runtime_ipc/long-handles-builder-regression.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':report['passed'],'total':report['total'],'path':str(path)},indent=2))
