"""Independently read completed short native coupon pair; do not run solver."""
from pathlib import Path
import json,hashlib,numpy as np
P=Path(__file__).resolve().parents[1];base=P/'runtime_ipc/cotton-linen-coupon-probe-01'
r=json.loads((base/'report.json').read_text());assert r['status']=='completed'
summary={'scope':'Independent numeric reduction of completed native 0.5N diagonal-load coupons; not material calibration or bag-task success','parent_report_sha256':hashlib.sha256((base/'report.json').read_bytes()).hexdigest(),'checks':{},'cases':[],'limitations':r['limitations'],'physical_success':False}
reference=None
for name,E,mu in [('baseline',4e6,50.),('candidate',2e7,1000.)]:
 d=np.load(base/name/'states.npz');a=d['metrics'];cb=d['callbacks'];rest=d['rest'];f=d['faces'];x=d['positions'];v=d['velocities'];
 checks={
 'exact_401_states':len(a)==401 and x.shape==(401,231,3),
 'exact_400_callback_frames':np.array_equal(cb[:,0],np.arange(1,401)),
 'same_times':np.allclose(a[:,0],np.arange(401)*.005,rtol=0,atol=1e-12),
 'all_arrays_finite':all(np.isfinite(d[k]).all() for k in d.files),
 'clamp_fixed':np.max(np.linalg.norm(x[:,d['fixed_ids']]-rest[d['fixed_ids']],axis=-1))<=1e-10,
 'mu':np.allclose(d['native_mu'],mu,rtol=0,atol=1e-12),
 'lambda':np.allclose(d['native_lambda'],E*.0006/(1-.3**2),rtol=1e-13,atol=0),
 'rate':np.all(d['native_strain_rate']==100),
 'bending':np.all(d['native_bending_stiffness']==5e-5),
 'radius':np.all(d['native_thickness']==.0003),
 'mass_6g':abs(d['native_volume'].sum()*500-.006)<1e-12,
 'force_only_41_to160':np.array_equal(cb[:,2],((cb[:,0]>=41)&(cb[:,0]<=160)).astype(float)),
 'force_total_and_direction':np.max(np.abs(cb[:,3:6]-cb[:,2,None]*np.array([.5/np.sqrt(2),.5/np.sqrt(2),0])))<1e-12,
 'load_zero_after_release':np.max(np.abs(a[a[:,0]>.8,13:16]))<1e-12,
 }
 if reference is None: reference={k:d[k].copy() for k in ['rest','faces','edges','fixed_ids','loaded_ids','native_volume']}
 else: checks['identical_geometry_clamp_mass']=all(np.array_equal(reference[k],d[k]) for k in reference)
 assert all(checks.values()),checks
 # independent principal metric from positions/rest, with flat XY rest basis
 Dm=np.stack((rest[f[:,1],:2]-rest[f[:,0],:2],rest[f[:,2],:2]-rest[f[:,0],:2]),axis=-1)
 Ds=np.stack((x[:,f[:,1]]-x[:,f[:,0]],x[:,f[:,2]]-x[:,f[:,0]]),axis=-1)
 S=np.linalg.svd(Ds@np.linalg.inv(Dm),compute_uv=False)
 checks['stored_principal_matches']=bool(np.allclose(S[:,: ,0].max(axis=1),a[:,5],rtol=1e-12,atol=1e-12))
 tip=np.linalg.norm(a[:,1:4],axis=1);endidx=np.flatnonzero(np.isclose(a[:,0],.8))[0];late=(a[:,0]>=.6)&(a[:,0]<=.8);post=a[:,0]>.8;tail=a[:,0]>=1.5
 vals={'name':name,'source_npz_sha256':hashlib.sha256((base/name/'states.npz').read_bytes()).hexdigest(),'checks':{k:bool(v) for k,v in checks.items()},'peak_principal_extension_pct':float((a[:,5].max()-1)*100),'peak_edge_extension_pct':float((a[:,4].max()-1)*100),'late_load_window_s':[.6,.8],'late_load_mean_max_principal_extension_pct':float((a[late,5].mean()-1)*100),'at_unload_max_principal_extension_pct':float((a[endidx,5]-1)*100),'peak_tip_displacement_mm':float(tip.max()*1000),'tip_displacement_at_unload_mm':float(tip[endidx]*1000),'after_unload_max_tip_displacement_mm':float(tip[post].max()*1000),'after_unload_peak_speed_m_s':float(a[post,9].max()),'max_out_of_plane_displacement_m':float(np.max(np.abs(x[:,:,2]))),'unloaded_before_force_max_displacement_m':float(np.max(np.linalg.norm(x[a[:,0]<=.2]-rest,axis=-1))),'min_area_ratio':float(a[:,8].min()),'after_unload_peak_kinetic_energy_J':float(a[post,10].max()),'late_unloaded_window_s':[1.5,2.0],'late_unloaded_tip_displacement_RMS_mm':float(np.sqrt(np.mean(tip[tail]**2))*1000),'late_unloaded_max_tip_displacement_mm':float(tip[tail].max()*1000),'final_max_vertex_speed_m_s':float(a[-1,9]),'final_tip_displacement_mm':float(tip[-1]*1000),'note':'Late-load window is a measured averaging window, not proof of static equilibrium; tip amplitude is relative to undeformed rest.'}
 summary['cases'].append(vals)
summary['checks']={'native_pair_completed':True,'all_readback_and_independent_geometry_checks':all(all(c['checks'].values()) for c in summary['cases'])}
b,c=summary['cases'];summary['ratios_candidate_over_baseline']={k:c[k]/b[k] if b[k] else None for k in ['peak_principal_extension_pct','peak_tip_displacement_mm','after_unload_peak_speed_m_s','after_unload_peak_kinetic_energy_J','late_unloaded_tip_displacement_RMS_mm']}
out=base/'independent-summary.json';assert not out.exists();out.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));print('SHA256',hashlib.sha256(out.read_bytes()).hexdigest())
