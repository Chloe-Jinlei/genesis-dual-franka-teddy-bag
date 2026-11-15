"""Explicitly PARTIAL failed-full37 versus successful-full30 common time prefix.

Use the immutable baseline's exact triangle and direction definitions. Do not
call its main or the completed-run wrapper. No simulation or production edits.
"""
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runtime_ipc/full37-partial-cloth-material-comparison.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    wrapper=ROOT/'runtime_ipc/compare_cloth_material_run.py'
    spec=importlib.util.spec_from_file_location('unchanged_cloth_method_loader',wrapper)
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    method,_=loader.load_method(ROOT/'runs/two_panel_full_37',ROOT/'runtime_ipc/unused_partial_method_main.json')
    # load_method loads definitions only. Neither completed main is executed.
    frames=method['triangle_frames'];intrinsic=method['intrinsic_inverse'];deform=method['deformation'];quantiles=method['weighted_quantiles']
    report37=json.loads((ROOT/'runs/two_panel_full_37/report.json').read_text())
    cutoff=float(report37['simulated_seconds'])
    assert report37['status']=='failed' and not Path('/proc/339905').exists()
    checks={'frozen_reference_method_hash':sha(loader.REFERENCE_SCRIPT)==loader.SCRIPT_SHA256,
            'immutable_baseline_report_hash':sha(loader.REFERENCE_REPORT)==loader.REPORT_SHA256}
    inputs={str(loader.REFERENCE_SCRIPT):sha(loader.REFERENCE_SCRIPT),str(loader.REFERENCE_REPORT):sha(loader.REFERENCE_REPORT)}
    output={};reference_rest=None
    for run_name in ['two_panel_full_30','two_panel_full_37']:
        run=ROOT/'runs'/run_name;cfg=json.loads((run/'config.json').read_text());events=json.loads((run/'events.json').read_text())
        asset_path=run/'source/assets'/cfg['bag']['npz_file']
        protected=[run/n for n in ['config.json','rest.npz','states.npz','events.json','report.json','soft_metrics.npy']]+[asset_path]
        rest=dict(np.load(run/'rest.npz'));asset=dict(np.load(asset_path));states=np.load(run/'states.npz')
        t=states['time'];keep=np.r_[np.diff(t)>1e-9,True]&(t<=cutoff+1e-9)
        times=t[keep];positions=states['bag'][keep];origin=['states.npz']*len(times)
        if run_name=='two_panel_full_37':
            failure_path=run/'failed_soft_state.npz';failure=np.load(failure_path);protected.append(failure_path)
            assert abs(float(failure['time'])-cutoff)<1e-9 and float(failure['time'])>times[-1]
            times=np.r_[times,float(failure['time'])];positions=np.concatenate((positions,failure['bag'][None]))
            origin.append('failed_soft_state.npz: actual terminal soft state')
        inputs.update({str(p):sha(p) for p in protected})
        faces=rest['bag_faces'];points=rest['bag']
        if reference_rest is None:reference_rest=rest
        else:checks['actual_cloth_rest_and_faces_identical']=bool(np.array_equal(points,reference_rest['bag']) and np.array_equal(faces,reference_rest['bag_faces']))
        checks[run_name+'_source_native_identity']=bool(np.array_equal(faces,asset['faces']) and np.max(abs(points-asset['points']-cfg['bag']['position']))<1e-8)
        groups={name:asset[name+'_face_indices'] for name in ['panel_front','panel_back']};body=set(asset['body_face_indices'].tolist())
        for side in ['front','back']:
            members=set(asset['handle_'+side].tolist())
            groups['handle_'+side]=np.array([i for i,f in enumerate(faces) if i not in body and set(f.tolist())<=members])
        assert np.array_equal(np.sort(np.concatenate(list(groups.values()))),np.arange(len(faces)))
        coords=np.full((len(faces),3,2),np.nan)
        for name,ids in groups.items():
            if name.startswith('panel_'):grid=asset[name+'_grid'];width=asset['material_x'];length=asset['material_height']
            else:grid,width,length=method['reconstruction_strip'](asset,ids,name.removeprefix('handle_'),checks)
            uv=np.full((len(points),2),np.nan);uv[grid]=np.stack(np.meshgrid(width,length),axis=-1);coords[ids]=uv[faces[ids]]
        material_inv=np.linalg.inv(np.stack((coords[:,1]-coords[:,0],coords[:,2]-coords[:,0]),axis=-1))
        edges=frames(points,faces);inverse,area=intrinsic(edges);ref=edges@material_inv;refnorm=np.linalg.norm(ref,axis=1)
        angle0=np.arccos(np.clip(np.sum(ref[:,:,0]*ref[:,:,1],axis=1)/np.prod(refnorm,axis=1),-1.,1.))
        metrics={name:np.empty((len(times),len(faces))) for name in ['principal_min','principal_max','area_ratio','width_ratio','length_ratio','shear_angle_change_deg']}
        for j,x in enumerate(positions):
            edge=frames(x,faces);singular,_=deform(edge,inverse);dirs=edge@material_inv;norm=np.linalg.norm(dirs,axis=1)
            angle=np.arccos(np.clip(np.sum(dirs[:,:,0]*dirs[:,:,1],axis=1)/np.prod(norm,axis=1),-1.,1.))
            metrics['principal_min'][j],metrics['principal_max'][j]=singular[:,0],singular[:,1]
            metrics['area_ratio'][j]=np.prod(singular,axis=1);metrics['width_ratio'][j]=norm[:,0]/refnorm[:,0];metrics['length_ratio'][j]=norm[:,1]/refnorm[:,1];metrics['shear_angle_change_deg'][j]=np.degrees(angle-angle0)
        checks[run_name+'_all_metrics_finite']=all(np.isfinite(v).all() for v in metrics.values())
        def snapshot(ids,j):
            return {'actual_time_s':float(times[j]),'evidence':origin[j],
                    'area_weighted_triangle_quantiles':{k:quantiles(v[j,ids],area[ids]) for k,v in metrics.items()}}
        def extrema(ids,rows):
            result={}
            for name,value in metrics.items():
                part=value[np.ix_(rows,ids)];objective=abs(part) if name=='shear_angle_change_deg' else part
                k=np.argmin(objective) if name=='principal_min' else np.argmax(objective);j,f=np.unravel_index(k,part.shape);j=int(rows[j]);f=int(ids[f])
                result[name]={'value':float(value[j,f]),'absolute_value':float(abs(value[j,f])),'time_s':float(times[j]),'face_index':f,'vertices':faces[f].tolist(),'evidence':origin[j]}
            return result
        completed={};partial={}
        for i,event in enumerate(events):
            start=event['start'];end=events[i+1]['start'] if i+1<len(events) else start+np.ceil(event['duration']/cfg['simulation']['dt'])*cfg['simulation']['dt']
            if start>cutoff+1e-8:continue
            rows=np.flatnonzero((times>=start-1e-8)&(times<=min(end,cutoff)+1e-8))
            if not len(rows):continue
            if end<=cutoff+1e-8:
                exact=np.flatnonzero(abs(times-end)<1e-8);assert len(exact)
                completed[event['phase']]={'rows':rows,'endpoint':int(exact[-1])}
            else:partial[event['phase']]=rows
        g={}
        for name,ids in groups.items():
            g[name]={'face_count':len(ids),'prefix_extrema':extrema(ids,np.arange(len(times))),
                     'last_observed_snapshot_NOT_final_task_state':snapshot(ids,len(times)-1),
                     'completed_event_states':{phase:{'extrema':extrema(ids,v['rows']),'endpoint':snapshot(ids,v['endpoint'])} for phase,v in completed.items()},
                     'unfinished_phase_prefix':{phase:{'extrema':extrema(ids,rows),'last_observed':snapshot(ids,int(rows[-1]))} for phase,rows in partial.items()}}
        soft=np.load(run/'soft_metrics.npy');sr=soft[:,0]<=cutoff+1e-9
        output[run_name]={'status':'PARTIAL_COMMON_PREFIX','source_run':run_name,'configuration':cfg['bag'],'first_saved_time_s':float(times[0]),'last_saved_time_s':float(times[-1]),'frame_count':len(times),'maximum_saved_interval_s':float(max(np.diff(times))),
                          'groups':g,'per_step_mesh_edge_ratio_peak_in_prefix':float(max(soft[sr,4])),
                          'completed_phase_names':list(completed),'unfinished_phase_names':list(partial)}
    checks['explicit_requested_run_binding_full37']=set(output)=={'two_panel_full_30','two_panel_full_37'} and output['two_panel_full_37']['configuration']['E']==10000000. and output['two_panel_full_37']['configuration']['shear_stiffness']==50. and output['two_panel_full_37']['configuration']['rho']==333.3333333333333 and output['two_panel_full_37']['configuration']['bending_stiffness']==1e-5 and output['two_panel_full_37']['configuration']['friction_mu']==.5 and output['two_panel_full_37']['configuration']['handle_shear_stiffness']==1000.
    old,new=output['two_panel_full_30'],output['two_panel_full_37'];summary={}
    common=set(old['completed_phase_names'])&set(new['completed_phase_names'])
    for name in loader.GROUPS:
        a,b=old['groups'][name],new['groups'][name]
        summary[name]={'full30_prefix_maximum_principal_ratio':a['prefix_extrema']['principal_max']['value'],
            'full37_prefix_maximum_principal_ratio':b['prefix_extrema']['principal_max']['value'],
            'full30_prefix_peak_time_s':a['prefix_extrema']['principal_max']['time_s'],
            'full37_prefix_peak_time_s':b['prefix_extrema']['principal_max']['time_s'],
            'maximum_principal_extension_change_percentage_points':100*(b['prefix_extrema']['principal_max']['value']-a['prefix_extrema']['principal_max']['value']),
            'last_observed_maximum_principal_ratio':{run:output[run]['groups'][name]['last_observed_snapshot_NOT_final_task_state']['area_weighted_triangle_quantiles']['principal_max']['maximum'] for run in output},
            'same_completed_event_endpoint_maximum_principal_ratio':{phase:{'full30':a['completed_event_states'][phase]['endpoint']['area_weighted_triangle_quantiles']['principal_max']['maximum'],'full37':b['completed_event_states'][phase]['endpoint']['area_weighted_triangle_quantiles']['principal_max']['maximum'],'full30_actual_time_s':a['completed_event_states'][phase]['endpoint']['actual_time_s'],'full37_actual_time_s':b['completed_event_states'][phase]['endpoint']['actual_time_s']} for phase in old['completed_phase_names'] if phase in common}}
    # Explicit named-stage comparison reuses already computed exact metrics.
    # Full30 completes insertion and starts release before the absolute cutoff;
    # full37 has only a partial insertion. These are not matched-progress loads.
    for name in loader.GROUPS:
        old_insert=old['groups'][name]['completed_event_states']['insert_teddy']
        new_insert=new['groups'][name]['unfinished_phase_prefix']['insert_teddy']
        summary[name]['same_named_insertion_observed_window']={
            'full30_completed_stage_maximum_principal_ratio':old_insert['extrema']['principal_max']['value'],
            'full37_partial_stage_maximum_principal_ratio':new_insert['extrema']['principal_max']['value'],
            'full30_completed_endpoint':old_insert['endpoint'],
            'full37_last_observed_before_planned_endpoint':new_insert['last_observed'],
            'scope':'Same named insertion stage, different observed durations/progress/contact loading. Full30 is complete here, full37 is partial. No interpolation or mechanically matched-load claim.'}
    checks['all_inputs_and_immutable_reference_unchanged']=all(sha(p)==h for p,h in inputs.items())
    assert all(checks.values()),checks
    result={'kind':'explicitly_PARTIAL_full37_cloth_material_comparison','created_utc':datetime.now(timezone.utc).isoformat(),'status':'PARTIAL_FAILED_BEFORE_RELEASE','physical_task_success':False,'full37_reported_failure':report37['error'],'common_absolute_cutoff_s':cutoff,
            'method_source':str(loader.REFERENCE_SCRIPT),'method_source_sha256':loader.SCRIPT_SHA256,'baseline_report_unchanged_sha256':loader.REPORT_SHA256,
            'exact_method_scope':'Use identical reference triangle functions, strip reconstruction, orthonormal F singular values, construction direction norms, rest-angle subtraction, rest-area quantiles and last-duplicate policy. Only the time window and explicit partial event reporting differ. No completed-run main is executed.',
            'sampling_note':'No interpolation: full30 last complete saved frame is earlier than the common cutoff; full37 also includes its separately saved actual failure soft snapshot. Compare the stated actual times. Neither last snapshot is described as a final task result.',
            'checks':checks,'all_diagnostic_checks_passed':all(checks.values()),'input_sha256':inputs,'script_sha256':sha(__file__),'summary':summary,'runs':output,
            'limitations':['Different density/inertia, contact/drape responses and actual motion durations change loads; this is not an equal-load material experiment. At the common absolute cutoff full30 has passed release whereas full37 is still in insertion; use the separately reported common completed endpoints and same-named insertion window with their own stated timings.','Principal extension and construction-grid axis extension are not interchangeable; no measured yarn orientation is declared.','Saved triangle metrics may miss inter-frame maxima. Per-step edge maxima are a different observable.','The new run did not reach release, withdrawal or final settle, so the full30 final state cannot serve as a matched new-material comparison.','Lower measured strain does not waive the original torque or any physical gate, and does not establish low restitution or material calibration.']}
    result['insertion_phase_timing']={}
    for run_name in ['two_panel_full_30','two_panel_full_37']:
        rr=ROOT/'runs'/run_name
        ev=next(e for e in json.loads((rr/'events.json').read_text()) if e['phase']=='insert_teddy')
        dt=json.loads((rr/'config.json').read_text())['simulation']['dt']
        planned_end=ev['start']+float(np.ceil(ev['duration']/dt))*dt
        result['insertion_phase_timing'][run_name]={'start_s':ev['start'],'declared_duration_s':ev['duration'],'planned_discrete_end_s':planned_end,'observed_end_s':min(planned_end,cutoff),'observed_elapsed_s':min(planned_end,cutoff)-ev['start'],'completed_by_cutoff':planned_end<=cutoff+1e-8}
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'path':str(OUTPUT),'sha256':sha(OUTPUT),'status':result['status'],'cutoff':cutoff,'last_actual_times':{k:v['last_saved_time_s'] for k,v in output.items()},'summary':{k:{n:v for n,v in values.items() if n!='same_completed_event_endpoint_maximum_principal_ratio'} for k,values in summary.items()}},indent=2))


if __name__=='__main__':main()
