"""Read-only scoped follow-up; preserves the original full-route rejection."""
from pathlib import Path
import json,time
import numpy as np
from scipy.spatial import cKDTree
import continuous_frozen_handles_review as cert

BASE=Path(__file__).resolve().parents[1]
D=BASE/'runs/two_panel_handoff_upright_offset_01/independent-hybrid-minus30-positive-y34p5'
SOURCE=BASE/'runs/two_panel_handoff_upright_offset_01/source/assets/two_panel_bag_wide_tall_coarse.npz'
OUT=D/'independent-continuous-handles-scoped-followup.json'

def check(v,tets,triangles,regions,delta,name):
    required=.002
    lo=np.minimum(v.min(1),v.min(1)+delta);hi=np.maximum(v.max(1),v.max(1)+delta)
    out={'scope':name,'required_clearance_m':required,'tetrahedra':len(tets),'handle_triangles':len(triangles),
         'displacement_m':delta,'aabb_certified_pairs':0,'fine_pair_count':0,'fine_pairs':[],
         'minimum_aabb_gap_lower_m':None,'minimum_fine_gap_lower_m':None,'status':'running'}
    started=time.monotonic();rejected=False
    for f,region in zip(triangles,regions):
        gaps=np.maximum(np.maximum(lo-f.max(0),f.min(0)-hi),0.)
        dist=np.linalg.norm(gaps,axis=1)-cert.GUARD;far=dist>=required
        out['aabb_certified_pairs']+=int(far.sum())
        if far.any():
            m=float(dist[far].min());old=out['minimum_aabb_gap_lower_m']
            out['minimum_aabb_gap_lower_m']=m if old is None else min(m,old)
        for i in np.flatnonzero(~far):
            proof=cert.pair(v[i],f,delta,required)
            item={'tetrahedron_index':int(i),'tet_vertices':tets[i].tolist(),**region,**proof}
            out['fine_pairs'].append(item);out['fine_pair_count']+=1
            m=proof['conservative_euclidean_projection_gap_lower_m'];old=out['minimum_fine_gap_lower_m']
            out['minimum_fine_gap_lower_m']=m if old is None else min(m,old)
            if proof['status']!='clear':
                rejected=True;out['first_failure']=item;break
        if rejected:break
    out['status']='rejected' if rejected else 'clear'
    out['processed_pairs']=out['aabb_certified_pairs']+out['fine_pair_count']
    out['total_pairs']=len(tets)*len(triangles)
    out['complete_pair_coverage']=out['processed_pairs']==out['total_pairs']
    out['minimum_complete_gap_lower_m']=None if rejected else min(x for x in [out['minimum_aabb_gap_lower_m'],out['minimum_fine_gap_lower_m']] if x is not None)
    out['wall_seconds']=time.monotonic()-started
    if not rejected:assert out['complete_pair_coverage'] and out['minimum_complete_gap_lower_m']>=required
    print(name,out['status'],out['minimum_complete_gap_lower_m'],'fine',out['fine_pair_count'],flush=True)
    return out

def main():
    paths=[D/'rest.npz',D/'hybrid_hypothetical_observation.npz',D/'guarded_candidate.json',D/'diagnostic_config.json',D/'independent-continuous-handles-review.json',SOURCE,Path(cert.__file__),Path(__file__)]
    hashes={str(p):cert.sha(p) for p in paths}
    rest=dict(np.load(paths[0]));state=dict(np.load(paths[1]));candidate=json.loads(paths[2].read_text());cfg=json.loads(paths[3].read_text());prior=json.loads(paths[4].read_text());src=dict(np.load(SOURCE))
    assert candidate['status']=='candidate' and prior['status']=='rejected'
    error,mapping=cKDTree(rest['bag']).query(src['points']+cfg['bag']['position'])
    assert error.max()<1e-6 and len(np.unique(mapping))==len(mapping)
    lookup={tuple(sorted(f)):i for i,f in enumerate(rest['bag_faces'])};bodyset=set(src['body_face_indices'].tolist())
    faces={};regions={}
    for name in ['front','back']:
        ids=[i for i,f in enumerate(src['faces']) if i not in bodyset and np.isin(f,src['handle_'+name]).all()]
        assert len(ids)==352
        faces[name]=mapping[src['faces'][ids]]
        regions[name]=[{'handle':name,'source_face_index':i,'native_face_index':lookup[tuple(sorted(mapping[src['faces'][i]]))],'native_vertices':mapping[src['faces'][i]].tolist()} for i in ids]
    stages=candidate['route']['stages'];rots=np.array([s['rotation_xyzw'] for s in stages]);assert np.array_equal(rots, np.tile(rots[0],(len(rots),1)))
    start=np.array(stages[-2]['right_tcp_m']);end=np.array(stages[-1]['right_tcp_m']);delta=end-start
    assert np.max(np.abs(delta[:2]))<1e-14 and delta[2]<0
    bear=state['bear']+start-state['tcp_right'];tets=rest['bear_tets'];bag=state['bag'];rim=bag[mapping[src['rim']]];body=bag[mapping[src['body']]]
    fraction=(float(rim[:,2].max())-float(bear[:,2].min()))/delta[2]
    assert 0<fraction<1
    out={'kind':'frozen_hybrid_scoped_handle_followup','native_execution':False,'production_edited':False,'physical_success':False,
         'original_full_route_result_remains':'rejected','original_rejection_report':str(paths[4]),
         'input_sha256':hashes,'source_to_native_max_error_m':float(error.max()),
         'front_entire_descent':check(bear[tets],tets,bag[faces['front']],regions['front'],delta,'held_front_handle_entire_descent'),
         'back_before_bear_bottom_reaches_highest_rim':check(bear[tets],tets,bag[faces['back']],regions['back'],delta*fraction,'free_back_handle_pre_rim_height_descent'),
         'prefix_boundary':{'definition':'whole frozen teddy minZ first equals highest actual body-rim vertex Z; vertical height condition only, not actual mouth crossing or containment',
                            'full_descent_fraction':fraction,'transport_bear_bbox_min_m':bear.min(0),'transport_bear_bbox_max_m':bear.max(0),
                            'rim_min_z_m':float(rim[:,2].min()),'rim_max_z_m':float(rim[:,2].max()),'body_max_z_m':float(body[:,2].max()),
                            'tcp_at_boundary_m':start+fraction*delta}}
    bad=prior['stages'][-1]['first_failure'];w=bad['strict_intersection'];f=float(w['path_fraction']);at=bear+f*delta
    local=(rest['bear']-np.asarray(cfg['bear']['position']))/.85
    ti=bad['tetrahedron_index'];tetids=tets[ti];restp=np.asarray(w['tet_barycentric'])@local[tetids]
    shapes=[('torso',(0,-.025,.045),(.065,.075,.040)),('head',(0,.080,.051),(.058,.055,.047)),('left_ear',(-.047,.126,.057),(.023,.023,.022)),('right_ear',(.047,.126,.057),(.023,.023,.022)),('left_arm',(-.076,-.010,.040),(.040,.035,.028)),('right_arm',(.076,-.010,.040),(.040,.035,.028)),('left_foot',(-.040,-.102,.034),(.036,.041,.030)),('right_foot',(.040,-.102,.034),(.036,.041,.030)),('muzzle',(0,.068,.091),(.031,.024,.017)),('nose',(0,.083,.106),(.012,.008,.006))]
    scores={n:float((np.linalg.norm((restp-np.array(c))/np.array(r))-1)*min(r)) for n,c,r in shapes}
    # Reconstruct the true strip rows from source topology and verify every quad;
    # no assumption about native index ordering is made.
    internal=np.setdiff1d(src['handle_back'],src['body']);nc=len(src['handle_back_root_left'])
    assert len(internal)%nc==0
    strip=np.vstack([src['handle_back_root_left'],internal.reshape(-1,nc),src['handle_back_root_right'][::-1]])
    expected=set()
    for j in range(len(strip)-1):
        for i in range(nc-1):
            a,b,c,d=strip[j,i],strip[j,i+1],strip[j+1,i],strip[j+1,i+1]
            expected.update([tuple(sorted([a,b,d])),tuple(sorted([a,d,c]))])
    true=set(tuple(sorted(src['faces'][r['source_face_index']])) for r in regions['back'])
    assert expected==true and len(expected)==352
    srcface=src['faces'][bad['source_face_index']]
    assert np.array_equal(mapping[srcface],bad['native_vertices'])
    positions={int(v):(j,i) for j,row in enumerate(strip) for i,v in enumerate(row)}
    row=np.array([positions[int(v)][0] for v in srcface]);bary=np.array(w['triangle_barycentric'])
    arc={}
    for label,pts in [('rest',src['points']),('actual_frozen',bag[mapping])]:
        centers=pts[strip].mean(1);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centers,axis=0),axis=1))]
        loc=float(bary@s[row]);arc[label]={'strip_centerline_length_m':float(s[-1]),'witness_material_interpolated_distance_from_left_root_m':loc,
                                       'distance_from_right_root_m':float(s[-1]-loc),'fraction_from_left_root':loc/float(s[-1]),
                                       'row_centerline_arclengths_m':s}
    out['previous_strict_witness_localization']={'not_first_contact_time':True,'full_descent_fraction':f,'point_world_m':w['bear_point_world_m'],
           'interior_margin_m':w['minimum_interior_face_or_edge_margin_m'],'recomputed_residual_m':w['recomputed_coordinate_residual_m'],
           'teddy_bbox_min_m':at.min(0),'teddy_bbox_max_m':at.max(0),'teddy_vertex_mean_m':at.mean(0),'tcp_m':start+f*delta,
           'bbox_minZ_minus_rim_maxZ_m':float(at[:,2].min()-rim[:,2].max()),'bbox_minZ_minus_rim_minZ_m':float(at[:,2].min()-rim[:,2].min()),
           'bbox_maxZ_minus_rim_maxZ_m':float(at[:,2].max()-rim[:,2].max()),'bbox_maxZ_minus_rim_minZ_m':float(at[:,2].max()-rim[:,2].min()),
           'witnessZ_minus_rim_maxZ_m':float(np.array(w['bear_point_world_m'])[2]-rim[:,2].max()),'witnessZ_minus_rim_minZ_m':float(np.array(w['bear_point_world_m'])[2]-rim[:,2].min()),
           'tetrahedron_index':ti,'tet_vertices':tetids,'unscaled_material_local_tet_vertices_m':local[tetids],
           'unscaled_material_local_witness_m':restp,'nearest_original_ellipsoid_component':min(scores,key=scores.get),'component_field_scores_m':scores,
           'classification_caveat':'Nearest original implicit ellipsoid field component at material witness, not an independent anatomical segmentation; smooth unions overlap.',
           'back_strap_source_face_index':bad['source_face_index'],'source_face_vertices':srcface,'strip_row_indices':row,'strip_cross_indices':[positions[int(v)][1] for v in srcface],
           'face_shares_body_or_sewn_root_vertex':bool(np.isin(srcface,src['body']).any()),'strip_verified_against_all_352_true_face_triplets':True,'arclength':arc}
    out['limitations']=['Hybrid source combines historical lifted teddy and new settled bag; not a joint native state.',
        'Only constant-orientation frozen straight translations are certified; actual cloth motion, teddy deformation, grip slip and joint-FK tracking are not covered.',
        'The pre-rim boundary uses height only; it does not establish crossing the open mouth or containment.',
        'Original both-handles full-descent result remains rejected, unchanged. This report distinguishes scopes, not physical threshold relaxation.',
        'Previously found strict witness is first encountered in enumeration, not the earliest path contact.',
        'Original robot/force/material gates are unchanged. No dynamic success is claimed.',
        'Reported lower bound is conservative separating-geometry bound, not exact global nearest distance; floating point guard is not interval arithmetic.']
    out['inputs_unchanged']=all(cert.sha(p)==h for p,h in hashes.items())
    OUT.write_text(json.dumps(out,indent=2,default=cert.serial,allow_nan=False)+'\n')
    print('REPORT',OUT,flush=True)
    print(json.dumps({k:out[k] for k in ['prefix_boundary','previous_strict_witness_localization']},indent=2,default=cert.serial))

if __name__=='__main__':main()
