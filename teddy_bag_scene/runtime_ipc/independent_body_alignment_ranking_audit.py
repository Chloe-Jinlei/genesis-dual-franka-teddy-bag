"""Independent CPU-only review of the separate body-ranking prototype."""
from pathlib import Path
import copy, hashlib, importlib.util, itertools, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/two_panel_full_17';sys.path.insert(0,str(ROOT/'runtime_ipc'))
import body_alignment_ranking as model
reference=json.loads((RUN/'independent-body-ranking-reference.json').read_text())
actual=json.loads((RUN/'independent-body-alignment-ranking.json').read_text())
rows=copy.deepcopy(actual['candidates']);original=copy.deepcopy(rows)
checks=[]
def check(name,okay,**details):checks.append(dict(name=name,passed=bool(okay),**details))
def probe(name,fn,predicate):
 try:
  value=fn();check(name,predicate(value))
 except Exception as exc:check(name,False,exception=type(exc).__name__,message=str(exc))

by_index={r['index']:r for r in rows}
max_delta=0.
for r in reference['rows']:
 m=by_index[r['index']]['body_metrics']
 check('actual_node_count_'+str(r['index']),m['inside_node_count']==r['inside_node_count'] and m['node_count']==r['total_nodes'])
 diffs=[abs(m['signed_com_body_distance_m']-r['actual_com_signed_body_distance_m']),abs(m['maximum_outside_node_body_distance_m']-r['max_outside_body_distance_m'])]
 max_delta=max(max_delta,max(diffs));check('actual_distances_'+str(r['index']),max(diffs)<1e-12,maximum_difference_m=max(diffs))
pure=model.rank_candidates(rows);band=model.rank_with_original_score_band(rows)
check('actual_pure_fronts_match_independent_reference',pure['pareto_fronts']==reference['pure_pareto_layers'])
check('actual_band_fronts_match_independent_reference',band['pareto_fronts']==reference['one_mm_band']['layers'] and band['retained_indices']==reference['one_mm_band']['in_band_indices'])
check('actual_band_cutoff_matches_independent_reference',band['inclusive_original_score_cutoff_m']==reference['one_mm_band']['cutoff_m'])
check('all_120_input_permutations_preserve_pure_and_band_results',all(model.rank_candidates(list(p))==pure and model.rank_with_original_score_band(list(p))==band for p in itertools.permutations(rows)))

valid=copy.deepcopy(rows[0]['frozen_query_diagnostic']);n=rows[0]['body_metrics']['node_count']
for label,gap,expected in [('exact_2mm',.002,'valid'),('below_2mm',np.nextafter(.002,-np.inf),'unknown')]:
 d=copy.deepcopy(valid);d['height_precondition']['minimum_query_height_gap_m']=float(gap)
 probe('height_'+label,lambda d=d:model.metrics_from_diagnostic(d,n),lambda x:x['status']==expected)
for label,dist,expected in [('exact_1um',1e-6,'unknown'),('above_1um',np.nextafter(1e-6,np.inf),'valid'),('negative',-.001,'unknown')]:
 d=copy.deepcopy(valid);d['physical_cloth_query_distance']['body_distance_m'][0]=float(dist)
 probe('body_distance_'+label,lambda d=d:model.metrics_from_diagnostic(d,n),lambda x:x['status']==expected)
d=copy.deepcopy(valid)
for k in ['mean_apex_raw_winding','high_apex_raw_winding']:d['cap_winding'][k]=(-np.asarray(d['cap_winding'][k])).tolist()
probe('coherent_negative_orientation_has_same_point_metrics',lambda:model.metrics_from_diagnostic(d,n),lambda x:x['status']=='valid' and x['inside_node_count']==rows[0]['body_metrics']['inside_node_count'])
d=copy.deepcopy(valid);d['physical_cloth_query_distance']['full_cloth_distance_m']=[0.]*(n+1)
probe('handle_distance_cannot_change_actual_body_metric',lambda:model.metrics_from_diagnostic(d,n),lambda x:x['status']=='valid' and x['signed_com_body_distance_m']==rows[0]['body_metrics']['signed_com_body_distance_m'])
for key,value in [('actual_topology_orientation',None),('actual_spatial_rim',[]),('cap_winding',None),('physical_cloth_query_distance',None)]:
 d=copy.deepcopy(valid);d[key]=value
 probe('malformed_diagnostic_'+key,lambda d=d:model.metrics_from_diagnostic(d,n),lambda x:x['status']=='unknown' and all(x[k] is None for k in model.METRIC_KEYS))
for label,value in [('missing',None),('none',None),('list',[]),('nonfinite_metric',{'status':'valid','node_count':n,'inside_node_count':1,'node_inside_fraction':1/n,'signed_com_body_distance_m':float('nan'),'maximum_outside_node_body_distance_m':0.})]:
 bad=copy.deepcopy(rows)
 if label=='missing':bad[-1].pop('body_metrics')
 else:bad[-1]['body_metrics']=value
 probe('malformed_metrics_'+label,lambda bad=bad:model.rank_with_original_score_band(bad),lambda x:x['method']=='whole_group_original_score_fallback' and x['ranked_indices']==pure['original_score_order'] and not x['band_applied'])

# One otherwise accepted, outside-band hypothesis becoming numerically ambiguous
# must discard all body ordering, rather than omit that candidate.
bad=copy.deepcopy(rows);i=next(j for j,r in enumerate(bad) if r['index']==3)
d=copy.deepcopy(bad[i]['frozen_query_diagnostic']);d['physical_cloth_query_distance']['body_distance_m'][0]=1e-6
bad[i]['body_metrics']=model.metrics_from_diagnostic(d,n)
fallback=model.rank_with_original_score_band(bad)
check('actual_outside_band_unknown_preserves_entire_original_order',fallback['method']=='whole_group_original_score_fallback' and fallback['ranked_indices']==pure['original_score_order'] and fallback['unknown_candidate_indices']==[3])

# Diagnostic exception handling is exercised through the complete evaluator;
# existing raw diagnostic evidence is reused for the other four candidates.
real_diagnose=model.low_region.diagnose
def one_exception(run,path,*a,**k):
 index=int(Path(path).stem.rsplit('_',1)[1])
 if index==3:raise RuntimeError('Independent injected geometric query failure')
 return copy.deepcopy(by_index[index]['frozen_query_diagnostic'])
try:
 model.low_region.diagnose=one_exception
 probe('one_geometry_exception_causes_whole_group_fallback',lambda:model.evaluate(RUN,RUN/'lower_teddy_to_rim.npz',RUN/'independent-adaptive-actual-center.json',ROOT/'config_guarded_adaptive_alignment_trial.json'),
  lambda x:x['ranking']['method']=='whole_group_original_score_fallback' and x['optional_original_score_band_ranking']['method']=='whole_group_original_score_fallback' and x['ranking']['ranked_indices']==pure['original_score_order'] and x['ranking']['unknown_candidate_indices']==[3])
finally:model.low_region.diagnose=real_diagnose

rng=np.random.default_rng(819);metrics=[]
for _ in range(50):
 metrics.append(dict(inside_node_count=int(rng.integers(0,10)),signed_com_body_distance_m=float(rng.integers(-5,6))*0.5e-6,maximum_outside_node_body_distance_m=float(rng.integers(0,10))*0.5e-6))
transitive=True;antecedents=0
for ia,ib,ic in rng.integers(0,len(metrics),size=(5000,3)):
 a,b,c=[metrics[i] for i in [ia,ib,ic]]
 if model.dominates(a,b) and model.dominates(b,c):
  antecedents+=1;transitive &= model.dominates(a,c)
check('one_sided_tolerance_random_transitivity',transitive,nonvacuous_antecedents=antecedents)
check('candidate_inputs_not_mutated',rows==original)
source=ROOT/'runtime_ipc/body_alignment_ranking.py';sourcehash=hashlib.sha256(source.read_bytes()).hexdigest()
check('actual_output_generated_by_current_prototype',actual['input_sha256'][str(source)]==sourcehash)
report=dict(kind='independent_body_alignment_ranking_audit',physical_success=False,runtime_execution_performed=False,
 reviewed_source_sha256=sourcehash,actual_report_sha256=hashlib.sha256((RUN/'independent-body-alignment-ranking.json').read_bytes()).hexdigest(),
 check_count=len(checks),all_checks_passed=all(c['passed']for c in checks),checks=checks,remaining_failures=[c for c in checks if not c['passed']],
 maximum_independent_actual_metric_difference_m=max_delta,
 expected_pure_fronts=reference['pure_pareto_layers'],expected_score_band=reference['one_mm_band'],
 mathematical_scope=['Fixed oriented body plus two caps above all queries gives cap-independent low-region pointwise winding. Closed topology does not prove embedding or absence of body self-intersection.',
 'Node fraction is unweighted on the common mesh, not mass, volume, collision clearance, or success. COM distance is signed by guarded unit winding and measured only to actual body triangles. Exterior max excludes COM because COM is its separate metric.',
 'All components exactly no worse plus at least one improvement beyond tolerance is a transitive strict partial order; symmetric epsilon dominance is not used.',
 'A 1mm original-score loss band is an explicit offline heuristic. It constrains recorded center/insert/release sampled-rigid and COM-envelope surpluses only, not relative withdraw clearance, continuous FK, future deformation or actual contact.',
 'All old physical gates and the active full19 run remain unchanged. No whole tetrahedral containment, release or settling proof is delivered.'])
out=RUN/'independent-body-alignment-ranking-audit.json';out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k]for k in ['reviewed_source_sha256','check_count','all_checks_passed','remaining_failures','maximum_independent_actual_metric_difference_m']},indent=2));print(out)
