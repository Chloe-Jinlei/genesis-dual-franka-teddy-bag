"""AST-isolated CPU flow audit. Does not import Genesis or execute physics."""
import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]
source=(HERE/"run.py").read_text()
tree=ast.parse(source)
method=next(n for c in tree.body if isinstance(c,ast.ClassDef)
            for n in c.body if isinstance(n,ast.FunctionDef) and n.name=="run_guarded_contact_trial")
class GateFailure(Exception):pass
namespace={"np":np,"GateFailure":GateFailure,"positions":lambda x:np.zeros((1,3)),"tcp":lambda x:np.zeros(3)}
exec(compile(ast.Module(body=[method],type_ignores=[]),"isolated_run_guarded_contact_trial","exec"),namespace)


def scenario(fresh_statuses, distances=(.025,.025), correction_status="candidate", bad_total=None, stage_xy=None):
    logs=[]; holder={"guards":[],"fresh":0,"corrections":[]}
    def stage(name,xy=(0.,0.)):
        return {"name":name,"right_tcp_m":[*xy,1.],"rotation_xyzw":[1.,0.,0.,0.],"duration_s":1.}
    first={"status":"candidate","id":"initial","stages_to_execute":[stage("height_at_current_xy"),stage("translate_outside_mouth")],"observed_time_s":0.}
    def initial(*args,**kwargs):return copy.deepcopy(first)
    def fresh(rest,state,config,selected,**kwargs):
        index=holder["fresh"];holder["fresh"]+=1
        logs.append({"op":"fresh","time":state["time"],"selected":selected["id"]})
        assert state["time"]>selected["observed_time_s"]
        status=fresh_statuses[min(index,len(fresh_statuses)-1)]
        return {"status":status,"insert_stage":stage("insert"),"outside_tcp_m":[0.,0.,1.]}
    def correction(rest,state,config,selected,*,attempt_index,cumulative_xy_m,**kwargs):
        holder["corrections"].append({"attempt":attempt_index,"previous_distance":cumulative_xy_m})
        distance=distances[attempt_index-1]
        return {"status":correction_status,"id":f"correction_{attempt_index}","observed_time_s":state["time"],
            "correction_distance_m":distance,"cumulative_xy_m_after":cumulative_xy_m+distance if bad_total is None else bad_total,
            "stages_to_execute":[stage("height_at_current_xy"),stage("translate_outside_mouth",(distance if stage_xy is None else stage_xy,0.))]}
    helpers=types.ModuleType("contact_insertion")
    helpers.plan_guarded_contact_trial=initial;helpers.reobserve_guarded_contact_trial=fresh;helpers.recenter_guarded_contact_trial=correction
    guard_module=types.ModuleType("contact_guard")
    class Guard:
        def __init__(self,scene):self.baseline_count=0;holder["guards"].append(self)
        def set_baseline(self,t):self.baseline_count+=1;return {"time_s":t}
        def sample(self,t):return {"right_finger_opening_projection_n":[4.,4.]},None
    guard_module.ContactGuard=Guard
    sys.modules["contact_insertion"]=helpers;sys.modules["contact_guard"]=guard_module
    class Sim:
        def __init__(self):
            self.rest={};self.frames=[{"time":0.}];self.config={};self.out=Path("/tmp")
            self.report={};self.scene=object();self.time=0.;self.bag=object();self.bear=object()
            self.left_grasp_material_ids=[0];self.robots={"left":object(),"right":object()}
            self.contact_guard_trace=[];self.must_hold_teddy=True;self.must_hold_bag=True
        def move_guarded(self,label,stage):
            logs.append({"op":"move","label":label,"guard_id":id(self.contact_guard),"must_hold_teddy":self.must_hold_teddy})
            self.time+=1.;self.frames.append({"time":self.time})
        def hold(self,label,duration,**fingers):
            logs.append({"op":"hold","label":label,"guard_id":id(self.contact_guard),"must_hold_teddy":self.must_hold_teddy})
            self.time+=duration;self.frames.append({"time":self.time})
    sim=Sim();error=None
    try:namespace["run_guarded_contact_trial"](sim)
    except GateFailure as exc:error=str(exc)
    return {"logs":logs,"error":error,"fresh_calls":holder["fresh"],"correction_calls":holder["corrections"],
        "single_guard_and_single_baseline":len(holder["guards"])==1 and holder["guards"][0].baseline_count==1,
        "completed":sim.report.get("sequence_completed",False),"report":sim.report}


results={}
for name,statuses in (("immediate_accept",["candidate"]),("one_correction",["rejected","candidate"]),
                     ("two_corrections",["rejected","rejected","candidate"]),
                     ("exhausted_budget",["rejected","rejected","rejected"])):
    result=scenario(statuses);results[name]=result
    assert result["single_guard_and_single_baseline"]
    inserted=any(x.get("label")=="insert_teddy" for x in result["logs"])
    assert inserted==(name!="exhausted_budget")
    assert result["completed"]==inserted
    assert len(result["correction_calls"])==min(statuses.index("candidate") if "candidate" in statuses else 2,2)
    if inserted:
        ops=result["logs"];index=next(i for i,x in enumerate(ops) if x.get("label")=="insert_teddy")
        assert ops[index-1]["op"]=="fresh"
        assert ops[index]["must_hold_teddy"]
        assert all(not x["must_hold_teddy"] for x in ops[index+1:] if x["op"] in ("move","hold"))
for name,kwargs in (("single_over_30mm",{"distances":(.031,.01)}),
                    ("total_over_50mm",{"bad_total":.051}),
                    ("negative_distance",{"distances":(-.001,.01)}),
                    ("nan_distance",{"distances":(float("nan"),.01)}),
                    ("correction_rejected",{"correction_status":"rejected"})):
    result=scenario(["rejected","candidate"],**kwargs);results[name]=result
    assert result["error"] and not result["completed"]
    assert not any(x.get("label","").startswith("recenter") for x in result["logs"])
# Deliberately dishonest helper metadata: expose which layer verifies actual
# geometric travel, without claiming the real helper behaves this way.
counterexample=scenario(["rejected","candidate"],distances=(.01,.01),stage_xy=.08)
results["metadata_only_budget_counterexample"]={
    "controller_accepted_inconsistent_stage_target":counterexample["completed"],
    "declared_distance_m":.01,"supplied_stage_target_xy_m":[.08,0.],
    "interpretation":"Controller checks returned distance metadata; helper must derive it from stage geometry, or controller should cross-check actual target displacement."}
output=HERE/"runtime_ipc/recenter-flow-audit.json"
output.write_text(json.dumps({"runtime_source_sha256":hashlib.sha256(source.encode()).hexdigest(),
    "scope":"CPU stub verifies control flow, not physical simulation", "results":results},indent=2))
print(json.dumps({"cases":len(results),"output":str(output),
    "metadata_only_budget_counterexample":results["metadata_only_budget_counterexample"]},indent=2))
