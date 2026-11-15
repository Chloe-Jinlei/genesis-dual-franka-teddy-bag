"""CPU AST audit of corrected recenter bookkeeping and stage contracts."""
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
exec(compile(ast.Module(body=[method],type_ignores=[]),"isolated_guarded_contact_trial_v2","exec"),namespace)


def scenario(fresh_statuses, *, distances=(.025,.025), endpoint_errors=(0.,0.),
             correction_status="candidate", mode=None, strategy="direct", initial_fault=None):
    logs=[]; holder={"guards":[],"fresh":0,"corrections":[]}
    def stage(name,target):
        return {"name":name,"right_tcp_m":np.asarray(target).tolist(),
                "rotation_xyzw":[1.,0.,0.,0.],"duration_s":1.}
    first={"status":"candidate","id":"initial","stages_to_execute":[stage("height_at_current_xy",[0,0,1]),stage("translate_outside_mouth",[0,0,1])],"observed_time_s":0.}
    if strategy=="y_then_x":
        first["stages_to_execute"]=[stage("height_at_current_xy",[0,0,1.1]),
            stage("transport_y_at_current_x",[0,.2,1.1]),stage("translate_outside_mouth",[-.3,.2,1.1])]
    if initial_fault=="missing_waypoint": first["stages_to_execute"].pop(1)
    if initial_fault=="diagonal_first_leg": first["stages_to_execute"][1]["right_tcp_m"][0]=.01
    if initial_fault=="wrong_y": first["stages_to_execute"][1]["right_tcp_m"][1]=.21
    if initial_fault=="varying_height": first["stages_to_execute"][1]["right_tcp_m"][2]+=.01
    if initial_fault=="moving_height_xy": first["stages_to_execute"][0]["right_tcp_m"][0]=.01
    if initial_fault=="nonfinite": first["stages_to_execute"][-1]["right_tcp_m"][0]=float("nan")
    def initial(*args,**kwargs):return copy.deepcopy(first)
    def fresh(rest,state,config,selected,**kwargs):
        index=holder["fresh"];holder["fresh"]+=1
        logs.append({"op":"fresh","time":state["time"],"selected":selected["id"]})
        assert state["time"]>selected["observed_time_s"]
        status=fresh_statuses[min(index,len(fresh_statuses)-1)]
        return {"status":status,"insert_stage":stage("insert",state["tcp_right"]),"outside_tcp_m":state["tcp_right"].tolist()}
    def correction(rest,state,config,selected,*,attempt_index,cumulative_xy_m,**kwargs):
        # Mirror the real API contract: caller passes the last declared total,
        # then helper updates the previous leg using the new actual endpoint.
        if attempt_index==1:
            assert cumulative_xy_m==0.;before=0.
        else:
            assert abs(cumulative_xy_m-selected["cumulative_xy_m_after"])<1e-9
            before=selected["cumulative_xy_m_before"]+float(np.linalg.norm(
                state["tcp_right"][:2]-np.asarray(selected["observed_tcp_xy_m"])))
        holder["corrections"].append({"attempt":attempt_index,"caller_declared_total":cumulative_xy_m,"observed_before":before})
        distance=distances[attempt_index-1];delta=np.array([distance,0.])
        start=state["tcp_right"].copy();target=start.copy();target[:2]+=delta
        height=start.copy();new_total=before+distance
        if mode=="bad_target":target[0]=start[0]+.08
        if mode=="bad_height_xy":height[0]+=.001
        if mode=="bad_total":new_total=.051
        if mode=="bad_before":before+=.001;new_total=before+distance
        stages=[stage("height_at_current_xy",height),stage("translate_outside_mouth",target)]
        if mode=="premature_insert":stages[-1]["name"]="insert_candidate_after_fresh_observation"
        if mode=="third_stage":stages.append(stage("insert_teddy",target))
        return {"status":correction_status,"id":f"correction_{attempt_index}","observed_time_s":state["time"],
            "correction_distance_m":distance,"correction_xy_m":delta.tolist(),
            "cumulative_xy_m_before":before,"cumulative_xy_m_after":new_total,
            "observed_tcp_xy_m":start[:2].tolist(),"stages_to_execute":stages}
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
            self.rest={};self.frames=[{"time":0.,"tcp_right":np.array([0.,0.,1.])}]
            self.config={"task":{"contact_trial":{"transport_xy_strategy":strategy}}};self.out=Path("/tmp");self.report={};self.scene=object();self.time=0.
            self.bag=object();self.bear=object();self.left_grasp_material_ids=[0]
            self.robots={"left":object(),"right":object()};self.contact_guard_trace=[]
            self.must_hold_teddy=True;self.must_hold_bag=True
        def move_guarded(self,label,stage):
            logs.append({"op":"move","label":label,"guard_id":id(self.contact_guard),"must_hold_teddy":self.must_hold_teddy})
            self.time+=1.;position=np.asarray(stage["right_tcp_m"],float).copy()
            if label.startswith("recenter_teddy_xy_"):
                index=int(label.rsplit("_",1)[1])-1;position[0]+=endpoint_errors[index]
            self.frames.append({"time":self.time,"tcp_right":position})
        def hold(self,label,duration,**fingers):
            logs.append({"op":"hold","label":label,"guard_id":id(self.contact_guard),"must_hold_teddy":self.must_hold_teddy})
            self.time+=duration;self.frames.append({"time":self.time,"tcp_right":self.frames[-1]["tcp_right"].copy()})
    sim=Sim();error=None
    try:namespace["run_guarded_contact_trial"](sim)
    except GateFailure as exc:error=str(exc)
    return {"logs":logs,"error":error,"fresh_calls":holder["fresh"],"correction_calls":holder["corrections"],
        "single_guard_and_single_baseline":len(holder["guards"])==1 and holder["guards"][0].baseline_count==1,
        "completed":sim.report.get("sequence_completed",False),"report":sim.report}


results={}
success_cases={"dogleg_immediate":(["candidate"],{"strategy":"y_then_x"}),
    "dogleg_then_correction":(["rejected","candidate"],{"strategy":"y_then_x"}),
    "immediate_accept":(["candidate"],{}),
    "one_correction":(["rejected","candidate"],{}),
    "two_corrections":(["rejected","rejected","candidate"],{}),
    "first_tracking_undershoot_5micrometres":(["rejected","rejected","candidate"],{"endpoint_errors":(-5e-6,0.)}),
    "first_tracking_overshoot_5micrometres_with_budget_room":(["rejected","rejected","candidate"],{"endpoint_errors":(5e-6,0.),"distances":(.024,.024)})}
for name,(statuses,kwargs) in success_cases.items():
    result=scenario(statuses,**kwargs);results[name]=result
    assert result["completed"] and not result["error"],(name,result)
    assert result["single_guard_and_single_baseline"]
    ops=result["logs"];idx=next(i for i,x in enumerate(ops) if x.get("label")=="insert_teddy")
    assert ops[idx-1]["op"]=="fresh" and ops[idx]["must_hold_teddy"]
    assert all(not x["must_hold_teddy"] for x in ops[idx+1:] if x["op"] in ("move","hold"))
failure_cases={"exhausted_two_corrections":(["rejected"]*3,{}),
    "misreported_80mm_target":(["rejected","candidate"],{"mode":"bad_target","distances":(.01,.01)}),
    "height_stage_changes_xy":(["rejected","candidate"],{"mode":"bad_height_xy"}),
    "premature_insert_stage":(["rejected","candidate"],{"mode":"premature_insert"}),
    "third_preparation_stage":(["rejected","candidate"],{"mode":"third_stage"}),
    "single_over_30mm":(["rejected","candidate"],{"distances":(.031,.01)}),
    "declared_total_over_50mm":(["rejected","candidate"],{"mode":"bad_total"}),
    "dishonest_observed_before":(["rejected","candidate"],{"mode":"bad_before"}),
    "negative_distance":(["rejected","candidate"],{"distances":(-.001,.01)}),
    "nan_distance":(["rejected","candidate"],{"distances":(float("nan"),.01)}),
    "correction_rejected":(["rejected","candidate"],{"correction_status":"rejected"}),
    "actual_single_move_over_30mm":(["rejected","candidate"],{"endpoint_errors":(.006,0.)}),
    "actual_cumulative_over_50mm":(["rejected","rejected","candidate"],{"endpoint_errors":(0.,5e-6)})}
failure_cases["invalid_strategy"] = (["candidate"],{"strategy":"unknown"})
for fault in ["missing_waypoint","diagonal_first_leg","wrong_y","varying_height","moving_height_xy","nonfinite"]:
    failure_cases["initial_"+fault]=(["candidate"],{"strategy":"y_then_x","initial_fault":fault})
for name,(statuses,kwargs) in failure_cases.items():
    result=scenario(statuses,**kwargs);results[name]=result
    assert result["error"] and not result["completed"],(name,result)
    assert not any(x.get("label")=="insert_teddy" for x in result["logs"])
    if name not in ("exhausted_two_corrections","actual_single_move_over_30mm","actual_cumulative_over_50mm"):
        assert not any(x.get("label","").startswith("recenter") for x in result["logs"])

# Confirm the actual helper's saved successful proposal uses the reviewed API.
real=json.loads((HERE/"runs/two_panel_full_07/guarded_recenter_actual_endpoint_candidate.json").read_text())
assert [s["name"] for s in real["stages_to_execute"]]==["height_at_current_xy","translate_outside_mouth"]
target=np.asarray(real["stages_to_execute"][-1]["right_tcp_m"][:2])
start=np.asarray(real["observed_tcp_xy_m"])
assert np.linalg.norm(target-start-np.asarray(real["correction_xy_m"]))<1e-9
assert abs(np.linalg.norm(target-start)-real["correction_distance_m"])<1e-9
output=HERE/"runtime_ipc/recenter-flow-audit-v3.json"
output.write_text(json.dumps({"runtime_source_sha256":hashlib.sha256(source.encode()).hexdigest(),
    "scope":"CPU AST flow and arithmetic audit; no dynamics", "all_cases_passed":True,
    "actual_helper_stage_contract_compatible":True,"results":results},indent=2))
print(json.dumps({"cases":len(results),"all_cases_passed":True,"actual_helper_stage_contract_compatible":True,"output":str(output)},indent=2))
