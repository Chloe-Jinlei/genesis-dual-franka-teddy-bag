"""Bounded CPU-only high transport test from the actual full07 lift state."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUN = HERE / "runs/two_panel_full_07"
spec = importlib.util.spec_from_file_location("frozen_high_transport_motion", RUN / "source/motion.py")
motion = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = motion
spec.loader.exec_module(motion)


def clearance_actual_fingers(planner, qs, fingers):
    for side in ("left", "right"):
        for i, value in enumerate(fingers[side], 1):
            planner.d.qpos[planner.m.joint(f"{side}_finger_joint{i}").qposadr[0]] = value
    # None preserves the individually assigned actual finger joints.
    return planner.clearance(qs, None)


def main():
    config = json.loads((RUN / "config.json").read_text())
    state = np.load(RUN / "lift_teddy.npz")
    previous = json.loads((RUN / "guarded_contact_candidate.json").read_text())["route"]
    base_q = {s: state["q_"+s][:7].copy() for s in ("left", "right")}
    actual_fingers = {s: state["q_"+s][7:].copy() for s in ("left", "right")}
    paired = {s: float(q.mean()) for s,q in actual_fingers.items()}
    tcp0 = state["tcp_right"]
    bag_max = float(state["bag"][:,2].max())
    bear_min = float(state["bear"][:,2].min())
    rows = []
    for clearance in (.02, .04):
        planner = motion.Planner(config)
        _, rotation = planner.fk("right", base_q["right"])
        height = bag_max + clearance + float(tcp0[2]) - bear_min
        raised = tcp0.copy(); raised[2] = height
        outside = np.asarray(previous["outside_tcp_m"]); outside[2] = height
        inserted = np.asarray(previous["insert_tcp_m"])
        row = {"bag_material_clearance_m": clearance, "bag_max_z_m": bag_max,
            "bear_initial_min_z_m": bear_min, "required_tcp_z_m": height,
            "lift_from_observed_tcp_m": height-float(tcp0[2]),
            "planned_bear_min_z_m": bag_max+clearance,
            "rotation_xyzw": rotation.as_quat().tolist(), "stages": [],
            "direct_endpoint_ik": {}}
        for label, target in (("raise_at_current_xy",raised),("transport_above_all_bag_material",outside)):
            q,error,angle = planner.solve("right",target,rotation,base_q["right"],max_nfev=250)
            row["direct_endpoint_ik"][label] = {"target_m":target.tolist(),
                "position_error_m":error,"rotation_error_rad":angle,"q":q.tolist(),
                "within_0p5mm_0p002rad": bool(error<=.0005 and angle<=.002)}
        qs = {s:q.copy() for s,q in base_q.items()}
        try:
            for label,target,duration in (("raise_at_current_xy",raised,3.),
                    ("transport_above_all_bag_material",outside,6.),
                    ("insert_from_high_transport",inserted,8.)):
                plan = planner.trajectory(qs,{"right":target},rotations={"right":rotation},
                    duration=duration,finger_start=paired,finger_end=paired,
                    max_joint_velocity=.35,max_joint_acceleration=.5,clearance_margin=.008)
                minima = {}
                for time in np.linspace(0,plan.duration,101):
                    point=plan.sample(float(time))
                    checks=clearance_actual_fingers(planner,point["q"],actual_fingers)
                    for name,value in checks.items():
                        if name not in minima or value["distance"]<minima[name]["distance"]:
                            minima[name]={**value,"time_s":float(time)}
                        if value["distance"] < .008:
                            raise ValueError(f"{label}: actual individual fingers have {name} clearance {value['distance']}")
                qs=plan.end_q()
                row["stages"].append({"name":label,"target_m":target.tolist(),
                    "duration_s":plan.duration,"minimum_actual_finger_clearance_by_category":minima,
                    "analytic_joint_extrema":plan.joint_extrema(),
                    "q_end":{s:q.tolist() for s,q in qs.items()}})
            release_min = .2
            for fraction in np.linspace(0,1,21):
                fingers={s:q.copy() for s,q in actual_fingers.items()}
                fingers["right"]=(1-fraction)*actual_fingers["right"]+fraction*.04
                margin=min(v["distance"] for v in clearance_actual_fingers(planner,qs,fingers).values())
                release_min=min(release_min,margin)
                if margin < .008: raise ValueError(f"release actual-finger clearance {margin}")
            opened=dict(paired);opened["right"]=.04
            plan=planner.trajectory(qs,{"right":outside},rotations={"right":rotation},duration=8.,
                finger_start=opened,finger_end=opened,max_joint_velocity=.35,
                max_joint_acceleration=.5,clearance_margin=.008)
            withdraw_min=.2
            for time in np.linspace(0,plan.duration,101):
                fingers={s:q.copy() for s,q in actual_fingers.items()};fingers["right"]=np.full(2,.04)
                margin=min(v["distance"] for v in clearance_actual_fingers(planner,plan.sample(float(time))["q"],fingers).values())
                withdraw_min=min(withdraw_min,margin)
                if margin < .008: raise ValueError(f"withdraw actual-finger clearance {margin}")
            row.update(status="hypothetical_robot_route_passed",release_minimum_clearance_m=release_min,
                withdraw_minimum_clearance_m=withdraw_min,withdraw_duration_s=plan.duration)
        except ValueError as error:
            row.update(status="rejected",reason=str(error))
        rows.append(row)
        print(json.dumps({k:row[k] for k in ("bag_material_clearance_m","required_tcp_z_m","status")} |
                         {"reason":row.get("reason"),"endpoint_ik":row["direct_endpoint_ik"]}),flush=True)
    result={"source_state":str(RUN/"lift_teddy.npz"),"source_time_s":float(state["time"]),
        "status":"bounded_hypothetical_robot_reach_study","rows":rows,
        "minimum_tested_feasible_clearance_m":next((r["bag_material_clearance_m"] for r in rows if r["status"]=="hypothetical_robot_route_passed"),None),
        "limits":["No Genesis initialization, GPU work or runtime/source/config changes.",
            "Observed bear shape is rigidly translated for the height bound; deformation and inertial sag can change its actual minimum height.",
            "Includes all observed cloth vertices, both handles and body, in the transport-height bound.",
            "Actual unequal finger joint values are checked individually at 101 path samples and 21 release samples.",
            "Robot self-pair model omits opposing fingers of the same hand; the observed positive gap is retained until release.",
            "Existing initial-section policy applies only at the observed lift shape; this does not guarantee passage or strap clearance during descent."]}
    output=RUN/"independent-high-transport-preflight.json"
    output.write_text(json.dumps(result,indent=2));print(str(output))


if __name__=="__main__":main()
