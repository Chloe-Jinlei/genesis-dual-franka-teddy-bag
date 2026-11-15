"""Run and record the native-contact teddy bag experiment with fail-closed gates."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import traceback

from scene import HERE, build, read_config, positions, physics_metadata
from motion import Planner, DOWN, TCP, hold_trajectory, command_pd
from physics_config import coupling_diagnostics
from contact_grasp import cloth_grasp_frame
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as imageio


class GateFailure(RuntimeError):
    pass


def tcp(robot):
    hand = robot.get_link("hand")
    quat = hand.get_quat().cpu().numpy().reshape(4)
    return hand.get_pos().cpu().numpy().reshape(3) + Rotation.from_quat(quat[[1,2,3,0]]).apply(TCP)


class Experiment:
    def __init__(self, config, args):
        self.config, self.args = config, args
        self.out = HERE/"runs"/(args.output or datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/"config.json").write_text(json.dumps(config,indent=2))
        src=self.out/"source"
        src.mkdir()
        for f in ("scene.py","run.py","motion.py","validate.py","assets_builder.py","two_panel_builder.py","physics_config.py","contact_grasp.py","insertion_candidates.py","remesh_bear.py","physical_monitor.py"):
            if (HERE/f).exists(): shutil.copy2(HERE/f,src/f)
        self.meta=json.loads((HERE/"assets"/config["bag"].get("metadata","metadata.json")).read_text())
        mesh_audit=self.meta["bag"]["mesh"]
        if mesh_audit["connected_components"]!=1 or mesh_audit["nonmanifold_edges"]!=0:
            raise GateFailure("Bag must be one connected manifold cloth shell before simulation")
        asset_snapshot=src/"assets"
        asset_snapshot.mkdir()
        bag_file=config["bag"].get("mesh","two_panel_bag.obj")
        bear_file=config["bear"].get("mesh","bear_physics.obj")
        for name in (bag_file,Path(bag_file).with_suffix(".mtl").name,
                     config["bag"].get("npz_file","two_panel_bag.npz"),
                     config["bag"].get("metadata","metadata_two_panel.json"),
                     bear_file,Path(bear_file).with_suffix(".mtl").name,"teddy_fur.png","cotton_canvas.png","panda_tactile.xml"):
            shutil.copy2(HERE/"assets"/name,asset_snapshot/name)
        self.planner=Planner(config)
        bear_grasp_quat=config.get("task",{}).get("bear_grasp_rotation_xyzw")
        self.bear_grasp_rotation=Rotation.from_quat(bear_grasp_quat) if bear_grasp_quat is not None else DOWN
        self.qs={}
        self.finger=dict.fromkeys(("left","right"),.001)
        self.rotations=dict.fromkeys(("left","right"),DOWN)
        bag_source=np.load(HERE/"assets"/config["bag"]["npz_file"])
        grasp_frames={}
        # Configuration is assigned once before integration. The initial hands
        # surround the two free handles; grasping occurs through finite-force PD.
        for side,group in (("left","grasp_front"),("right","grasp_back")):
            grasp_spec=config.get("task",{}).get("handle_grasps",{}).get(side,{})
            group=grasp_spec.get("group",group)
            frame=cloth_grasp_frame(bag_source["points"],bag_source["faces"],bag_source[group],
                shell_radius=config["bag"]["thickness"],
                preferred_approach=grasp_spec.get("preferred_approach",[0,0,-1]))
            grasp_frames[side]=frame.as_dict()
            self.rotations[side]=frame.rotation
            self.finger[side]=max(.001,frame.minimum_half_gap)
            target=np.asarray(config["bag"]["position"])+frame.position
            if args.task in ("grasp","grasp_rotate") and side=="right":
                target=np.asarray(config["bear"]["position"])+config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034])
                target[2]+=.09
                self.finger[side]=.04
                self.rotations[side]=self.bear_grasp_rotation
            q,err,angle=self.planner.solve(side,target,self.rotations[side])
            if err>.001 or angle>.005: raise GateFailure(f"Initial {side} pose unreachable: {err}, {angle}")
            self.qs[side]=q
        initial_q={s:np.r_[q,self.finger[s],self.finger[s]] for s,q in self.qs.items()}
        self.scene,self.robots,self.bag,self.bear,self.cameras=build(config,
            viewer=args.viewer,width=args.width,height=args.height,initial_q=initial_q)
        self.dofs={s:np.arange(9) for s in self.robots}
        self.rest={"bag":positions(self.bag),"bear":positions(self.bear),
            "bag_faces":np.asarray(self.bag.elems),"bear_tets":np.asarray(self.bear.elems)}
        np.savez_compressed(self.out/"rest.npz",**self.rest)
        from validate import signed_volumes
        self.rest_tetra_volumes=signed_volumes(self.rest["bear"],self.rest["bear_tets"])
        self.dt=config["simulation"]["dt"]
        self.time=0.
        self.frames=[]
        self.joint_trace=[]
        self.soft_metrics=[]
        self.must_hold_teddy=False
        from physical_monitor import PhysicalMonitor
        self.physical_monitor=PhysicalMonitor(self.rest,config)
        self.events=[]
        self.report={"status":"running","task_passed":False,"physics":physics_metadata(config,self.bag,self.bear)}
        self.report["initialization"]=self.scene.teddy_initialization_report
        self.report["initial_cloth_grasp_geometry"]=grasp_frames
        self.report["shell_conversion"]=self.scene.teddy_shell_report
        self.report["stc_reaction_feedback"]=self.scene.teddy_stc_feedback_report
        self.report["rigid_contact_pairing"]=self.scene.teddy_rigid_contact_report
        self.coupling_trace=[]
        self.writer=imageio.get_writer(str(self.out/"simulation.mp4"),fps=20,codec="libx264",quality=7)
        self.last_frame_time=-1.
        self.wall=time.monotonic()
        self.last_progress=self.wall
        self.observe_physics()
        self.capture("initial",True)

    def observe_physics(self):
        bag,bear=positions(self.bag),positions(self.bear)
        row=self.physical_monitor.measure(self.time,bag,bear)
        self.soft_metrics.append(row)
        error=None
        if not bool(row[5]): error="Nonfinite soft-body state"
        elif row[1]<.05: error=f"Teddy element excessively compressed or inverted: volume ratio {row[1]:.4f}"
        elif max(row[2:4])>.005: error="A soft body penetrated the actual tabletop footprint by over 5 mm"
        elif row[4]>1.15: error=f"Cloth edge stretch exceeded 15 percent: ratio {row[4]:.4f}"
        if error:
            np.savez_compressed(self.out/"failed_soft_state.npz",time=self.time,bag=bag,bear=bear,metrics=row)
            raise GateFailure(error)

    def capture(self, phase, image=False):
        bp, tp=positions(self.bag), positions(self.bear)
        if not np.isfinite(bp).all() or not np.isfinite(tp).all(): raise GateFailure("Nonfinite soft-body state")
        from validate import signed_volumes
        minimum_volume_ratio=float(np.min(signed_volumes(tp,self.rest["bear_tets"])/self.rest_tetra_volumes))
        self.report["minimum_observed_tetra_volume_ratio"]=min(
            self.report.get("minimum_observed_tetra_volume_ratio",1.),minimum_volume_ratio)
        row={"time":self.time,"bag":bp,"bear":tp}
        for side,robot in self.robots.items():
            row["q_"+side]=robot.get_qpos().cpu().numpy().copy()
            row["tcp_"+side]=tcp(robot)
            row["force_"+side]=robot.get_dofs_control_force().cpu().numpy().copy()
        self.frames.append(row)
        self.coupling_trace.append({"time":self.time,**coupling_diagnostics(self.scene)})
        rgb=np.asarray(self.cameras["overview"].render()[0])
        self.writer.append_data(rgb)
        if image:
            Image.fromarray(rgb).save(self.out/(phase+".png"))
            detail=np.asarray(self.cameras["detail"].render()[0])
            Image.fromarray(detail).save(self.out/(phase+"-detail.png"))
            np.savez_compressed(self.out/(phase+".npz"),**row)
        self.last_frame_time=self.time
        if minimum_volume_ratio<.05:
            raise GateFailure(f"Teddy element excessively compressed or inverted: minimum volume ratio {minimum_volume_ratio:.4f}")
        if self.must_hold_teddy and np.linalg.norm(tp-tcp(self.robots["right"]),axis=1).min()>.045:
            raise GateFailure("Teddy slipped out of the right finger pads during transport")
        return row

    def execute(self,label,plan):
        print(f"PHASE {label} duration={plan.duration:.3f}s",flush=True)
        self.events.append({"phase":label,"start":self.time,"duration":plan.duration})
        for i in range(1,int(np.ceil(plan.duration/self.dt))+1):
            sample=plan.sample(min(i*self.dt,plan.duration))
            command_pd(self.robots,self.dofs,sample)
            self.scene.step()
            self.time+=self.dt
            self.joint_trace.append(np.r_[self.time,*[r.get_qpos().cpu().numpy().reshape(-1) for r in self.robots.values()]])
            self.observe_physics()
            if self.time-self.last_frame_time>=.0499: self.capture(label)
            now=time.monotonic()
            if now-self.last_progress>15:
                print(f"PROGRESS {label} simulation={self.time:.2f}s wall={now-self.wall:.1f}s",flush=True)
                self.last_progress=now
            if self.args.max_seconds and self.time>=self.args.max_seconds:
                raise GateFailure("Stopped at the requested diagnostic duration; task not complete")
        self.qs=plan.end_q()
        self.finger=plan.finger_end.copy()
        self.capture(label,True)
        (self.out/"events.json").write_text(json.dumps(self.events,indent=2))
        (self.out/"latest_contact_diagnostics.json").write_text(json.dumps(self.coupling_trace[-1],indent=2))

    def hold(self,label,duration,**fingers):
        finish=self.finger|fingers
        self.execute(label,hold_trajectory(self.qs,duration,self.finger,finish))

    def close_teddy_gripper(self):
        task=self.config.get("task",{})
        force=float(task.get("bear_finger_force_limit",4.))
        gap=float(task.get("bear_finger_target",.027))
        self.robots["right"].set_dofs_force_range(
            np.array([-force,-force]),np.array([force,force]),dofs_idx_local=np.array([7,8]))
        self.report["teddy_grip_limits"]={"force_per_finger_n":force,"target_per_finger_m":gap}
        self.hold("close_teddy_gripper",1.,right=gap)
        self.must_hold_teddy=True

    def move(self,label,targets,duration=2.5,rotations=None):
        finish_rotations={s:self.rotations[s] for s in targets}
        if rotations: finish_rotations.update(rotations)
        speed=float(self.config.get("task",{}).get("motion_speed_scale",1.))
        plan=self.planner.trajectory(self.qs,targets,duration=duration/speed,
            rotations=finish_rotations,finger_start=self.finger,
            finger_end=self.finger,clearance_margin=.005,
            max_joint_velocity=.55*speed,max_joint_acceleration=speed**2)
        self.execute(label,plan)
        self.rotations.update(finish_rotations)

    def near_material(self,entity,side,radius=.028):
        p=positions(entity)
        distances=np.linalg.norm(p-tcp(self.robots[side]),axis=1)
        return {"near_vertices":int(np.sum(distances<radius)),"nearest_m":float(distances.min())}

    def run(self):
        if self.args.task=="settle":
            self.hold("settle",self.args.settle_seconds)
            return
        if self.args.task in ("grasp","grasp_rotate"):
            self.hold("grasp_probe_settle",.5,left=0.)
            center=np.asarray(self.config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034]))
            rest=self.rest["bear"]-np.asarray(self.config["bear"]["position"])
            arm=np.where(np.linalg.norm(rest-center,axis=1)<.032)[0]
            target=np.asarray(self.config["bear"]["position"])+center+(positions(self.bear)[arm]-self.rest["bear"][arm]).mean(0)
            target[2]=max(target[2],self.config["table"]["height"]+.025)
            self.move("teddy_grasp_approach",{"right":target},1.5,
                rotations={"right":self.bear_grasp_rotation})
            self.close_teddy_gripper()
            self.move("lift_teddy",{"right":[target[0],target[1],1.17]},3.)
            self.hold("grasp_hold",float(self.config.get("task",{}).get("grasp_probe_hold_seconds",1.)))
            if self.args.task=="grasp_rotate":
                self.move("grasp_rotate",{"right":[target[0],target[1],1.17]},6.,
                    rotations={"right":Rotation.from_euler("x",90,degrees=True)})
                self.hold("rotated_grasp_hold",1.)
            self.report["grasp_probe_lift_minimum_z"]=float(positions(self.bear)[:,2].min())
            if positions(self.bear)[:,2].min()<self.config["table"]["height"]+.04:
                raise GateFailure("Teddy friction grasp did not lift it clear of the table")
            self.report["grasp_probe_lifted"]=True
            return
        self.hold("close_both_handles",float(self.config.get("task",{}).get("handle_close_seconds",.5)),left=0.,right=0.)
        evidence={s:self.near_material(self.bag,s) for s in self.robots}
        self.report["initial_handle_proximity"]=evidence
        if any(x["near_vertices"]<2 for x in evidence.values()):
            raise GateFailure("Handle slipped away before physical closure")
        self.hold("drape_under_gravity",.6)
        self.report["settled_handle_proximity"]={s:self.near_material(self.bag,s) for s in self.robots}
        if any(v["nearest_m"]>.018 for v in self.report["settled_handle_proximity"].values()):
            raise GateFailure("A handle slipped out of the finger pads while settling")
        open_targets=self.config.get("task",{}).get("open_targets",{"left":[-.25,-.13,1.25],"right":[-.10,.13,1.25]})
        self.move("spread_bag",open_targets,2.5)
        self.report["spread_handle_proximity"]={s:self.near_material(self.bag,s) for s in self.robots}
        if any(x["near_vertices"]<2 for x in self.report["spread_handle_proximity"].values()):
            raise GateFailure("One of the handles is not held after spreading")
        if self.args.task=="open": return
        self.hold("release_right_handle",.4,right=.04)
        self.move("right_clear_handle",{"right":[.12,.20,1.32]},2.,rotations={"right":DOWN})
        # Follow the actual soft upper arm, not a scripted teddy pose.
        rest=self.rest["bear"]-np.asarray(self.config["bear"]["position"])
        center=np.asarray(self.config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034]))
        arm=np.where(np.linalg.norm(rest-center,axis=1)<.032)[0]
        if len(arm)<3: raise GateFailure("No narrow teddy upper-arm grasp region")
        target=np.asarray(self.config["bear"]["position"])+center+(positions(self.bear)[arm]-self.rest["bear"][arm]).mean(0)
        target[2]=max(target[2],self.config["table"]["height"]+.025)
        self.report["teddy_grasp_target"]=target.tolist()
        self.move("teddy_pregrasp",{"right":target+[0,0,.15]},3.,
            rotations={"right":self.bear_grasp_rotation})
        self.move("teddy_grasp_approach",{"right":target},2.,
            rotations={"right":self.bear_grasp_rotation})
        self.close_teddy_gripper()
        self.move("lift_teddy",{"right":[target[0],target[1],1.37]},4.)
        bear=positions(self.bear)
        self.report["lift_minimum_z"]=float(bear[:,2].min())
        if bear[:,2].min()<self.config["table"]["height"]+.04:
            raise GateFailure("Teddy did not lift clear of the table; real friction grasp failed")
        from insertion_candidates import choose_insertion_route,reobserve_insertion
        choice=choose_insertion_route(self.rest,self.frames[-1],self.config,
            output=self.out/"insertion_candidates.json")
        route=choice["route"]
        self.report["selected_insertion_candidate"]=route
        phase_names={"height_at_current_xy":"adjust_teddy_height",
            "rotate_held_teddy":"orient_teddy","translate_outside_mouth":"transport_teddy"}
        for stage in route["stages"]:
            if stage["name"]=="insert_candidate": break
            self.move(phase_names[stage["name"]],{"right":stage["right_tcp_m"]},
                stage["duration_s"],rotations={"right":Rotation.from_quat(stage["rotation_xyzw"])})
        fresh=reobserve_insertion(self.rest,self.frames[-1],self.config,route,
            output=self.out/"reobserved_insertion_candidates.json")
        self.report["reobserved_insertion"]=fresh
        insert=fresh["insert_stage"]
        self.move("insert_teddy",{"right":insert["right_tcp_m"]},insert["duration_s"],
            rotations={"right":Rotation.from_quat(insert["rotation_xyzw"])})
        self.must_hold_teddy=False
        self.hold("release_teddy",.5,right=.04)
        self.move("withdraw_right_arm",{"right":fresh["outside_tcp_m"]},3.)
        self.hold("final_settle",2.)
        self.report["sequence_completed"]=True

    def finish(self,error=None):
        self.writer.close()
        self.report.update(status="failed" if error else "completed",error=error,
            simulated_seconds=self.time,wall_seconds=time.monotonic()-self.wall,
            events=self.events)
        frames={k:np.asarray([f[k] for f in self.frames]) for k in self.frames[0]}
        trace=np.asarray(self.joint_trace)
        np.savez_compressed(self.out/"states.npz",**frames)
        np.save(self.out/"joint_trace.npy",trace)
        np.save(self.out/"soft_metrics.npy",np.asarray(self.soft_metrics))
        (self.out/"soft_metrics_columns.json").write_text(json.dumps(self.physical_monitor.columns,indent=2))
        (self.out/"events.json").write_text(json.dumps(self.events,indent=2))
        if len(trace)>3:
            from validate import audit_arrays
            try:
                audit=audit_arrays(self.rest,frames,self.config,
                    task_completed=bool(self.report.get("sequence_completed",False)),
                    joint_trace=trace,events=self.events,soft_metrics=np.asarray(self.soft_metrics))
                (self.out/"validation.json").write_text(json.dumps(audit,indent=2))
                self.report["task_passed"]=bool(audit.get("task_success",False))
                if self.args.task in ("grasp","grasp_rotate"):
                    checks=audit.get("checks",{})
                    self.report["grasp_probe_passed"]=bool(self.report.get("grasp_probe_lifted",False)
                        and all(checks.get(k,False) for k in ("finite_saved_states",
                        "no_tetrahedron_inversion","table_penetration_below_2mm",
                        "joint_motion_within_configured_limits","motion_sample_interval_at_most_20ms")))
            except Exception as exc:
                self.report["validation_error"]=f"{type(exc).__name__}: {exc}"
        (self.out/"report.json").write_text(json.dumps(self.report,indent=2))
        (self.out/"coupling_trace.json").write_text(json.dumps(self.coupling_trace,indent=2))
        print("OUTPUT "+str(self.out),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=("settle","open","grasp","grasp_rotate","full"),default="full")
    ap.add_argument("--viewer",action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--output")
    ap.add_argument("--width",type=int,default=960)
    ap.add_argument("--height",type=int,default=720)
    ap.add_argument("--max-seconds",type=float,default=0)
    ap.add_argument("--settle-seconds",type=float,default=.5)
    args=ap.parse_args()
    exp=Experiment(read_config(args.config),args)
    error=None
    try:
        exp.run()
    except (Exception, KeyboardInterrupt) as exc:
        error=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        exp.finish(error)
    if error: raise SystemExit(1)


if __name__=="__main__": main()
