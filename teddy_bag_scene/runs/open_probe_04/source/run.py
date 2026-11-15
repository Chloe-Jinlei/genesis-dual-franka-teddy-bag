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
        for f in ("scene.py","run.py","motion.py","validate.py","assets_builder.py","physics_config.py"):
            if (HERE/f).exists(): shutil.copy2(HERE/f,src/f)
        self.meta=json.loads((HERE/"assets/metadata.json").read_text())
        self.planner=Planner(config)
        self.qs={}
        self.finger=dict.fromkeys(("left","right"),.004)
        # Configuration is assigned once before integration. The initial hands
        # surround the two free handles; grasping occurs through finite-force PD.
        for side,group in (("left","grasp_front"),("right","grasp_back")):
            target=np.asarray(config["bag"]["position"])+self.meta["bag"]["grasp_centers_local_m"][group]
            q,err,angle=self.planner.solve(side,target,DOWN)
            if err>.001 or angle>.005: raise GateFailure(f"Initial {side} pose unreachable: {err}, {angle}")
            self.qs[side]=q
        initial_q={s:np.r_[q,self.finger[s],self.finger[s]] for s,q in self.qs.items()}
        self.scene,self.robots,self.bag,self.bear,self.cameras=build(config,
            viewer=args.viewer,width=args.width,height=args.height,initial_q=initial_q)
        self.dofs={s:np.arange(9) for s in self.robots}
        self.rest={"bag":positions(self.bag),"bear":positions(self.bear),
            "bag_faces":np.asarray(self.bag.elems),"bear_tets":np.asarray(self.bear.elems)}
        np.savez_compressed(self.out/"rest.npz",**self.rest)
        self.dt=config["simulation"]["dt"]
        self.time=0.
        self.frames=[]
        self.joint_trace=[]
        self.events=[]
        self.report={"status":"running","task_passed":False,"physics":physics_metadata(config,self.bag,self.bear)}
        self.report["initialization"]=self.scene.teddy_initialization_report
        self.report["shell_conversion"]=self.scene.teddy_shell_report
        self.coupling_trace=[]
        self.writer=imageio.get_writer(str(self.out/"simulation.mp4"),fps=20,codec="libx264",quality=7)
        self.last_frame_time=-1.
        self.wall=time.monotonic()
        self.last_progress=self.wall
        self.capture("initial",True)

    def capture(self, phase, image=False):
        bp, tp=positions(self.bag), positions(self.bear)
        if not np.isfinite(bp).all() or not np.isfinite(tp).all(): raise GateFailure("Nonfinite soft-body state")
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
        if min(bp[:,2].min(),tp[:,2].min())<self.config["table"]["height"]-.005:
            raise GateFailure("A soft body has penetrated the tabletop by more than 5 mm")
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

    def hold(self,label,duration,**fingers):
        finish=self.finger|fingers
        self.execute(label,hold_trajectory(self.qs,duration,self.finger,finish))

    def move(self,label,targets,duration=2.5):
        plan=self.planner.trajectory(self.qs,targets,duration=duration,
            finger_start=self.finger,finger_end=self.finger,clearance_margin=.005)
        self.execute(label,plan)

    def near_material(self,entity,side,radius=.028):
        p=positions(entity)
        distances=np.linalg.norm(p-tcp(self.robots[side]),axis=1)
        return {"near_vertices":int(np.sum(distances<radius)),"nearest_m":float(distances.min())}

    def run(self):
        if self.args.task=="settle":
            self.hold("settle",self.args.settle_seconds)
            return
        self.hold("close_both_handles",.20,left=0.,right=0.)
        evidence={s:self.near_material(self.bag,s) for s in self.robots}
        self.report["initial_handle_proximity"]=evidence
        if any(x["near_vertices"]<2 for x in evidence.values()):
            raise GateFailure("Handle slipped away before physical closure")
        self.move("spread_bag",{"left":[-.25,-.16,1.23],"right":[-.13,.15,1.23]},2.5)
        self.report["spread_handle_proximity"]={s:self.near_material(self.bag,s) for s in self.robots}
        if any(x["near_vertices"]<2 for x in self.report["spread_handle_proximity"].values()):
            raise GateFailure("One of the handles is not held after spreading")
        if self.args.task=="open": return
        self.hold("release_right_handle",.4,right=.04)
        self.move("right_clear_handle",{"right":[.12,.20,1.32]},2.)
        # Follow the actual soft upper arm, not a scripted teddy pose.
        rest=self.rest["bear"]-np.asarray(self.config["bear"]["position"])
        center=np.asarray(self.meta["bear"]["grasp_arm_right_local_m"])
        arm=np.where(np.linalg.norm(rest-center,axis=1)<.032)[0]
        if len(arm)<3: raise GateFailure("No narrow teddy upper-arm grasp region")
        target=positions(self.bear)[arm].mean(0)
        target[2]=max(target[2],self.config["table"]["height"]+.025)
        self.report["teddy_grasp_target"]=target.tolist()
        self.move("teddy_pregrasp",{"right":target+[0,0,.15]},3.)
        self.move("teddy_grasp_approach",{"right":target},2.)
        self.hold("close_teddy_gripper",.6,right=0.)
        self.move("lift_teddy",{"right":[target[0],target[1],1.37]},4.)
        bear=positions(self.bear)
        self.report["lift_minimum_z"]=float(bear[:,2].min())
        if bear[:,2].min()<self.config["table"]["height"]+.04:
            raise GateFailure("Teddy did not lift clear of the table; real friction grasp failed")
        self.move("transport_teddy",{"right":[-.10,.04,1.37]},5.)
        self.move("insert_teddy",{"right":[-.10,.04,1.10]},4.)
        self.hold("release_teddy",.5,right=.04)
        self.move("withdraw_right_arm",{"right":[-.10,.04,1.38]},3.)
        self.hold("final_settle",2.)
        self.report["sequence_completed"]=True

    def finish(self,error=None):
        self.writer.close()
        self.report.update(status="failed" if error else "completed",error=error,
            simulated_seconds=self.time,wall_seconds=time.monotonic()-self.wall,
            events=self.events)
        np.savez_compressed(self.out/"states.npz",**{k:np.asarray([f[k] for f in self.frames]) for k in self.frames[0]})
        np.save(self.out/"joint_trace.npy",np.asarray(self.joint_trace))
        (self.out/"report.json").write_text(json.dumps(self.report,indent=2))
        (self.out/"coupling_trace.json").write_text(json.dumps(self.coupling_trace,indent=2))
        print("OUTPUT "+str(self.out),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=("settle","open","full"),default="full")
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
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        exp.finish(error)
    if error: raise SystemExit(1)


if __name__=="__main__": main()
