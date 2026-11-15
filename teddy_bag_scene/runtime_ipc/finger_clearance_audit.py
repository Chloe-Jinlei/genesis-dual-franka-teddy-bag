"""CPU-only opposing-finger geometry audit; never initializes Genesis."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from motion import Planner


def audit(run: Path):
    config = json.loads((run / "config.json").read_text())
    planner = Planner(config)
    asset = Path(config["robots"].get("mjcf_path", HERE / "assets/panda_tactile.xml"))
    trace = np.load(run / "joint_trace.npy")
    groups = {
        side: {finger: [g for g in planner.geoms[side]
                        if planner.body_name(g) == f"{side}_{finger}_finger"]
               for finger in ("left", "right")}
        for side in ("left", "right")
    }

    def distance(qs):
        for side in groups:
            planner.d.qpos[planner.qindices[side]] = qs[side][:7]
            for finger in (1, 2):
                planner.d.qpos[planner.m.joint(f"{side}_finger_joint{finger}").qposadr[0]] = qs[side][6 + finger]
        mujoco.mj_forward(planner.m, planner.d)
        result = {}
        for side in groups:
            value, ga, gb = min((float(mujoco.mj_geomDistance(planner.m, planner.d, a, b, .2, None)), a, b)
                               for a in groups[side]["left"] for b in groups[side]["right"])
            result[side] = {"distance_m": value, "geom_ids": [ga, gb],
                            "finger_q_m": qs[side][-2:].tolist()}
        return result

    stages = {}
    for name in ("initial", "close_both_handles", "drape_under_gravity", "spread_bag",
                 "right_clear_handle", "close_teddy_gripper", "orient_teddy"):
        path = run / f"{name}.npz"
        if path.exists():
            data = np.load(path)
            stages[name] = {"time": float(data["time"]),
                            "distances": distance({s: data[f"q_{s}"] for s in groups})}
    zero = distance({s: np.r_[planner.home, [0., 0.]] for s in groups})
    minima = {}
    for side, offset in (("left", 8), ("right", 17)):
        # Fingers translate along opposing parallel axes, hence relative distance
        # depends only on q1+q2. Both actual q are nonnegative in this recording.
        index = int(np.argmin(trace[:, offset] + trace[:, offset + 1]))
        row = trace[index]
        minima[side] = {"time": float(row[0]),
                        **distance({"left": row[1:10], "right": row[10:19]})[side]}

    # A positive separating interval is a rigorous nonintersection witness for
    # the complete collision union, even for slightly nonrigid IPC transforms.
    finger_mesh = HERE.parent / "genesis-world/genesis/assets/xml/franka_emika_panda/assets/finger_0.obj"
    vertices = [np.asarray(trimesh.load(finger_mesh, process=False).vertices)]
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=3)))
    for gid in groups["left"]["left"]:
        if planner.m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, planner.m.geom_quat[gid])
            vertices.append((signs * planner.m.geom_size[gid]) @ matrix.reshape(3, 3).T + planner.m.geom_pos[gid])
    vertices = np.concatenate(vertices)
    proxy = {s: {"min_separating_interval_m": float("inf")} for s in groups}
    coupling = json.loads((run / "coupling_trace.json").read_text())
    for frame in coupling:
        for side in groups:
            links = {link["link"]: link for link in frame["links"] if link["entity"] == f"{side}_franka"}
            left, right = links["left_finger"], links["right_finger"]
            lt, rt = np.asarray(left["ipc_transform"]), np.asarray(right["ipc_transform"])
            axis = np.asarray(left["genesis_current_rotation_matrix"])[:, 1]
            lv = vertices @ lt[:3, :3].T + lt[:3, 3]
            rv = vertices @ rt[:3, :3].T + rt[:3, 3]
            interval = float((lv @ axis).min() - (rv @ axis).max())
            if interval < proxy[side]["min_separating_interval_m"]:
                proxy[side] = {"min_separating_interval_m": interval, "time": frame["time"]}
    result = {
        "method": "MuJoCo signed distance over 36 opposing-finger collision-geometry pairs at stage frames and minimum actual q-sum; full IPC geometry separating-plane witness at every saved coupling frame.",
        "asset": str(asset), "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "joint_trace_rows": len(trace), "coupling_samples": len(coupling),
        "central_pad_inner_face_local_y_m": 0.,
        "pad_modification": "Half-size Y=.0055 and center Y=.0055: inward face is 0, not -0.0015 m.",
        "zero_opening_distances": zero,
        "stages": stages, "full_trace_minima": minima, "ipc_geometry_separation": proxy,
        "all_checked_opposing_fingers_disjoint": all(x["distance_m"] > 0 for x in minima.values()) and all(x["min_separating_interval_m"] > 0 for x in proxy.values()),
        "cloth_clearance": "A flat 0.6 mm shell fits nominally at q1=q2=0.3 mm. IPC barrier activation and proxy compliance increase observed holding gap; they do not imply rigid pad overlap.",
        "limitations": "IPC transform samples are about 20 Hz, not every solver step. The separating interval is a nonintersection witness, not an exact Euclidean distance. No actual native rigid contact forces were recorded by full05.",
        "neutral_collision_recommendation": {"scope": "scene RigidOptions", "enable_neutral_collision": True, "enable_self_collision": True, "enable_adjacent_collision": False},
    }
    output = run / "independent-finger-clearance.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "minima": minima, "ipc_separation": proxy,
                      "all_checked_opposing_fingers_disjoint": result["all_checked_opposing_fingers_disjoint"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    audit(parser.parse_args().run.resolve())
