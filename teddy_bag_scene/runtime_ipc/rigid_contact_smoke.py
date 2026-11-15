"""Bounded native IPC + Genesis rigid-contact regression, without rendering."""
from pathlib import Path
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "vendor")]
os.environ.setdefault("GS_CACHE_FILE_PATH", str(HERE / ".cache/genesis"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
import numpy as np
import genesis as gs
from physics_config import install_exact_stc_feedback, install_rigid_collision_fallback


def main():
    wall = time.perf_counter()
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    dt, duration, top = .005, .4, .1
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=1),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=.001, contact_resistance=1e7, two_way_coupling=True,
            enable_rigid_rigid_contact=False, enable_rigid_ground_contact=False,
            constraint_strength_translation=100., constraint_strength_rotation=100.,
            newton_semi_implicit_enable=False, sanity_check_enable=True),
        show_viewer=False)
    table = scene.add_entity(name="tabletop",
        morph=gs.morphs.Box(pos=(0, 0, .05), size=(.4, .4, .1), fixed=True),
        material=gs.materials.Rigid(coup_type="ipc_only", friction=.7, coup_friction=.7))
    cube = scene.add_entity(name="falling_box",
        morph=gs.morphs.Box(pos=(0, 0, .13), size=(.04, .04, .04)),
        material=gs.materials.Rigid(coup_type="two_way_soft_constraint", rho=1000.,
                                    friction=.7, coup_friction=.7))
    pairing = install_rigid_collision_fallback(scene)
    feedback = install_exact_stc_feedback(scene)
    print("BUILD_START", flush=True)
    scene.build()
    print("BUILD_DONE", flush=True)
    mass = float(cube.get_mass().cpu().numpy().reshape(-1)[0])
    rows = []
    contacts_seen = []
    for step in range(int(round(duration / dt))):
        scene.step()
        vertices = cube.get_verts().cpu().numpy().reshape(-1, 3)
        force = cube.get_links_net_contact_force().cpu().numpy().reshape(-1, 3).sum(0)
        contacts = cube.get_contacts(with_entity=table)
        count = len(contacts["geom_a"])
        data = scene.sim.coupler._coupling_data
        ipc_force = np.asarray(data.out_forces).sum(axis=(0, 1))
        position = cube.get_pos().cpu().numpy().reshape(3)
        rows.append([(step+1)*dt, vertices[:, 2].min(), *position, *force, count, *ipc_force])
        contacts_seen.append(count)
    a = np.asarray(rows)
    last = a[a[:, 0] >= duration-.1]
    normal = float(last[:, 7].mean())
    weight = mass*9.81
    penetration = float(max(0., top-a[:, 1].min()))
    resting_height_variation = float(np.ptp(last[:, 4]))
    checks = {"pair_builder_verified": pairing["verified"],
              "native_rigid_contacts_remain_disabled": not scene.sim.coupler.options.enable_rigid_rigid_contact,
              "finite_states_and_forces": bool(np.isfinite(a).all()),
              "actual_contacts_detected": bool(max(contacts_seen) > 0),
              "final_contact_sustained": bool(np.all(last[:, 8] > 0)),
              "maximum_penetration_below_2mm": penetration <= .002,
              "final_normal_force_supports_weight_within_10_percent": abs(normal-weight) <= .1*weight,
              "final_height_variation_below_1mm": resting_height_variation <= .001}
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "purpose": "Simulation regression of restored contact pairs; not physical material or impact calibration.",
              "initial_bottom_gap_m": .01, "dt_s": dt, "duration_s": duration,
              "box_mass_kg": mass, "weight_n": weight, "final_mean_normal_support_n": normal,
              "maximum_penetration_m": penetration,
              "final_height_variation_m": resting_height_variation,
              "maximum_genesis_contact_points": max(contacts_seen),
              "maximum_ipc_feedback_force_norm_n": float(np.linalg.norm(a[:, 9:12], axis=1).max()),
              "rigid_contact_pairing": pairing, "exact_stc_feedback_enabled": feedback["enabled"],
              "wall_seconds": time.perf_counter()-wall,
              "columns": ["time", "bottom_z", "center_x", "center_y", "center_z",
                          "force_x", "force_y", "force_z", "contact_count",
                          "ipc_force_x", "ipc_force_y", "ipc_force_z"]}
    np.savez_compressed(HERE / "runtime_ipc/rigid-contact-dynamic.npz", trace=a)
    (HERE / "runtime_ipc/rigid-contact-dynamic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
