"""One authorized complete-scene build and 0.1 s physical material smoke."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "vendor")]
os.environ.setdefault("GS_CACHE_FILE_PATH", str(HERE / ".cache/genesis"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import uipc
from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot
from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
from scene import build, read_config, positions
from physical_monitor import PhysicalMonitor

started = time.perf_counter()
config_path = HERE / "config_guarded_webbing_alignment_trial.json"
config = read_config(config_path)
assert config["bag"]["handle_shear_stiffness"] == 250.
initial_path = HERE / "runs/two_panel_full_16/initial.npz"
with np.load(initial_path) as data:
    initial_q = {side: data["q_" + side] for side in ("left", "right")}
scene, arms, bag, bear, cameras = build(config, width=160, height=120, initial_q=initial_q)
rest = {"bag": positions(bag), "bear": positions(bear),
        "bag_faces": np.asarray(bag.elems), "bear_tets": np.asarray(bear.elems)}
monitor = PhysicalMonitor(rest, config)
rows = [monitor.measure(0., positions(bag), positions(bear))]
native = {}
fixed_counts = {}
for slot in SceneVisitor(scene.sim.coupler._ipc_scene).geometries():
    if not isinstance(slot, SimplicialComplexSlot):
        continue
    geom = slot.geometry()
    meta = read_ipc_geometry_metadata(geom)
    if meta is None or meta[0] not in ("cloth", "fem"):
        continue
    fixed_counts[str(meta)] = int(np.count_nonzero(uipc.view(geom.vertices().find("is_fixed"))))
    if meta[0] == "cloth":
        assert not native
        native = {"native_mu": np.array(uipc.view(geom.triangles().find("mu"))),
                  "native_lambda": np.array(uipc.view(geom.triangles().find("lambda"))),
                  "native_strain_rate": np.array(uipc.view(geom.triangles().find("strain_rate"))),
                  "native_thickness": np.array(uipc.view(geom.vertices().find("thickness"))),
                  "native_volume": np.array(uipc.view(geom.vertices().find("volume"))),
                  "native_bending": np.array(uipc.view(geom.edges().find("bending_stiffness"))),
                  "native_triangles": np.array(uipc.view(geom.triangles().topo())).reshape(-1, 3)}
        cloth_geometry = geom
steps = round(.1 / config["simulation"]["dt"])
for step in range(1, steps + 1):
    scene.step()
    rows.append(monitor.measure(step * config["simulation"]["dt"], positions(bag), positions(bear)))
rows = np.asarray(rows)
entity_report = scene.teddy_shell_report["entities"][0]
region_ids = entity_report["material_region_mapping"]["regions"]
region_values = {name: np.unique(native["native_mu"][region["native_face_indices"]]).tolist()
                 for name, region in region_ids.items()}
checks = {
    "native_built_and_twenty_steps_completed": steps == 20 and len(rows) == 21,
    "native_body_2960_at_50": region_ids["body"]["face_count"] == 2960 and region_values["body"] == [50.],
    "native_each_handle_352_at_250": all(region_ids[k]["face_count"] == 352
        and region_values[k] == [250.] for k in ("handle_front", "handle_back")),
    "no_fixed_soft_vertices": not any(fixed_counts.values()),
    "protected_attributes_unchanged": all(entity_report["protected_attributes_unchanged"].values()),
    "native_mu_same_after_advance": bool(np.array_equal(native["native_mu"],
        uipc.view(cloth_geometry.triangles().find("mu")))),
    "all_soft_states_finite": bool(np.all(rows[:, 5] == 1)),
    "minimum_J_at_least_005": bool(rows[:, 1].min() >= .05),
    "table_penetration_at_most_2mm": bool(rows[:, 2:4].max() <= .002),
    "cloth_edge_stretch_at_most_115": bool(rows[:, 4].max() <= 1.15),
}
np.savez_compressed(HERE / "runtime_ipc/handle-shear-native-smoke.npz", **native,
    soft_metrics=rows, final_bag=positions(bag), final_bear=positions(bear),
    initial_q_left=initial_q["left"], initial_q_right=initial_q["right"])
result = {"passed": all(checks.values()), "checks": checks,
          "simulated_seconds": .1, "wall_seconds": time.perf_counter() - started,
          "native_region_values_n_per_m": region_values,
          "native_lambda_unique": np.unique(native["native_lambda"]).tolist(),
          "native_thickness_radius_unique_m": np.unique(native["native_thickness"]).tolist(),
          "native_bending_unique_nm": np.unique(native["native_bending"]).tolist(),
          "fixed_soft_vertex_counts": fixed_counts,
          "minimum_J": float(rows[:, 1].min()),
          "maximum_table_penetration_m": float(rows[:, 2:4].max()),
          "maximum_cloth_edge_stretch_ratio": float(rows[:, 4].max()),
          "shell_conversion": scene.teddy_shell_report,
          "initialization": scene.teddy_initialization_report,
          "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (HERE / "physics_config.py", HERE / "scene.py", config_path, initial_path)},
          "scope": "Complete native scene with saved correct initial robot pose, then 0.1 s physical integration. No full task, grasp transfer, insertion or calibrated material claim."}
(HERE / "runtime_ipc/handle-shear-native-smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k not in ("shell_conversion", "initialization")}, indent=2), flush=True)
raise SystemExit(0 if result["passed"] else 1)
