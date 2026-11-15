"""Small isolated IPC runtime check; uses scene-local vendor dependencies."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
OUT = ROOT / "runtime_ipc"
OUT.mkdir(exist_ok=True)

import numpy as np
import genesis as gs
from importlib.metadata import version


def main():
    started = time.perf_counter()
    gs.init(backend=gs.cpu, logging_level="warning")
    points = np.array([[x, y, .159] for y in np.linspace(-.06, .06, 7)
                       for x in np.linspace(-.06, .06, 7)])
    faces = []
    for j in range(6):
        for i in range(6):
            a = j * 7 + i
            faces.extend([[a, a+1, a+8], [a, a+8, a+7]])
    obj = OUT / "cloth.obj"
    obj.write_text("".join("v %.9f %.9f %.9f\n" % tuple(p) for p in points)
                   + "".join("f %d %d %d\n" % tuple(np.asarray(f)+1) for f in faces))
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=.01, substeps=1),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=.002, contact_resistance=1e7,
            contact_friction_enable=True, sanity_check_enable=True,
            newton_max_iterations=100, newton_semi_implicit_enable=False,
            enable_rigid_rigid_contact=False, two_way_coupling=True),
        show_viewer=False)
    scene.add_entity(morph=gs.morphs.Box(pos=(0, 0, .05), size=(.4, .4, .1), fixed=True),
        material=gs.materials.Rigid(coup_type="ipc_only", coup_friction=.5))
    cube = scene.add_entity(morph=gs.morphs.Box(pos=(0,0,.131), size=(.05,.05,.05)),
        material=gs.materials.FEM.Elastic(E=2e4, nu=.3, rho=100,
            friction_mu=.5, model="stable_neohookean"))
    cloth = scene.add_entity(morph=gs.morphs.Mesh(file=str(obj),
            file_meshes_are_zup=True, decimate=False, convexify=False),
        material=gs.materials.FEM.Cloth(E=6e4, nu=.3, rho=300, thickness=.0006,
            bending_stiffness=.01, friction_mu=.5),
        surface=gs.surfaces.Default(double_sided=True))
    print("BUILD_START", flush=True)
    scene.build()
    print("BUILD_DONE", flush=True)
    begin = time.perf_counter()
    previous = {}
    for i in range(10):
        if i == 9:
            previous = {name:entity.get_state().pos.detach().cpu().numpy().reshape(-1,3).copy()
                        for name,entity in (("cube",cube),("cloth",cloth))}
        scene.step()
        print("IPC_STEP", i+1, flush=True)
    result = {"status":"passed", "steps":10, "simulation_seconds":.1,
        "genesis_version":version("genesis-world"), "pyuipc_version":version("pyuipc"),
        "initial_build_seconds":begin-started,
        "step_wall_seconds":time.perf_counter()-begin,
        "velocity_note":"Genesis IPC syncs FEM positions only; velocity is measured from position differences.",
        "entities":{}}
    for name, entity in (("cube",cube),("cloth",cloth)):
        state=entity.get_state()
        p=state.pos.detach().cpu().numpy().reshape(-1,3)
        v=state.vel.detach().cpu().numpy().reshape(-1,3)
        assert np.isfinite(p).all() and np.isfinite(v).all()
        assert p[:,2].min() > .099
        result["entities"][name]={"vertices":len(p), "min_z":float(p[:,2].min()),
            "max_speed_finite_difference":float(np.linalg.norm((p-previous[name])/.01,axis=1).max())}
    (OUT / "smoke-report.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    try:
        main()
    except BaseException as exc:
        (OUT / "smoke-error.json").write_text(json.dumps({"status":"failed",
            "error_type":type(exc).__name__,"error":str(exc)},indent=2))
        raise
