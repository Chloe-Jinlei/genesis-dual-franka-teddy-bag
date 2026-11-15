"""One isolated native baseline/candidate cloth coupon comparison.

Prescribed load only: no viscous drag, no Genesis, no robot or production assets.
Default zero gravity isolates membrane deformation and recoil; --gravity-z -9.81
adds a gravity-loaded bending response. This is a diagnostic, not fabric calibration.
Do not launch until the supervising task authorizes execution.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "vendor"))

import numpy as np


DT = .005
DURATION = 2.0
LOAD_START, LOAD_END = .2, .8
FORCE_N = .5
RADIUS, DENSITY, NU, BENDING, STRAIN_RATE = .0003, 500., .3, 5e-5, 100.
CASES = (("baseline", 4e6, 50.), ("candidate", 20e6, 1000.))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")


def coupon_mesh():
    # Width 0.1 m, length 0.2 m; clamp the y=0 short edge.
    nx, ny = 10, 20
    points = np.array([[i*.01, j*.01, 0.] for j in range(ny+1)
                       for i in range(nx+1)], dtype=np.float64)
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i
            b, c, d = a+1, a+nx+1, a+nx+2
            faces.extend(((a, b, d), (a, d, c)))
    faces = np.asarray(faces, dtype=np.int32)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]],
        faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    return points, faces, edges, np.arange(nx+1), np.arange(ny*(nx+1), (ny+1)*(nx+1))


def run_case(label, youngs, shear, out, gravity_z, wall_deadline):
    # Native imports are deliberately inside the launch path.
    import uipc
    from uipc import Engine, Scene, World, builtin, view
    from uipc.constitution import (DiscreteShellBending, ElasticModuli2D,
        FiniteElementExternalForce, StrainLimitingBaraffWitkinShell)
    from uipc.core import FiniteElementStateAccessorFeature
    from uipc.geometry import label_surface, trimesh

    started = time.monotonic()
    points, faces, edges, fixed_ids, load_ids = coupon_mesh()
    rest_edges = np.linalg.norm(points[edges[:, 1]]-points[edges[:, 0]], axis=1)
    tri0 = points[faces]
    dm = np.stack((tri0[:, 1, :2]-tri0[:, 0, :2],
                   tri0[:, 2, :2]-tri0[:, 0, :2]), axis=2)
    dm_inv = np.linalg.inv(dm)
    rest_twice_area = np.linalg.norm(np.cross(tri0[:, 1]-tri0[:, 0],
                                             tri0[:, 2]-tri0[:, 0]), axis=1)
    case_out = out / label
    case_out.mkdir()
    result = dict(label=label, status="initializing", E_Pa=youngs,
        requested_mu_N_per_m=shear, density_kg_m3=DENSITY,
        radius_m=RADIUS, full_thickness_m=2*RADIUS, nu=NU,
        requested_bending_Nm=BENDING, requested_strain_rate=STRAIN_RATE,
        gravity_z_m_s2=gravity_z, native_errors=[], callback_count=0)
    pos_history, vel_history, rows, frame_stats, callback_rows = [], [], [], [], []
    native = {}
    engine = world = scene = feature = state = obj = geom = rest_geom = None
    try:
        uipc.Timer.disable_all()
        uipc.Logger.set_level(uipc.Logger.Level.Error)
        engine_config = Engine.default_config()
        engine_config["extras"]["gui"]["enable"] = False
        engine = Engine("cuda", str(case_out / "native"), engine_config)
        world = World(engine)
        cfg = Scene.default_config()
        cfg["dt"] = DT
        cfg["gravity"] = [[0.], [0.], [gravity_z]]
        cfg["contact"]["enable"] = False
        cfg["contact"]["constitution"] = "ipc"
        cfg["integrator"]["type"] = "bdf1"
        cfg["newton"]["semi_implicit"]["enable"] = False
        cfg["newton"]["max_iter"] = 128
        cfg["newton"]["velocity_tol"] = 1e-4
        cfg["newton"]["transrate_tol"] = 1e-4
        cfg["line_search"]["max_iter"] = 32
        cfg["extras"]["strict_mode"]["enable"] = True
        cfg["sanity_check"]["enable"] = True
        scene = Scene(cfg)
        result["scene_config"] = cfg
        membrane = StrainLimitingBaraffWitkinShell()
        bending = DiscreteShellBending()
        external = FiniteElementExternalForce()
        for constitution in (membrane, bending, external):
            scene.constitution_tabular().insert(constitution)
        mesh = trimesh(points, faces)
        label_surface(mesh)
        membrane.apply_to(mesh, moduli=ElasticModuli2D.youngs_poisson(youngs, NU),
            mass_density=DENSITY, thickness=RADIUS, strain_rate=STRAIN_RATE)
        view(mesh.triangles().find("mu"))[:] = shear
        bending.apply_to(mesh, bending_stiffness=BENDING)
        fixed = mesh.vertices().find(builtin.is_fixed)
        if fixed is None:
            fixed = mesh.vertices().create(builtin.is_fixed, 0)
        view(fixed).reshape(-1)[:] = 0
        view(fixed).reshape(-1)[fixed_ids] = 1
        external.apply_to(mesh, np.zeros(3, dtype=np.float64), enable=False)
        obj = scene.objects().create(label)
        geo_slot, rest_slot = obj.geometries().create(mesh)
        geom, rest_geom = geo_slot.geometry(), rest_slot.geometry()
        direction = np.array([1., 1., 0.]) / np.sqrt(2.)
        # Trapezoidal line weights: uniform traction over the free short edge.
        weights = np.ones(len(load_ids))
        weights[[0, -1]] = .5
        weights /= weights.sum()
        first_loaded_frame = round(LOAD_START/DT)+1
        last_loaded_frame = round(LOAD_END/DT)

        def apply_prescribed_force(info):
            active_geom = info.geo_slots()[0].geometry()
            force = view(active_geom.vertices().find("external_force")).reshape(-1, 3)
            mask = view(active_geom.vertices().find("external_force/is_constrained")).reshape(-1)
            force[:] = 0.
            mask[:] = 0
            frame = int(info.frame())
            active = first_loaded_frame <= frame <= last_loaded_frame
            if active:
                mask[load_ids] = 1
                force[load_ids] = FORCE_N*weights[:, None]*direction
            callback_rows.append([frame, float(info.dt()), int(active),
                                  *force.sum(axis=0), int(np.count_nonzero(mask))])

        scene.animator().insert(obj, apply_prescribed_force)
        scene.animator().substep(1)
        world.init(scene)
        if not world.is_valid() or engine.status().has_error():
            raise RuntimeError(f"Native initialization failed: {engine.status().to_json()}")
        for name, attrs in (("mu", geom.triangles()), ("lambda", geom.triangles()),
                            ("strain_rate", geom.triangles()), ("thickness", geom.vertices()),
                            ("volume", geom.vertices()), ("bending_stiffness", geom.edges()),
                            ("is_fixed", geom.vertices())):
            attr = attrs.find(name)
            if attr is None:
                raise RuntimeError(f"Native attribute missing: {name}")
            native[name] = np.array(view(attr), copy=True)
        expected_lambda = youngs*(2*RADIUS)/(1-NU**2)
        checks = dict(mu=np.allclose(native["mu"], shear, rtol=1e-12, atol=1e-12),
            lambda_=np.allclose(native["lambda"], expected_lambda, rtol=1e-12),
            strain_rate=np.allclose(native["strain_rate"], STRAIN_RATE),
            thickness=np.allclose(native["thickness"], RADIUS),
            bending=np.allclose(native["bending_stiffness"], BENDING),
            clamp_count=np.count_nonzero(native["is_fixed"]) == len(fixed_ids))
        result["native_coefficient_checks"] = {k: bool(v) for k, v in checks.items()}
        result["native_coefficient_unique"] = {k: np.unique(v).tolist()
            for k, v in native.items() if k != "volume"}
        result["native_vertex_volume_sum_m3"] = float(native["volume"].sum())
        masses = native["volume"].reshape(-1)*DENSITY
        result["native_volume_times_declared_density_mass_kg"] = float(masses.sum())
        if not all(checks.values()):
            raise RuntimeError(f"Native coefficient readback mismatch: {checks}")
        feature = world.features().find(FiniteElementStateAccessorFeature)
        if feature is None:
            raise RuntimeError("Native FEM state accessor unavailable")
        if feature.vertex_count() != len(points):
            raise RuntimeError("Unexpected native vertex count")
        state = feature.create_geometry()
        state.vertices().create(builtin.position, np.zeros(3, dtype=np.float64))
        state.vertices().create(builtin.velocity, np.zeros(3, dtype=np.float64))

        def sample(frame):
            feature.copy_to(state)
            x = np.array(view(state.vertices().find(builtin.position))).reshape(-1, 3)
            v = np.array(view(state.vertices().find(builtin.velocity))).reshape(-1, 3)
            if not (np.isfinite(x).all() and np.isfinite(v).all()):
                raise RuntimeError(f"Nonfinite native state at frame {frame}")
            tri = x[faces]
            ds = np.stack((tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=2)
            F = ds @ dm_inv
            sigma = np.linalg.svd(F, compute_uv=False)
            cos = np.sum(F[:, :, 0]*F[:, :, 1], axis=1) / np.maximum(
                np.linalg.norm(F[:, :, 0], axis=1)*np.linalg.norm(F[:, :, 1], axis=1), 1e-30)
            edge_ratio = np.linalg.norm(x[edges[:, 1]]-x[edges[:, 0]], axis=1)/rest_edges
            area_ratio = np.linalg.norm(np.cross(ds[:, :, 0], ds[:, :, 1]), axis=1)/rest_twice_area
            tip_displacement = x[load_ids].mean(0)-points[load_ids].mean(0)
            clamp_error = float(np.linalg.norm(x[fixed_ids]-points[fixed_ids], axis=1).max())
            kinetic = float(.5*np.sum(masses[:, None]*v*v))
            force_now = np.array(view(geom.vertices().find("external_force"))).reshape(-1, 3).sum(0)
            rows.append([frame*DT, *tip_displacement, float(edge_ratio.max()),
                float(sigma[:, 0].max()), float(sigma[:, 1].min()), float(np.abs(cos).max()),
                float(area_ratio.min()), float(np.linalg.norm(v, axis=1).max()), kinetic,
                clamp_error, float(x[:, 2].min()), *force_now])
            pos_history.append(x)
            vel_history.append(v)
            if clamp_error > 1e-8:
                raise RuntimeError(f"Clamp drift {clamp_error} m at frame {frame}")
            if area_ratio.min() <= 1e-6 or np.linalg.norm(x-points, axis=1).max() > 1.:
                raise RuntimeError(f"Coupon geometric validity guard failed at frame {frame}")

        sample(0)
        print(f"{label}: initialized {len(points)} vertices, {len(faces)} triangles", flush=True)
        for frame in range(1, round(DURATION/DT)+1):
            if time.monotonic() > wall_deadline:
                raise TimeoutError("Bounded coupon wall-time budget exceeded")
            world.advance()
            world.retrieve()
            stats = engine.frame_stats()
            frame_stats.append(stats)
            if engine.status().has_error() or not stats.get("completed", False):
                raise RuntimeError(f"Native step failed at frame {frame}: "
                    f"status={engine.status().to_json()}, stats={stats}")
            if stats.get("hit_newton_limit") or stats.get("hit_line_search_limit"):
                raise RuntimeError(f"Native iteration limit at frame {frame}: {stats}")
            sample(frame)
            if frame % 40 == 0:
                print(f"{label}: t={frame*DT:.3f}s edge_max={rows[-1][4]:.6f} "
                      f"speed_max={rows[-1][9]:.6f}m/s", flush=True)
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["native_errors"].append(dict(type=type(exc).__name__, message=str(exc),
                                           traceback=traceback.format_exc()))
        print(f"{label}: FAILED {type(exc).__name__}: {exc}", flush=True)
    finally:
        result["wall_seconds"] = time.monotonic()-started
        result["callback_count"] = len(callback_rows)
        result["saved_state_count"] = len(rows)
        result["completed_simulated_seconds"] = rows[-1][0] if rows else 0.
        result["frame_stats"] = frame_stats
        result["metric_columns"] = ["time_s", "tip_dx_m", "tip_dy_m", "tip_dz_m",
            "max_edge_stretch", "max_principal_stretch", "min_principal_stretch",
            "max_absolute_material_axis_cosine", "min_triangle_area_ratio",
            "max_vertex_speed_m_s", "kinetic_energy_J", "max_clamp_error_m",
            "min_z_m", "force_x_N", "force_y_N", "force_z_N"]
        result["callback_columns"] = ["frame", "dt_s", "load_active", "force_x_N",
                                       "force_y_N", "force_z_N", "enabled_vertex_count"]
        if rows:
            a = np.array(rows)
            release = a[:, 0] > LOAD_END+DT/2
            result["max_edge_stretch"] = float(a[:, 4].max())
            result["max_principal_stretch"] = float(a[:, 5].max())
            result["peak_tip_displacement_m"] = float(np.linalg.norm(a[:, 1:4], axis=1).max())
            result["final_tip_displacement_m"] = a[-1, 1:4]
            result["final_max_vertex_speed_m_s"] = float(a[-1, 9])
            result["final_kinetic_energy_J"] = float(a[-1, 10])
            result["load_zero_after_release"] = bool(np.all(np.abs(a[release, 13:16]) < 1e-12))
            if release.any():
                result["post_release_peak_speed_m_s"] = float(a[release, 9].max())
                result["post_release_peak_kinetic_energy_J"] = float(a[release, 10].max())
        np.savez_compressed(case_out / "states.npz", rest=points, faces=faces, edges=edges,
            fixed_ids=fixed_ids, loaded_ids=load_ids, positions=np.array(pos_history),
            velocities=np.array(vel_history), metrics=np.array(rows), callbacks=np.array(callback_rows),
            **{"native_"+k: v for k, v in native.items()})
        write_json(case_out / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=HERE / "runtime_ipc/cotton-linen-coupon-probe")
    parser.add_argument("--gravity-z", type=float, default=0.)
    parser.add_argument("--max-wall-seconds", type=float, default=300.)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing probe directory: {args.output}")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    results = []
    for label, E, mu in CASES:
        result = run_case(label, E, mu, args.output, args.gravity_z, start+args.max_wall_seconds)
        results.append(result)
        gc.collect()
        if result["status"] != "completed":
            break
    summary = dict(status="completed" if len(results) == 2 and all(r["status"] == "completed"
        for r in results) else "failed", dt_s=DT, duration_per_coupon_s=DURATION,
        dimensions_m=[.1, .2], nominal_grid_m=.01, force_total_N=FORCE_N,
        force_direction=[float(1/np.sqrt(2)), float(1/np.sqrt(2)), 0.],
        load_interval_s=[LOAD_START, LOAD_END], gravity_z_m_s2=args.gravity_z,
        contact_enabled=False, added_viscous_damping=False, production_modified=False,
        physical_fabric_calibration=False, task_success_certified=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wall_seconds=time.monotonic()-start,
        limitations=["One coarse prescribed-load comparison; no fitted fabric parameters.",
            "BDF1 has numerical dissipation; recoil is not a physical damping calibration.",
            "Zero-gravity planar load does not validate bending drape.",
            "Contact is disabled; no self-contact, grasp, robot, or bag insertion is tested.",
            "Principal and edge stretch are diagnostics; the full-task 1.15 guard is unchanged."],
        cases=[{k: v for k, v in r.items() if k not in ("frame_stats", "scene_config")}
               for r in results])
    write_json(args.output / "report.json", summary)
    print(json.dumps(summary, indent=2, default=serial, allow_nan=False), flush=True)
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
