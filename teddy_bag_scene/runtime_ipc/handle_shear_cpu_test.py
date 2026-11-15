"""CPU-only native attribute and mesh-region checks; no World or GPU initialization."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE / "vendor")]
import numpy as np
import uipc
from uipc.constitution import DiscreteShellBending, ElasticModuli2D, StrainLimitingBaraffWitkinShell
from physics_config import _map_shell_material_regions, _set_shell_shear_coefficients

source_path = HERE / "assets/two_panel_bag_wide_tall_coarse.npz"
with np.load(source_path) as data:
    source = {name: np.array(data[name]) for name in data.files}
translation = np.array([-.13, 0., .742])
points = source["points"] + translation
faces = source["faces"]
checks = {}


def geometry(p, f):
    mesh = uipc.geometry.trimesh(np.ascontiguousarray(p), np.ascontiguousarray(f, dtype=np.int32))
    StrainLimitingBaraffWitkinShell().apply_to(mesh,
        moduli=ElasticModuli2D.youngs_poisson(4e6, .3), mass_density=500., thickness=.0003)
    DiscreteShellBending().apply_to(mesh, bending_stiffness=5e-5)
    return mesh


mesh = geometry(points, faces)
regions, mapping = _map_shell_material_regions(source_path, translation,
    uipc.view(mesh.positions()), uipc.view(mesh.triangles().topo()))
report = _set_shell_shear_coefficients(mesh, 50., handle_shear_stiffness_n_per_m=250., regions=regions)
checks["current_asset_exact_partition"] = {k: len(v) for k, v in regions.items()} == {
    "body": 2960, "handle_front": 352, "handle_back": 352}
checks["native_handle_coefficients_and_other_properties"] = (
    all(report["protected_attributes_unchanged"].values())
    and all(r["all_faces_equal"] for r in report["region_coefficients"].values())
    and report["region_coefficients"]["body"]["shear_stiffness_n_per_m"] == 50.
    and all(report["region_coefficients"][k]["shear_stiffness_n_per_m"] == 250.
            for k in ("handle_front", "handle_back")))
plain = geometry(points, faces)
_set_shell_shear_coefficients(plain, 50.)
checks["default_stays_uniform"] = bool(np.all(uipc.view(plain.triangles().find("mu")) == 50.))

rng = np.random.default_rng(16)
vertex_order = rng.permutation(len(points))
inverse = np.argsort(vertex_order)
face_order = rng.permutation(len(faces))
shuffled_faces = inverse[faces[face_order]][:, [2, 0, 1]]
permuted = geometry(points[vertex_order], shuffled_faces)
mapped, shuffled_report = _map_shell_material_regions(source_path, translation,
    uipc.view(permuted.positions()), uipc.view(permuted.triangles().topo()))
_set_shell_shear_coefficients(permuted, 50., handle_shear_stiffness_n_per_m=250., regions=mapped)
checks["permuted_vertices_faces_and_corner_orders"] = all(
    set(face_order[mapped[k]]) == set(regions[k]) for k in regions)


def rejected(call):
    try:
        call()
    except ValueError:
        return True
    return False

bad_points = points.copy()
bad_points[1] = bad_points[0]
bad_faces = faces.copy()
bad_faces[1] = bad_faces[0]
checks["reject_nonunique_vertex_or_face_mapping"] = (
    rejected(lambda: _map_shell_material_regions(source_path, translation, bad_points, faces))
    and rejected(lambda: _map_shell_material_regions(source_path, translation, points, bad_faces)))
with tempfile.TemporaryDirectory() as tmp:
    bad_source = dict(source)
    bad_source["body_face_indices"] = np.r_[source["body_face_indices"], regions["handle_front"][0]]
    path = Path(tmp) / "bad_regions.npz"
    np.savez(path, **bad_source)
    checks["reject_body_handle_overlap"] = rejected(
        lambda: _map_shell_material_regions(path, translation, points, faces))
checks["reject_invalid_coefficients"] = all(rejected(
    lambda value=value: _set_shell_shear_coefficients(mesh, 50.,
        handle_shear_stiffness_n_per_m=value, regions=regions)) for value in (0., -1., np.nan, np.inf))
result = {"passed": all(checks.values()), "checks": checks,
          "native_attribute_report": report,
          "mapping": {k: v for k, v in mapping.items() if k != "regions"},
          "permuted_mapping": {k: v for k, v in shuffled_report.items() if k != "regions"},
          "source_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest()
                            for p in ("physics_config.py", "scene.py")},
          "scope": "Seven CPU-only native geometry checks; no World.init or simulation"}
(HERE / "runtime_ipc/handle-shear-cpu-tests.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
