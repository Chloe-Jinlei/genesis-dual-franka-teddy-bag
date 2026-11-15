"""Create a separate, uniformly sized teddy rest asset; never touch a live state."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import trimesh


def physical_mesh(lines):
    # OBJ texture seams use separate vt indices, not separate physical nodes.
    vertices = [list(map(float, line.split()[1:4])) for line in lines if line.startswith("v ")]
    faces = [[int(word.split("/")[0])-1 for word in line.split()[1:]]
             for line in lines if line.startswith("f ")]
    if any(len(face) != 3 or min(face) < 0 for face in faces):
        raise ValueError("Expected positive-indexed triangle faces")
    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def build(source: Path, destination: Path, scale: float):
    if not np.isfinite(scale) or not .5 <= scale <= 1.5:
        raise ValueError("Declare a finite scale in [0.5, 1.5]")
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("A new independent destination is required")
    if destination.suffix != ".obj" or not source.is_file():
        raise ValueError("Expected an existing OBJ source and a new OBJ destination")
    material = destination.with_suffix(".mtl")
    if material.exists():
        raise ValueError("Destination material already exists")
    original = source.read_text().splitlines(keepends=True)
    output = []
    for line in original:
        if line.startswith("v "):
            xyz = np.asarray(line.split()[1:4], float) * scale
            output.append("v " + " ".join(f"{x:.17g}" for x in xyz) + "\n")
        elif line.startswith("mtllib "):
            output.append(f"mtllib {material.name}\n")
        else:
            output.append(line)
    old = physical_mesh(original)
    if not old.is_watertight or not old.is_winding_consistent:
        raise ValueError("Source must be a closed, consistently oriented teddy")
    destination.write_text("".join(output))
    material.write_bytes(source.with_suffix(".mtl").read_bytes())
    new = physical_mesh(output)
    error = float(np.max(np.abs(new.vertices - scale * old.vertices)))
    if not np.array_equal(new.faces, old.faces) or error > 1e-14 \
            or not new.is_watertight or not new.is_winding_consistent:
        raise ValueError("Uniform rest asset transformation did not preserve topology")
    report = dict(kind="independent_uniform_teddy_rest_asset", scale=scale,
        source=str(source.resolve()), output=str(destination.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        output_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        unchanged_faces=True, unchanged_uv_lines=True, maximum_vertex_error_m=error,
        bounds_m=new.bounds.tolist(), dimensions_m=new.extents.tolist(),
        enclosed_volume_m3=float(new.volume),
        same_density_mass_ratio=scale**3,
        simulation_state_modified=False, physical_task_success=False,
        required_config_changes=["mesh", "maxvolume scaled cubically",
            "local grasp point and tracking radius scaled linearly"],
        scope="New initial geometry only. Native volume meshing, friction grasp, motion and containment must be checked in a fresh simulation.")
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.scale), indent=2))
