"""Offline asset checks and a separate coarse diagnostic tote mesh.

No simulation or GPU work is performed. This script never modifies production
assets. Run --build-coarse to produce bag_coarse* and metadata_coarse.json;
--genesis-cpu additionally checks Genesis' mesh/material loader on the CPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from assets_builder import ASSETS, topology, write_obj


def build_coarse_bag(output_dir=ASSETS):
    output_dir = Path(output_dir)
    production = json.loads((ASSETS / "metadata.json").read_text())["bag"]
    width, depth, height, strap_width = .38, .22, .32, .025
    spacing = .02
    across = np.unique(np.round(np.r_[np.linspace(-.0125, .0125, 6), [-.01, 0., .01]], 10))
    xs = np.unique(np.round(np.r_[np.linspace(-width/2, width/2, 20),
                                  -.12 + across, .12 + across], 10))
    ys = np.linspace(-depth/2, depth/2, 12)
    zs = np.linspace(0, height, 17)
    points, faces, lookup = [], [], {}
    def vertex(p):
        key = tuple(np.round(p, 9))
        if key not in lookup:
            lookup[key] = len(points)
            points.append(p)
        return lookup[key]
    def quad(a,b,c,d,flip=False):
        fs = [[a,b,d],[a,d,c]]
        faces.extend([f[::-1] for f in fs] if flip else fs)
    perimeter = ([(x,-depth/2) for x in xs[:-1]]
                 + [(width/2,y) for y in ys[:-1]]
                 + [(x,depth/2) for x in xs[::-1][:-1]]
                 + [(-width/2,y) for y in ys[::-1][:-1]])
    rings = np.array([[vertex([x,y,z]) for x,y in perimeter] for z in zs])
    for k in range(len(zs)-1):
        for j in range(len(perimeter)):
            nxt = (j+1)%len(perimeter)
            quad(rings[k,j],rings[k,nxt],rings[k+1,j],rings[k+1,nxt])
    bottom = np.array([[vertex([x,y,0.]) for x in xs] for y in ys])
    for j in range(len(ys)-1):
        for i in range(len(xs)-1):
            quad(bottom[j,i],bottom[j,i+1],bottom[j+1,i],bottom[j+1,i+1],flip=True)
    groups = {"rim":rings[-1].tolist(), "bottom":np.unique(bottom).tolist(),
              "body":list(range(len(points)))}
    # Twenty arc intervals preserve the same exact arch peak and total shape;
    # five mm cross-strip samples preserve the physical grip-patch resolution.
    arc = np.linspace(np.pi,0,21)
    path = np.r_[np.column_stack([np.full(2,-.12),np.linspace(.32,.36,3)[:-1]]),
                 np.column_stack([.12*np.cos(arc), .36+.12*np.sin(arc)]),
                 np.column_stack([np.full(2,.12),np.linspace(.36,.32,3)[1:]])]
    tangent = np.gradient(path,axis=0)
    tangent[0],tangent[-1] = [0,1],[0,-1]
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    normal = np.column_stack([tangent[:,1],-tangent[:,0]])
    for side,y in (("front",-depth/2),("back",depth/2)):
        strip = np.array([[vertex([p[0]+a*n[0],y,p[1]+a*n[1]]) for a in across]
                          for p,n in zip(path,normal)])
        for j in range(len(path)-1):
            for i in range(len(across)-1):
                quad(strip[j,i],strip[j,i+1],strip[j+1,i],strip[j+1,i+1],flip=side=="back")
        groups["handle_"+side] = np.unique(strip).tolist()
        groups["handle_"+side+"_root_left"] = strip[0].tolist()
        groups["handle_"+side+"_root_right"] = strip[-1].tolist()
        groups["grasp_"+side] = np.unique(strip[np.abs(path[:,0])<=.02]).tolist()
    points,faces = np.asarray(points,float),np.asarray(faces,np.int32)
    mesh = trimesh.Trimesh(points,faces,process=False)
    trimesh.repair.fix_normals(mesh)
    faces = mesh.faces.copy()
    stats = topology(points,faces)
    assert stats["connected_components"] == 1 and stats["nonmanifold_edges"] == 0
    assert stats["boundary_loops"] == 3 and stats["boundary_degree_two"]
    assert stats["consistent_winding"] and stats["degenerate_triangles"] == 0
    assert all(set(groups["handle_"+s+"_root_"+e])<=set(groups["rim"])
               for s in ("front","back") for e in ("left","right"))
    edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    edges,counts = np.unique(edges,axis=0,return_counts=True)
    bottom_edges = np.all(np.isclose(points[edges,2],0),axis=1)
    assert np.all(counts[bottom_edges]==2)
    uv = np.column_stack([(points[:,0]+width/2)/width,(points[:,2]+points[:,1]*.15)/.5])
    initial_points = points.copy()
    initial_points[:,1] *= .5
    output_dir.mkdir(parents=True,exist_ok=True)
    write_obj(output_dir/"bag_coarse.obj",points,faces,uv,"cotton_canvas.png")
    write_obj(output_dir/"bag_coarse_collision.obj",points,faces)
    write_obj(output_dir/"bag_coarse_initial_half_depth.obj",initial_points,faces,uv,"cotton_canvas.png")
    np.savez_compressed(output_dir/"bag_coarse.npz",points=points,faces=faces,uv=uv,
                        initial_points=initial_points,
                        **{k:np.asarray(v,np.int32) for k,v in groups.items()})
    mass = production["estimated_mass_kg"]
    density_scale = production["mesh"]["triangle_area_m2"]/stats["triangle_area_m2"]
    material = dict(production["suggested_material"])
    material["surface_density_kg_m2"] = mass/stats["triangle_area_m2"]
    metadata = dict(production)
    metadata.update(file="bag_coarse.obj", collision_file="bag_coarse_collision.obj",
                    half_depth_preview_file="bag_coarse_initial_half_depth.obj",npz_file="bag_coarse.npz",
                    mesh=stats,vertex_groups=groups,suggested_material=material,
                    estimated_mass_kg=mass,production_rho_multiplier_for_equal_mass=density_scale,
                    body_spacing_m=spacing,handle_cross_strip_spacing_max_m=.005,
                    diagnostic_only=True,production_asset_unchanged=True,
                    grasp_centers_local_m={k:points[v].mean(0).tolist() for k,v in groups.items() if k.startswith("grasp")})
    result = {"units":"metres","generator":"asset_visual_check.py --build-coarse","bag":metadata}
    (output_dir/"metadata_coarse.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def inspect_obj(path):
    mesh = trimesh.load(path,force="mesh",process=False)
    uv = getattr(mesh.visual,"uv",None)
    image = getattr(getattr(mesh.visual,"material",None),"image",None)
    result = {"file":str(path),"vertices":len(mesh.vertices),"faces":len(mesh.faces),
              "texture_visual":mesh.visual.kind == "texture","has_uv":uv is not None,
              "uv_count":0 if uv is None else len(uv),
              "has_material_image":image is not None,
              "image_size":None if image is None else list(image.size)}
    if uv is not None:
        assert len(uv) == len(mesh.vertices) and np.all(np.isfinite(uv))
        result["uv_bounds"] = [uv.min(0).tolist(),uv.max(0).tolist()]
    return result


def genesis_cpu_check():
    import genesis as gs
    from genesis.utils.mesh import merge_submeshes
    gs.init(backend=gs.cpu,logging_level="error",performance_mode=True)
    results = []
    for filename,image in (("bag.obj","cotton_canvas.png"),("bear_physics.obj","teddy_fur.png")):
        morph = gs.morphs.Mesh(file=str(ASSETS/filename),file_meshes_are_zup=True,
                              decimate=False,convexify=False)
        for kind in ("explicit_color","explicit_image"):
            surface = (gs.surfaces.Default(color=(.56,.30,.13,1)) if kind=="explicit_color"
                       else gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_path=str(ASSETS/image))))
            # Scene.add_entity calls update_texture before mesh construction.
            surface.update_texture()
            meshes = gs.Mesh.from_morph_surface(morph,surface)
            points,faces,maps = merge_submeshes([m.verts for m in meshes],[m.faces for m in meshes])
            results.append({"asset":filename,"surface_option":kind,
                            "simulation_surface_vertices":len(points),"simulation_surface_triangles":len(faces),
                            "visual_vertices":sum(len(m.verts) for m in meshes),
                            "retained_uvs":all(m.uvs is not None for m in meshes),
                            "texture_type":[type(m.surface.texture).__name__ for m in meshes],
                            "max_visual_to_physical_mapping_error_m":float(max(np.linalg.norm(m.verts-points[mp],axis=1).max() for m,mp in zip(meshes,maps)))})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-coarse",action="store_true")
    p.add_argument("--genesis-cpu",action="store_true")
    args = p.parse_args()
    results = {"asset_load_checks":[inspect_obj(ASSETS/f) for f in ("bag.obj","bear.obj","bear_physics.obj")],
               "no_gpu_simulation":True,
               "high_resolution_visual_note":"Current FEMVisGeom uses one simulated vertex index per visual vertex. Separate high-poly deformation requires barycentric skinning in the FEM render path; simply changing the OBJ increases the physical TetGen input."}
    if args.build_coarse:
        coarse = build_coarse_bag()
        results["coarse_bag_summary"] = {k:coarse["bag"][k] for k in
            ("mesh","estimated_mass_kg","production_rho_multiplier_for_equal_mass","grasp_centers_local_m")}
    if args.genesis_cpu:
        results["genesis_loader_checks"] = genesis_cpu_check()
    (ASSETS/"visual_compatibility_check.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
