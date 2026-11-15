"""Scene-local Genesis 1.4.3 / pyuipc 0.0.29 compatibility and diagnostics.

Only the supplied scene instance is changed. No engine installation is edited.
"""
from __future__ import annotations

from importlib.metadata import version
import numpy as np


def stc_reaction_wrench(ipc_transform, aim_transform, mass, first_moment,
                        second_moment, translation_scale, rotation_scale):
    """Exact rigid-target virtual-work pullback of pyuipc 0.0.29 STC energy.

    Scales are eta_p/h^2 and eta_a/h^2. The native ABD mass moments use the
    link's original reference frame, including a nonzero centre of mass.
    Return world force and world torque *about the aim link origin*. Retain
    the full affine A: polar projection would change the energy derivative.
    """
    ipc, aim = np.asarray(ipc_transform), np.asarray(aim_transform)
    mc, second = np.asarray(first_moment).reshape(3), np.asarray(second_moment).reshape(3, 3)
    m = float(mass)
    if not np.isfinite(m) or m <= 0:
        raise ValueError("STC feedback needs a finite positive native ABD mass")
    dp, da = ipc[:3, 3]-aim[:3, 3], ipc[:3, :3]-aim[:3, :3]
    effective_second = rotation_scale*second + (translation_scale-rotation_scale)*np.outer(mc, mc)/m
    force = translation_scale*(m*dp + da@mc)
    ga = translation_scale*np.outer(dp, mc) + da@effective_second
    torque = np.cross(aim[:3, :3].T, ga.T).sum(axis=0)
    return force, torque


def install_exact_stc_feedback(scene):
    """Opt in to scene-local, exact STC reaction; never patch installed code.

    Genesis 1.4.3's default approximation uses link-origin displacement and
    MJCF mass/inertia. libuipc's STC instead uses its ABD mass moments and COM.
    This adapter replaces that one scene's feedback callback with the actual
    energy derivative. Forces act at the Genesis link origin; the torque
    already includes the COM lever arm, so no extra r-cross-F is added.
    Only two_way_soft_constraint, nonfixed links receive reactions. A native
    animator substep of 1 is required so aim_transforms is the actual target.
    """
    if scene.is_built or version("genesis-world")!="1.4.3" or version("pyuipc")!="0.0.29":
        raise RuntimeError("Exact STC adapter requires an unbuilt, version-matched scene")
    import uipc
    from genesis.engine.couplers.ipc_coupler.data import COUPLING_TYPE
    coupler=scene.sim.coupler
    report={"method":"Exact native STC energy gradient, rigid-target virtual work",
            "enabled":True,"applied_steps":0,"links":[],
            "engine_installation_modified":False,"mass_source":"native IPC ABD dyadic moments",
            "wrench_reference":"world axes, torque about target link origin",
            "source":"libuipc v0.0.29 soft_transform_constraint.cu"}
    cache=None

    def exact_feedback():
        nonlocal cache
        if not coupler.options.two_way_coupling:
            return
        data=coupler._coupling_data
        if data is None:
            return
        if coupler._ipc_animator.substep()!=1:
            raise RuntimeError("Exact STC feedback requires native animator substep=1")
        if cache is None:
            cache=[]
            for index,link in enumerate(data.links):
                if link.is_fixed or coupler._coup_type_by_entity.get(link.entity)!=COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                    continue
                for env,slot in enumerate(coupler._abd_slots_by_link[link]):
                    meta=slot.geometry().meta()
                    m=float(np.asarray(uipc.view(meta.find("abd_mass"))).reshape(-1)[0])
                    mc=np.asarray(uipc.view(meta.find("abd_mass_x_bar"))).reshape(3).copy()
                    second=np.asarray(uipc.view(meta.find("abd_mass_x_bar_x_bar"))).reshape(3,3).copy()
                    if not np.isfinite(np.r_[m,mc,second.ravel()]).all() or m<=0:
                        raise RuntimeError("Invalid native ABD moments for exact STC feedback")
                    cache.append((env,index,m,mc,second))
                    report["links"].append({"entity":link.entity.name,"link":link.name,
                        "environment":env,"ipc_mass_kg":m,
                        "genesis_mass_kg":float(data.links_mass[index]),
                        "ipc_local_com_m":(mc/m).tolist()})
        data.out_forces.fill(0.)
        data.out_torques.fill(0.)
        for env,index,m,mc,second in cache:
            force,torque=stc_reaction_wrench(data.ipc_transforms[env,index],data.aim_transforms[env,index],
                m,mc,second,coupler._constraint_strength_translation_scaled,
                coupler._constraint_strength_rotation_scaled)
            data.out_forces[env,index]=force
            data.out_torques[env,index]=torque
        if not np.isfinite(data.out_forces).all() or not np.isfinite(data.out_torques).all():
            raise RuntimeError("Nonfinite exact STC feedback wrench")
        # During couple(), Genesis has not advanced since aim_transforms was
        # captured. Thus the current link origin is the target origin.
        coupler.rigid_solver.apply_links_external_wrench(
            data.out_forces if coupler.sim.n_envs>0 else data.out_forces[0],
            data.out_torques if coupler.sim.n_envs>0 else data.out_torques[0],
            links_idx=data.links_idx)
        report["applied_steps"]+=1

    coupler._apply_abd_coupling_forces=exact_feedback
    return report


def initialize_before_ipc(scene, robots, initial_q):
    """Install before scene.build(); preserve MJCF qpos0, set actual initial q.

Genesis joint.desc.init_qpos is the neutral reference qpos0, not an independent
start-pose field. The actual q must be applied after rigid buffers/FK exist but
before IPC constructs its world-space collision bodies.
"""
    if scene.is_built or version("genesis-world") != "1.4.3":
        raise RuntimeError("Initial-pose adapter requires an unbuilt Genesis 1.4.3 scene")
    coupler = scene.sim.coupler
    native_build = coupler.build
    report = {"method":"Actual qpos after rigid solver build, before IPC geometry build",
              "neutral_joint_reference_modified":False,"verified":False}

    def build_with_initial_pose():
        rigid = scene.sim.rigid_solver
        before = rigid.qpos0.to_numpy().copy()
        for side,robot in robots.items():
            q=np.asarray(initial_q[side],dtype=float)
            if q.shape != (robot.n_qs,):
                raise ValueError(f"{side} initial q has wrong shape: {q.shape}")
            rigid.set_qpos(q,qs_idx=np.arange(robot.q_start,robot.q_end),skip_forward=True)
        # Rigid update_forward_pos refreshes both links and collision geometry.
        rigid.update_forward_pos()
        if not np.array_equal(before,rigid.qpos0.to_numpy()):
            raise RuntimeError("Initial-pose adapter changed the neutral reference")
        native_build()
        # Populate IPC state buffers without advancing simulation time.
        coupler._store_gs_rigid_states()
        coupler._retrieve_rigid_states()
        data=coupler._coupling_data
        errors=np.linalg.norm(data.ipc_transforms[...,:3,3]-data.aim_transforms[...,:3,3],axis=-1)
        report["maximum_initial_ipc_genesis_translation_error_m"]=float(errors.max()) if errors.size else 0.
        report["verified"]=bool(not errors.size or errors.max()<1e-7)
        if not report["verified"]:
            raise RuntimeError(f"Initial IPC geometry disagrees with Genesis: {report}")

    coupler.build=build_with_initial_pose
    return report


def configure_shell_shear(scene, *, shear_stiffness_n_per_m=20.0):
    """Set independent membrane shear stiffness for this scene's cloth shells.

In pyuipc 0.0.29's Baraff-Witkin implementation, the triangle shear coefficient
is E_shear/[2(1+nu)] and multiplies triangle area; unlike stretch, its conversion
does not include thickness. Genesis currently passes the same E to both modes.
That makes the default shear much harder than membrane stretch for thin fabric.
This narrow compatibility adapter replaces only the generated per-triangle mu
coefficient before World.init. 20 N/m is an uncalibrated effective cloth value,
not a measured property; change it explicitly when calibration is available.
"""
    if scene.is_built or version("genesis-world")!="1.4.3" or version("pyuipc")!="0.0.29":
        raise RuntimeError("Shell shear adapter requires an unbuilt, version-matched scene")
    if not np.isfinite(shear_stiffness_n_per_m) or shear_stiffness_n_per_m<=0:
        raise ValueError("Effective shear stiffness must be positive")
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    import uipc
    coupler=scene.sim.coupler
    native_add=coupler._add_fem_entities_to_ipc
    report={"scope":"Per-scene pre-initialization FEM shell coefficient conversion",
            "calibrated":False,"shear_stiffness_n_per_m":float(shear_stiffness_n_per_m),
            "thickness_semantics":"IPC one-sided radius r; full thickness h=2r",
            "entities":[]}

    def add_with_independent_shear():
        native_add()
        for slot in SceneVisitor(coupler._ipc_scene).geometries():
            if not isinstance(slot,SimplicialComplexSlot):
                continue
            geometry=slot.geometry()
            meta=read_ipc_geometry_metadata(geometry)
            if meta is None or meta[0]!="cloth":
                continue
            _,env,index=meta
            entity=coupler.fem_solver.entities[index]
            mu=uipc.view(geometry.triangles().find("mu"))
            stretch=uipc.view(geometry.triangles().find("lambda"))
            old=float(mu.mean())
            mu[:]=shear_stiffness_n_per_m
            volume=uipc.view(geometry.vertices().find("volume"))
            report["entities"].append({"entity_index":int(index),"environment":int(env),
                "previous_shear_coefficient":old,"new_shear_coefficient":float(mu.mean()),
                "stretch_coefficient":float(stretch.mean()),
                "full_shell_thickness_m":float(2*entity.material.thickness),
                "native_rest_mass_kg":float(volume.sum()*entity.material.rho)})
        if not report["entities"]:
            raise RuntimeError("No IPC cloth geometry found for independent shear configuration")

    coupler._add_fem_entities_to_ipc=add_with_independent_shear
    return report


def coupling_diagnostics(scene):
    """Read latest IPC contact-body transforms against current Genesis links.

Use after build and/or step. Initialization helper populates the otherwise
uninitialized IPC arrays before the first step. No state positions are changed.
"""
    from scipy.spatial.transform import Rotation
    coupler=scene.sim.coupler
    if scene.sim.cur_step_global==0:
        # Scene.build performs one compilation step then reset(). IPC reset
        # restores frame 0 but does not refresh these Python accessor buffers.
        # Read the recovered world so t=0 diagnostics are not stale warmup data.
        coupler._store_gs_rigid_states()
        coupler._retrieve_rigid_states()
    data=coupler._coupling_data
    values=[]
    for index,link in enumerate(data.links):
        if link.entity.n_dofs==0:
            continue
        ipc=data.ipc_transforms[0,index]
        pos=link.get_pos().detach().cpu().numpy().reshape(3)
        q=link.get_quat().detach().cpu().numpy().reshape(4)
        current=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        u,s,vh=np.linalg.svd(ipc[:3,:3])
        polar=u@vh
        if np.linalg.det(polar)<0:
            u[:,-1]*=-1
            polar=u@vh
        row={"entity":link.entity.name,"link":link.name,
            "translation_error_m":float(np.linalg.norm(ipc[:3,3]-pos)),
            "rotation_error_rad":float(Rotation.from_matrix(polar@current.T).magnitude()),
            "affine_singular_value_deviation":float(np.abs(s-1).max())}
        if "finger" in link.name:
            # Preserve the raw buffers needed to distinguish an origin-based
            # feedback spring from IPC's centre-of-mass STC energy. These are
            # observations only; this function never changes coupling forces.
            import uipc
            geometry=coupler._abd_slots_by_link[link][0].geometry()
            meta=geometry.meta()
            mass=float(np.asarray(uipc.view(meta.find("mass"))).reshape(-1)[0])
            center=np.asarray(uipc.view(meta.find("mass_center"))).reshape(3)
            aim=data.aim_transforms[0,index]
            dc=(ipc[:3,3]+ipc[:3,:3]@center)-(aim[:3,3]+aim[:3,:3]@center)
            origin_delta=ipc[:3,3]-aim[:3,3]
            scale=float(coupler._constraint_strength_translation_scaled)
            origin_formula=scale*float(data.links_mass[index])*origin_delta
            com_formula=scale*mass*dc
            # Panda finger joint axis is local +Y on each mirrored finger.
            opening_axis=current[:,1]
            progressed=scene.sim.cur_step_global>0
            row.update({
                "ipc_transform":ipc.tolist(), "aim_transform":aim.tolist(),
                "genesis_current_position_m":pos.tolist(),
                "genesis_current_rotation_matrix":current.tolist(),
                "ipc_geometry_mass_kg":mass,
                "genesis_link_mass_kg":float(data.links_mass[index]),
                "ipc_geometry_local_center_of_mass_m":center.tolist(),
                "ipc_geometry_local_inertia":np.asarray(uipc.view(meta.find("inertia"))).reshape(3,3).tolist(),
                "origin_delta_ipc_minus_aim_m":origin_delta.tolist(),
                "com_delta_ipc_minus_aim_m":dc.tolist(),
                "opening_axis_world":opening_axis.tolist(),
                "aim_opening_axis_world":aim[:3,1].tolist(),
                "origin_feedback_formula_n":origin_formula.tolist(),
                "com_translation_gradient_formula_n":com_formula.tolist(),
                "origin_formula_along_opening_n":float(origin_formula@opening_axis),
                "com_formula_along_opening_n":float(com_formula@opening_axis),
                "origin_formula_along_aim_opening_n":float(origin_formula@aim[:3,1]),
                "com_formula_along_aim_opening_n":float(com_formula@aim[:3,1]),
                "last_feedback_force_world_n":data.out_forces[0,index].tolist() if progressed else None,
                "last_feedback_torque_world_nm":data.out_torques[0,index].tolist() if progressed else None,
                "translation_gradient_is_not_complete_rigid_wrench":True,
            })
        values.append(row)
    return {"links":values,
        "max_translation_error_m":max((v["translation_error_m"] for v in values),default=0.),
        "max_rotation_error_rad":max((v["rotation_error_rad"] for v in values),default=0.),
        "max_affine_deviation":max((v["affine_singular_value_deviation"] for v in values),default=0.)}
