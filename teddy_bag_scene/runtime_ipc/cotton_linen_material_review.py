"""Read-only native constitutive/source review and finite algebraic probes; no solver."""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parents[1]
paths=[P/'scene.py',P/'physics_config.py',P/'two_panel_builder.py',P/'vendor/uipc/_native/pyuipc/constitution.pyi',P.parent/'genesis-world/genesis/engine/couplers/ipc_coupler/coupler.py',P.parent/'genesis-world/genesis/engine/solvers/fem_solver.py']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
old={'E_Pa':4e6,'nu':.3,'rho_kg_m3':500.,'collision_radius_m':.0003,'body_mu_N_m':50.,'handle_mu_N_m':250.,'bending_N_m':5e-5,'strain_rate_dimensionless':100.}
new={**old,'E_Pa':2e7,'body_mu_N_m':1000.,'handle_mu_N_m':2000.}
for c in (old,new): c['stretch_lambda_N_m']=c['E_Pa']*2*c['collision_radius_m']/(1-c['nu']**2);c['areal_density_kg_m2']=c['rho_kg_m3']*2*c['collision_radius_m']
def W(c,u,v,dot,handle=False):
    er=lambda x:(x-1)**2+c['strain_rate_dimensionless']*max(x-1,0)**3
    return c['stretch_lambda_N_m']*(er(u)+er(v))+c['handle_mu_N_m' if handle else 'body_mu_N_m']*dot**2
# A rhombus with unit material columns: no a/b extension but nonzero principal extension.
c=1.15**2-1
cases=[]
for name,u,v,dot in [('identity',1.,1.,0.),('uniaxial_2pct',1.02,1.,0.),('unit_columns_rhombus_15pct_principal',1.,1.,c)]:
    cases.append({'name':name,'material_axis_lengths':[u,v],'dot_Fa_Fb':dot,'old_body_energy_J_m2':W(old,u,v,dot),'new_body_energy_J_m2':W(new,u,v,dot),'old_handle_energy_J_m2':W(old,u,v,dot,True),'new_handle_energy_J_m2':W(new,u,v,dot,True)})
assert cases[0]['old_body_energy_J_m2']==0
assert abs(cases[1]['new_body_energy_J_m2']/cases[1]['old_body_energy_J_m2']-5)<1e-12
assert abs(cases[2]['new_body_energy_J_m2']/cases[2]['old_body_energy_J_m2']-20)<1e-12
assert abs(cases[2]['new_handle_energy_J_m2']/cases[2]['old_handle_energy_J_m2']-8)<1e-12
commit='ac89e1009c7d8efb72d260790bb5ab31d7460542';base=f'https://github.com/spiriMirror/libuipc/blob/{commit}/'
r={'scope':'Read-only installed Genesis/libuipc material mapping + exact formula checks, not calibrated textile properties or a physical success certificate','production_files_unchanged':hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'source_sha256':hashes,'versions':{'genesis':'1.4.3','pyuipc':'0.0.29','pinned_cpp_commit':commit},'source_evidence':[
{'path':str(paths[0]),'lines':'68-77,99-103','finding':'Genesis Cloth passes E/nu/rho/radius/bending/friction; per-scene mu adapter runs before initialization.'},
{'path':str(paths[1]),'lines':'316-435','finding':'Mutates only triangles.mu; protects lambda/strain_rate/thickness/mass/is_fixed/bending/contact metadata; exact source/native region bijection.'},
{'path':str(paths[4]),'lines':'330-385','finding':'Cloth actually uses StrainLimitingBaraffWitkinShell plus independent DiscreteShellBending; stale NeoHookean docstring is not actual implementation.'},
{'path':str(paths[5]),'lines':'978-987','finding':'Genesis FEM substep is skipped for IPCCoupler; its FEM damping options do not control current cloth.'},
{'url':base+'src/constitution/strain_limiting_baraff_witkin.cpp','finding':'lambda=(Lamé lambda+2mu)*(2radius), mu=shear_moduli.mu(); native area weighting gives both effective coefficients units N/m.'},
{'url':base+'src/backends/cuda/finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h','finding':'W=lambda*sum(e^2+q*max(e,0)^3)+mu*(Fa dot Fb)^2, where e=norm(F a/b)-1. q=strain_rate is dimensionless, conservative, finite, not hard cap or viscous damping.'},
{'url':base+'src/backends/cuda/finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.cu','finding':'Rest triangle normal is rotated onto +Z to define Dm XY axes; a=(1,0),b=(0,1). Energy is W*restarea*dt². Backend copies coefficients at init.'},
{'url':base+'src/constitution/discrete_shell_bending.cpp','finding':'Explicit raw coefficient is independent from E; units N*m via E*(2r)^3/[12*(1-nu²)]. Increasing membrane E need not increase bend stiffness.'}],
'current_material':old,'one_recommended_candidate':new,'unchanged':['s663 teddy and all teddy physical parameters','two-panel U seam/zero-width bottom/geometry/handles','mass/density/thickness','friction/contact/integrator/dt','robot motion/grasp/force and every final validation gate'],
'algebraic_probes':cases,'rhombus_explanation':{'principal_stretches':[math.sqrt(1+c),math.sqrt(1-c)],'angle_between_material_axes_deg':math.degrees(math.acos(c)),'axis_lengths':[1,1],'claim':'This is an exact deformation-gradient counterexample: significant diagonal/principal extension can occur with zero axial stretching term. E alone cannot suppress it. It is not a prediction of any native run.'},
'damping_scope':{'built_in_viscous_material_parameter_found':False,'strain_rate_is_damping':False,'plastic_bending_is_viscous_damping':False,'external_force_animator_in_current_recommendation':False,'limitations':'Installed public constitutions expose no viscous/Rayleigh shell setting. Plastic bending changes rest curvature/yield memory; not generic cotton hysteresis. Explicit velocity drag is possible only with additional callback/state handling, excluded from this narrow material update. Unchanged implicit integration/contact friction dissipate energy, but reduced rebound must be measured: increasing stiffness alone is no damping guarantee.'},
'candidate_rationale':'Increase membrane axial coefficient 5x and body/handle shear 20x/8x; keep very small independent bending coefficient and same mass. These are engineering trial values with meaningful units, not measured cotton/linen calibration. q=100 already makes axial positive strain increasingly costly; retaining it avoids conflating stiffness and damping.',
'required_probe_evidence':['same coupon mesh/mass/clamp/time step and finite force protocol','read back native lambda/mu/strain_rate/thickness/bending/mass before world init','loaded principal stretch and a/b deformation; do not use 15% safety gate as material target','after unloading, actual displacement/kinetic-energy trend with stated observation window','same-motion bag handoff observes drape/grip/contact force; cloth-only coupon cannot certify manipulation'],
'physical_success':False}
out=P/'runtime_ipc/independent-cotton-linen-material-review.json';out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(out);print(hashlib.sha256(out.read_bytes()).hexdigest());print(json.dumps(cases,indent=2))
