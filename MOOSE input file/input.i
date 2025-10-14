# input file.
# Define mesh. 2-D system, simulation size 200*200.
[Mesh]
  type = GeneratedMesh
  dim = 2
  nx = 200
  xmax = 200
  ny = 200
  ymax = 200
[]
# variables. w: chemical potential, eta: order parameter, pot: applied overpotential.
[Variables]
  [w]
  []
  [eta]
  []
  [pot]
  []
[]
# Creating functions for initial conditions.
[Functions]
  [ic_func_eta] 
    type = ParsedFunction 
    expression = 0.5*(1.0-1.0*tanh((x-20)*2))
  []
  [ic_func_c]
    type = ParsedFunction
    expression = 0 
  []
  [ic_func_pot] 
    type = ParsedFunction
    expression = -0.225*(1.0-tanh((x-20)*2))
  []
[]
# Initial conditions.
[ICs]
  [eta]
    variable = eta
    type = FunctionIC
    function = ic_func_eta
  []
  [w]
    variable = w
    type = FunctionIC
    function = ic_func_c
  []
  [pot]
    variable = pot
    type = FunctionIC
    function = ic_func_pot
  []
[]
# Boundary conditions.
[BCs]
  [bottom_eta]
    type = NeumannBC
    variable = 'eta'
    boundary = 'bottom'
    value = 0
  []
  [top_eta]
    type = NeumannBC
    variable = 'eta'
    boundary = 'top'
    value = 0
  []
  [left_eta]
    type = DirichletBC
    variable = 'eta'
    boundary = 'left'
    value = 1
  []
  [right_eta]
    type = DirichletBC
    variable = 'eta'
    boundary = 'right'
    value = 0
  []
  [bottom_w]
    type = NeumannBC
    variable = 'w'
    boundary = 'bottom'
    value = 0
  []
  [top_w]
    type = NeumannBC
    variable = 'w'
    boundary = 'top'
    value = 0.0
  []
  [left_w]
    type = NeumannBC
    variable = 'w'
    boundary = 'left'
    value = 0
  []
  [right_w]
    type = DirichletBC
    variable = 'w'
    boundary = 'right'
    value = 0.0
  []
  [left_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'left'
    value = -0.47
  []
  [right_pot]
    type = DirichletBC
    variable = 'pot'
    boundary = 'right'
    value = 0
  []
[]

[Kernels]
  # First part of equation 3 in main text . chi*dw/dt
  [w_dot]
    type = SusceptibilityTimeDerivative
    variable = w
    f_name = chi
    coupled_variables = 'w'
  []
  # Intrinsic diffusion part of equation 3 in main text.
  [Diffusion1]
    type = MatDiffusion
    variable = w
    diffusivity = D
  []
  # Migration.
  [Diffusion2]
    type = Migration
    variable = w
    cv = eta
    Q_name = 0.
    QM_name = DN
    cp = pot
  []
  # Coupling between w and eta.
  [coupled_etadot]
    type = CoupledSusceptibilityTimeDerivative
    variable = w
    v = eta
    f_name = ft
    coupled_variables = 'eta'
  []
  # Conduction, left handside of equation 4 in main text.
  [Cond]
    type = Conduction
    variable = pot
    cp = eta
    cv = w
    Q_name = Le1
    QM_name = 0.
  []
  # Source term for Equation 4 in main text.
  [coupled_pos]
    type = CoupledSusceptibilityTimeDerivative
    variable = pot
    v = eta
    f_name = ft2
    coupled_variables = 'eta'
  []
  # Bulter-volmer equation, right hand side of Equation 1 in main text.
  [BV]
    type = Kinetics
    variable = eta
    f_name = G
    cp = pot
    cv = eta
  []
  # Driving force from switching barrier, right hand side of Equation 1 in main text.
  [AC_bulk]
    type = AllenCahn
    variable = eta
    f_name = FF
  []
  # interfacial energy
  [AC_int]
    type = ACInterface
    variable = eta
  []
  [Noiseeta]
    type = LangevinNoise
    variable = eta
    amplitude = 0.04
  []
  # deta/dt
  [e_dot]
    type = TimeDerivative
    variable = eta
  []
[]

[Materials]

  [constants]
    type = GenericConstantMaterial
    # kappa_op: gradient coefficient;  M0:diffucion coefficient of Li+ in electrolyte
    #  S1, S2 conductivity of electrode and electrolyte; L: kinetic coefficient; Ls: electrochemical kinetic coefficient; B: Barrier height;
    #  es, el: difference in the chemical potential of lithium and neutral components on the electrode/electrolyte phase at initial equilibrium state;
    # us, ul: free energy density of the electrode/electrolyte phases. Defined in Ref. 20 and 26 of the main text; A: prefactor; AA: nF/R; T: temperature (K);
    # dv is the ratio of site density for the electrode/electrolyte phases; ft2: normalized used in Equation 4.

    prop_names  = 'kappa_op   M0_ref   S1       S2     L      Ls_ref B     es      el     A     ul     us     AA        dv    ft2         T     R            T0    ED      EK'
    prop_values = '0.3        317.9    1000000  1.19   6.25   0.001  2.4   -13.8   2.631  1.0   0.0695 13.8   11605.124  5.5   0.0074    343   8.314462618 300   5.0e4  3.0e4'
  []
  # grand potential of electrolyte phase
  [liquid_GrandPotential]
    type = DerivativeParsedMaterial
    expression = 'ul-A*log(1+exp((w-el)/A))'
    coupled_variables = 'w'
    property_name = f1
    material_property_names = 'A ul el'
  []
  # grand potential of electrode phase
  [solid_GrandPotential]
    type = DerivativeParsedMaterial
    expression = 'us-A*log(1+exp((w-es)/A))'
    coupled_variables = 'w'
    property_name = f2
    material_property_names = 'A us es'
  []
  #interpolation function
  [switching_function]
    type = SwitchingFunctionMaterial
    eta = 'eta'
    h_order = HIGH
  []
  # Barrier function 
  [barrier_function]
    type = BarrierFunctionMaterial
    eta = eta
  []
  [total_GrandPotential]
    type = DerivativeTwoPhaseMaterial
    coupled_variables = 'w'
    eta = eta
    fa_name = f1
    fb_name = f2
    derivative_order = 2
    W = 2.4
  []
  # Coupling between eta and w
  [coupled_eta_function]
    type = DerivativeParsedMaterial
    expression = '-(cs*dv-cl)*dh' # in this code cs=-cs h=eta dh=1
    coupled_variables = 'w eta'
    property_name = ft
    material_property_names = 'dh:=D[h,eta] h dv cs:=D[f2,w] cl:=D[f1,w]'
    derivative_order = 1
  []
  [susceptibility]
    type = DerivativeParsedMaterial
    expression = '-d2F1*(1-h)-d2F2*h*dv'
    coupled_variables = 'w'
    property_name = chi
    derivative_order = 1
    material_property_names = 'h dv d2F1:=D[f1,w,w] d2F2:=D[f2,w,w]'
  []

  # --- Temperature-dependent parameters (Arrhenius) ---
  # AA/T already used in expressions; here we add M0T and LsT by Arrhenius laws
  [Arrhenius_M0]
    type = ParsedMaterial
    property_name = M0T
    material_property_names = 'M0_ref ED R T0 T'
    expression = 'M0_ref*exp( ED/R * (1./T0 - 1./T) )'
  []

  [Arrhenius_Ls]
    type = ParsedMaterial
    property_name = LsT
    material_property_names = 'Ls_ref EK R T0 T'
    expression = 'Ls_ref*exp( EK/R * (1./T0 - 1./T) )'
  []

  # Mobility defined by D*c/(R*T), whereR*T is normalized by the chemical potential
  # M0*(1-h) is the effective diffusion coefficient; cl*(1-h) is the ion concentration
  [Mobility_coefficient]
    type = DerivativeParsedMaterial
    expression = '-M0T*(1-h)*cl*(1-h)' #c is -c
    property_name = D
    coupled_variables = 'eta w'
    derivative_order = 1
    material_property_names = ' M0T cl:=D[f1,w] h'
  []
  # Energy of the barrier
  [Free]
    type = DerivativeParsedMaterial
    property_name = FF
    material_property_names = 'B'
    coupled_variables = 'eta'
    expression = 'B*eta*eta*(1-eta)*(1-eta)'
    derivative_order = 1
  []
  # Migration coefficient.
  [Migration_coefficient]
    type = DerivativeParsedMaterial
    # expression = '-cl*(1-h)*AA*M0*(1-h)'
    # expression = '-(1-t_plus)*cl*(1-h)*(AA/T)*M0T*(1 - h)' 
    expression = '-cl*(1-h)*(AA/T)*M0T*(1 - h)'
    coupled_variables = 'eta w'
    property_name = DN
    derivative_order = 1
    material_property_names = 'M0T AA cl:=D[f1,w] h t_plus T'
  []
  [Bultervolmer]
    type = DerivativeParsedMaterial
    expression = 'LsT*(exp(pot*AA/(2*T))+14.89*cl*(1-h)*exp(-pot*AA/(2*T)))*dh'
    coupled_variables = 'pot eta w'
    property_name = G
    derivative_order = 1
    material_property_names = 'LsT dh:=D[h,eta] h cl:=D[f1,w] AA T'
    outputs = exodus
  []
  # output the ion concentration
  [concentration]
    type = ParsedMaterial
    property_name = c
    coupled_variables = 'eta w'
    material_property_names = 'h dFl:=D[f1,w]'
    expression = '-dFl*(1-h)'
    outputs = exodus
  []
  # Effective conductivity
  [Le1]
    type = DerivativeParsedMaterial
    property_name = Le1
    coupled_variables = 'eta'
    material_property_names = 'S1 S2 h'
    expression = 'S1*h+S2*(1-h)'
    derivative_order = 1
  []
[]
[GlobalParams]
  enable_jit = false # Disable JIT
[]

[Preconditioning]
  [SMP]
    type = SMP
    full = true
    petsc_options_iname = '-pc_type -ksp_grmres_restart -sub_ksp_type -sub_pc_type -pc_asm_overlap'
    petsc_options_value = 'asm      121                  preonly       lu           8'
  []
[]

[Executioner]
  type = Transient
  scheme = bdf2
  #solve_type =Newton
  l_max_its = 50
  l_tol = 1e-4
  nl_max_its = 20
  nl_rel_tol = 1e-6
  nl_abs_tol = 1e-6
  dt = 0.02
  end_time = 1000
[]

[Outputs]
  exodus = true
  csv = true
  perf_graph = true
  #checkpoint = true
  #execute_on = 'TIMESTEP_END'
  [other] # creates input_other.e output every 30 timestep
    type = Exodus
    time_step_interval = 30
  []
  [Checkpoint]
    type = Checkpoint
    execute_on = 'TIMESTEP_END'  
    time_step_interval = 100      
    num_files = 3                 
  []
[]
