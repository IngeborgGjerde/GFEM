'''
Functions solving the Poisson equation

.. math:: 

 -\Delta u = f  \\text{ in } \Omega,  

         u = u_0 \\text{ in } \Omega,

using generalized finite element methods.

All functions take a function space :math:`V`, boundary data :math:`u_0` and right-hand side :math:`f`
and return the solution approximation and the condition number of the stiffness-matrix.

The generalized fems have an optional argument custom_quad. If custom_quad=True the 
quadrature is done on a refined mesh. See quadrature_utils.py for details.
'''


from xii import *   
import numpy as np
from fenics import *
from quadrature_utils import weighted_interpolation_matrix
from petsc4py import PETSc



def StandardFEM(V, u0, f, gamma=None, f_gamma=None):
    '''
    Standard FEM

    Args:
        V (df.function_space): Function space
        gamma (df.mesh): 1D mesh for circle
        u_a (df.expression): analytic solution (for boundary conditions)
        f (df.expression): right-hand side
        gamma (df.mesh): lower-dim domain for Dirac source
        g_famma (df.expression): intensity of Dirac source term

    Returns:
        u (df.function): FE solution 
        k( float): condition number of stiffness matrix
    '''
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Averaging surface is cylinder obtained by sweeping shape along 1d
    
    W = [V]

    a = block_form(W, 2)
    a[0][0] = inner(grad(u), grad(v))*dx

    L = block_form(W, 1)  # The second argument is arity
    L[0] = inner(f, v)*dx
    
    if gamma: 
        dx_ = Measure('dx', gamma)
        T_v = Trace(v, gamma)
        L[0] = inner(f_gamma, T_v)*dx_

    A, b = map(ii_assemble, (a, L))
    W_bcs = [[DirichletBC(V, u0, 'on_boundary')]]

    A, b = apply_bc(A, b, W_bcs)
    uh = ii_Function(W)
    solve(ii_convert(A), uh.vector(), ii_convert(b))
    
    k = np.linalg.cond(A[0][0].array())
    
    return uh[0], k
    
    
    

# The GFEM method will enrich using phi

# The stabilized GFEM will enrich using \bar{phi}=\phi - I(\phi)
# where I(\phi) is the interpolation of phi on the space of linear
# We implement this using a custom expression
class Phi_Bar(UserExpression):
    def __init__(self, degree, phi, phi1, **kwargs):
        self.degree=degree
        self.phi=phi # enrichment function phi
        self.phi1=phi1 # linear interpolant of phi
        super().__init__(**kwargs)
    
    def eval(self, value, x):
        # subtract the linear interpolant
        value[0] = (self.phi(x) -  self.phi1(x))
    
    def value_shape(self):
        return ()


## Generalized FEM
def GFEM(V, phi, mesh_f, u_a, f, custom_quad=True, gamma=None, f_gamma=None):
    '''
    Implentation of a GFEM method with enrichment function phi
    
    Args:
        V (df.function_space): P1 function space 
        phi (df.function): enrichment function
        mesh_f (df.mesh): a refined mesh for the quadrature
        u_a (df.expression): analytic solution (for boundary conditions)
        f (df.expression): rhs
        custom_quad (bool): do quadrature on refined mesh
        gamma (df.mesh): lower-dim domain for Dirac source
        g_famma (df.expression): intensity of Dirac source term
    
    Returns: 
        uh (df.function): Full solution on the refined mesh
        k (float): Condition number of the (full) stiffness matrix
    '''

    # TODO: Implement custom quadrature rule for circle source

    mesh = V.mesh()
        
    phi_i = interpolate(phi, FunctionSpace(mesh, 'CG', 2))
    
    W = [V, V]
    
    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)

    if gamma: 
        dx_ = Measure('dx', gamma)
        T_v1, T_v2, T_phi = (Trace(f, gamma) for f in (v1,v2, phi_i))
    
    u2 = u2*phi_i
    v2 = v2*phi_i
    
    a = block_form(W, 2)
    a[0][0] = inner(grad(u1), grad(v1))*dx
    a[1][0] = inner(grad(u1), grad(v2))*dx
    a[0][1] = inner(grad(u2), grad(v1))*dx
    a[1][1] = inner(grad(u2), grad(v2))*dx
    
    
    L = block_form(W, 1) 
    L[0] = inner(f, v1)*dx
    L[1] = inner(f, phi_i*v2)*dx

    if gamma: 
        L[0] = inner(f_gamma, T_phi*T_v1)*dx_
        L[0] = inner(f_gamma, T_phi*T_v2)*dx_

    
    AA, b = map(ii_assemble, (a, L))
    
    # The blocks we put in were computed using quadrature rules on
    # the coarse mesh
    
    # Using a projection operator from V -> V_f, we can do the quadrature
    # rule on a finer mesh and map the values back to our coarse mesh
    
    if custom_quad:
        # TODO: Implement custom quadrature rule for circle source
        assert gamma==None

        # Make fine functions
        VLf = FunctionSpace(mesh_f, 'CG', 1)
        uf = TrialFunction(VLf)
        vf = TestFunction(VLf)

        # Projection matrix from coarse to fine space
        Ps_m = weighted_interpolation_matrix(W[1], VLf, phi)
        P_m = weighted_interpolation_matrix(W[1], VLf, Constant(1.0))
        
        a11f = assemble(inner(grad(uf),grad(vf))*dx)
        L1f = assemble(inner(f, vf)*dx)
    
        Af, Ps, P = [as_backend_type(m).mat() for m in [a11f, Ps_m, P_m]]
        
        P_T = PETSc.Mat(); P.transpose(P_T)
        Ps_T = PETSc.Mat(); Ps.transpose(Ps_T)

        PsAf = Ps.matMult(Af)
        PAf = P.matMult(Af)
        
        A01 = Matrix(PETScMatrix( PAf.matMult(Ps_T) ))
        A10 = Matrix(PETScMatrix( PsAf.matMult(P_T) ))
        A11 = Matrix(PETScMatrix( PsAf.matMult(Ps_T) ))
        
        L1 = Function(W[1])
        L1 = L1.vector()
        Ps_m.mult(L1f, L1)
        
        
        AA[1][0] = A10
        AA[0][1] = A01
        AA[1][1] = A11
        
        b[1] = L1
    
    V1_bcs = [DirichletBC(W[0], u_a, 'on_boundary')]
    V2_bcs = [DirichletBC(W[1], Constant(0.0), 'on_boundary')]
    
    
    A, b = apply_bc(AA, b, [V1_bcs, V2_bcs])
    
    solver = PETScLUSolver()  # create LU solver
    ksp = solver.ksp()  # get ksp  solver
    pc = ksp.getPC()  # get pc
    pc.setType('hypre')  # set solver to LU
    #pc.setFactorSolverPackage('mumps')  # set LU solver to use mumps
    opts = PETSc.Options()  # get options
    opts['mat_mumps_icntl_4'] = 1  # set amount of info output
    opts['mat_mumps_icntl_14'] = 400 # set percentage increase in estimated working space
    ksp.setFromOptions()  # update ksp with options set above

    comm = mesh.mpi_comm()
    uh = Vector(comm, 2*V.dim())
    
    solver.solve(ii_convert(A), uh, ii_convert(b))  # solve system

    u1, u2 = Function(V), Function(V)

    u1v = u1.vector()
    u1v.set_local(uh.get_local()[:V.dim()])
    u1v.apply('insert')

    u2v = u2.vector()
    u2v.set_local(uh.get_local()[V.dim():(2*V.dim())])
    u2v.apply('insert')

    
    AA=np.bmat([[A[0][0].array(),A[0][1].array()],[A[1][0].array(),A[1][1].array()]])
    
    k = np.linalg.cond(AA)
    
    Vf = FunctionSpace(mesh_f, 'CG', 1)
    uhs = Function(Vf)
    Vfcoords = Vf.tabulate_dof_coordinates()
    for i, x in enumerate(Vfcoords):
        uhs.vector()[i]=u1(x) + phi(x)*u2(x)
    
    return uhs, k


## Stable generalized FEM
def Stable_GFEM(V, phi, mesh_f, u_a, f, custom_quad, gamma=None, f_gamma=None):
    '''
    Implentation of a stable GFEM method with enrichment function phi
    
    Args:
        V (df.function_space): P1 function space 
        phi (df.function): enrichment function
        mesh_f (df.mesh): a refined mesh for the quadrature
        u_a (df.expression): analytic solution (for boundary conditions)
        f (df.expression): rhs
        custom_quad (bool): do quadrature on refined mesh
        gamma (df.mesh): lower-dim domain for Dirac source
        g_famma (df.expression): intensity of Dirac source term
    
    Returns: 
        uh (df.function): Full solution on the refined mesh
        k (float): Condition number of the (full) stiffness matrix
    '''

    mesh = V.mesh()

    phi_i = interpolate(phi, FunctionSpace(mesh, 'CG', 2))
    
    W = [V, V]
    
    u1, u2 = map(TrialFunction, W) #u1 is standard solution, u2 is enriched part of solution
    v1, v2 = map(TestFunction, W) 

    if gamma: 
        dx_ = Measure('dx', gamma)
        T_v1, T_v2, T_phi = (Trace(f, gamma) for f in (v1,v2, phi_i))
    
    # Input blocks
    a = block_form(W, 2)
    a[0][0] = inner(grad(u1), grad(v1))*dx
    a[1][1] = inner(grad(phi_i*u2), grad(phi_i*v2))*dx
    
    # The SGFEM has zero non-diagional blocks (contrary to the GFEM)

    L = block_form(W, 1) 
    L[0] = inner(f, v1)*dx
    L[1] = inner(f, phi_i*v2)*dx
    
    AA, b = map(ii_assemble, (a, L))

    if gamma: 
        L[0] = inner(f_gamma, T_phi*T_v1)*dx_
        L[0] = inner(f_gamma, T_phi*T_v2)*dx_

    
    
    # The blocks we put in were computed using quadrature rules on
    # the coarse mesh
    
    # Using a projection operator from V -> V_f, we can do the quadrature
    # rule on a finer mesh and map the values back to our coarse mesh
    
    if custom_quad:
        # TODO: Implement custom quadrature rule for circle source
        assert gamma==None
        
        # Make fine functions
        VLf = FunctionSpace(mesh_f, 'CG', 1)
        uf = TrialFunction(VLf)
        vf = TestFunction(VLf)

        # Projection matrix from coarse to fine space
        Ps_m = weighted_interpolation_matrix(W[1], VLf, phi) # maps s from V to VLf
        
        # Make the stiffness matrix for fine basis functions
        a11f = assemble(inner(grad(uf),grad(vf))*dx)
        L1f = assemble(inner(f, vf)*dx)
    
        Af, Ps = [as_backend_type(m).mat() for m in [a11f, Ps_m]]
        
        Ps_T = PETSc.Mat(); Ps.transpose(Ps_T)

        # Map the stiffness matrix assembled on the fine mesh
        # onto the coarse mesh
        PsAf = Ps.matMult(Af)
        A11 = Matrix(PETScMatrix( PsAf.matMult(Ps_T) ))
        
        L1 = Function(W[1])
        L1 = L1.vector()
        Ps_m.mult(L1f, L1)
        
        AA[1][1] = A11
        b[1] = L1
        
    V1_bcs = [DirichletBC(W[0], u_a, 'on_boundary')]
    V2_bcs = [DirichletBC(W[1], Constant(0.0), 'on_boundary')]
    
    
    A, b = apply_bc(AA, b, [V1_bcs, V2_bcs])
    
    solver = PETScLUSolver()  # create LU solver
    ksp = solver.ksp()  # get ksp  solver
    pc = ksp.getPC()  # get pc
    pc.setType('hypre')  # set solver to LU
    #pc.setFactorSolverPackage('mumps')  # set LU solver to use mumps
    opts = PETSc.Options()  # get options
    opts['mat_mumps_icntl_4'] = 1  # set amount of info output
    opts['mat_mumps_icntl_14'] = 400 # set percentage increase in estimated working space
    ksp.setFromOptions()  # update ksp with options set above

    comm = mesh.mpi_comm()
    uh = Vector(comm, 2*V.dim())
    
    solver.solve(ii_convert(A), uh, ii_convert(b))  # solve system

    u1, u2 = Function(V), Function(V)

    u1v = u1.vector()
    u1v.set_local(uh.get_local()[:V.dim()])
    u1v.apply('insert')

    u2v = u2.vector()
    u2v.set_local(uh.get_local()[V.dim():(2*V.dim())])
    u2v.apply('insert')

    AA=np.bmat([[A[0][0].array(),A[0][1].array()],[A[1][0].array(),A[1][1].array()]])
    
    #LL=np.bmat([b[1]])
    k = np.linalg.cond(AA)
    
    Vf = FunctionSpace(mesh_f, 'CG', 1)
    uh_f = Function(Vf)
    Vfcoords = Vf.tabulate_dof_coordinates()
    for i, x in enumerate(Vfcoords):
        uh_f.vector()[i]=u1(x)+ phi(x)*u2(x)
        
    return uh_f, k
