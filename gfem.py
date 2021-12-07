from xii import *   
import numpy as np
from fenics import *


def refine_mesh(mesh, Nrefs, points):
    
    p1, p2 = points
    
    for i in range(Nrefs):
        
        tol = mesh.hmin()*5.0

        cell_markers = MeshFunction('bool', mesh, 1)
        cell_markers.set_all(False)
            
        for cell in cells(mesh):
            if (cell.midpoint().x()-p1)**2.0 < tol**2.0 or (cell.midpoint().x()-p2)**2.0 < tol**2.0:
                cell_markers[cell] = True
        #mesh = refine(mesh, cell_markers)
    
    mesh = refine(refine(refine(refine(refine(refine(mesh))))))

    return mesh

## Standard FEM
def StFEM(V, u_a, f):
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v))*dx
    L = f*v*dx
    
    A, b = assemble(a), assemble(L)
    
    # Define boundary condition
    bc = DirichletBC(V, u_a, 'on_boundary')

    # Compute solution
    u = Function(V)
    bc.apply(A, b)
    solve(A, u.vector(), b)
    
    k = np.linalg.cond(A.array())
    
    return u, k



## Generalized FEM
def GFEM(V, s, mesh_f, u_a, f, custom_quad):
    '''
    Implentation of a GFEM method
    
    Input: 
    V: P1 function space 
    s: enrichment function
    mesh_f: a refined mesh for the quadrature
    u_a: analytic solution (for boundary conditions)
    f: right-hand side 
    custom_quad (bool): do quadrature on refined mesh
    
    Returns: 
    uh: Full solution on the refined mesh
    k: Condition number of the (full) stiffness matrix
    '''

    mesh = V.mesh()
    h = mesh.hmin()
        
    si = interpolate(s, FunctionSpace(mesh, 'CG', 8))
    
    W = [V, V]
    
    u1, u2 = map(TrialFunction, W)
    v1, v2 = map(TestFunction, W)
    
    a = block_form(W, 2)
    a[0][0] = inner(grad(u1), grad(v1))*dx
    a[1][0] = inner(grad(u1), grad(si*v2))*dx
    a[0][1] = inner(grad(u2*si), grad(v1))*dx
    a[1][1] = inner(grad(si*u2), grad(si*v2))*dx
    
    
    L = block_form(W, 1) 
    L[0] = inner(f, v1)*dx
    L[1] = inner(f, si*v2)*dx
    
    AA, b = map(ii_assemble, (a, L))
    
    # The blocks we put in were computed using quadrature rules on
    # the coarse mesh
    
    # Using a projection operator from V -> V_f, we can do the quadrature
    # rule on a finer mesh and map the values back to our coarse mesh
    
    if custom_quad:
        # Make fine functions
        VLf = FunctionSpace(mesh_f, 'CG', 1)
        uf = TrialFunction(VLf)
        vf = TestFunction(VLf)

        # Projection matrix from coarse to fine space
        Ps_m = make_projection(W[1], VLf, s)
        P_m = make_projection(W[1], VLf, Constant(1.0))
        
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
    
    #LL=np.bmat([b[1]])
    k = np.linalg.cond(AA)
    
    Vf = FunctionSpace(mesh_f, 'CG', 1)
    uhs = Function(Vf)
    Vfcoords = Vf.tabulate_dof_coordinates()
    for i, x in enumerate(Vfcoords):
        uhs.vector()[i]=u1(x) + s(x)*u2(x)
    return uhs, k


## Stable generalized FEM
def Stable_GFEM(V, s, mesh_f, u_a, f, custom_quad):
    '''
    Implentation of a stable GFEM method
    
    Input: 
    V: P1 function space 
    s: enrichment function
    mesh_f: a refined mesh for the quadrature
    u_a: analytic solution (for boundary conditions)
    custom_quad (bool): do quadrature on refined mesh
    
    Returns: 
    uh: Full solution on the refined mesh
    k: Condition number of the (full) stiffness matrix
    '''
    mesh = V.mesh()

    si = interpolate(s, FunctionSpace(mesh, 'CG', 2))
    
    W = [V, V]
    
    u1, u2 = map(TrialFunction, W) #u1 is standard solution, u2 is enriched part of solution
    v1, v2 = map(TestFunction, W) 
    
    # Input blocks
    a = block_form(W, 2)
    a[0][0] = inner(grad(u1), grad(v1))*dx
    a[1][1] = inner(grad(si*u2), grad(si*v2))*dx
    
    L = block_form(W, 1) 
    L[0] = inner(f, v1)*dx
    L[1] = inner(f, si*v2)*dx
    
    AA, b = map(ii_assemble, (a, L))
    
    
    # The blocks we put in were computed using quadrature rules on
    # the coarse mesh
    
    # Using a projection operator from V -> V_f, we can do the quadrature
    # rule on a finer mesh and map the values back to our coarse mesh
    
    if custom_quad:
        # Make fine functions
        VLf = FunctionSpace(mesh_f, 'CG', 1)
        uf = TrialFunction(VLf)
        vf = TestFunction(VLf)

        # Projection matrix from coarse to fine space
        Ps_m = make_projection(W[1], VLf, s) # maps s from V to VLf
        
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
        uh_f.vector()[i]=u1(x)+ s(x)*u2(x)
        
    return uh_f, k



from petsc4py import PETSc
def make_projection(V, Vf, s):
    '''
    Projection function P: V -> Vf mapping a function s
    from a coarse mesh to a refined mesh
    
    Input:
    V: Coarse function space
    Vf: Fine function space
    s: 
    
    Returns: 
    P: Projection matrix
    '''
    comm = V.mesh().mpi_comm()
    P = PETSc.Mat()
    P.create(comm)
    P.setSizes([[V.dofmap().index_map().size(IndexMap.MapSize.OWNED),
                   V.dofmap().index_map().size(IndexMap.MapSize.GLOBAL)],
                 [Vf.dofmap().index_map().size(IndexMap.MapSize.OWNED),
                   Vf.dofmap().index_map().size(IndexMap.MapSize.GLOBAL)]])
    P.setType('aij')
    P.setUp()
    
    s = interpolate(s, Vf)

    # We loop through the dofs in V, setting at each iteration f = s N_i
    # (i.e the hat function corresponding to this dof)
    # We then interpolate f_i onto the finer function space
    # and record the dofs in this function space
    f = Function(V)
    for i in range(V.dim()):
        f.vector()[i]=1.0
        f_i = interpolate(f, Vf)
        
        col_values = f_i.vector().get_local()*s.vector().get_local()
        col_indices = np.array(range(Vf.dim()), dtype='int32')
        row_indices = [i]

        P.setValues(row_indices, col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)

        f.vector()[i]=0.0

    P.assemblyEnd()
    return PETScMatrix(P)