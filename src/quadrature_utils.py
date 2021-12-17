##################################
##        Helper functions      ##
##################################

from fenics import *
import numpy as np

from petsc4py import PETSc
def weighted_interpolation_matrix(V, Vf, weight):
    '''
    Projection function P: V -> Vf mapping a function s
    from a coarse mesh to a refined mesh
    
    Input:
    V: Coarse function space
    Vf: Fine function space
    weight: weight function 
    
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
    
    weight = interpolate(weight, Vf)

    # We loop through the dofs in V, setting at each iteration f = phi N_i
    # (i.e the hat function corresponding to this dof)
    # We then interpolate f_i onto the finer function space
    # and record the dofs in this function space
    f = Function(V)
    for i in range(V.dim()):
        f.vector()[i]=1.0
        f_i = interpolate(f, Vf)
        
        col_values = f_i.vector().get_local()*weight.vector().get_local()
        col_indices = np.array(range(Vf.dim()), dtype='int32')
        row_indices = [i]

        P.setValues(row_indices, col_indices, col_values, PETSc.InsertMode.INSERT_VALUES)

        f.vector()[i]=0.0

    P.assemblyEnd()
    return PETScMatrix(P)



def refine_mesh(mesh, Nrefs, points):
    
    p1, p2 = points
    
    for i in range(Nrefs):

        cell_markers = MeshFunction('bool', mesh, 1)
        cell_markers.set_all(False)
            
        for cell in cells(mesh):
            # Check if cell is in the interval (p1, p2)
            if cell.midpoint().x()-p1 >= DOLFIN_EPS or cell.midpoint().x()-p2<=DOLFIN_EPS:
                cell_markers[cell] = True
        mesh = refine(mesh, cell_markers)
    
    return mesh



### Unit test ###
def test_custom_quadrature_method():
    
    # We test the interpolation using different function weight functions phi

    # phi = 1
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, 'CG', 1)

    meshf = refine(refine(refine(mesh)))
    Vf = FunctionSpace(meshf, 'CG', 1)

    phi = Constant(1.0)
    P = weighted_interpolation_matrix(V, Vf, weight = phi)
    Af = assemble(inner(grad(TrialFunction(Vf)), grad(TestFunction(Vf)))*dx)
    
    P = as_backend_type(P).mat()
    Af = as_backend_type(Af).mat()
    P_T = PETSc.Mat(); P.transpose(P_T)

    # Map the stiffness matrix assembled on the fine mesh
    # onto the coarse mesh
    PsAf = P.matMult(Af)
    A = Matrix(PETScMatrix( PsAf.matMult(P_T) ))

    # A should now equal the standard stiffness matrix
    A_ref = assemble(inner(grad(TrialFunction(V)), grad(TestFunction(V)))*dx)

    assert np.linalg.norm(A_ref.array()-A.array()) < 1e-8

    # phi = e^x
    mesh = UnitIntervalMesh(2)
    c = mesh.coordinates()
    c[:,0] *= 2
    V = FunctionSpace(mesh, 'CG', 1)
    
    # hat function is now N=1-x on the leftmost cell
    
    # find corresponding dof on left boundary
    dof = np.where(V.tabulate_dof_coordinates()==0)[0][0]

    # Calculating we find (grad(e^x N_i), grad(e^x N_i))=(e^(2x)*(1-x)^2+2*e^(2x)*(x-1)+e^(2x)) for this dof
    # According to wolfram alpha, \int_0^1 (e^(2x)*(1-x)^2+2*e^(2x)*(x-1)+e^(2x))*dx = 0.25*(e^2-1) \approx 1.5973 
    # We check that the custom quadrature method returns something similar 
    
    meshf = refine(refine(refine(refine(refine(refine(refine(refine(mesh))))))))
    Vf = FunctionSpace(meshf, 'CG', 1)

    phi = Expression('exp(x[0])', degree=2)
    P = weighted_interpolation_matrix(V, Vf, weight = phi)
    Af = assemble(inner(grad(TrialFunction(Vf)), grad(TestFunction(Vf)))*dx)
    
    P = as_backend_type(P).mat()
    Af = as_backend_type(Af).mat()
    P_T = PETSc.Mat(); P.transpose(P_T)

    # Map the stiffness matrix assembled on the fine mesh
    # onto the coarse mesh
    PAf = P.matMult(Af)
    A = Matrix(PETScMatrix( PAf.matMult(P_T) ))

    value_dof = 0.25*(np.exp(1)**2-1)
    assert np.abs(A.array()[dof, dof] - value_dof) < 0.01
    print( 'quad_error custom', A.array()[dof, dof] - value_dof )

    # For comparison it's interesting to see how the standard quadrature does
    interpolation_degrees = [1, 2, 4, 8, 16]
    print('Quad error (fenics) with interpolation degree i')
    for i in interpolation_degrees:
        phi_i = interpolate(phi, FunctionSpace(mesh, 'CG', i))
        A = assemble(inner(grad(phi_i*TrialFunction(V)), grad(phi_i*TestFunction(V)))*dx)
        print( '  % 1.2e  '% (A.array()[dof, dof] - value_dof ))


def make_babushka_quadrature_table():

    # phi = x^alpha
    # Calculating we find int_0^1 (grad(x^alpha N_i), grad(x^alpha N_i))= a/(4a^2-1) for this dof
    # We check that the custom quadrature method returns something similar 

    # We test the assembly on the same type of mesh, checking the dof value corresponding to the left boundary
    
    # TODO: This fails if alpha -> 0.51 !!

    mesh = UnitIntervalMesh(2)
    c = mesh.coordinates()
    c[:,0] *= 2
    V = FunctionSpace(mesh, 'CG', 1)
    
    # hat function is now N=1-x on the leftmost cell
    
    # find corresponding dof on left boundary
    dof = np.where(V.tabulate_dof_coordinates()==0)[0][0]

    alpha = 0.8
    meshf = refine(mesh)   
    
    for i in range(0, 8):
        meshf = refine(meshf)
        Vf = FunctionSpace(meshf, 'CG', 1)
    
        phi = Expression('pow(x[0]+0.00000001, alpha)', degree=2, alpha=alpha)
        value_dof = alpha/(4*alpha**2.0-1)
    
        P = weighted_interpolation_matrix(V, Vf, weight = phi)
        Af = assemble(inner(grad(TrialFunction(Vf)), grad(TestFunction(Vf)))*dx)
    
        P = as_backend_type(P).mat()
        Af = as_backend_type(Af).mat()
        P_T = PETSc.Mat(); P.transpose(P_T)
    
        # Map the stiffness matrix assembled on the fine mesh
        # onto the coarse mesh
        PAf = P.matMult(Af)
        A = Matrix(PETScMatrix( PAf.matMult(P_T) ))
        
        #assert np.abs(A.array()[dof, dof] - value_dof) < 0.01, print(A.array()[dof, dof], value_dof)
        print( '%1.2e %1.2e'%(meshf.hmin(), A.array()[dof, dof] - value_dof))

    # For comparison it's interesting to see how the standard quadrature does
    interpolation_degrees = [1, 2, 4, 8, 16, 32]
    print('Quad error (fenics) with interpolation degree i')
    for i in interpolation_degrees:
        phi_i = interpolate(phi, FunctionSpace(mesh, 'CG', i))
        A = assemble(inner(grad(phi_i*TrialFunction(V)), grad(phi_i*TestFunction(V)))*dx)
        print( '%i  % 1.2e  '% (i, A.array()[dof, dof] - value_dof ))




def main():
    make_babushka_quadrature_table()

main()
    

