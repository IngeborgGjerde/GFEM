from xii import *   
import numpy as np
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt

from fenics import *
set_log_level(LogLevel.ERROR)
import fems as solvers
from quadrature_utils import refine_mesh
from mshr import *

## TODO: So far we only solve here for the circle source problem -> expand suite of test problems in 2D
## TODO: Fix plotting of 2D approximation

def main():

    '''
    Script for running and comparing GFEM methods on different test problems in 2D
    '''

    # Set up testproblem
    # So far we can choose between power function and mollifier
    import testproblems
    center=[0.5, 0.5]
    radius = 0.1

    u_a, f_gamma, phi = testproblems.fundsol_circlesource(center, radius, plot_sol=True)


    # Solve using standard FEM, GFEM and Stable GFEM
    # on finer and finer meshes, recording and printing errors as we go
    form_str = '%1.3f (%1.1f)  %1.3f(%1.1f)  %1.1e   '
    form_str = '%1.3f  '+form_str+form_str+form_str

    print('       Standard FEM                       Stable GFEM                        Enr. FEM ')
    print('h      |u_e|_L2     |u_e|_H1    k         |u_e|_L2     |u_e|_H1    k         |u_e|_L2    |u_e|_H1   k ')

    uhs, uh_enrs, uh_sgfems, error, error_enr, error_st_enr, hs = [],[],[],[],[],[],[]



    Ns = [2, 4, 8, 16]

    for ix_N, N in enumerate(Ns):
        
        # Set up mesh and function spaces
        mesh = UnitSquareMesh(N, N) #2d mesh
        mesh_f = refine_mesh(mesh, 3, [0.2, 0.8]) # refined mesh for quadrature rule

        # 1d mesh for circle source
        circlemesh = generate_mesh(Circle(Point(*center), radius), 80)
        gamma = BoundaryMesh(circlemesh, 'exterior')

        V, V3 = [FunctionSpace(mesh, 'CG', i) for i in [1,3]]
        
        
        # Enrichment functions
        phi1 = interpolate(phi, FunctionSpace(mesh, 'CG', 1)) # I(phi) on linear space
        phi_bar = solvers.Phi_Bar(degree=3, phi=phi, phi1=phi1) # phi_bar = phi-I(phi), used for Stab GFEM

        
        # Solve using different FE methods
        uh, k = solvers.StandardFEM(V, u_a, Constant(0.0), gamma=gamma, f_gamma=f_gamma)

        uh_enr, k_enr = solvers.GFEM(V, phi, mesh_f, u_a, f_gamma, custom_quad=False, gamma=gamma, f_gamma=f_gamma)
        
        uh_st_enr, k_st_enr = solvers.Stable_GFEM(V, phi_bar, mesh_f, u_a, f_gamma, custom_quad=False, gamma=gamma, f_gamma=f_gamma)
        
        uhs.append(uh); uh_enrs.append(uh_enr); uh_sgfems.append(uh_st_enr)
        
        
        # Interpolate analytic solution
        u_ai = interpolate(u_a, V3) 
        uf_ai = interpolate(u_a, FunctionSpace(mesh_f, 'CG', 3))
        
        
        # Calculate errors and convergence rates
        error.append([errornorm(uh, u_ai, norm) for norm in ['L2', 'H10']])
        error_enr.append([errornorm(uh_enr, uf_ai, norm) for norm in ['L2', 'H10']])
        error_st_enr.append([errornorm(uh_st_enr, uf_ai, norm) for norm in ['L2', 'H10']])
        hs.append(mesh.hmin())
        
        if ix_N==0: 
            r, r_enr, r_st_enr = [0,0], [0,0], [0,0]
        else:
            dh = np.log(hs[ix_N])-np.log(hs[ix_N-1])

            derror, derror_enr, derror_st_enr = [ [np.log(e_list[ix_N][n])-np.log(e_list[ix_N-1][n]) for n in [0,1]]
                                                                for e_list in [error, error_enr, error_st_enr] ]
            r, r_enr, r_st_enr = derror/dh, derror_enr/dh, derror_st_enr/dh
            
        # Make string showing the formatted errors
        print( form_str %(mesh.hmin(), error[-1][0], r[0], error[-1][1], r[1], k, 
                            error_st_enr[-1][0], r_st_enr[0], error_st_enr[-1][1], r_st_enr[1], k_st_enr,
                            error_enr[-1][0], r_enr[0], error_enr[-1][1], r_enr[1], k_enr) )

    
    
    # Plot solution
    fig, axs = plt.subplots(1,len(Ns),figsize=(15,5))
    fig.suptitle('FEM solutions')


    for i in range(len(axs)):
        axs[i].plot(uh_enrs[i].function_space().tabulate_dof_coordinates(), uh_enrs[i].vector().get_local(), 'b-', label='$u_h^{enr}$')
        
        axs[i].plot(uh_sgfems[i].function_space().tabulate_dof_coordinates(), uh_sgfems[i].vector().get_local(), 'g.-', label='$u_h^{gen, stable}$')

        axs[i].plot(uhs[i].function_space().tabulate_dof_coordinates(), uhs[i].vector().get_local(), 'ko-', label='$u_h$')
        
        uf_a1 = interpolate(u_a, uh_enrs[i].function_space())
        axs[i].plot(uf_a1.function_space().tabulate_dof_coordinates(), uf_a1.vector().get_local(), 'r', label='$u_a$')
        axs[i].legend()
        axs[i].set_xlabel('x')
        axs[i].set_title('N=%i'%Ns[i])
        
    fig.savefig('alpha.png')


main()