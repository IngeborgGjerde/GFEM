'''
Different manufactured testprolbmes problems solving :math:`-\Delta u_a = f` on the unit interval
'''


from fenics import *
import sympy as sym


def power_function(alpha = 0.51, plot_sol = False):
    '''
    Babushka example: :math:`u=x^{\\alpha}`  with :math:`0.5 < \\alpha < 1.5`

    Args:
        alpha (float): power coefficient, default value 0.51.
                       We assume 0.5 < alpha < 1.5 
        plot_sol (bool): plot solution, default=False

    Returns:
        u_a (df.expression): analytic solution
        f (df.expression): corresponding rhs
        phi (df.expression): corresponding enrichment function

    '''

    assert 0.5 < alpha < 1.5, 'Need 0.5 < alpha < 1.5'
    x = sym.symbols('x[0]')

    
    eps  = 1.0e-3 # we add in an epsilon to avoid dividing by 0 in f
    
    u_a = (x+eps)**alpha
    f = -alpha*(alpha-1.0)*(x+eps)**(alpha-2.0) # corresponding rhs
    
    # Enrichment function
    phi = u_a #
    
    # Convert from sympy to fenics expressions
    u_a, phi, f = [sym.printing.ccode(expr) for expr in [u_a, phi, f]]
    u_a, phi, f = [Expression(func, degree=2) for func in [u_a, phi, f]]
    
    if plot_sol:
        # mesh to plot on 
        Vfine = FunctionSpace(UnitIntervalMesh(200), 'CG', 1)
        plot_analytic_sol(u_a, f, Vfine)

    return u_a, f, phi

def steep_mollifier(beta = 2.5, plot_sol = False):
    '''
    Mollifier with steep gradients: :math:`u= \\exp(-1/((1-\\beta^2)(x-0.5)^2)) / \\exp(-1)`.

    Args:
        beta (float): steepness coefficient, default value 2.5
        plot_sol (bool): plot solution, default=False

    Returns:
        u_a (df.expression): analytic solution
        f (df.expression): corresponding rhs
        phi (df.expression): corresponding enrichment function
    '''
    

    x = sym.symbols('x[0]')

    u_a = -sym.exp(-1.0/(1.0-beta**2.0*(x-0.5)**2.0))
    u_a *= 1.0/sym.exp(-1.0) #normalize to 1
    
    f = -sym.diff(sym.diff(u_a,x),x) # corresponding rhs
    
    # Enrichment function
    phi = u_a

    # We need a cut-off function as u_a is unstable in the region it should be zero
    Ie = sym.LessThan(sym.Abs(x-0.5), 1.0/beta - 0.01) # Cut-off function

    # Convert from sympy to fenics expressions
    u_a, phi, f, Ie = [sym.printing.ccode(expr) for expr in [u_a, phi, f, Ie]]

    u_a = '(' + u_a + ')*' + '(' + Ie + ')' # sympy cannot convert expressions 
    phi = '(' + phi + ')*' + '(' + Ie + ')' # involving "less than" to ccode
    f = '(' + f + ')*' + '(' + Ie + ')'     # so we add the cut-off function ourselves

    u_a, phi, f = [Expression(func, degree=2) for func in [u_a, phi, f]]

    if plot_sol: 
        # mesh to plot on 
        Vfine = FunctionSpace(UnitIntervalMesh(200), 'CG', 1)
        plot_analytic_sol(u_a, f, Vfine)

    return u_a, f, phi



def fundsol_circlesource(center, radius, plot_sol = False):
    '''
    Solution for the Poisson equation in a 2D domain with a circle source: 
    :math:`-\\Delta u= f \\delta_\\Gamma` where :math:`\\delta_\\Gamma` is the circle source (1D),

    Args:
        center (list): [x,y]-coordinates of the circle center
        radius (float): circle radius
        plot_sol (bool): plot solution, default=False

    Returns:
        u_a (df.expression): analytic solution
        f (df.expression): corresponding rhs
        phi (df.expression): corresponding enrichment function
    '''

    x, y = sym.symbols('x[0] x[1]')
    r = sym.sqrt( (x-center[0])**2.0 + (y-center[1])**2.0 + DOLFIN_EPS) 

    f = 10.0/radius
    Ie = sym.LessThan(radius, r)
    G = -radius*sym.ln(r/radius)
    u_a = f*G

    u_a, G, Ie = [sym.printing.ccode(expr) for expr in [u_a, G, Ie]]

    u_a = '(' + u_a + ')*' + '(' + Ie + ')' # sympy cannot convert expressions 
    G = '(' + G + ')*' + '(' + Ie + ')'     # involving "less than" to ccode

    R_val = 0.1

    G = Expression(G.replace('log', 'std::log'), degree=3, R=R_val)
    u_a = Expression(u_a.replace('log', 'std::log'), degree=3, R=R_val)
    f = 10.0/R_val
    f = Constant(f)

    phi = u_a

    if plot_sol: 
        # mesh to plot on 
        Vfine = FunctionSpace(UnitSquareMesh(100, 100), 'CG', 1)
        plot_analytic_sol(u_a, f, Vfine)

    return u_a, f, phi



def plot_analytic_sol(u_a, f, Vfine, fname = 'analytic_solution'):
    '''
    Plot analytic solution with matplotlib

    Args:
        u_a (function): analytic solution
        f (function): corresponding rhs
        fname (str): plot file name, default=analytic_solution
    '''
    import matplotlib.pyplot as plt


    uai = interpolate(u_a, Vfine)
    fi = interpolate(f, Vfine)

    if Vfine.mesh().geometric_dimension() == 1:
        fig, axs = plt.subplots(1,2, figsize=(10,5))
        axs[0].plot(Vfine.tabulate_dof_coordinates(), uai.vector().get_local(), 'b', label='$u_a$')
        axs[1].plot(Vfine.tabulate_dof_coordinates(), fi.vector().get_local(), 'r', label='$f$')
        axs[0].legend(); axs[1].legend()
    else: 
        fig, axs = plt.subplots(1,1, figsize=(10,5))
        c = plot(uai, mode='color')
        plt.colorbar(c)
        axs.legend() 
    
    fig.suptitle('Analytic solution')
    fig.savefig(fname + '.png')



