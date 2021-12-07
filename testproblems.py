from fenics import *
import sympy as sym


def power_function(alpha = 0.51, plot_sol = False):
    ## Babushka example, u=x^alpha with 0.5 < alpha < 1.5

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
        plot_analytic_sol(u_a, f)

    return u_a, f, phi

def steep_mollifier(beta = 2.5, plot_sol = False):
    ## Mollifier function with steep gradients

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
        plot_analytic_sol(u_a, f)

    return u_a, f, phi

def plot_analytic_sol(u_a, f):
    ## Plot solution
    import matplotlib.pyplot as plt
    Vfine = FunctionSpace(UnitIntervalMesh(200), 'CG', 1)
    uai = interpolate(u_a, Vfine)
    fi = interpolate(f, Vfine)

    fig, axs = plt.subplots(1,2, figsize=(10,5))
    fig.suptitle('Analytic solution')
    axs[0].plot(Vfine.tabulate_dof_coordinates(), uai.vector().get_local(), 'b', label='$u_a$')
    axs[1].plot(Vfine.tabulate_dof_coordinates(), fi.vector().get_local(), 'r', label='$f$')
    axs[0].legend(); axs[1].legend()
    fig.savefig('analytic_solution.png')


