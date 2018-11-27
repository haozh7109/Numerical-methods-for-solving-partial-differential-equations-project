"""
Solve the Nonlinear diffusion equation by Picard method
rho*u_t = grad(a(u).grad(u))+f(x,t), with u(x,0)=I(x) and du/dn=0
"""

from dolfin import *
from math import log
import numpy,time, sys

def picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T,u_exact=None,user_action=None):
    """
    picard method to solve the PDE, in current case we only used 1 iteration for each time level.
    """

    # Define the mesh and function space
    # The equal length assigned to 2D and 3D for current case
    if dimension == "1D":
        mesh = UnitIntervalMesh(Ncell)
        V    = FunctionSpace(mesh, "Lagrange", degree)
    elif dimension == "2D":
        mesh = UnitSquareMesh(Ncell,Ncell)
        V    = FunctionSpace(mesh, "Lagrange", degree)
    else:
        mesh = UnitCubeMesh(Ncell,Ncell,Ncell)
        V    = FunctionSpace(mesh, "Lagrange", degree)
            
    # Define variational form for the PDE
    u   = TrialFunction(V)
    v   = TestFunction(V)
    u_1 = interpolate(u0,V)
    a   = dt/rho*inner(alpha(u_1)*nabla_grad(u), nabla_grad(v))*dx + u*v*dx 
    L   = u_1*v*dx + dt/rho*f*v*dx 
    u   = Function(V)
    t   = dt

    # Picard iteration for solve the PDE (1 iteration applied only)
    while t <= T:
        
        # update th source term by each time level     
        f.t=t
        # solve the linear system A*c = b
        solve(assemble(a),u.vector(),assemble(L))    
        # derive the exact solution
        if u_exact is not None:
            u_exact.t = t
            u_e = interpolate(u_exact, V)
        else:
            u_e = None
        # plot or save the data
        if user_action is not None:
            user_action(u,u_e,t)
            
        t+=dt
        u_1.assign(u)

    u=u_1
    return u, u_e

def Proj_D_constant_solution():
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")    
    print (">>>>>>>>>>>>>>>>>>>>>>>>>> start the running of Proj_D_constant_solution <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< ")  
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # define the initial condition and source term
    C  = 10
    u0 = Constant(C)
    f  = Constant(0.0)

    # define the a(u)function
    def alpha(u):
        return u
        
    # define PDE picard solving parameters
    rho=Constant(1.0)
    dimension = "1D"
    degree = 1
    Ncell=10
    T=1
    dt=0.1
    
    # verification test for constant solution in 1D, 2D and 3D with P1 and P2 element
    for dimension in ["1D","2D","3D"]:
        for degree in [1,2]:
            u,u_e = picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T)
            diff = abs(u.vector().array() - C)
            tol = 1E-10
            print "\n ========== Start Constant Test: %s with P%d element =================\n" % (dimension, degree)
            print "\n ---------- Calclulated error from the sovlved PDE ---------\n", diff
            assert diff.max() < tol, 'error of constant solution Testing'
            print "\n ========== Completed Constant Test: %s with P%d element completed without error========\n" % (dimension, degree)

def Proj_E_analytical_solution_NoSourceTerm():
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>> start the running of Proj_E_analytical_solution_NoSourceTerm <<<<<<<<<<<<<<<<<<<<<<<<") 
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # define the initial condition and source term
    u0=Expression('cos(pi*x[0])')
    f  = Constant(0.0)
    
    # define the a(u)function
    def alpha(u):
        return 1.0
        
    # define PDE picard solving parameters
    rho=Constant(1.0)
    dimension = "2D"
    degree = 1
    Ncell=10
    T=1
    h = 0.01
    # define the exact solution
    u_exact = Expression('exp(-pi*pi*t)*cos(pi*x[0])',t=T)

    # Convergence rate analysis
    for test in [1,2,3,4,5]:
        if test==1:
            dt=h=0.01;    dx=dy=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==2:
            dt=h=0.005;   dx=dy=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==3:
            dt=h=0.0025;  dx=dy=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==4:
            dt=h=0.00125; dx=dy=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==5:
            dt=h=0.000625;dx=dy=sqrt(dt); Ncell =int(round(1.0/dx))
            
        # define the calculation error       
        u,u_e = picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T,u_exact)
        e   = u_e.vector().array()-u.vector().array()
        E   = sqrt(sum(e**2)/u.vector().array().size)
        Eh  = E/h
        print "\n ========== Current Test E: %f; h: %f and E/h: %f =================\n" % (E,h,Eh)


def Proj_F_analytical_solution_withSourceTerm():
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start the running of Proj_F_analytical_solution_withSourceTerm <<<<<<<<<<<<<<<<")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")    
    # define the initial condition and source term
    u0 = Expression('t*x[0]*x[0]*(0.5-x[0]/3.0)',t=0)
    f  = Expression('-rho*pow(x[0],3)/3.0 + rho*pow(x[0],2)/2.0 + 8*pow(t,3)*pow(x[0],7)/9.0 - 28*pow(t,3)*pow(x[0],6)/9.0 + 7*pow(t,3)*pow(x[0],5)/2.0 - 5*pow(t,3)*pow(x[0],4)/4.0+2*t*x[0]-t',rho=1.0,t=0)
    
    # define the a(u)function
    def alpha(u):
        return 1 + u**2

    # define PDE picard solving parameters
    rho=Constant(1.0)
    dimension = "1D"
    degree = 1
    T=1
    dt = 0.1
    Ncell=20
    
    # define the exact solution
    u_exact = Expression('t*x[0]*x[0]*(0.5-x[0]/3.0)',t=0)

    # define the plot function
    def plot_solution(u,u_e,t):
        time.sleep(1)
        plot(u, key='u',title='Numerical solution of Nonlinear Diffusion Equation at t:%s'%t)
        plot(u_e, key='u_e',title='Exact_solution of Nonlinear Diffusion Equation at t:%s'%t)
        e   = u_e.vector().array()-u.vector().array()
        E   = sqrt(sum(e**2)/u.vector().array().size)
        print "\n ========== Current Test on Time: %f; the accumulated error E: %f =================\n" % (t,E)
    
    # define the calculation error       
    u,u_e = picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T,u_exact,user_action=plot_solution)


def Proj_H_convergence_rate():
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start the running of Proj_H_convergence_rate <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # define the initial condition and source term
    u0 = Expression('t*x[0]*x[0]*(0.5-x[0]/3.0)',t=0)

    # define the a(u)function
    def alpha(u):
        return 1 + u**2

    # define PDE picard solving parameters
    rho=Constant(1.0)
    dimension = "1D"
    degree = 1
    T  = 1
    h  = 0.01
    
    # define the exact solution
    u_exact = Expression('t*x[0]*x[0]*(0.5-x[0]/3.0)',t=0)

    # Convergence rate analysis
    E_array = []
    h_array = []
    for test in [1,2,3,4,5,6]:
        if test==1:
            dt=h=0.01;       dx=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==2:
            dt=h=0.005;      dx=sqrt(dt); Ncell =int(round(1.0/dx))            
        if test==3:
            dt=h=0.0025;     dx=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==4:
            dt=h=0.00125;    dx=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==5:
            dt=h=0.000625;   dx=sqrt(dt); Ncell =int(round(1.0/dx))
        if test==6:
            dt=h=0.0003125;  dx=sqrt(dt); Ncell =int(round(1.0/dx))
            
        f  = Expression('rho*pow(x[0],2)*(-2*x[0]+3)/6.0 - (-12*t*x[0]+3*t*(-2*x[0]+3))*\
            (pow(x[0],4)*pow(-dt+t,2)*pow(-2*x[0]+3,2)+36)/324.0-(-6*t*x[0]*x[0]+6*t*x[0]*(-2*x[0]+3))*(36*pow(x[0],4)*\
            pow(-dt+t,2)*(2*x[0]-3)+36*pow(x[0],3)*pow(-dt+t,2)*pow(-2*x[0]+3,2))/5832.0',rho=1.0,t=0,dt=dt)
            
        # define the calculation error       
        u,u_e = picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T,u_exact)
        e   = u_e.vector().array()-u.vector().array()
        E   = sqrt(sum(e**2)/u.vector().array().size)
        E_array.append(E)
        h_array.append(h)
        
    r = []
    for i in range(5):
        r.append(log(E_array[i]/E_array[i+1]) / log(h_array[i]/h_array[i+1]))
        print r[i]
    print ("===Starting Convergence rate analysis ====")
    print (">>>Testing derived Convergence rate:%.2f" % r[-1])

def Proj_I_diffusion_gaussian():
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start the running of Proj_I_diffusion_gaussian <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # define the initial condition and source term
    u0=Expression('exp(-(1./(2.*sigma*sigma))*(x[0]*x[0]+x[1]*x[1]))',sigma=1.0)
    f=Constant(0.0)

    # define the a(u)function
    beta = Constant(0.1)
    def alpha(u):
        return 1 + beta*u**2

    # define PDE picard solving parameters
    rho=Constant(1.0)
    dimension = "2D"
    degree = 1
    T  = 1
    dt = 0.1
    Ncell=10
   
    # define the plot function
    def plot_solution(u,u_e,t):
        time.sleep(1)
        plot(u,key='k',rescale=False,title='Simulation of Nonlinear diffusion of Gaussian function at time:%s' %t)

    # define the calculation error       
    u = picard_iteration_solver(dimension,degree,Ncell,u0,rho,f,alpha,dt,T,u_exact=None,user_action=plot_solution)
    print ("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>< Job running completed :)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
   
if __name__ == '__main__':
   Proj_D_constant_solution()
   Proj_E_analytical_solution_NoSourceTerm()
   Proj_F_analytical_solution_withSourceTerm()
   Proj_H_convergence_rate()
   Proj_I_diffusion_gaussian()
