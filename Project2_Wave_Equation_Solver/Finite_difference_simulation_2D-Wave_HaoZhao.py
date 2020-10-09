#!/usr/bin/env python
"""
2D wave equation solved by finite differences::
  ----> 2D wave equation simulation with constant velocity and without damping factor <--------
  dt, cpu_time = solver(I, V, f, c, Lx, Ly, Nx, Ny, dt, T,user_action=None, version='scalar')
  
"""

import time, sys
import numpy as np
from scitools.std import *

def solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T,exact_solution=None,user_action=None, version='vectorized'):

    if version == 'vectorized':
        advance = advance_vectorized
    elif version == 'scalar':
        advance = advance_scalar

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xv = x[:,newaxis]          # for vectorized function evaluations
    yv = y[newaxis,:]
    
    q  = zeros((Nx+1,Ny+1))   # define Spatial variant velocity array
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            q[i,j] = c(x[i],y[j])**2    
    Ca = np.array(q)
    c_max = sqrt(max(Ca.flatten()))

    stability_limit = (1/float(c_max))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)

    Nt  = int(round(T/float(dt)))
    t   = linspace(0, Nt*dt, Nt+1)             # mesh points in time
    Cx2 = (dt/dx)**2;  Cy2 = (dt/dy)**2        # help variables
    dt2 = dt**2

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else lambda x, y, t: zeros((x.shape[0], y.shape[1]))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else lambda x, y: zeros((x.shape[0], y.shape[1]))

    order = 'Fortran' if version == 'f77' else 'C'
    u   = zeros((Nx+1,Ny+1), order=order)   # solution array
    u_e = zeros((Nx+1,Ny+1), order=order)   # exact solution array
    u_1 = zeros((Nx+1,Ny+1), order=order)   # solution at t-dt
    u_2 = zeros((Nx+1,Ny+1), order=order)   # solution at t-2*dt
    f_a = zeros((Nx+1,Ny+1), order=order)   # for compiled loops
    v_a = zeros((Nx+1,Ny+1), order=order)   # for compiled loops
    err = zeros(Nt+1)                       # Simulation error for each time
    error = 0                               # predefine error value

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])

    import time; t0 = time.clock()          # for measuring CPU time
#========= Part-1: Load initial condition into u_1=========================================

    if version == 'scalar':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
                u_e[i,j] = exact_solution(x[i], y[j],t[0])
        err[0] = abs(u_e - u_1).max()

    else: # use vectorized version
        u_1[:,:] = I(xv, yv)

    if user_action is not None:
        user_action(u_1, x, y, xv, yv, Lx, Ly, dx, dy, t, 0)
        
#========= Part-2: Special formula for first time step======================================
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2, b, q, V,step1=True)
    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        v_a[:,:] = V(xv, yv)
        u = advance_vectorized(u, u_1, u_2, f_a,Cx2, Cy2, dt2, b, q, v_a, step1=True)

    for i in Ix:
        for j in Iy:
            u_e[i,j] = exact_solution(x[i], y[j],t[0])
    err[1] = abs(u_e - u).max()
            
    if user_action is not None:
        user_action(u, x, y, xv, yv, Lx, Ly, dx, dy, t, 1)

#========= Part-3: Update data structures for next step======================================
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2, b, q, V)
        else:
            f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
            u = advance(u, u_1, u_2, f_a, Cx2, Cy2, dt2, b, q, v_a)

        for i in Ix:
            for j in Iy:
                u_e[i,j] = exact_solution(x[i], y[j],t[n+1])
        err[n+1] = abs(u_e - u).max()
                
        if user_action is not None:
            if user_action(u, x, y, xv, yv, Lx, Ly, dx, dy, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to set u = u_1 if u is to be returned!
    t1 = time.clock()
    u = u_1
    error = err.max()
    cpu_time = t0 - time.clock()
    # dt might be computed in this function so return the value
    return u, x, t, cpu_time, error

def advance_scalar(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2, b, q, V=None, step1=False, boundary='Neuman'):

    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = sqrt(dt2)  # save
    
    if boundary == 0:    
       # Boundary condition u=0        
       
       if step1:    
          D1 = 1;  D2 = 0; D3 = 0.5;  D4 = 0.5; D5 = 1-0.5*b*dt;
       else:
          D1 = 2/(1+0.5*b*dt);  D2 = (1-0.5*b*dt)/(1+0.5*b*dt); D3 = 1/(1+0.5*b*dt); D4 = 1/(1+0.5*b*dt); D5 = 0;
          
       for i in Ix[1:-1]:
           for j in Iy[1:-1]:
               u_xx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j]) 
               u_yy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1]) 
               u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f(x[i], y[j], t[n])) + D5*dt*V(x[i], y[j])
                                
       j = Iy[0]
       for i in Ix: u[i,j] = 0
       j = Iy[-1]
       for i in Ix: u[i,j] = 0
       i = Ix[0]
       for j in Iy: u[i,j] = 0 
       i = Ix[-1] 
       for j in Iy: u[i,j] = 0
       
       
    if boundary == 'Neuman':
       # Neuman Boundary condition 
       
       if step1:    
          D1 = 1;  D2 = 0; D3 = 0.5;  D4 = 0.5; D5 = 1-0.5*b*dt;
       else:
          D1 = 2/(1+0.5*b*dt);  D2 = (1-0.5*b*dt)/(1+0.5*b*dt); D3 = 1/(1+0.5*b*dt); D4 = 1/(1+0.5*b*dt); D5 = 0;

       for i in Ix[:]:
           for j in Iy[:]:
               ip1 = i+1;im1 = i-1;jp1 = j+1;jm1 = j-1 
               ip1 = im1 if ip1 > Ix[-1] else ip1; im1 = ip1 if im1 < Ix[0] else im1
               jp1 = jm1 if jp1 > Iy[-1] else jp1; jm1 = jp1 if jm1 < Iy[0] else jm1
               u_xx = 0.5*(q[i,j] + q[ip1,j])*(u_1[ip1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[im1,j])*(u_1[i,j] - u_1[im1,j]) 
               u_yy = 0.5*(q[i,j] + q[i,jp1])*(u_1[i,jp1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,jm1])*(u_1[i,j] - u_1[i,jm1]) 
               u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f(x[i], y[j], t[n])) + D5*dt*V(x[i], y[j])
    return u
    

def advance_vectorized(u, u_1, u_2, f_a, Cx2, Cy2, dt2, b, q, v_a, step1=False, boundary='Neuman'):
    
    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])
    dt = sqrt(dt2)  # save
    
    if step1:    
      D1 = 1;  D2 = 0; D3 = 0.5;  D4 = 0.5; D5 = 1-0.5*b*dt;
    else:
      D1 = 2/(1+0.5*b*dt);  D2 = (1-0.5*b*dt)/(1+0.5*b*dt); D3 = 1/(1+0.5*b*dt); D4 = 1/(1+0.5*b*dt); D5 = 0;
    
    u_xx = 0.5*(q[1:-1,1:-1] + q[2:,1:-1])*(u_1[2:,1:-1] - u_1[1:-1,1:-1]) - 0.5*(q[1:-1,1:-1] + q[:-2,1:-1])*(u_1[1:-1,1:-1] - u_1[:-2,1:-1]) 
    u_yy = 0.5*(q[1:-1,1:-1] + q[1:-1,2:])*(u_1[1:-1,2:] - u_1[1:-1,1:-1]) - 0.5*(q[1:-1,1:-1] + q[1:-1,:-2])*(u_1[1:-1,1:-1] - u_1[1:-1,:-2]) 
    u[1:-1,1:-1] = D1*u_1[1:-1,1:-1] - D2*u_2[1:-1,1:-1] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[1:-1,1:-1]) + D5*dt*v_a[1:-1,1:-1]

    # Boundary condition u=0    
    if boundary == 0:  
        # Set the Boundary values according to the Boundary condition u=0
       j = 0
       u[:,j] = 0
       j = u.shape[1]-1
       u[:,j] = 0
       i = 0
       u[i,:] = 0
       i = u.shape[0]-1
       u[i,:] = 0
        
    if boundary == 'Neuman':
       # Set the Boundary values according to the Boundary condition du/dx = 0 and du/dy = 0  
       #(1)set the boundary when j = Iy[0] 
       j = Iy[0]
       u_xx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - 0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j])
       u_yy = 0.5*(q[1:-1,j] + q[1:-1,j+1])*(u_1[1:-1,j+1] - u_1[1:-1,j]) - 0.5*(q[1:-1,j] + q[1:-1,j+1])*(u_1[1:-1,j] - u_1[1:-1,j+1])          
       u[1:-1,j] = D1*u_1[1:-1,j] - D2*u_2[1:-1,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[1:-1,j]) + D5*dt*v_a[1:-1,j]

       #(2)set the boundary when j = Iy[-1]
       j = Iy[-1]
       u_xx = 0.5*(q[1:-1,j] + q[2:,j])*(u_1[2:,j] - u_1[1:-1,j]) - 0.5*(q[1:-1,j] + q[:-2,j])*(u_1[1:-1,j] - u_1[:-2,j]) 
       u_yy = 0.5*(q[1:-1,j] + q[1:-1,j-1])*(u_1[1:-1,j-1] - u_1[1:-1,j]) - 0.5*(q[1:-1,j] + q[1:-1,j-1])*(u_1[1:-1,j] - u_1[1:-1,j-1]) 
       u[1:-1,j] = D1*u_1[1:-1,j] - D2*u_2[1:-1,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[1:-1,j]) + D5*dt*v_a[1:-1,j]

       #(3)set the boundary when i = Ix[0]   
       i = Ix[0]
       u_xx = 0.5*(q[i,1:-1] + q[i+1,1:-1])*(u_1[i+1,1:-1] - u_1[i,1:-1]) - 0.5*(q[i,1:-1] + q[i+1,1:-1])*(u_1[i,1:-1] - u_1[i+1,1:-1]) 
       u_yy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - 0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2]) 
       u[i,1:-1] = D1*u_1[i,1:-1] - D2*u_2[i,1:-1] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,1:-1]) + D5*dt*v_a[i,1:-1]
          
       #(4)set the boundary when i = Ix[-1]   
       i = Ix[-1]
       u_xx = 0.5*(q[i,1:-1] + q[i-1,1:-1])*(u_1[i-1,1:-1] - u_1[i,1:-1]) - 0.5*(q[i,1:-1] + q[i-1,1:-1])*(u_1[i,1:-1] - u_1[i-1,1:-1]) 
       u_yy = 0.5*(q[i,1:-1] + q[i,2:])*(u_1[i,2:] - u_1[i,1:-1]) - 0.5*(q[i,1:-1] + q[i,:-2])*(u_1[i,1:-1] - u_1[i,:-2]) 
       u[i,1:-1] = D1*u_1[i,1:-1] - D2*u_2[i,1:-1] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,1:-1]) + D5*dt*v_a[i,1:-1]

       # Neuman Boundary condition Ut = 0 for the 4 conner points
       i = Ix[0]; j = Iy[0]
       u_xx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[i+1,j])*(u_1[i,j] - u_1[i+1,j])
       u_yy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j] - u_1[i,j+1])          
       u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,j]) + D5*dt*v_a[i,j]

       i = Ix[-1];j = Iy[0]
       u_xx = 0.5*(q[i,j] + q[i-1,j])*(u_1[i-1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j])
       u_yy = 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,j+1])*(u_1[i,j] - u_1[i,j+1])          
       u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,j]) + D5*dt*v_a[i,j]

       i = Ix[0]; j = Iy[-1]
       u_xx = 0.5*(q[i,j] + q[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[i+1,j])*(u_1[i,j] - u_1[i+1,j]) 
       u_yy = 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j-1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1]) 
       u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,j]) + D5*dt*v_a[i,j]

       i = Ix[-1];j = Iy[-1]
       u_xx = 0.5*(q[i,j] + q[i-1,j])*(u_1[i-1,j] - u_1[i,j]) - 0.5*(q[i,j] + q[i-1,j])*(u_1[i,j] - u_1[i-1,j]) 
       u_yy = 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j-1] - u_1[i,j]) - 0.5*(q[i,j] + q[i,j-1])*(u_1[i,j] - u_1[i,j-1]) 
       u[i,j] = D1*u_1[i,j] - D2*u_2[i,j] + D3*(Cx2*u_xx + Cy2*u_yy) + D4*(dt2*f_a[i,j]) + D5*dt*v_a[i,j]
       
    return u

def viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution=None,version='vectorized', animate=True, Title=None):
    """Run solver and visualize u at each time level."""
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure()
    
    def plot_u(u, x, y, xv, yv, Lx, Ly, dx, dy, t, n):
        
        if n!=0:
           plt.clf()
       
        ax = fig.gca(projection='3d')
        X = np.arange(0, Lx + dx, dx)
        Y = np.arange(0, Ly + dy, dy)
        Y, X = np.meshgrid(Y, X)
        Z = u

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        
        ax.set_zlim(umin, umax)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if Title != None:
           ax.set_title(Title)


        plt.draw()
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(1) if t[n] == 0 else time.sleep(0.01)


    user_action = plot_u if animate else None

    u, x, t, cpu, error = solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T,exact_solution,user_action=plot_u, version=version)
                          
    return cpu
    
#===========================================================================================================
#========= Part-Final: Start Using the Scheme to Solving Practical Problems=================================
#===========================================================================================================
import nose.tools as nt

def test_constant():
    """Exact discrete solution of the scheme."""

    def exact_solution(x, y, t): 
        return 10  # randomly choose a non-zero constant value for testing

    def I(x, y):
        return exact_solution(x, y, 0)

    def V(x, y):
        return 0

    def f(x, y, t):
        return 0

    def c(x, y):
        return 1.0

    b  = 0  # choose the dump(b!=0) or undumped(b=0) version   
    Lx = 5
    Ly = 5
    Nx = 20
    Ny = 20
    T  = 10
    dt = -1 # use longest possible steps

    def assert_no_error(u, x, y, xv, yv, Lx, Ly, dx, dy, t, n):
        u_e = exact_solution(xv, yv, t[n])
        diff = abs(u - u_e).max()
        #print "Running in time step:" n, "The corresponding difference:" diff
        print ("Running in time step: %d The corresponding difference: %f " % (n, diff))
        nt.assert_almost_equal(diff, 0, places=12,msg='diff=%g, step %d, time=%g' % (diff, n, t[n]))
    
    print "----> start the constant solution testing, scalar version testing......"
    solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, exact_solution,user_action=assert_no_error, version='scalar') 
    print "----> start the constant solution testing, vectorized version testing......"
    solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, exact_solution,user_action=assert_no_error, version='vectorized') 
    print "----> completed the constant solution testing, the code is running without bugs :)"

def test_plug():
    """Check that an initial plug is correct back after one period."""

    print "----> start the plug wave simulation from the X direction: ......"
    def exact_solution(x, y, t): 
        return 0  
        
    def I(x, y):
        if (abs(x-Lx/2.0) > 0.1) :
           return 0
        else:
           return 1
           
    def V(x, y):
        return 0

    def f(x, y, t):
        return 0

    def c(x, y):
        return Vel
        
    b  = 0 # choose the dump(b!=0) or undumped(b=0) version   
    Vel= 1.0
    Lx = 1.0
    Ly = 1.0
    Nx = 50
    Ny = 50
    T  = 1
    dx = Lx/Nx
    dy = Ly/Ny
    dt = dx/Vel # set Courant number eaquals to 1 

    umin = -1.5;  umax = -umin
    Title = "Plug-wave simulation in X direction"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='scalar', animate=True, Title=Title)

    print "----> start the plug wave simulation from the Y direction: ......"
    def I(x, y):

        if (abs(y-Ly/2.0) > 0.1) :
           return 0
        else:
           return 1
           
    def V(x, y):
        return 0

    def f(x, y, t):
        return 0

    def c(x, y):
        return Vel

    b  = 0 # choose the dump(b!=0) or undumped(b=0) version   
    Vel= 1.0
    Lx = 1.0
    Ly = 1.0
    Nx = 50
    Ny = 50
    T  = 1
    dx = Lx/Nx
    dy = Ly/Ny
    dt = dx/Vel # set Courant number eaquals to 1 

    umin = -1.5;  umax = -umin
    Title = "Plug-wave simulation in Y direction"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='scalar', animate=True, Title=Title)

def test_standing_wave_undamped():
    """Check the undapmed standing wave."""

    Lx = 5
    Ly = 5
    Nx = 40
    Ny = 40
    dt = 0.04
    Vel= 1.0 

    A  = 1.0
    Mx = 2.0
    My = 2.0
   
    Kx = Mx * pi / Lx
    Ky = My * pi / Ly

    W = sqrt(Kx**2 + Ky**2) * Vel
    T = 2

    def exact_solution(x, y, t): 
        return A*cos(Kx*x)*cos(Ky*y)*cos(W*t)  

    def I(x, y):
        return A*cos(Kx*x)*cos(Ky*y)

    def V(x, y):
        return  0

    b  = 0
    
    def c(x, y):
        return Vel

    def f(x,y,t):
        return 0
    
    print ("===Starting ploting the standing wave simulation ====")
    umin = -1.5;  umax = -umin
    Title = "undamped standing wave simulation"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='vectorized', animate=True, Title=Title)
    print ("===completed ploting the standing wave simulation ====")
    
    # Convergence rate analysis
    E_array = []
    dt_array = []
    test_values = [1,2,3]

    for test in test_values:
	if test==1:
           Nx = 10; Ny = 10 ; dt= 0.08
	if test==2:
           Nx = 20; Ny = 20 ; dt= 0.04
	if test==3:
           Nx = 40; Ny = 40 ; dt= 0.02
        errors_in_time = []
        u, x, t, cpu, error = solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T,exact_solution,user_action=None, version='vectorized')
        E_array.append(error)
        dt_array.append(dt)
        print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print ("===Starting Convergence rate analysis ====")
    print (">>>Testing derived Convergence rate:%.2f" % r[-1])



import sympy as sym
 
def Manuafactured_solution():
    """Check the manuafactured solution of standing wave."""

    Lx = 5
    Ly = 5
    Nx = 40
    Ny = 40
    dt = -1

    A  = 1.0
    B  = 1.0
    b  = 1.0
    C  = b/2.0
    Mx = 2.0
    My = 2.0
   
    Kx = Mx * pi / Lx
    Ky = My * pi / Ly

    W = 1
    T = 1

    def exact_solution(x, y, t): 
        return (A*cos(W*t) + B*sin(W*t))*exp(-t*C)*cos(Kx*x)*cos(Ky*y)

    def I(x, y):
        return exact_solution(x, y, 0)
        
    def V(x,y):
        Vfc = V_term()
        return Vfc(x,y,0) # derive the initial velocity function (t=0)

    b  = 0.1

    def c(x, y):  
        return x  # set the spatial variant velocity as c = x

    def source_term():
        x, y, t = sym.symbols('x y t')
        kx = Mx*sym.pi/Lx
        ky = My*sym.pi/Ly
        u  = (A*sym.cos(W*t) + B*sym.sin(W*t))*sym.exp(-t*C)*sym.cos(Kx*x)*sym.cos(Ky*y)
        ut = sym.diff(u,t);utt= sym.diff(u,t,t)
        q  = x**2
        xqux = sym.diff(q*sym.diff(u,x),x) ## derive (q*ux)_x
        yquy = sym.diff(q*sym.diff(u,y),y) ## derive (q*uy)_y
        s_term  = utt + b*ut -xqux -yquy
        return sym.lambdify((x,y,t),s_term,'numpy')
        
    def V_term():
        x, y, t = sym.symbols('x y t')
        kx = Mx*sym.pi/Lx
        ky = My*sym.pi/Ly
        u  = (A*sym.cos(W*t) + B*sym.sin(W*t))*sym.exp(-t*C)*sym.cos(Kx*x)*sym.cos(Ky*y)
        ut = sym.diff(u,t)
        return sym.lambdify((x,y,t),ut,'numpy')

    
    f = source_term() # derive the source term based 
    
    print ("===Starting ploting the standing damped wave simulation ====")
    umin = -1.5;  umax = -umin
    Title = "damped standing wave simulation"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='vectorized', animate=True, Title=Title)
    print ("===Completed ploting the standing damped wave simulation ====")
    
    # Convergence rate analysis
    E_array = []
    dt_array = []
    test_values = [1,2,3]

    for test in test_values:
	if test==1:
           Nx = 10; Ny = 10 ; dt= 0.04
	if test==2:
           Nx = 20; Ny = 20 ; dt= 0.02
	if test==3:
           Nx = 40; Ny = 40 ; dt= 0.01
        errors_in_time = []
        u, x, t, cpu, error = solver(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T,exact_solution,user_action=None, version='vectorized')
        E_array.append(error)
        dt_array.append(dt)
        print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print ("===Starting Convergence rate analysis ====")
    print (">>>Testing derived Convergence rate:%.2f" % r[-1])
    
def Tsunami_simulation():
    
    Lx = 100
    Ly = 100
    Nx = 100
    Ny = 100
    
    b  =  0
    dt =  -1 # let the program itself to choose the best time step
    T  = 10
    g  = 9.8 # define the gravitational acceleration
    H0 = 10  # define the average water bottom depth
    
    def exact_solution(x, y, t):    # set exact solution as 0 for the practical problems
        return 0                     
    
    def I(x,y):
        return 5*exp(-(x)**2/10.0)  # define the initial condition with a 2d gausian wave
        
    def V(x,y):        
        return 0
        
    def f(x,y,t):        
        return 0
        
    def B(x,y):
        
        B0  = 0
        Ba  = 5
        Bmx = 20
        Bmy = 50
        Bs  = 10
        Bb  = 1
        
        if hill == 'gaussian':
           return B0 + Ba*exp(-((x-Bmx)/Bs)**2 -((y-Bmy)/(Bb*Bs))**2)
            
        if hill == 'cosinhat':
           r = sqrt((x-Bmx)**2 + (y-Bmy)**2)
           if r>= 0 and r<=Bs:        
              return B0 + Ba*cos(pi*(x-Bmx)/(2*Bs))*cos(pi*(y-Bmy)/(2*Bs))
           else:
              return B0 
              
        if hill == 'box':
           if x >= Bmx - Bs and x<=Bmx + Bs and y >= Bmy - Bb*Bs and y <= Bmy + Bb*Bs:
              return Ba
           else:
              return B0
              
    def c(x,y):
        return sqrt(g * (H0 - B(x,y)))
 
     #def sub_surface():          # define the sub sea hill's formation
    
    hill = 'gaussian'    
    print ("===Starting ploting the Tsunami wave simulation, the current subsurface hill is 'Gaussian' tape ====")
    umin = -10;  umax = -umin
    Title = "Tsunami wave simulation,subsurface hill used Gaussian form"
    
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='vectorized', animate=True, Title=Title)
    print ("===Completed ploting the Tsunami  wave simulation,  ====")
    
    hill = 'cosinhat'
    print ("===Starting ploting the Tsunami wave simulation, the current subsurface hill is 'cosin hat' tape ====")
    umin = -10;  umax = -umin
    Title = "Tsunami wave simulation,subsurface hill used cosinhat form"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='vectorized', animate=True, Title=Title)
    print ("===Completed ploting the Tsunami  wave simulation,  ====")
    
    hill = 'box'
    print ("===Starting ploting the Tsunami wave simulation, the current subsurface hill is 'Box' tape ====")
    umin = -10;  umax = -umin
    Title = "Tsunami wave simulation,subsurface hill used Box form"
    viz(I, V, f, b, c, Lx, Ly, Nx, Ny, dt, T, umin, umax, exact_solution, version='vectorized', animate=True, Title=Title)
    print ("===Completed ploting the Tsunami  wave simulation,  ====")
    
    
    
if __name__ == '__main__':
    test_constant()
    test_plug()
    test_standing_wave_undamped()
    Manuafactured_solution()
    Tsunami_simulation()

