#!/usr/bin/env python
"""
1D wave equation with Dirichlet or Neumann conditions and variable wave velocity::

 u, x, t, cpu = solver(I, V, f, c, U_0, U_L, L, dt, C, T, user_action=None, version='scalar',stability_safety_factor=1.0)

"""
import time, glob, shutil
from scitools.std import *
import nose.tools as nt

def solver(I, V, f, c, U_0, U_L, L, dt, C, T,method,user_action=None,version='vectorized',stability_safety_factor=1.0):
    """Solve u_tt=(c^2*u_x)_x + f on (0,L)x(0,T]."""

    Nt = int(round(T/dt))
    t = linspace(0, Nt*dt, Nt+1)      # Mesh points in time
    dx = dt*c_max/(stability_safety_factor*C)
    Nx = int(round(L/dx))
    x = linspace(0, L, Nx+1)          # Mesh points in space
    C2 = (dt/dx)**2; dt2 = dt*dt      # Help variables in the scheme

    # Wrap user-given f, I, V, U_0, U_L if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else lambda x, t: zeros(x.shape)

    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else lambda x: zeros(x.shape)

    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else lambda x: zeros(x.shape)

    if U_0 is not None:
        if isinstance(U_0, (float,int)) and U_0 == 0:
            U_0 = lambda t: 0
    if U_L is not None:
        if isinstance(U_L, (float,int)) and U_L == 0:
            U_L = lambda t: 0

    u   = zeros(Nx+1)   # Solution array at new time level
    u_1 = zeros(Nx+1)   # Solution at 1 time level back
    u_2 = zeros(Nx+1)   # Solution at 2 time levels back
    u_e = zeros(Nx+1)   # Exact Solution array at new time level
    q   = zeros(Nx+1)   # Spatial variant velocity array
    err = zeros(Nx+1)   # Simulation error array
    error = 0           # define the sum error

    # Load the spatial variant velocity
    for i in range(0,Nx+1):
        q[i] = c(x[i])**2

    import time;  t0 = time.clock()  # CPU time measurement

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1 and calculate the exact solution and simulation error
    for i in range(0,Nx+1):
        u_1[i] = I(x[i])
        u_e[i] = cos(pi*x[i]/L)*cos(W*t[0]) # calculate the exact solution
        err[i] = (u_e[i] - u_1[i])**2       # calculate the individual error
    error  = error + sum(err)               # calculate the sum error
    
    if user_action is not None:
        user_action(u_1, u_e, x, t, 0)

    # Special formula for the first step
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i]) - 0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + 0.5*dt2*f(x[i], t[0])

    i = Ix[0]
    if U_0 is None:
        # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
        # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        if  method == 'method_1': # the scheme from the equation (55)
           u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + 0.5*dt2*f(x[i], t[0])
        elif method == 'method_2':# the scheme from the equation (52)  
           u[i] = u_1[i] + dt*V(x[i]) + C2*q[i]*(u_1[ip1] - u_1[i]) + 0.5*dt2*f(x[i], t[0])
	elif method == 'method_3':# the scheme from the exercise_c 
	   u[i] = u[ip1]
	elif method == 'method_4':# the scheme from the exercise_d 
	   u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])) + 0.5*dt2*f(x[i], t[0])
        else:
            raise ValueError('method=%s' % method)
    else:
        u[i] = U_0(dt)

    i = Ix[-1]
    if U_L is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        if  method == 'method_1': # the scheme from the equation (55)
           u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + 0.5*dt2*f(x[i], t[0])
        elif method == 'method_2':# the scheme from the equation (52)  
           u[i] = u_1[i] + dt*V(x[i]) + C2*q[i]*(u_1[im1] - u_1[i]) + 0.5*dt2*f(x[i], t[0])
	elif method == 'method_3':# the scheme from the exercise_c 
	   u[i] = u[im1]
	elif method == 'method_4':# the scheme from the exercise_d 
	   u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(- 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + 0.5*dt2*f(x[i], t[0])
        else:
            raise ValueError('method=%s' % method)
    else:
        u[i] = U_L(dt)
    
    # calculate the exact solution and the error
    for i in range(0,Nx+1):
        u_e[i] = cos(pi*x[i]/L)*cos(W*t[1])
        err[i] = (u_e[i] - u[i])**2          # calculate the individual error
    error  = error + sum(err)                # calculate the sum error

    if user_action is not None:
        user_action(u, u_e, x, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        if version == 'scalar':
            for i in Ix[1:-1]:
                u[i] = - u_2[i] + 2*u_1[i] + C2*(0.5*(q[i] + q[i+1])*(u_1[i+1] - u_1[i])  - 0.5*(q[i] + q[i-1])*(u_1[i] - u_1[i-1])) + dt2*f(x[i], t[n])
        elif version == 'vectorized':
            u[1:-1] = - u_2[1:-1] + 2*u_1[1:-1] + C2*(0.5*(q[1:-1] + q[2:])*(u_1[2:] - u_1[1:-1]) -0.5*(q[1:-1] + q[:-2])*(u_1[1:-1] - u_1[:-2])) + dt2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Insert boundary conditions
        i = Ix[0]
        if U_0 is None:
            # Set boundary values
            # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
            # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
            ip1 = i+1
            im1 = ip1
            if  method == 'method_1': # the scheme from the equation (55)
                u[i] = - u_2[i] + 2*u_1[i] + C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + dt2*f(x[i], t[n])
            elif method == 'method_2': # the scheme from the equation (52)
                u[i] = - u_2[i] + 2*u_1[i] + C2*2*q[i]*(u_1[ip1] - u_1[i]) + dt2*f(x[i], t[0])
	    elif method == 'method_3':# the scheme from the exercise_c 
	        u[i] = u[ip1]
	    elif method == 'method_4':# the scheme from the exercise_d 
                u[i] = - u_2[i] + 2*u_1[i] + C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])) + dt2*f(x[i], t[n])
            else:
                raise ValueError('method=%s' % method)
        else:
            u[i] = U_0(t[n+1])

        i = Ix[-1]
        if U_L is None:
            im1 = i-1
            ip1 = im1
            if  method == 'method_1': # the scheme from the equation (55)
                u[i] = - u_2[i] + 2*u_1[i] + C2*(0.5*(q[i] + q[ip1])*(u_1[ip1] - u_1[i])  - 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + dt2*f(x[i], t[n])
            elif method == 'method_2': # the scheme from the equation (52)
                u[i] = - u_2[i] + 2*u_1[i] + C2*2*q[i]*(u_1[im1] - u_1[i]) + dt2*f(x[i], t[0])
	    elif method == 'method_3':# the scheme from the exercise_c 
	        u[i] = u[im1]
	    elif method == 'method_4':# the scheme from the exercise_d 
	        u[i] = - u_2[i] + 2*u_1[i] + C2*(- 0.5*(q[i] + q[im1])*(u_1[i] - u_1[im1])) + dt2*f(x[i], t[n])
            else:
                raise ValueError('method=%s' % method)
        else:
            u[i] = U_L(t[n+1])

        # calculate the exact solution and the error
        for i in range(0,Nx+1):
            u_e[i] = cos(pi*x[i]/L)*cos(W*t[n+1])
            err[i] = (u_e[i] - u[i])**2          # calculate the individual error
        error  = error + sum(err)                # calculate the sum error

        if user_action is not None:
            if user_action(u, u_e, x, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = t0 - time.clock()
    error    = sqrt(dt*dx*error)   # calculate the squre root error based on the L2 norm

    return u, x, t, cpu_time,error

def viz(I, V, f, c, U_0, U_L, L, dt, C, T, umin, umax,method,version='scalar', animate=True):
    """Run solver and visualize u at each time level."""
    import scitools.std as plt, time, glob, os
    if callable(U_0):
        bc_left = 'u(0,t)=U_0(t)'
    elif U_0 is None:
        bc_left = 'du(0,t)/dx=0'
    else:
        bc_left = 'u(0,t)=0'
    if callable(U_L):
        bc_right = 'u(L,t)=U_L(t)'
    elif U_L is None:
        bc_right = 'du(L,t)/dx=0'
    else:
        bc_right = 'u(L,t)=0'

    def plot_u(u, u_e, x, t, n):
        """user_action function for solver."""
        plt.plot(x, u, 'r-', x, u_e, 'b-',
                 xlabel='x', ylabel='u',
                 axis=[0, L, umin, umax],
                 title='t=%.3f, %s, %s' % (t[n], bc_left, bc_right))
        # Let the initial condition stay on the screen for 2
        # seconds, else insert a pause of 0.2 s between each plot
        time.sleep(0.1) if t[n] == 0 else time.sleep(0.000001)

    user_action = plot_u if animate else None
    method = method
    u, x, t, cpu, error = solver(I, V, f, c, U_0, U_L, L, dt, C, T,method,user_action,version='vectorized',stability_safety_factor=1.0)

    return cpu

import nose.tools as nt

def Exercise_a():
    """discretizations of a Neumann condition: Exercise-a"""

    print ("=====================Starting Exercise_a ================================")
    # set the initial condition for the wave simulation
    L  = 10.0
    T  = 3
    Nx = 150
    global W
    W  = 1
    C  = 1 

    def u_exact(x,t):
        return cos(pi*x/L)*cos(W*t)
  
    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0

    def c(x):
        return sqrt(1+(x-L/2)**4)

    def f(x,t):
        return -W**2*cos(t*W)*cos(pi*x/L) +4*pi*(-L/2 + x)**3*sin(pi*x/L)*cos(t*W)/L + pi**2*((-L/2 + x)**4 + 1)*cos(t*W)*cos(pi*x/L)/L**2

    global c_max
    c_max = 25                        # define the max C based on sqrt(1+(x-L/2)**4)
    dt    = (L/Nx)/c_max              # dt = (L/Nx)/c  # choose the stability limit with given Nx,Lower C will then use this dt, but smaller Nx 
    
    U_0 = None # Neumann condition
    U_L = None # Neumann condition

    print ("===Starting ploting the wave simulation ====")
    umin = -1.5;  umax = -umin
    viz(I, V, f, c, U_0, U_L, L, dt, C, T, umin, umax,method='method_1',version='vectorized', animate=True) # display the wave simulation 
    close()

    print ("===Starting Divergence rate analysis ====")
    print ("===The Divergence rate analysis for Exercise_a is slow, please be patient :)coming in 2 minutes  ====")
    # Divergence rate analysis
    E_array = []
    dt_array = []
    Nx_values = [10,20,40,80,160,320,640]
    for Nx in Nx_values:
        errors_in_time = []
        dt = (L/Nx)/c_max 
        u, x, t, cpu, error = solver(I, V, f, c, U_0, U_L, L, dt, C, T,method='method_1',user_action=None,version='vectorized',stability_safety_factor=1.0)
        E_array.append(error)
        dt_array.append(dt)
        #print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print (">>>Testing derived Divergence rate:%.2f" % r[-1])

def Exercise_b():
    """discretizations of a Neumann condition: Exercise-b"""
    print ("=====================Starting Exercise_b ================================")
    # set the initial condition for the wave simulation
    L  = 10.0
    T  = 3
    Nx = 150
    global W
    W  = 1
    C  = 1 

    def u_exact(x,t):
        return cos(pi*x/L)*cos(W*t)
  
    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0

    def c(x):
        return sqrt(1+cos(pi*x/L))

    def f(x,t):
        return -W**2*cos(t*W)*cos(pi*x/L) + pi**2*(cos(pi*x/L) + 1)*cos(t*W)*cos(pi*x/L)/L**2 - pi**2*sin(pi*x/L)**2*cos(t*W)/L**2

    global c_max
    c_max = 1.414                     # define the max C based on sqrt(1+cos(pi*x/L))
    dt    = (L/Nx)/c_max              # dt = (L/Nx)/c  # choose the stability limit with given Nx,Lower C will then use this dt, but smaller Nx 
    
    U_0 = None # Neumann condition
    U_L = None # Neumann condition

    
    print ("===Starting ploting the wave simulation ====")
    umin = -1.5;  umax = -umin
    viz(I, V, f, c, U_0, U_L, L, dt, C, T, umin, umax,method='method_1',version='vectorized', animate=True) # display the wave simulation 
    close()

    # Divergence rate analysis
    E_array = []
    dt_array = []
    Nx_values = [10,20,40,80,160,320,640]
    for Nx in Nx_values:
        errors_in_time = []
        dt = (L/Nx)/c_max 
        u, x, t, cpu, error = solver(I, V, f, c, U_0, U_L, L, dt, C, T,method='method_1',user_action=None,version='vectorized',stability_safety_factor=1.0)
        E_array.append(error)
        dt_array.append(dt)
        #print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print ("===Starting Divergence rate analysis ====")
    print (">>>Testing derived Divergence rate:%.2f" % r[-1])

def Exercise_c():
    """discretizations of a Neumann condition: Exercise-c"""
    print ("=====================Starting Exercise_c ================================")
    # set the initial condition for the wave simulation
    L  = 10.0
    T  = 3
    Nx = 150
    global W
    W  = 1
    C  = 1 

    def u_exact(x,t):
        return cos(pi*x/L)*cos(W*t)
  
    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0

    def c(x):
        return sqrt(1+cos(pi*x/L))

    def f(x,t):
        return -W**2*cos(t*W)*cos(pi*x/L) + pi**2*(cos(pi*x/L) + 1)*cos(t*W)*cos(pi*x/L)/L**2 - pi**2*sin(pi*x/L)**2*cos(t*W)/L**2

    global c_max
    c_max = 1.414                     # define the max C based on sqrt(1+cos(pi*x/L))
    dt    = (L/Nx)/c_max              # dt = (L/Nx)/c  # choose the stability limit with given Nx,Lower C will then use this dt, but smaller Nx 
    
    U_0 = None # Neumann condition
    U_L = None # Neumann condition

    print ("===Starting ploting the wave simulation ====")
    umin = -1.5;  umax = -umin
    viz(I, V, f, c, U_0, U_L, L, dt, C, T, umin, umax,method='method_3',version='vectorized', animate=True) # display the wave simulation 
    close()

    # Divergence rate analysis
    E_array = []
    dt_array = []
    Nx_values = [10,20,40,80,160,320,640]
    for Nx in Nx_values:
        errors_in_time = []
        dt = (L/Nx)/c_max 
        u, x, t, cpu, error = solver(I, V, f, c, U_0, U_L, L, dt, C, T,method='method_3',user_action=None,version='vectorized',stability_safety_factor=1.0)
        E_array.append(error)
        dt_array.append(dt)
        #print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print ("===Starting Divergence rate analysis ====")
    print (">>>Testing derived Divergence rate:%.2f" % r[-1])


def Exercise_d():
    """discretizations of a Neumann condition: Exercise-c"""
    print ("=====================Starting Exercise_d ================================")
    # set the initial condition for the wave simulation
    L  = 10.0
    T  = 3
    Nx = 150
    global W
    W  = 1
    C  = 1 

    def u_exact(x,t):
        return cos(pi*x/L)*cos(W*t)
  
    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0

    def c(x):
        return sqrt(1+cos(pi*x/L))

    def f(x,t):
        return -W**2*cos(t*W)*cos(pi*x/L) + pi**2*(cos(pi*x/L) + 1)*cos(t*W)*cos(pi*x/L)/L**2 - pi**2*sin(pi*x/L)**2*cos(t*W)/L**2

    global c_max
    c_max = 1.414                     # define the max C based on sqrt(1+cos(pi*x/L))
    dt    = (L/Nx)/c_max              # dt = (L/Nx)/c  # choose the stability limit with given Nx,Lower C will then use this dt, but smaller Nx 
    
    U_0 = None # Neumann condition
    U_L = None # Neumann condition

    print ("===Starting ploting the wave simulation ====")
    umin = -1.5;  umax = -umin
    viz(I, V, f, c, U_0, U_L, L, dt, C, T, umin, umax,method='method_4',version='vectorized', animate=True) # display the wave simulation 
    close()

    # Divergence rate analysis
    E_array = []
    dt_array = []
    Nx_values = [10,20,40,80,160,320,640]
    for Nx in Nx_values:
        errors_in_time = []
        dt = (L/Nx)/c_max 
        u, x, t, cpu, error = solver(I, V, f, c, U_0, U_L, L, dt, C, T,method='method_4',user_action=None,version='vectorized',stability_safety_factor=1.0)
        E_array.append(error)
        dt_array.append(dt)
        #print dt_array[-1], E_array[-1]

    m = len(E_array)
    r = zeros(m-1)

    for i in range(0,m-1):
        r[i] = log(E_array[i]/E_array[i+1]) / log(dt_array[i]/dt_array[i+1])
        #print r[i]
    print ("===Starting Divergence rate analysis ====")
    print (">>>Testing derived Divergence rate:%.2f" % r[-1])

if __name__ == '__main__':
    Exercise_a()
    Exercise_b()
    Exercise_c()
    Exercise_d()

