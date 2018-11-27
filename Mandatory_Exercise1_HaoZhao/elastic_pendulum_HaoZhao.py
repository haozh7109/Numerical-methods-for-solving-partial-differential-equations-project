import numpy as np
import matplotlib.pyplot as plt


def simulate(
    beta=0.9, 			# dimensionless parameter
    Theta=30, 			# initial angle in degrees
    epsilon=0, 			# initial stretch of wire
    num_periods=6, 		# simulate for num_periods
    time_steps_per_period=60, 	# time step resolution
    plot=True, 			# make plots or not
    ):

    # set the initial parameters
    Theta_rad = np.deg2rad(Theta)
    T  = num_periods * 2 * np.pi 
    dt = 2 * np.pi / time_steps_per_period
    Nt = np.int(T/dt)
    T  = Nt * dt

    x = np.zeros(Nt+1)
    y = np.zeros(Nt+1)
    L = np.zeros(Nt+1)
    thetas = np.zeros(Nt+1)
    coef  = np.zeros(Nt+1)

    x_nep = np.zeros(Nt+1)
    y_nep = np.zeros(Nt+1)
    L_nep = np.zeros(Nt+1)
    thetas_nep = np.zeros(Nt+1)
    coef_nep  = np.zeros(Nt+1)

    t = np.linspace(0, T, Nt+1)

  # Part-1: Numerical calculation for the Elastic Pendulum

    # set the initial conditions
    thetas[0] = Theta
    x[0]    = (1+epsilon)*np.sin(thetas[0])
    y[0]    = 1-(1+epsilon)*np.cos(thetas[0])
    L[0]    = np.sqrt( x[0]**2 + (y[0]-1)**2 )
 
    coef[0] = -(beta/(1-beta))*(1-beta/L[0])
    thetas[0]  = np.arctan2(x[0], 1-y[0]) / np.pi *180

    # derive the first time value
    x[1] = 0.5*coef[0]*x[0]*dt**2 + x[0]
    y[1] = 0.5*(coef[0]*(y[0]-1)-beta)*dt**2 + y[0]
    thetas[1]  = np.arctan2(x[1], 1-y[1]) / np.pi *180

    # derive the all time value iteratively
    for n in range(1, Nt):
        L[n]    = np.sqrt( x[n]**2 + (y[n]-1)**2 )
        coef[n] = -(beta/(1-beta)) * (1-beta/L[n])
        x[n+1]  = coef[n]*x[n]*dt**2 + 2*x[n] - x[n-1]
        y[n+1]  = (coef[n]*(y[n]-1)-beta)*dt**2 + 2*y[n] - y[n-1]
        thetas[n+1]  = np.arctan2(x[n+1], 1-y[n+1]) / np.pi *180


  # Part-2: Numerical calculation for the Non-Elastic Pendulum
    
    # set the beta close to 1 to simulate the Non-Elastic Pendulum

    beta= 0.99
    # set the initial conditions
    thetas_nep[0] = Theta
    x_nep[0]    = (1+epsilon)*np.sin(thetas_nep[0])
    y_nep[0]    = 1-(1+epsilon)*np.cos(thetas_nep[0])
    L_nep[0]    = np.sqrt( x_nep[0]**2 + (y_nep[0]-1)**2 )
   

    coef_nep[0] = -(beta/(1-beta))*(1-beta/L_nep[0])
    #coef_nep[0] = -(beta/L_nep[0])
    thetas_nep[0]  = np.arctan2(x_nep[0], 1-y_nep[0]) / np.pi *180

    # derive the first time value
    x_nep[1] = 0.5*coef_nep[0]*x_nep[0]*dt**2 + x_nep[0]
    y_nep[1] = 0.5*(coef_nep[0]*(y_nep[0]-1)-beta)*dt**2 + y_nep[0]
    thetas[1]  = np.arctan2(x_nep[1], 1-y_nep[1]) / np.pi *180

    # derive the all time value iteratively
    for n in range(1, Nt):
        L_nep[n]    = np.sqrt( x_nep[n]**2 + (y_nep[n]-1)**2 )
	coef_nep[n] = -(beta/(1-beta))*(1-beta/L_nep[n])        
	#coef_nep[n] = -(beta/L_nep[n])
        x_nep[n+1]  = coef_nep[n]*x_nep[n]*dt**2 + 2*x_nep[n] - x_nep[n-1]
        y_nep[n+1]  = (coef_nep[n]*(y_nep[n]-1)-beta)*dt**2 + 2*y_nep[n] - y_nep[n-1]
        thetas_nep[n+1]  = np.arctan2(x_nep[n+1], 1-y_nep[n+1]) / np.pi *180


    # Plot the Pendulum motion and T-Angle graph
    if plot == True:
      plt.figure(1)
      plt.subplot(211)
      plt.plot(x,y)
      plt.legend(["Y(X)"])
      plt.gca().set_aspect('equal')
      plt.title("Physical Motion of Elastic Pendulum")
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.subplot(212)
      plt.plot(t,thetas)
      plt.xlabel('T')
      plt.ylabel('Theta')
      plt.legend(["Theta(t)"])
      plt.show()

      # if Theta less than 10, calculate the motion with non-elastic method
      if Theta < 10:
        # Compare elastic vs non-elastic pendulum
        plt.figure(2)
        plt.subplot(211)
        plt.gca().set_aspect('equal') # Set axis equal
        plt.plot(x,y,'b',x_nep,y_nep,'r')
        plt.legend(["Elastic","Non-elastic"])
        plt.title("Comparison of elastic pendulum with non-elastic pendulum.")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.subplot(212)
        plt.plot(t,thetas,'b',t,thetas_nep,'r')
        plt.xlabel('T')
        plt.ylabel('Theta')
        plt.legend(["Elastic","Non-elastic"])
        plt.show()
         
    return t,x,x_nep,y,y_nep,thetas,thetas_nep,L,L_nep

def test_equilibrium():
    """Zero test by using x=y=theta=0."""
    t,x,x_nep,y,y_nep,thetas,thetas_nep,L,L_nep = simulate(
    beta=0.9,
    Theta=0,
    epsilon=0,
    num_periods=6,
    time_steps_per_period=60,
    plot=False,
    )
    tol = 1E-14
    assert np.abs(x.max()) < tol
    assert np.abs(y.max()) < tol
    assert np.abs(thetas.max()) < tol

def test_vertical_move():
    """test the vertical motion of elastic pendulum"""
    beta = 0.9
    W = np.sqrt(beta/(1-beta))
    period = 2*np.pi/W
    # We want T = N*period
    N = 5
    # simulate function has T = 2*pi*num_periods
    num_periods = N/W
    n = 600
    time_steps_per_period = W*n

    y_exact = lambda t: -0.1*np.cos(W*t)

    t,x,x_nep,y,y_nep,thetas,thetas_nep,L,L_nep = simulate(
	 beta=beta, Theta=0, epsilon=0.1,
         num_periods=num_periods,
         time_steps_per_period=time_steps_per_period,plot=False)

    tol = 0.00055 # ok tolerance for the above resolution
    # No motion in x direction is epxected
    assert np.abs(x.max()) < tol
    # Check motion in y direction
    y_e = y_exact(t)
    diff = np.abs(y_e - y).max()

    if diff > tol: 
       plt.plot(t, y,'b', t, y_e,'r')
    assert diff < tol, 'diff=%g' % diff    


def demo(beta=0.999, Theta=40, num_periods=3):
    t,x,x_nep,y,y_nep,thetas,thetas_nep,L,L_nep = simulate(
        beta=beta, 
        Theta=Theta, 
        epsilon=0,
        num_periods=num_periods,
        time_steps_per_period=600,
        plot=True,)

if __name__ == '__main__':
    print "==================job start running now=============================="
    # 1-simulate the elastic pendulum
    simulate()
    # 2-simulate both the elastic & non-elastic pendulum and make comparison
    simulate(
    beta=0.9, 	# beta = 0.9 for elastic, and beta = 0.99 for non-elastic(fixed term in job)		
    Theta=8, 			
    epsilon=0, 			
    num_periods=6, 		
    time_steps_per_period=60, 	
    plot=True,) 
    # 3-equilibrium test and ertical_move test
    test_equilibrium()
    test_vertical_move()
    # 4-plot the demo fuction
    demo()
    print "==================job completed without errors :), Hao =============================="

