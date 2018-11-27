import odespy
import numpy as np
import scitools.std as pl

def simulate(
  beta=0.9, # dimensionless parameter
  Theta=30, # initial angle in degrees
  epsilon=0, # initial stretch of wire
  num_periods=6, # simulate for num_periods
  time_steps_per_period=60, # time step resolution
  plot=True, # make plots or no

  #from math import sin, cos, pi
  Theta = Theta*np.pi/180 # convert to radians
  # Initial position and velocity
  # (we order the equations such that Euler-Cromer in odespy
  # can be used, i.e., vx, x, vy, y)
  ic = [0,(1 + epsilon)*np.sin(Theta)),0,1 - (1 + epsilon)*np.cos(Theta),]

def f(u, t, beta):
  vx, x, vy, y = u
  L = np.sqrt(x**2 + (y-1)**2)
  h = beta/(1-beta)*(1 - beta/L) # help factor
  return [-h*x, vx, -h*(y-1) - beta, vy]

# Non-elastic pendulum (scaled similarly in the limit beta=1)
# solution Theta*cos(t)
  P = 2*pi
  dt = P/time_steps_per_period
  T = num_periods*P
  omega = 2*pi
  time_points = np.linspace(
  0, T, num_periods*time_steps_per_period+1)
  solver = odespy.EulerCromer(f, f_args=(beta,))
  solver.set_initial_condition(ic)
  u, t = solver.solve(time_points)
  x = u[:,1]
  y = u[:,3]
  theta = np.arctan(x/(1-y)

if plot:
  plt.figure()
  plt.plot(x, y, ’b-’, title=’Pendulum motion’,
  daspect=[1,1,1], daspectmode=’equal’,
  axis=[x.min(), x.max(), 1.3*y.min(), 1])
  plt.savefig(’tmp_xy.png’)
  plt.savefig(’tmp_xy.pdf’)
  
  # Plot theta in degrees
  plt.figure()
  plt.plot(t, theta*180/np.pi, ’b-’,
  title=’Angular displacement in degrees’)
  plt.savefig(’tmp_theta.png’)
  plt.savefig(’tmp_theta.pdf’)

if abs(Theta) < 10*pi/180:
  # Compare theta and theta_e for small angles (<10 degrees)
  theta_e = Theta*np.cos(omega*t) # non-elastic scaled sol.
  plt.figure()
  plt.plot(t, theta, t, theta_e,
  legend=[’theta elastic’, ’theta non-elastic’],
  title=’Elastic vs non-elastic pendulum, ’\
  ’beta=%g’ % beta)
  plt.savefig(’tmp_compare.png’)
  plt.savefig(’tmp_compare.pdf’)
  # Plot y vs x (the real physical motion)
  return x, y, theta, 

