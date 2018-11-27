import sympy as sym

V, t, I, w, n, dt, a, b, c, d = sym.symbols('V t I w n dt a b c d') #global symbols
f = None # global variable for the source term in the ODE

def ode_source_term(u):
  """Return the terms in the ODE that the source term
  must balance, here u'' + w**2*u. u is symbolic Python function of t.
  """
  return sym.diff(u(t), t, t) + w**2*u(t)

def ode_source_term_discrete(u):
  """Return the terms in the ODE that the source term
  must balance, here u'' + w**2*u. u is symbolic Python function of t.
  """
  return DtDt(u, dt) + w**2*u(t).subs(t,n*dt)

def residual_discrete_eq(u):
  """Return the residual of the discrete eq. with u inserted."""
  I,V = InitialCond(u)
  # the residual of ODE can be expressed as : R = u'' + W**2*u - f(t), so we can derive equation residual as below 
  R = DtDt(u, dt) + w**2*u(t).subs(t,n*dt)- ode_source_term_discrete(u)

  return sym.simplify(R)


def residual_discrete_eq_step1(u):
  """Return the residual of the discrete eq. at the first
  step with u inserted."""
  I,V = InitialCond(u)
  ue1 =  u(t).subs(t, 1*dt)
  u1  =  0.5*dt**2*(ode_source_term_discrete(u).subs(n,0)-w**2*u(t).subs(t,0)) + u(t).subs(t,0)+dt*V
  R   =  ue1 -u1  

  return sym.simplify(R)


def DtDt(u, dt):
  """Return 2nd-order finite difference for u_tt.
  u is a symbolic Python function of t.
  """
  dtdt = (u(t).subs(t,(n+1)*dt) - 2*u(t).subs(t,n*dt) + u(t).subs(t,(n-1)*dt))/dt**2
  return sym.simplify(dtdt)


def InitialCond(u):
  """based on the initial condition to define the constant values.
  """
  I = u(t).subs(t,0)
  V = sym.diff(u(t), t).subs(t, 0)
  return I,V 

def solver(I, V, f, w, dt, T):
   """
   Solve u'' + w**2*u = f for t in (0,T], u(0)=I and u'(0)=V,
   by a central finite difference method with time step dt.
   """
   dt = float(dt)
   Nt = int(round(T/dt))
   u = np.zeros(Nt+1)
   t = np.linspace(0, Nt*dt, Nt+1)

   u[0] = I
   u[1] = u[0] - 0.5*dt**2*w**2*u[0] + 0.5*dt**2*f(t[0]) + dt*V

   for n in range(1, Nt):
        u[n+1] = 2*u[n] - u[n-1] - dt**2*w**2*u[n] + dt**2*f(t[n])
   return u, t

import nose.tools as nt
import numpy as np

def test_quadratic_exact_solution():
    # Transform global symbolic variables to functions and numbers
    # for numerical computations
    global b, V, I, w
    b, V, I, w = 2, 1, 3, 1.5
    global f, t
    u_e = lambda t: b*t**2 + V*t + I 
    f = ode_source_term(u_e)         
    f = sym.lambdify(t, f)            

    dt = 2./w
    T  = 5 
    u, t = solver(I=I, V=V, f=f, w=w, dt=dt, T=T)
    u_e = u_e(t)
    error = np.abs(u - u_e).max()
    nt.assert_almost_equal(error, 0, delta=1E-12)
    print 'Round off Error in computing a quadratic solution:', error


def main(u):
  """
  Given some chosen solution u (as a function of t, implemented
  as a Python function), use the method of manufactured solutions
  to compute the source term f, and check if u also solves
  the discrete equations."""
 
  print "=============== Start Job Execution =======================" 
  print "=== Testing exact solution: %s ===" % u(t)
  print "=== Testing 2nd-order finite difference for u_tt: %s ===" % DtDt(u, dt)
  print "=== Initial conditions u(0)=%s, u'(0)=%s:" % (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))
  # Method of manufactured solution requires fitting f
  
  global f # source term in the ODE
  f = sym.simplify(ode_source_term(u))
  print "=== source term: %s ===" % f
  # Residual in discrete equations (should be 0)
  print "residual step1:", residual_discrete_eq_step1(u)
  print "residual:", residual_discrete_eq(u)
  print "=============== Completed Job Execution ====================" 

def linear():
  main(lambda t: V*t + I)

def quadratic():
  main(lambda t: b*t**2 + c*t + d)

def cubic():
  main(lambda t: a*t**3 + b*t**2 + c*t + d)
  
if __name__ == "__main__":
  linear()
  quadratic()
  cubic()
  test_quadratic_exact_solution()
