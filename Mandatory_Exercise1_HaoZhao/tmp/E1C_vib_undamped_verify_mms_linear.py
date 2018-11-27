import sympy as sym
V, t, I, w, dt = sym.symbols('V t I w dt') #global symbols
f = None # global variable for the source term in the ODE

def ode_source_term(u):
  """Return the terms in the ODE that the source term
  must balance, here u'' + w**2*u. u is symbolic Python function of t.
  """
  return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
  """Return the residual of the discrete eq. with u inserted."""
  R = DtDt(u, t) + w**2*u(t)- ode_source_term(u)

  return sym.simplify(R)


def residual_discrete_eq_step1(u):
  """Return the residual of the discrete eq. at the first
  step with u inserted."""
  # the first step u1 can be expressed as : u1 = 0.5*dt**2(f(t0)-w**2*u0) + u0 + dt*v, so we can derive residual as below 
  R = u(t).subs(t, 1*dt) - (0.5*dt**2*(ode_source_term(u).subs(t,0*dt)-w**2*u(t).subs(t,0*dt))+u(t).subs(t,0*dt)+dt*V) 
  return sym.simplify(R)


def DtDt(u, t):
  """Return 2nd-order finite difference for u_tt.
  u is a symbolic Python function of t.
  """
  return sym.diff(u(t), t, t)


def main(u):
  """
  Given some chosen solution u (as a function of t, implemented
  as a Python function), use the method of manufactured solutions
  to compute the source term f, and check if u also solves
  the discrete equations.
  """

  print "=== Testing exact solution: %s ===" % u(t)
  print "Initial conditions u(0)=%s, u'(0)=%s:" % (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))
  # Method of manufactured solution requires fitting f
  
  global f # source term in the ODE
  f = sym.simplify(ode_source_term(u))

  # Residual in discrete equations (should be 0)
  print "residual step1:", residual_discrete_eq_step1(u)
  print "residual:", residual_discrete_eq(u)

def linear():
  main(lambda t: V*t + I)

if __name__ == "__main__":
  linear()
