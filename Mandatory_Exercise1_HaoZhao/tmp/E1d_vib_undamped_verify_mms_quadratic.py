import sympy as sym
V, t, I, w, dt = sym.symbols('V t I w dt') #global symbols
b, c, d = sym.symbols('b c d')
f = None # global variable for the source term in the ODE

def ode_source_term(u):
  """Return the terms in the ODE that the source term
  must balance, here u'' + w**2*u. u is symbolic Python function of t.
  """
  return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
  """Return the residual of the discrete eq. with u inserted."""
  # the residual of ODE can be expressed as : R = u'' + W**2*u - f(t), so we can derive equation residual as below 
  R = DtDt(u, t) + w**2*u(t)- ode_source_term(u)

  return sym.simplify(R)


def residual_discrete_eq_step1(u):
  """Return the residual of the discrete eq. at the first
  step with u inserted."""
  # the first step u1 can be expressed as : u1 = 0.5*dt**2(f(t0)-w**2*u0) + u0 + dt*v, so we can derive residual as below 
  # R1 is the discrete exact solution from quadratic, R2 is the discrete solution from ODE, so we can get the residual 
  R1 = u(t).subs(t, 1*dt) 
  R2 = (0.5*dt**2*(ode_source_term(u).subs(t,0*dt)-w**2*u(t).subs(t,0*dt))+u(t).subs(t,0*dt)+dt*c)
  R  = R1 - R2
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

  # print "the source term f:", f
  # Residual in discrete equations (should be 0)
  print "residual step1:", residual_discrete_eq_step1(u)
  print "residual:", residual_discrete_eq(u)

def linear():
  main(lambda t: V*t + I)

def quadratic():
  main(lambda t: b*t**2 + c*t + d)

if __name__ == "__main__":
  quadratic()
