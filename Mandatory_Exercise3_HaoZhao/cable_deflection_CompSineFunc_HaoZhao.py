
#!/usr/bin/env python
"""
----> solve partial differential equation with FEM <--------
"""

import sympy as sym
import time, sys
import numpy as np
from scitools.std import *

x, L, C, D, c_0, c_1, i = sym.symbols('x L C D c_0 c_1 i')

def solver(f,L,C,D):
   """solve the PDE : u'' = f(x),u(0)=C,u'(1)=D"""
   # integrate twice
   u_x = sym.integrate(f, (x, 0, x)) + c_0
   u = sym.integrate(u_x, (x, 0, x)) + c_1
   # setup 2 equations from the boundary conditions and solve the constants c_0 and c-1
   r = sym.solve([u.subs(x,0)-C, sym.diff(u,x).subs(x,L)-D],[c_0,c_1])
   #substitution the constants in the solution
   u = u.subs(c_0,r[c_0]).subs(c_1,r[c_1])
   #u = sym.simplify(sym.expand(u))
   return u


def E2_A_exact_solution():
    f = 1
    u = solver(f,1,0,0)
    print 'solved PDE:',u


def E2_C_visualize_solution():    
    """Compare the numerical and exact solution in a plot."""
    
    def exact_u(x):
        return x**2/2-x
        
    def numerical_u():
        c_i   = -16/(pi**3*(2*i + 1)**3)
        psi_i = sym.sin((2*i+1)*sym.pi*x/2)
        fun_i = c_i*psi_i
        
        f_N0   = sym.summation(fun_i,(i,0,0))
        f_N1   = sym.summation(fun_i,(i,0,1))
        f_N20   = sym.summation(fun_i,(i,0,20))
        
        u_N0 = sym.lambdify((x),f_N0,'numpy')
        u_N1 = sym.lambdify((x),f_N1,'numpy')
        u_N20 = sym.lambdify((x),f_N20,'numpy')
        return u_N0,u_N1,u_N20
        
    t_e  = linspace(0, 1, 1001) # fine mesh for u_e
    t_n  = linspace(0, 1, 21)   # coarse mesh for u
    
    u_e  = exact_u(t_e)
    u0,u1,u20 = numerical_u()

    plot(t_n, u0(t_n), 'r--o')
    hold('on')
    plot(t_n, u1(t_n), 'g--o')
    hold('on')
    plot(t_n, u20(t_n),'b--o')
    hold('on')
    plot(t_e, u_e,'k-')
    
    legend(['numerical solution N=0','numerical solution N=1','numerical solution N=20', 'exact solution'])
    xlabel('t')
    ylabel('u')
    title('computation of deflection of cable with sine function')
    savefig('cable_deflection_E2C.png')
    
def E2_D_visualize_solution():    
    """Compare the solution based on different basis funciton"""
    
    def exact_u(x):
        return x**2/2-x
        
    def numerical_u():        
        f_old = -16/(sym.pi**3)* sym.sin(sym.pi*x/2) -16/(27*sym.pi**3)* sym.sin(3*sym.pi*x/2)
        f_new = (48-144*sym.pi)/(-16*sym.pi**2+9*sym.pi**4)* sym.sin(sym.pi*x/2) + (96-18*sym.pi)/(-16*sym.pi**2+9*sym.pi**4)* sym.sin(sym.pi*x)
        
        u_old = sym.lambdify((x),f_old,'numpy')
        u_new = sym.lambdify((x),f_new,'numpy')

        return u_old,u_new
        
    t_e  = linspace(0, 1, 1001) # fine mesh for u_e
    t_n  = linspace(0, 1, 21)   # coarse mesh for u
    
    u_e  = exact_u(t_e)
    u0,u1 = numerical_u()
    
    figure()
    plot(t_n, u0(t_n), 'r--o')
    hold('on')
    plot(t_n, u1(t_n), 'g--o')
    hold('on')
    plot(t_e, u_e,'k-')
    
    legend(['numerical solution with basis function: sin((2i+1)*pi*x/2)','numerical solution with basis function: sin((i+1)*pi*x/2)', 'exact solution'])
    xlabel('t')
    ylabel('u')
    title('computation of deflection of cable with sine function')
    savefig('cable_deflection_E2D.png')

def E2_E_visualize_solution():    
    """Compare the numerical and exact solution in a plot."""
    
    def exact_u(x):
        return x**2/2-x
        
    def numerical_u():
        c_i   = -8*((-1)**i+1)/(pi**3*(i+1)**3)
        psi_i = sym.sin((i+1)*sym.pi*x/2)
        fun_i = c_i*psi_i
        
        f_N0   = sym.summation(fun_i,(i,0,0))
        f_N1   = sym.summation(fun_i,(i,0,1))
        f_N20   = sym.summation(fun_i,(i,0,20))
        
        u_N0 = sym.lambdify((x),f_N0,'numpy')
        u_N1 = sym.lambdify((x),f_N1,'numpy')
        u_N20 = sym.lambdify((x),f_N20,'numpy')
        return u_N0,u_N1,u_N20
        
    t_e  = linspace(0, 2, 1001) # fine mesh for u_e
    t_n  = linspace(0, 2, 21)   # coarse mesh for u
    
    u_e  = exact_u(t_e)
    u0,u1,u20 = numerical_u()
    
    figure()
    plot(t_n, u0(t_n), 'r--o')
    hold('on')
    plot(t_n, u1(t_n), 'g--o')
    hold('on')
    plot(t_n, u20(t_n),'b--o')
    hold('on')
    plot(t_e, u_e,'k-')
    
    legend(['numerical solution N=0','numerical solution N=1','numerical solution N=20', 'exact solution'])
    xlabel('t')
    ylabel('u')
    title('computation of deflection of cable with sine function')
    savefig('cable_deflection_E2E.png')

    
if __name__ == '__main__':
   E2_A_exact_solution()
   E2_C_visualize_solution()
   E2_D_visualize_solution()
   E2_E_visualize_solution()