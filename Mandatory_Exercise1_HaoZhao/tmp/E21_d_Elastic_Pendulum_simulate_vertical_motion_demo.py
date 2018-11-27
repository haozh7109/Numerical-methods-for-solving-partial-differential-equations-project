def demo(beta=0.999, Theta=40, num_periods=3):
  x, y, theta, t = simulate(
  beta=beta, Theta=Theta, epsilon=0,
  num_periods=num_periods, time_steps_per_period = 600,plot=true)
