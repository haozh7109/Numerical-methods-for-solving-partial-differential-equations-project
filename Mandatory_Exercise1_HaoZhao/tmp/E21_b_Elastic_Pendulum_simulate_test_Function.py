def test_equilibrium():
  """Test that the elastic pendulum simulation by using condition x=y=theta=0."""
  x, y, theta, t = simulate(beta=0.9, Theta=0, epsilon=0, num_periods=6, ime_steps_per_period=10, plot = false)
  tol = 1E-10
  assert np.abs(x.max()) < tol
  assert np.abs(y.max()) < tol
  assert np.abs(theta.max()) < tol
