import numpy as np
import miepython as mie


def exp_num_psd(r, N, lam, m, beta):

  lr = len(r)

  Q_ext_r = np.empty(lr)
  Q_sca_r = np.empty(lr)
  Q_back_r = np.empty(lr)
  g_r = np.empty(lr)

  # Do main Mie loop first to find Mie kernel - convert to diameter and lambda in [nm] - vectorised
  Q_ext_r[:], Q_sca_r[:], Q_back_r[:], g_r[:] = mie.efficiencies(m, r[:]*1e3*2.0, lam*1e3)


  r_cm = r * 1e-4
  beta_cm = beta * 1e-4

  # Find number density per size (size distribution) [cm-3 cm-1], assuming exponential distribution - vectorised
  f_r = np.empty(lr)
  f_r[:] = N/beta_cm * np.exp(-r_cm[:]/beta_cm)


  # Now integrate to find the total beta, ssa, and g
  xsec = np.pi * r_cm**2
  b_ext = np.trapezoid(Q_ext_r * f_r * xsec, r_cm)

  b_sca = np.trapezoid(Q_sca_r * f_r * xsec, r_cm)

  ssa = b_sca/b_ext

  g = np.trapezoid(Q_sca_r * g_r * f_r * xsec, r_cm)/b_sca

  return b_ext, ssa, g
