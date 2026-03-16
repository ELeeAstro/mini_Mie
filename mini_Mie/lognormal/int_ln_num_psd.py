import numpy as np
import miepython as mie


def ln_num_psd(r, N, lam, m, mu, sig):

  lr = len(r)

  # Do main Mie loop first to find Mie kernel - convert to diameter and lambda in [nm] - vectorised
  r_cm = r * 1e-4
  d_cm = 2.0 * r_cm # Diameter in cm
  lam_cm = lam * 1e-4 # wavelength in cm
  mu_cm = mu * 1e-4 # Median radius in cm
  sig_ln = np.log(sig) # log sigma_g

  # Evaluate the Mie efficiencies at the quadrature radii - vectorised
  Q_ext_r, Q_sca_r, _, g_r = mie.efficiencies(m, d_cm, lam_cm)

  # Find number density per size (size distribution) [cm-3 cm-1], assuming log-normal distribution - vectorised
  f_r = np.maximum(N/(r_cm*np.sqrt(2.0*np.pi)*sig_ln) * np.exp(-np.log(r_cm/mu_cm)**2/(2.0*sig_ln**2)),1e-199)


  # Now integrate to find the total beta, ssa, and g
  xsec = np.pi * r_cm**2
  b_ext = np.trapezoid(Q_ext_r * f_r * xsec, r_cm)
  b_sca = np.trapezoid(Q_sca_r * f_r * xsec, r_cm)

  ssa = b_sca/b_ext

  g = np.trapezoid(Q_sca_r * g_r * f_r * xsec, r_cm)/b_sca

  return b_ext, ssa, g
