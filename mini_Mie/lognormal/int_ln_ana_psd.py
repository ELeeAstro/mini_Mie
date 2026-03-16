import numpy as np
import miepython as mie

# Important - precompute Gauss-Hermite nodes and weights outside the timed region
n_quad = 16
x_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)

def ln_ana_psd(N, lam, m, mu, sig):

  # Use Gauss-Hermite quadrature after transforming the log-normal radius PDF
  mu_cm = mu * 1e-4 # r_med in cm
  lam_cm = lam * 1e-4 # wavelength in cm

  r_cm = np.exp(np.log(mu_cm) + np.sqrt(2.0) * np.log(sig) * x_quad)
  d_cm = 2.0 * r_cm # Diameter in cm

  # Evaluate the Mie efficiencies at the quadrature radii - vectorised
  Q_ext_r, Q_sca_r, _, g_r = mie.efficiencies(m, d_cm, lam_cm)

  weighted_xsec = w_quad * (np.pi * r_cm**2)
  norm = N / np.sqrt(np.pi)

  b_ext = norm * np.sum(weighted_xsec * Q_ext_r)
  b_sca = norm * np.sum(weighted_xsec * Q_sca_r)

  ssa = b_sca/b_ext

  g = norm * np.sum(weighted_xsec * Q_sca_r * g_r) / b_sca

  return b_ext, ssa, g
