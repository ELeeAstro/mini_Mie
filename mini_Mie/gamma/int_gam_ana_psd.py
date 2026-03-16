import numpy as np
import miepython as mie
from scipy.special import gamma

# Precompute Gauss-Laguerre nodes and weights outside the timed region.
n_quad = 16
x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
PI = np.pi
UM_TO_CM = 1e-4
LAM_TO_NM = 1e3
DIAM_CM_TO_NM = 2.0e7


def gam_ana_psd(N, lam, m, alpha, beta):

  # Use Gauss-Laguerre quadrature and include x^(alpha-1) in the integrand.
  beta_cm = beta * UM_TO_CM
  r_cm = beta_cm * x_quad
  diam_nm = DIAM_CM_TO_NM * r_cm

  # Evaluate the Mie efficiencies at the quadrature radii - vectorised
  Q_ext_r, Q_sca_r, _, g_r = mie.efficiencies(m, diam_nm, lam * LAM_TO_NM)

  wt = w_quad * x_quad**(alpha - 1.0) / gamma(alpha)
  weighted_xsec = wt * (PI * r_cm * r_cm)
  b_ext = N * np.dot(weighted_xsec, Q_ext_r)
  sca_weighted = weighted_xsec * Q_sca_r
  b_sca = N * np.sum(sca_weighted)

  ssa = b_sca/b_ext
  g = N * np.dot(sca_weighted, g_r) / b_sca

  return b_ext, ssa, g
