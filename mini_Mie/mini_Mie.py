import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

os.environ["MIEPYTHON_USE_JIT"] = "1"

import numpy as np
from time import perf_counter

from mini_Mie.lognormal.int_ln_num_psd import ln_num_psd
from mini_Mie.lognormal.int_ln_ana_psd import ln_ana_psd
from mini_Mie.exponential.int_exp_num_psd import exp_num_psd
from mini_Mie.exponential.int_exp_ana_psd import exp_ana_psd
from mini_Mie.gamma.int_gam_num_psd import gam_num_psd
from mini_Mie.gamma.int_gam_ana_psd import gam_ana_psd

def main():


  # First we define particle size distribution properties and Mie parameters

  # Define particle size grid [um]
  n_r = 10000
  r_min = 1e-3 
  r_max = 100.0
  r = np.logspace(np.log10(r_min), np.log10(r_max),n_r)


  # Define total number density of cloud particles [cm-3]
  N = 1.0

  # Define single wavelength [um]
  lam = 0.5

  # Define (complex) single n and k constants
  m = 4.0 - 0.1j

  # log-normal parameters
  mu = 1.0
  sig = 2.0

  # exponential parameters
  beta = 1.0

  # gamma parameters
  alpha = 3.0
  beta_gam = 0.5

  # Call log-normal calculations
  t0 = perf_counter()
  b_ext_num, ssa_num, g_num = ln_num_psd(r, N, lam, m, mu, sig)
  t_num = perf_counter() - t0

  t0 = perf_counter()
  b_ext_ana, ssa_ana, g_ana = ln_ana_psd(N, lam, m, mu, sig)
  t_ana = perf_counter() - t0

  print("Numerical:", b_ext_num, ssa_num, g_num, "time [s] =", t_num)

  print("Analytical:", b_ext_ana, ssa_ana, g_ana, "time [s] =", t_ana)


  # Call exponential calculations
  t0 = perf_counter()
  b_ext_num, ssa_num, g_num = exp_num_psd(r, N, lam, m, beta)
  t_num = perf_counter() - t0

  t0 = perf_counter()
  b_ext_ana, ssa_ana, g_ana = exp_ana_psd(N, lam, m, beta)
  t_ana = perf_counter() - t0

  print("Numerical:", b_ext_num, ssa_num, g_num, "time [s] =", t_num)
  print("Analytical:", b_ext_ana, ssa_ana, g_ana, "time [s] =", t_ana)

  # Gamma parameters
  t0 = perf_counter()
  b_ext_num, ssa_num, g_num = gam_num_psd(r, N, lam, m, alpha, beta_gam)
  t_num = perf_counter() - t0

  t0 = perf_counter()
  b_ext_ana, ssa_ana, g_ana = gam_ana_psd(N, lam, m, alpha, beta_gam)
  t_ana = perf_counter() - t0

  print("Numerical:", b_ext_num, ssa_num, g_num, "time [s] =", t_num)
  print("Analytical:", b_ext_ana, ssa_ana, g_ana, "time [s] =", t_ana)

  return


main()
