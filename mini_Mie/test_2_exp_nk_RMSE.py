import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

os.environ["MIEPYTHON_USE_JIT"] = "1"
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter

from mini_Mie.exponential.int_exp_num_psd import exp_num_psd
from mini_Mie.exponential.int_exp_ana_psd import exp_ana_psd, n_quad as exp_n_quad


def load_nk_table(file_path):
  data = np.loadtxt(file_path, skiprows=5)
  lam_data = data[:, 0]
  n_data = data[:, 1]
  k_data = data[:, 2]
  return lam_data, n_data, k_data


def interpolate_m(lam, lam_data, n_data, k_data):
  lam = np.asarray(lam, dtype=float)

  if np.any(lam < lam_data[0]) or np.any(lam > lam_data[-1]):
    raise ValueError(
      f"Wavelengths must lie within the SiO2 table range "
      f"[{lam_data[0]}, {lam_data[-1]}] um."
    )

  n_interp = np.interp(lam, lam_data, n_data)
  k_interp = np.maximum(np.interp(lam, lam_data, k_data), 1e-6)
  return n_interp - 1j * k_interp


def rmse(reference, test):
  reference = np.asarray(reference, dtype=float)
  test = np.asarray(test, dtype=float)
  return np.sqrt(np.mean((test - reference) ** 2))


def relative_rmse_percent(reference, test):
  reference = np.asarray(reference, dtype=float)
  test = np.asarray(test, dtype=float)
  rel_err = (test - reference) / reference
  return 100.0 * np.sqrt(np.mean(rel_err ** 2))


def main():
  test_t0 = perf_counter()
  results_dir = Path(__file__).resolve().parent / "results"
  results_dir.mkdir(parents=True, exist_ok=True)

  r_min = 1e-3
  r_max = 100.0
  N = 1.0

  n_lam = 100
  lam_min = 0.3
  lam_max = 30.0
  lam = np.logspace(np.log10(lam_min), np.log10(lam_max), n_lam)

  nk_path = Path(__file__).resolve().parents[1] / "nk_data" / "SiO2.txt"
  lam_data, n_data, k_data = load_nk_table(nk_path)
  m_vals = interpolate_m(lam, lam_data, n_data, k_data)

  beta_vals = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0], dtype=float)

  n_r_baseline = 10000
  r_baseline = np.logspace(np.log10(r_min), np.log10(r_max), n_r_baseline)

  n_r_reduced = 100
  r_reduced = np.logspace(np.log10(r_min), np.log10(r_max), n_r_reduced)

  rows = []
  total_mie_eval_baseline = 0
  total_mie_eval_reduced = 0
  total_mie_eval_analytical = 0
  total_time_baseline = 0.0
  total_time_reduced = 0.0
  total_time_analytical = 0.0

  for beta in beta_vals:
    b_ext_num = np.empty_like(lam)
    b_ext_num_2 = np.empty_like(lam)
    b_ext_ana = np.empty_like(lam)

    total_t_num = 0.0
    total_t_num_2 = 0.0
    total_t_ana = 0.0

    for i, (lam_i, m_i) in enumerate(zip(lam, m_vals)):
      t0 = perf_counter()
      b_ext_num[i], _, _ = exp_num_psd(r_baseline, N, lam_i, m_i, beta)
      total_t_num += perf_counter() - t0

      t0 = perf_counter()
      b_ext_num_2[i], _, _ = exp_num_psd(r_reduced, N, lam_i, m_i, beta)
      total_t_num_2 += perf_counter() - t0

      t0 = perf_counter()
      b_ext_ana[i], _, _ = exp_ana_psd(N, lam_i, m_i, beta)
      total_t_ana += perf_counter() - t0

    rmse_num = rmse(b_ext_num, b_ext_num_2)
    rmse_ana = rmse(b_ext_num, b_ext_ana)

    rel_rmse_num = relative_rmse_percent(b_ext_num, b_ext_num_2)
    rel_rmse_ana = relative_rmse_percent(b_ext_num, b_ext_ana)

    rows.append({
      "beta [um]": beta,
      "rmse_b_ext_num": rmse_num,
      "rmse_b_ext_ana": rmse_ana,
      "rel_rmse_b_ext_num [%]": rel_rmse_num,
      "rel_rmse_b_ext_ana [%]": rel_rmse_ana,
      "t_baseline [s]": total_t_num,
      "t_reduced [s]": total_t_num_2,
      "t_analytical [s]": total_t_ana,
    })

    print(
      f"beta = {beta:6.3f} um, "
      f"RMSE reduced = {rmse_num:.8e}, RMSE analytical = {rmse_ana:.8e}"
    )

    total_mie_eval_baseline += n_lam * n_r_baseline
    total_mie_eval_reduced += n_lam * n_r_reduced
    total_mie_eval_analytical += n_lam * exp_n_quad
    total_time_baseline += total_t_num
    total_time_reduced += total_t_num_2
    total_time_analytical += total_t_ana

  df = pd.DataFrame(rows)
  pd.set_option("display.float_format", "{:.4e}".format)
  print()
  print(df[[
    "beta [um]",
    "rmse_b_ext_num",
    "rmse_b_ext_ana",
    "rel_rmse_b_ext_num [%]",
    "rel_rmse_b_ext_ana [%]",
  ]].to_string(index=False))

  output_csv = results_dir / "exp_rmse_sweep.csv"
  df.to_csv(output_csv, index=False)
  print(f"\nSaved RMSE table to {output_csv.name}")

  log_rel_num = np.log10(np.maximum(df["rel_rmse_b_ext_num [%]"].values, 1e-300))
  log_rel_ana = np.log10(np.maximum(df["rel_rmse_b_ext_ana [%]"].values, 1e-300))

  contour_min = -2.0
  contour_max = 2.0
  y_ticks = np.arange(int(contour_min), int(contour_max) + 1, 1)

  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

  axes[0].plot(beta_vals, log_rel_num, marker="o")
  axes[0].set_xlabel(r"$r_{0}$ [$\mu$m]", fontsize=16)
  axes[0].set_ylabel(r"$\beta_{\rm ext}$ [$\log_{10}$ % relative error]", fontsize=16)
  axes[0].set_xscale("log")
  axes[0].set_ylim(contour_min, contour_max)
  axes[0].set_yticks(y_ticks)
  axes[0].grid(True, alpha=0.3)
  axes[0].tick_params(axis="both", labelsize=14)

  axes[1].plot(beta_vals, log_rel_ana, marker="o")
  axes[1].set_xlabel(r"$r_{0}$ [$\mu$m]", fontsize=16)
  axes[1].set_xscale("log")
  axes[1].set_ylim(contour_min, contour_max)
  axes[1].set_yticks(y_ticks)
  axes[1].grid(True, alpha=0.3)
  axes[1].tick_params(axis="both", labelsize=14)

  plt.tight_layout()
  output_png = results_dir / "exp_rmse_sweep.png"
  plt.savefig(output_png, dpi=300, bbox_inches="tight")
  output_pdf = results_dir / "exp_rmse_sweep.pdf"
  plt.savefig(output_pdf, dpi=300, bbox_inches="tight")
  print(f"Saved plot to {output_pdf.name}")

  total_test_time = perf_counter() - test_t0
  print()
  print("Total Mie evaluations and timing:")
  print(f"  Baseline numerical : {total_mie_eval_baseline:d} evaluations, {total_time_baseline:.6f} s")
  print(f"  Reduced numerical  : {total_mie_eval_reduced:d} evaluations, {total_time_reduced:.6f} s")
  print(f"  Analytical         : {total_mie_eval_analytical:d} evaluations, {total_time_analytical:.6f} s")
  print(f"  Total test runtime : {total_test_time:.6f} s")
  plt.show()


main()
