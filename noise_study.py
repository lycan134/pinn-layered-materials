import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from inverse_fit import (
    ROOT, DATA_FILE, OBS_TIMES, LAYOUTS,
    T_INITIAL, LOWER, UPPER, predict_temperatures,
)


OUTPUT = ROOT / "outputs" / "noise_pilot"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_TRIALS = 30
NOISE_STD_K = 0.01
SEED = 42

# Fixed initial guess for every trial and layout.
INITIAL_GUESS = np.log([0.5, 0.002])

# Used only to evaluate the fitted parameters.
TRUE_K2 = 1.0
TRUE_R = 0.001


def main():
    data = pd.read_csv(DATA_FILE)
    times = data["time_s"].to_numpy()

    if (
        not np.all(np.diff(times) > 0)
        or times[0] > OBS_TIMES[0]
        or times[-1] < OBS_TIMES[-1]
    ):
        raise ValueError("Input times must increase and cover 0.5–60 s.")

    columns = ["T_2mm_K", "T_4mm_K", "T_7mm_K"]

    clean = {
        column: np.interp(OBS_TIMES, times, data[column]) - T_INITIAL
        for column in columns
    }

    rng = np.random.default_rng(SEED)
    results = []
    measurements = []

    for trial in range(N_TRIALS):
        # Generate one noise trace per sensor.
        # Both layouts reuse the same noisy 2 mm trace.
        noisy = {
            column: clean[column] + rng.normal(
                0.0, NOISE_STD_K, size=len(OBS_TIMES)
            )
            for column in columns
        }

        for j, observation_time in enumerate(OBS_TIMES):
            measurements.append({
                "trial": trial + 1,
                "time_s": observation_time,
                **{
                    column: noisy[column][j] + T_INITIAL
                    for column in columns
                },
            })

        for layout, sensors in LAYOUTS.items():
            positions = [position for position, _ in sensors]
            observed = np.array([
                noisy[column] for _, column in sensors
            ])

            def residual(log_parameters):
                return (
                    predict_temperatures(log_parameters, positions)
                    - observed
                ).ravel() / NOISE_STD_K

            started = time.perf_counter()

            fit = least_squares(
                residual,
                x0=INITIAL_GUESS,
                bounds=(LOWER, UPPER),
                diff_step=1e-4,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=100,
            )

            elapsed = time.perf_counter() - started
            k_est, r_est = np.exp(fit.x)

            # Flag estimates close to the allowed parameter limits.
            near_bound = bool(np.any(
                np.minimum(fit.x - LOWER, UPPER - fit.x) < 1e-3
            ))

            results.append({
                "trial": trial + 1,
                "layout": layout,
                "estimated_k2": k_est,
                "estimated_R": r_est,
                "k2_error_percent": 100 * (k_est / TRUE_K2 - 1),
                "R_error_percent": 100 * (r_est / TRUE_R - 1),
                "temperature_rmse_K": (
                    np.sqrt(np.mean(fit.fun ** 2)) * NOISE_STD_K
                ),
                "optimizer_success": fit.success,
                "near_parameter_bound": near_bound,
                "function_evaluations": fit.nfev,
                "elapsed_s": elapsed,
                "message": fit.message,
            })

            print(
                f"Trial {trial + 1:02d}/{N_TRIALS} | {layout} | "
                f"k2={k_est:.6f} | R={r_est:.8f} | "
                f"success={fit.success}",
                flush=True,
            )

        # Save progress after each paired trial.
        pd.DataFrame(results).to_csv(
            OUTPUT / "noise_trials.csv", index=False
        )
        pd.DataFrame(measurements).to_csv(
            OUTPUT / "noisy_measurements.csv", index=False
        )

    trials = pd.DataFrame(results)
    summary_rows = []

    for layout, group in trials.groupby("layout", sort=False):
        valid = group[
            group["optimizer_success"]
            & ~group["near_parameter_bound"]
        ]

        row = {
            "layout": layout,
            "noise_std_K": NOISE_STD_K,
            "n_trials": len(group),
            "optimizer_failures": int(
                (~group["optimizer_success"]).sum()
            ),
            "near_bound_count": int(
                group["near_parameter_bound"].sum()
            ),
            "n_valid_for_summary": len(valid),
        }

        for parameter in ["k2", "R"]:
            errors = valid[f"{parameter}_error_percent"].to_numpy()

            row[f"{parameter}_bias_percent"] = (
                float(np.mean(errors)) if len(errors) else np.nan
            )
            row[f"{parameter}_error_sd_percent"] = (
                float(np.std(errors, ddof=1))
                if len(errors) > 1 else np.nan
            )
            row[f"{parameter}_rmse_percent"] = (
                float(np.sqrt(np.mean(errors ** 2)))
                if len(errors) else np.nan
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "noise_summary.csv", index=False)

    print("\nNoise-study summary:")
    print(summary.to_string(index=False))
    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
