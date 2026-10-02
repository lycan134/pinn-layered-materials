from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "outputs" / "sensor_temperatures_80_0.025.csv"
OUTPUT = ROOT / "outputs" / "inverse_pilot"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Known properties and experimental conditions.
L1 = L2 = 0.005
K1 = 10.0
CV = 2.0e6
T_INITIAL = 293.15
HEAT_FLUX = 1.0e4
PULSE_DURATION = 10.0

# Different spatial resolution from the synthetic-data generator.
CELLS_PER_LAYER = 60

# Sample the informative transient, every 0.5 seconds.
OBS_TIMES = np.arange(0.5, 60.01, 0.5)

LAYOUTS = {
    "same_layer": [
        (0.002, "T_2mm_K"),
        (0.004, "T_4mm_K"),
    ],
    "across_interface": [
        (0.002, "T_2mm_K"),
        (0.007, "T_7mm_K"),
    ],
}

# Trial starting points, not supplied ground-truth values.
STARTS = [
    (0.5, 0.002),
    (2.0, 0.0002),
    (5.0, 0.005),
]

LOWER = np.log([0.1, 1.0e-5])
UPPER = np.log([10.0, 1.0e-2])


def predict_temperatures(log_parameters, positions):
    """Return sensor temperature rises: shape (sensors, times).

    Uses the finite-volume spatial model with an exact modal time
    solution for the prescribed rectangular heat pulse.
    """
    k2, r_interface = np.exp(log_parameters)

    n = 2 * CELLS_PER_LAYER
    dx = (L1 + L2) / n
    capacity = CV * dx

    conductivity = np.r_[
        np.full(CELLS_PER_LAYER, K1),
        np.full(CELLS_PER_LAYER, k2),
    ]

    resistance = (
        dx / (2 * conductivity[:-1])
        + dx / (2 * conductivity[1:])
    )
    resistance[CELLS_PER_LAYER - 1] += r_interface
    conductance = 1.0 / resistance

    diagonal = (
        np.r_[conductance, 0.0]
        + np.r_[0.0, conductance]
    ) / capacity

    rates, modes = eigh_tridiagonal(
        diagonal, -conductance / capacity
    )

    # An insulated slab has one uniform, zero-decay mode.
    rates[0] = 0.0

    heating_time = np.minimum(OBS_TIMES, PULSE_DURATION)
    cooling_time = np.maximum(OBS_TIMES - PULSE_DURATION, 0.0)

    response = np.empty((n, len(OBS_TIMES)))
    response[0] = heating_time

    response[1:] = (
        -np.expm1(-rates[1:, None] * heating_time)
        / rates[1:, None]
    )
    response *= np.exp(-rates[:, None] * cooling_time)

    # Interpolate cell-center temperatures to sensor positions.
    x = (np.arange(n) + 0.5) * dx
    weights = np.zeros((len(positions), n))

    for j, position in enumerate(positions):
        i = np.searchsorted(x, position) - 1
        if i < 0 or i >= n - 1:
            raise ValueError("Sensor must lie between cell centers.")

        fraction = (position - x[i]) / dx
        weights[j, i] = 1.0 - fraction
        weights[j, i + 1] = fraction

    source_modes = (HEAT_FLUX / capacity) * modes[0, :]

    return (
        (weights @ modes) * source_modes[None, :]
    ) @ response


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file: {DATA_FILE}\n"
            "Copy your 80-cell, 0.025 s dataset to this filename."
        )

    data = pd.read_csv(DATA_FILE)
    times = data["time_s"].to_numpy()

    if (
        not np.all(np.diff(times) > 0)
        or times[0] > OBS_TIMES[0]
        or times[-1] < OBS_TIMES[-1]
    ):
        raise ValueError("Input times must increase and cover 0.5–60 s.")

    if not np.allclose(np.diff(times), 0.025, atol=1e-9):
        raise ValueError("Expected the dataset with DT = 0.025 s.")

    summaries = []
    start_results = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (name, sensors) in zip(axes, LAYOUTS.items()):
        positions = [position for position, _ in sensors]

        observed = np.array([
            np.interp(OBS_TIMES, times, data[column]) - T_INITIAL
            for _, column in sensors
        ])

        def residual(log_parameters):
            return (
                predict_temperatures(log_parameters, positions)
                - observed
            ).ravel()

        fits = []

        for k_start, r_start in STARTS:
            print(
                f"{name}: starting k2={k_start:g}, "
                f"R={r_start:g}",
                flush=True,
            )

            fit = least_squares(
                residual,
                x0=np.log([k_start, r_start]),
                bounds=(LOWER, UPPER),
                diff_step=1e-4,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                max_nfev=100,
            )
            fits.append(fit)

            k_est, r_est = np.exp(fit.x)
            start_results.append({
                "layout": name,
                "initial_k2": k_start,
                "initial_R": r_start,
                "estimated_k2": k_est,
                "estimated_R": r_est,
                "rmse_K": np.sqrt(np.mean(fit.fun ** 2)),
                "success": fit.success,
                "message": fit.message,
            })

        best = min(fits, key=lambda result: result.cost)
        k_est, r_est = np.exp(best.x)
        predicted = predict_temperatures(best.x, positions)

        # Local sensitivity diagnostic in log-parameter coordinates.
        singular_values = np.linalg.svd(
            best.jac, compute_uv=False
        )
        condition = (
            singular_values[0] / singular_values[-1]
            if singular_values[-1] > 0 else np.inf
        )
        j0, j1 = best.jac[:, 0], best.jac[:, 1]
        sensitivity_cosine = (
            np.dot(j0, j1)
            / (np.linalg.norm(j0) * np.linalg.norm(j1))
        )

        summaries.append({
            "layout": name,
            "estimated_k2_W_mK": k_est,
            "estimated_R_m2K_W": r_est,
            "rmse_K": np.sqrt(np.mean(best.fun ** 2)),
            "log_parameter_jacobian_condition": condition,
            "sensitivity_cosine": sensitivity_cosine,
            "optimizer_success": best.success,
            "all_starts_success": all(f.success for f in fits),
        })

        saved = {"time_s": OBS_TIMES}

        for j, (position, _) in enumerate(sensors):
            label = f"{position * 1000:.0f} mm"
            line, = ax.plot(
                OBS_TIMES, predicted[j], label=f"Fit: {label}"
            )
            ax.scatter(
                OBS_TIMES[::4], observed[j, ::4],
                color=line.get_color(), s=15,
                label=f"Data: {label}",
            )
            saved[f"observed_{label}_rise_K"] = observed[j]
            saved[f"predicted_{label}_rise_K"] = predicted[j]

        pd.DataFrame(saved).to_csv(
            OUTPUT / f"{name}_predictions.csv", index=False
        )

        ax.set(
            xlabel="Time [s]",
            ylabel="Temperature rise [K]",
            title=name.replace("_", " ").title(),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "fit_summary.csv", index=False)
    pd.DataFrame(start_results).to_csv(
        OUTPUT / "multistart_results.csv", index=False
    )

    fig.tight_layout()
    fig.savefig(OUTPUT / "inverse_fit.png", dpi=200)

    print("\nFit summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved results to: {OUTPUT}")

    # Ground truth is used only for evaluation after fitting.
    print("\nBenchmark ground truth:")
    print("k2 = 1.0 W/(m K)")
    print("R_interface = 0.001 m^2 K/W")

    plt.show()


if __name__ == "__main__":
    main()
