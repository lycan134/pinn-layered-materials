import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pinn_inverse import (
    OUT, DEVICE, InversePINN, tensor, derivatives,
)


def main():
    model = InversePINN().to(DEVICE)
    model.load_state_dict(torch.load(
        OUT / "checkpoint.pt",
        map_location=DEVICE,
        weights_only=True,
    ))
    model.eval()

    # Exclude t=0 and the exact pulse-switching time from residual metrics.
    times = np.unique(np.r_[
        np.linspace(0.01, 0.99, 100),
        np.linspace(1.0, 9.99, 150),
        np.linspace(10.01, 60.0, 250),
    ])
    tau = times / 10
    left = tensor(np.column_stack([np.zeros_like(tau), tau]))
    right = tensor(np.column_stack([np.ones_like(tau), tau]))

    _, left_gradient = derivatives(model.layer1, left)
    predicted_flux = -left_gradient.detach().cpu().numpy().ravel()
    imposed_flux = (times < 10).astype(float)
    flux_error = predicted_flux - imposed_flux

    pd.DataFrame({
        "time_s": times,
        "imposed_flux_W_m2": imposed_flux * 1e4,
        "predicted_flux_W_m2": predicted_flux * 1e4,
        "normalized_flux_error": flux_error,
    }).to_csv(OUT / "boundary_diagnostic.csv", index=False)

    # Evaluate PDE residuals on a deterministic interior grid.
    z = np.linspace(0.001, 0.999, 60)
    zz, tt = np.meshgrid(z, tau)
    points = tensor(np.column_stack([zz.ravel(), tt.ravel()]))

    k2, _ = model.properties()
    residuals = []

    for network, coefficient in [
        (model.layer1, 2.0),
        (model.layer2, 0.2 * k2),
    ]:
        _, _, ut, uzz = derivatives(network, points, second=True)
        residual = (ut - coefficient * uzz)
        residuals.append(
            residual.detach().cpu().numpy().reshape(zz.shape)
        )

    predictions = pd.read_csv(OUT / "pinn_predictions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(
        times, imposed_flux * 1e4, "k--", label="Imposed"
    )
    axes[0, 0].plot(
        times, predicted_flux * 1e4, label="PINN"
    )
    axes[0, 0].set(
        xlabel="Time [s]", ylabel="Heat flux [W/m²]",
        title="Left-boundary heat flux",
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    for sensor in ["2mm", "7mm"]:
        error = (
            predictions[f"predicted_{sensor}_rise_K"]
            - predictions[f"observed_{sensor}_rise_K"]
        )
        axes[0, 1].plot(
            predictions.time_s, error, label=sensor
        )

    axes[0, 1].set(
        xlabel="Time [s]", ylabel="Prediction − data [K]",
        title="Sensor errors",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    maximum = max(np.abs(r).max() for r in residuals)

    for ax, residual, title in zip(
        axes[1], residuals, ["Layer 1", "Layer 2"]
    ):
        plot = ax.pcolormesh(
            z, times, residual,
            shading="auto", cmap="RdBu_r",
            vmin=-maximum, vmax=maximum,
        )
        ax.set(
            xlabel="Local position z",
            ylabel="Time [s]",
            title=f"{title}: dimensionless PDE residual",
        )
        fig.colorbar(plot, ax=ax)

    fig.tight_layout()
    fig.savefig(OUT / "pinn_diagnostics.png", dpi=200)

    # These metrics use this diagnostic grid, not the random
    # validation distribution used during training.
    for label, mask in [
        ("Early heating: 0–1 s", times < 1),
        ("Heating: 1–9 s", (times >= 1) & (times < 9)),
        ("Pulse switch: 9–11 s", (times >= 9) & (times <= 11)),
        ("Redistribution: >11 s", times > 11),
    ]:
        print(
            f"{label}: normalized flux RMS error = "
            f"{np.sqrt(np.mean(flux_error[mask] ** 2)):.6f}"
        )

    print(f"Saved diagnostics to: {OUT}")
    plt.show()


if __name__ == "__main__":
    main()
