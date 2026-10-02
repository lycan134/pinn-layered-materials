from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized


# Illustrative benchmark properties—not a specific real material.
L1 = 0.005                  # Layer 1 thickness [m]
L2 = 0.005                  # Layer 2 thickness [m]
K1 = 10.0                   # Layer 1 conductivity [W/(m K)]
K2 = 1.0                    # Layer 2 conductivity [W/(m K)]
CV = 2.0e6                  # Volumetric heat capacity [J/(m^3 K)]
R_INTERFACE = 1.0e-3         # Interface resistance [m^2 K/W]
T_INITIAL = 293.15           # Initial temperature [K]

HEAT_FLUX = 1.0e4            # Inward left-boundary heat flux [W/m^2]
PULSE_DURATION = 10.0        # Heating duration [s]
END_TIME = 200.0             # Observation duration [s]

CELLS_PER_LAYER = 80
DT = 0.025                    # Time step [s]

OUTPUT = Path(__file__).resolve().parent / "outputs"
OUTPUT.mkdir(exist_ok=True)


def simulate():
    n = 2 * CELLS_PER_LAYER
    dx = (L1 + L2) / n
    x = (np.arange(n) + 0.5) * dx

    # This benchmark uses equal layer thicknesses and uniform cells.
    conductivity = np.where(x < L1, K1, K2)

    # Resistance between neighboring cell centers.
    resistance = (
        dx / (2.0 * conductivity[:-1])
        + dx / (2.0 * conductivity[1:])
    )

    interface_face = CELLS_PER_LAYER - 1
    resistance[interface_face] += R_INTERFACE
    conductance = 1.0 / resistance

    capacity = np.full(n, CV * dx)  # Capacity per unit slab area

    # Insulated right boundary; left heat flux enters through the RHS.
    diagonal = (
        np.r_[conductance, 0.0]
        + np.r_[0.0, conductance]
    )

    matrix = diags(
        [
            -conductance,
            capacity / DT + diagonal,
            -conductance,
        ],
        offsets=[-1, 0, 1],
        format="csc",
    )
    solve_step = factorized(matrix)

    steps = int(round(END_TIME / DT))
    pulse_steps = int(round(PULSE_DURATION / DT))
    time = np.arange(steps + 1) * DT

    # Solve for temperature rise to improve numerical conditioning.
    rise = np.zeros((steps + 1, n))
    energy_input = np.zeros(steps + 1)

    for j in range(steps):
        flux = HEAT_FLUX if j < pulse_steps else 0.0

        rhs = capacity / DT * rise[j]
        rhs[0] += flux

        rise[j + 1] = solve_step(rhs)
        energy_input[j + 1] = energy_input[j] + flux * DT

    energy_stored = rise @ capacity
    energy_error = energy_stored - energy_input

    # Reconstruct temperatures immediately on either side of interface.
    # Cell-center differences also include conduction within half-cells.
    interface_flux = (
        rise[:, interface_face] - rise[:, interface_face + 1]
    ) * conductance[interface_face]

    interface_left = (
        T_INITIAL
        + rise[:, interface_face]
        - interface_flux * dx / (2.0 * K1)
    )
    interface_right = (
        T_INITIAL
        + rise[:, interface_face + 1]
        + interface_flux * dx / (2.0 * K2)
    )

    return (
        x, time, rise, energy_input, energy_stored,
        energy_error, interface_left, interface_right,
    )


def main():
    (
        x, time, rise, energy_input, energy_stored,
        energy_error, interface_left, interface_right,
    ) = simulate()

    temperature = T_INITIAL + rise

    # Virtual sensors at prescribed positions.
    sensor_positions = [0.002, 0.004, 0.007]
    data = {"time_s": time}

    for position in sensor_positions:
        # Linear interpolation between neighboring cell centers.
        values = np.array([
            np.interp(position, x, row)
            for row in temperature
        ])
        data[f"T_{position * 1000:.0f}mm_K"] = values

    data["T_interface_left_K"] = interface_left
    data["T_interface_right_K"] = interface_right
    data["energy_input_J_m2"] = energy_input
    data["energy_stored_J_m2"] = energy_stored

    pd.DataFrame(data).to_csv(
        OUTPUT / "sensor_temperatures.csv", index=False
    )

    np.savez_compressed(
        OUTPUT / "temperature_field.npz",
        x_m=x,
        time_s=time,
        temperature_K=temperature,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for position in sensor_positions:
        label = f"T_{position * 1000:.0f}mm_K"
        axes[0, 0].plot(
            time, data[label] - T_INITIAL,
            label=f"x = {position * 1000:.0f} mm",
        )

    axes[0, 0].axvline(
        PULSE_DURATION, color="gray", linestyle="--",
        label="Heating ends",
    )
    axes[0, 0].set(
        xlabel="Time [s]",
        ylabel="Temperature rise [K]",
        title="Virtual sensor responses",
    )
    axes[0, 0].legend()

    for snapshot in [5, 10, 30, 100, 200]:
        j = int(round(snapshot / DT))
        axes[0, 1].plot(
            x[:CELLS_PER_LAYER] * 1000,
            rise[j, :CELLS_PER_LAYER],
            label=f"{snapshot} s",
        )
        color = axes[0, 1].lines[-1].get_color()
        axes[0, 1].plot(
            x[CELLS_PER_LAYER:] * 1000,
            rise[j, CELLS_PER_LAYER:],
            color=color,
        )
        # Show actual interface traces without joining the jump.
        axes[0, 1].scatter(
            [L1 * 1000, L1 * 1000],
            [
                interface_left[j] - T_INITIAL,
                interface_right[j] - T_INITIAL,
            ],
            color=color, s=15,
        )

    axes[0, 1].axvline(
        L1 * 1000, color="gray", linestyle="--"
    )
    axes[0, 1].set(
        xlabel="Position [mm]",
        ylabel="Temperature rise [K]",
        title="Temperature profiles",
    )
    axes[0, 1].legend()

    axes[1, 0].plot(time, interface_left - interface_right)
    axes[1, 0].set(
        xlabel="Time [s]",
        ylabel="Interface temperature jump [K]",
        title="Temperature difference across the joint",
    )

    axes[1, 1].plot(
        time, energy_input, label="Energy supplied"
    )
    axes[1, 1].plot(
        time, energy_stored, "--", label="Energy stored"
    )
    axes[1, 1].set(
        xlabel="Time [s]",
        ylabel="Energy per unit area [J/m²]",
        title="Energy conservation",
    )
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT / "forward_results.png", dpi=200)

    expected_mean_rise = (
        HEAT_FLUX * PULSE_DURATION / (CV * (L1 + L2))
    )
    relative_energy_error = (
        np.max(np.abs(energy_error)) / energy_input[-1]
    )

    print(f"Output folder: {OUTPUT}")
    print(f"Maximum relative energy error: {relative_energy_error:.3e}")
    print(f"Expected final mean rise: {expected_mean_rise:.6f} K")
    print(f"Computed final mean rise: {rise[-1].mean():.6f} K")
    print("Saved sensor data, temperature field, and figure.")

    plt.show()


if __name__ == "__main__":
    main()
