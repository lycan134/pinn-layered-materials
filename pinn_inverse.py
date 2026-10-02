from pathlib import Path
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "sensor_temperatures_80_0.025.csv"
OUT = ROOT / "outputs" / "pinn_inverse_hard_flux_seed42_lbfgs2000"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
ADAM_STEPS = 5000
LBFGS_STEPS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)

# Dimensionless coordinates:
# z = local position / 0.005 m, tau = time / 10 s
# u = temperature rise / 5 K
# PDE: u_tau - 0.2*k*u_zz = 0
# Normalized rightward heat flux: q = -0.1*k*u_z
# Interface: u1-u2 = 2000*R*q
# Left heating: q=1 for tau<1, then q=0


class TemperatureNet(nn.Module):
    """One smooth network for one time interval."""
    def __init__(self, heating, exterior):
        super().__init__()
        self.heating = heating
        self.exterior = exterior
        layers = [nn.Linear(2, 32), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(32, 32), nn.Tanh()]
        layers += [nn.Linear(32, 1)]
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates):
        z = coordinates[:, :1]
        tau = coordinates[:, 1:2]
        local_time = tau if self.heating else (tau - 1) / 5
        # Even spatial coordinates give exactly zero correction gradient
        # at the appropriate exterior, without prescribing its temperature.
        spatial = z.square() if self.exterior == "left" else (1 - z).square()
        features = torch.cat([2 * spatial - 1, 2 * local_time - 1], dim=1)
        value = self.net(features)
        # Preserve the previous exact initial condition for heating.
        # The cooling initial field is connected by a temporal loss.
        return tau / (tau + 0.1) * value if self.heating else value


def flux_step_temperature(z, elapsed):
    """Unit flux response for u_tau = 2 u_zz on a half-line.

    Used only as a boundary lift. The network must still learn the finite
    slab and contact interface correction; this is not their solution.
    For elapsed > 0: F_z(0,elapsed) = -1 and F_tau - 2 F_zz = 0.
    At elapsed <= 0 its temperature is zero. PDE/flux derivatives at the
    incompatible boundary corners tau=0 and tau=1 are not assessed.
    """
    active = elapsed > 0
    safe = torch.where(active, elapsed, torch.ones_like(elapsed))
    a = 2.0  # Fixed known layer 1 diffusivity in dimensionless units.
    eta = z / (2 * torch.sqrt(a * safe))
    value = 2 * torch.sqrt(a * safe / torch.pi) * torch.exp(-eta.square())
    value = value - z * torch.erfc(eta)
    return torch.where(active, value, torch.zeros_like(value))


def pulse_temperature(coordinates):
    z = coordinates[:, :1]
    tau = coordinates[:, 1:2]
    return flux_step_temperature(z, tau) - flux_step_temperature(z, tau - 1)


class TimeSplitLayer(nn.Module):
    def __init__(self, exterior):
        super().__init__()
        self.exterior = exterior
        self.heating = TemperatureNet(heating=True, exterior=exterior)
        self.cooling = TemperatureNet(heating=False, exterior=exterior)

    def forward(self, coordinates):
        correction = torch.where(
            coordinates[:, 1:2] < 1,
            self.heating(coordinates),
            self.cooling(coordinates),
        )
        if self.exterior == "left":
            return pulse_temperature(coordinates) + correction
        return correction

    def temporal_jump(self, coordinates):
        # The analytical lift is shared by both time branches and cancels.
        return self.cooling(coordinates) - self.heating(coordinates)


class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TimeSplitLayer(exterior="left")
        self.layer2 = TimeSplitLayer(exterior="right")

        # Trainable material properties, initialized away from reference values.
        self.log_k2 = nn.Parameter(torch.tensor(np.log(0.5)))
        self.log_r = nn.Parameter(torch.tensor(np.log(0.002)))

    def properties(self):
        return self.log_k2.exp(), self.log_r.exp()

def tensor(values):
    return torch.as_tensor(
        values, dtype=torch.float64, device=DEVICE
    )


def derivatives(network, points, second=False):
    points = points.detach().clone().requires_grad_(True)
    u = network(points)

    grad = torch.autograd.grad(
        u, points, torch.ones_like(u), create_graph=True
    )[0]
    uz = grad[:, :1]
    ut = grad[:, 1:2]

    if second:
        uzz = torch.autograd.grad(
            uz, points, torch.ones_like(uz), create_graph=True
        )[0][:, :1]
        return u, uz, ut, uzz

    return u, uz


def sample_points(n):
    # Restore balanced heating / redistribution sampling. The open
    # intervals avoid evaluating the PDE at the discontinuous flux switch.
    n_heat = n // 2
    eps = 1e-6
    heating_tau = eps + (1 - 2 * eps) * torch.rand(n_heat, 1, device=DEVICE)
    cooling_tau = 1 + eps + (5 - 2 * eps) * torch.rand(n - n_heat, 1, device=DEVICE)
    tau = torch.cat([heating_tau, cooling_tau])
    z = torch.rand(n, 1, device=DEVICE)
    return torch.cat([z, tau], dim=1)


def boundary_points(n=256):
    tau = sample_points(n)[:, 1:2]
    left = torch.cat([torch.zeros_like(tau), tau], dim=1)
    right = torch.cat([torch.ones_like(tau), tau], dim=1)
    return left, right


def mse(value):
    return value.square().mean()


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    frame = pd.read_csv(DATA)
    times = np.arange(0.5, 60.01, 0.5)

    if frame.time_s.min() > times.min() or frame.time_s.max() < times.max():
        raise ValueError("Dataset must cover the observation times.")

    observed1 = tensor(
        np.interp(times, frame.time_s, frame.T_2mm_K)
        - 293.15
    ).reshape(-1, 1) / 5

    observed2 = tensor(
        np.interp(times, frame.time_s, frame.T_7mm_K)
        - 293.15
    ).reshape(-1, 1) / 5

    # Both sensors have local coordinate z=0.4 in their own layer.
    sensor_points = tensor(
        np.column_stack([np.full(len(times), 0.4), times / 10])
    )

    model = InversePINN().to(DEVICE)
    # Independent temporal connections inside each material layer.
    # Do not connect the two materials to each other here: their physical
    # interface may have a temperature jump due to contact resistance.
    seam_z = torch.linspace(0, 1, 129, device=DEVICE).reshape(-1, 1)
    seam_points = torch.cat([seam_z, torch.ones_like(seam_z)], dim=1)
    history = []
    started = time.perf_counter()

    def losses(interior1, interior2, left, right):
        k2, resistance = model.properties()

        _, _, ut1, uzz1 = derivatives(
            model.layer1, interior1, second=True
        )
        _, _, ut2, uzz2 = derivatives(
            model.layer2, interior2, second=True
        )

        pde = (
            mse(ut1 - 2.0 * uzz1)
            + mse(ut2 - 0.2 * k2 * uzz2)
        )

        # Left exterior boundary is z=0 in layer 1.
        _, gradient_left = derivatives(model.layer1, left)
        imposed_flux = (left[:, 1:2] < 1).to(torch.float64)
        bc_left = mse(-gradient_left - imposed_flux)

        # Right exterior boundary is z=1 in layer 2.
        _, gradient_right = derivatives(model.layer2, right)
        bc_right = mse(-0.1 * k2 * gradient_right)

        # Interface: z=1 in layer 1; z=0 in layer 2.
        u1, gradient1 = derivatives(model.layer1, right)
        u2, gradient2 = derivatives(model.layer2, left)

        q1 = -gradient1
        q2 = -0.1 * k2 * gradient2

        flux_continuity = mse(q1 - q2)
        interface_jump = mse(
            u1 - u2 - 2000 * resistance * q1
        )

        data_loss = (
            mse(model.layer1(sensor_points) - observed1)
            + mse(model.layer2(sensor_points) - observed2)
        )

        temporal_continuity = (
            mse(model.layer1.temporal_jump(seam_points))
            + mse(model.layer2.temporal_jump(seam_points))
        )

        total = (
            # Exterior flux is exact by construction. Keep its residuals
            # as diagnostics rather than training objectives.
            pde
            + flux_continuity + interface_jump
            + 10 * data_loss + 10 * temporal_continuity
        )

        parts = {
            "total": total,
            "pde": pde,
            "left_bc": bc_left,
            "right_bc": bc_right,
            "flux_continuity": flux_continuity,
            "interface_jump": interface_jump,
            "data": data_loss,
            "temporal_continuity": temporal_continuity,
        }
        return total, parts

    def record(stage, step, parts):
        k2, resistance = model.properties()
        row = {
            "stage": stage,
            "step": step,
            "k2": k2.item(),
            "R_interface": resistance.item(),
            **{name: value.item() for name, value in parts.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            OUT / "training_history.csv", index=False
        )
        torch.save(model.state_dict(), OUT / "checkpoint.pt")
        print(
            f"{stage} {step:5d} | loss={row['total']:.3e} | "
            f"k2={row['k2']:.6f} | R={row['R_interface']:.7f}",
            flush=True,
        )

    print(f"Device: {DEVICE}", flush=True)
    print("Fixed-parameter diagnostic: exact exterior flux away from pulse corners; temporal continuity weight=10.", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, ADAM_STEPS + 1):
        if step == 1 or step % 100 == 0:
            points1 = sample_points(512)
            points2 = sample_points(512)
            left, right = boundary_points()

        optimizer.zero_grad()
        total, parts = losses(points1, points2, left, right)

        if not torch.isfinite(total):
            raise RuntimeError("Non-finite loss; inspect training history.")

        total.backward()
        optimizer.step()

        if step == 1 or step % 250 == 0:
            # Recalculate so logged loss and parameters match.
            _, logged = losses(points1, points2, left, right)
            record("Adam", step, logged)

    # Fixed collocation points during L-BFGS.
    points1 = sample_points(1024)
    points2 = sample_points(1024)
    left, right = boundary_points(512)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=LBFGS_STEPS,
        history_size=50,
        line_search_fn="strong_wolfe",
    )
    evaluations = 0

    def closure():
        nonlocal evaluations
        optimizer.zero_grad()
        total, parts = losses(points1, points2, left, right)
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite L-BFGS loss.")
        total.backward()
        evaluations += 1
        if evaluations % 100 == 0:
            print(
                f"L-BFGS evaluation {evaluations}: "
                f"loss={total.item():.3e}",
                flush=True,
            )
        return total

    optimizer.step(closure)
    _, parts = losses(points1, points2, left, right)
    record("LBFGS", evaluations, parts)

    # Independent collocation sample for physics checks.
    validation1 = sample_points(2048)
    validation2 = sample_points(2048)
    validation_left, validation_right = boundary_points(1024)
    _, validation = losses(
        validation1, validation2, validation_left, validation_right
    )

    with torch.no_grad():
        prediction1 = (
            model.layer1(sensor_points).cpu().numpy().ravel() * 5
        )
        prediction2 = (
            model.layer2(sensor_points).cpu().numpy().ravel() * 5
        )

    actual1 = observed1.cpu().numpy().ravel() * 5
    actual2 = observed2.cpu().numpy().ravel() * 5
    rmse = np.sqrt(np.mean(np.r_[
        prediction1 - actual1, prediction2 - actual2
    ] ** 2))

    # Check the temporal connection on a finer grid than used for training.
    check_z = torch.linspace(0, 1, 513, device=DEVICE).reshape(-1, 1)
    check_seam = torch.cat([check_z, torch.ones_like(check_z)], dim=1)
    with torch.no_grad():
        jump1 = model.layer1.temporal_jump(check_seam).cpu().numpy().ravel() * 5
        jump2 = model.layer2.temporal_jump(check_seam).cpu().numpy().ravel() * 5
    pd.DataFrame({
        "local_z": check_z.cpu().numpy().ravel(),
        "layer1_temporal_jump_K": jump1,
        "layer2_temporal_jump_K": jump2,
    }).to_csv(OUT / "temporal_connection.csv", index=False)
    k2, resistance = model.properties()
    summary = {
        "seed": SEED,
        "device": str(DEVICE),
        "k2": k2.item(),
        "R_interface": resistance.item(),
        "sensor_rmse_K": rmse,
        "temporal_jump_layer1_rmse_K": np.sqrt(np.mean(jump1 ** 2)),
        "temporal_jump_layer2_rmse_K": np.sqrt(np.mean(jump2 ** 2)),
        "temporal_jump_max_abs_K": max(np.max(np.abs(jump1)), np.max(np.abs(jump2))),
        "elapsed_s": time.perf_counter() - started,
        **{
            f"validation_{name}": value.item()
            for name, value in validation.items()
        },
    }
    pd.DataFrame([summary]).to_csv(
        OUT / "pinn_summary.csv", index=False
    )
    pd.DataFrame({
        "time_s": times,
        "observed_2mm_rise_K": actual1,
        "predicted_2mm_rise_K": prediction1,
        "observed_7mm_rise_K": actual2,
        "predicted_7mm_rise_K": prediction2,
    }).to_csv(OUT / "pinn_predictions.csv", index=False)

    print("\nFinal summary:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"\nSaved to: {OUT}")


def smoke_test():
    """Check both branches, coordinate derivatives, and optimizer gradients."""
    model = InversePINN().to(DEVICE)
    points = tensor([[0.2, 0.01], [0.4, 0.5], [0.7, 0.999],
                     [0.2, 1.001], [0.4, 3.0], [0.7, 5.99]])
    seam = tensor([[0.0, 1.0], [0.4, 1.0], [1.0, 1.0]])
    initial = tensor([[0.0, 0.0], [0.4, 0.0], [1.0, 0.0]])
    loss = torch.zeros((), device=DEVICE)
    for layer in (model.layer1, model.layer2):
        assert torch.all(layer(initial) == 0), "Initial condition failed"
        u, uz, ut, uzz = derivatives(layer, points, second=True)
        for value in (u, uz, ut, uzz):
            assert torch.isfinite(value).all(), "Non-finite derivatives"
        # Compare autodiff to finite differences away from the time seam.
        h = 1e-4
        with torch.no_grad():
            dz = torch.zeros_like(points); dz[:, 0] = h
            ht = 1e-7
            dt = torch.zeros_like(points); dt[:, 1] = ht
            numerical_z = (layer(points + dz) - layer(points - dz)) / (2 * h)
            numerical_t = (layer(points + dt) - layer(points - dt)) / (2 * ht)
            numerical_zz = (layer(points + dz) - 2 * layer(points) + layer(points - dz)) / h**2
        assert torch.allclose(uz, numerical_z, atol=1e-5, rtol=1e-4)
        assert torch.allclose(ut, numerical_t, atol=1e-4, rtol=1e-3)
        assert torch.allclose(uzz, numerical_zz, atol=1e-5, rtol=1e-3)
        loss = loss + mse(u) + mse(uz) + mse(ut) + mse(uzz) + mse(layer.temporal_jump(seam))
    # Exterior flux checks for both branches, without the corner times.
    boundary_tau = tensor([0.0001, 0.01, 0.5, 0.9999, 1.0001, 1.5, 6.0]).reshape(-1, 1)
    left = torch.cat([torch.zeros_like(boundary_tau), boundary_tau], dim=1)
    right = torch.cat([torch.ones_like(boundary_tau), boundary_tau], dim=1)
    _, gradient_left = derivatives(model.layer1, left)
    _, gradient_right = derivatives(model.layer2, right)
    expected = (boundary_tau < 1).to(torch.float64)
    assert torch.allclose(-gradient_left, expected, atol=1e-11, rtol=0), "Left flux failed"
    assert torch.allclose(gradient_right, torch.zeros_like(gradient_right), atol=1e-11, rtol=0), "Insulation failed"
    # The lift itself satisfies the layer 1 PDE at regular points.
    _, _, lift_t, lift_zz = derivatives(pulse_temperature, points, second=True)
    assert torch.allclose(lift_t, 2 * lift_zz, atol=1e-9, rtol=1e-8), "Lift PDE failed"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Invalid gradient: {name}"
    optimizer.step()
    k2, resistance = model.properties()
    assert abs(k2.item() - 1) < 1e-12 and abs(resistance.item() - 0.001) < 1e-12
    clone = InversePINN().to(DEVICE)
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(model.layer1(points), clone.layer1(points))
    print(f"Smoke test passed on {DEVICE}: branches, derivatives, gradients, fixed properties, state reload, exact exterior flux and lift PDE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
    else:
        main()
