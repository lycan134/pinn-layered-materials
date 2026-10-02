from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "sensor_temperatures_80_0.025.csv"
OUT = ROOT / "outputs" / "pinn_forward_focused"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
ADAM_STEPS = 5000
LBFGS_STEPS = 500
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
    def __init__(self):
        super().__init__()
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
        features = torch.cat(
            [2 * z - 1, tau / 3 - 1], dim=1
        )
        # Enforce initial temperature rise exactly.
        return tau / (tau + 0.1) * self.net(features)


class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TemperatureNet()
        self.layer2 = TemperatureNet()

        # Fixed known properties for this diagnostic run.
        self.register_buffer(
            "log_k2", torch.tensor(np.log(1.0))
        )
        self.register_buffer(
            "log_r", torch.tensor(np.log(0.001))
        )

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
    counts = [n // 4] * 3
    counts.append(n - sum(counts))

    # Dimensionless times: startup, pulse switch, and full interval.
    tau = torch.cat([
        0.1 * torch.rand(counts[0], 1, device=DEVICE),
        0.9 + 0.1 * torch.rand(counts[1], 1, device=DEVICE),
        1.0 + 0.1 * torch.rand(counts[2], 1, device=DEVICE),
        6.0 * torch.rand(counts[3], 1, device=DEVICE),
    ])

    # Include more points near the heated exterior boundary.
    z = torch.rand(n, 1, device=DEVICE)
    z[:n // 2] = z[:n // 2].square()

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

        total = (
            pde + bc_left + bc_right
            + flux_continuity + interface_jump
            + 10 * data_loss
        )

        parts = {
            "total": total,
            "pde": pde,
            "left_bc": bc_left,
            "right_bc": bc_right,
            "flux_continuity": flux_continuity,
            "interface_jump": interface_jump,
            "data": data_loss,
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

    k2, resistance = model.properties()
    summary = {
        "seed": SEED,
        "device": str(DEVICE),
        "k2": k2.item(),
        "R_interface": resistance.item(),
        "sensor_rmse_K": rmse,
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


if __name__ == "__main__":
    main()
