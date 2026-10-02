# Layered-material PINN research

Synthetic benchmark for estimating layer-2 conductivity and
interfacial thermal resistance from transient temperatures.

Reference values:
- k2 = 1 W/(m K)
- R = 0.001 m² K/W

## Experiment progression

| Experiment | Sensor RMSE (K) | Main finding |
|---|---:|---|
| Original fixed-parameter PINN | 0.02148 | Boundary errors near pulse transitions |
| Focused sampling | 0.03977 | Temperature accuracy worsened |
| Split time, continuity weight 10 | 0.02757 | Temporal connection remained inaccurate |
| Split time, continuity weight 100 | 0.05163 | Better connection, worse temperature fit |
| Direct exterior flux, fixed parameters | 0.00495 | Improved temperature approximation |
| Direct exterior flux, inverse, seed 42 | 0.00521 | k2 = 0.998804; R = 0.00100158 |

These are exploratory pilot results, not a completed robustness study.
Earlier code versions were not necessarily preserved.

## Current investigation

Repeat clean-data inverse training with different seeds.
Then evaluate initialization sensitivity, sensor layouts, and noise.

## Reproducibility

Preserve each experiment in its own output directory.
Record the code commit, seed, starting parameters, training settings,
software versions, and reference dataset for every future run.
