"""Train the forecasting agent, then simulate a fleet over a test period with the full agent loop.
Reports forecast MAE/RMSE/MAPE/R2 vs persistence, and fleet-level scheduling error /
curtailment / work-order stats for: (a) persistence-driven ops, (b) multi-agent system."""
import argparse, json, os
import numpy as np, torch
from .data import synthetic_scada, load_scada_csv, add_features, make_windows, chrono_split, Scaler
from .metrics import regression_metrics
from .trainutil import fit, predict
from .forecaster import CNNBiLSTMAttention, power_curve
from .agents.orchestrator import OrchestratorAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/synthetic_scada.csv" if os.path.exists("data/synthetic_scada.csv") else None, help="SCADA-style CSV (default: committed synthetic CSV)"); ap.add_argument("--turbines", type=int, default=10)
    ap.add_argument("--hours", type=int, default=168); ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--grid_limit_kw", type=float, default=12000); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--interactive", action="store_true", help="ask a human to approve each bundle")
    ap.add_argument("--out", default="runs"); a = ap.parse_args()
    if a.quick: a.epochs, a.hours, a.turbines = 4, 48, 6
    rng = np.random.default_rng(0); lb = 24
    # --- forecaster training on one turbine's SCADA history ---
    base = load_scada_csv(a.csv) if a.csv else synthetic_scada(24 * 90)
    df = add_features(base); X, y = make_windows(df, lookback=lb); tr, va, te = chrono_split(len(X))
    sc = Scaler().fit(X[tr]); Xs = sc.transform(X).astype("float32"); ym, ys = y[tr].mean(), y[tr].std()
    m = fit(CNNBiLSTMAttention(X.shape[2]), Xs[tr], ((y[tr] - ym) / ys).astype("float32"),
            Xs[va], ((y[va] - ym) / ys).astype("float32"), epochs=a.epochs)
    ws_i = list(df.columns).index("wind_speed")
    res = {"forecast_test": regression_metrics(y[te], predict(m, Xs[te]) * ys + ym),
           "persistence_test": regression_metrics(y[te], X[te][:, -1, ws_i])}
    forecaster = lambda W: predict(m, sc.transform(W).astype("float32")) * ys + ym
    # --- fleet: each turbine = test-period windows with a site offset + noise ---
    N, H = a.turbines, min(a.hours, len(X[te]) - 1)
    fleetX = np.stack([X[te][:H + 1].copy() for _ in range(N)], 1)       # (H+1, N, lb, F)
    scale = rng.uniform(0.85, 1.15, N); fleetX[..., ws_i] *= scale[None, :, None]
    actual_ws = np.stack([y[te][:H + 1] * s for s in scale], 1)
    vib = 2 + rng.gamma(2, 0.6, (H + 1, N)); temp = 55 + rng.normal(0, 4, (H + 1, N))
    for k in rng.choice(N, max(1, N // 4), replace=False):                 # inject degrading turbines
        ramp = np.linspace(0, 1, H + 1); vib[:, k] += 6 * ramp ** 2; temp[:, k] += 30 * ramp
    approver = (lambda b: input(f"approve {len(b['work_orders'])} orders, curtail={b['dispatch']['curtail_factor']}? [y/N] ").lower() == "y") \
        if a.interactive else (lambda b: True)
    orch = OrchestratorAgent(forecaster, power_curve, approver)
    sched_err_mas, sched_err_pers, curtail, orders = [], [], [], 0
    stopped_until = {}
    for t in range(H):
        hist = np.stack([vib[max(0, t - 24):t + 1], temp[max(0, t - 24):t + 1]], -1)
        cur = np.stack([vib[t], temp[t]], -1)
        bundle, ok = orch.cycle(fleetX[t], hist, cur, vib[t], temp[t], a.grid_limit_kw)
        stops = {o["turbine"] for o in bundle["work_orders"] if o["action"] == "stop_and_inspect"}
        orders += len(bundle["work_orders"])
        for k in stops:
            if k not in stopped_until: stopped_until[k] = t; vib[t + 1:, k] = 2 + rng.gamma(2, .6, H - t); temp[t + 1:, k] = 55
        avail = np.array([k not in stops for k in range(N)])
        actual_p = power_curve(actual_ws[t + 1]) * avail
        sched_err_mas.append(abs(bundle["dispatch"]["scheduled_kw"] - min(actual_p.sum(), a.grid_limit_kw)))
        fpers = power_curve(fleetX[t][:, -1, ws_i])
        sched_err_pers.append(abs(min(fpers.sum(), a.grid_limit_kw) - min(power_curve(actual_ws[t + 1]).sum(), a.grid_limit_kw)))
        curtail.append(1 - bundle["dispatch"]["curtail_factor"])
    res["fleet_sim"] = {"turbines": N, "hours": H, "mean_abs_schedule_error_kw_multi_agent": float(np.mean(sched_err_mas)),
                        "mean_abs_schedule_error_kw_persistence": float(np.mean(sched_err_pers)),
                        "mean_curtailment_fraction": float(np.mean(curtail)), "work_orders_raised": orders,
                        "turbines_stopped_for_inspection": sorted(int(k) for k in stopped_until),
                        "bundles_committed": len(orch.committed)}
    os.makedirs(a.out, exist_ok=True); orch.bus.dump(f"{a.out}/audit_log.json")
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=2); print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
