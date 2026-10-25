import numpy as np
from .core import Agent
from .retrieval import Retriever


class ForecastingAgent(Agent):
    """Wraps a trained forecaster; emits per-turbine next-hour wind speed and power."""
    name = "forecasting"

    def __init__(self, bus, forecaster, power_curve):
        super().__init__(bus); self.f, self.pc = forecaster, power_curve

    def run(self, windows):
        ws = self.f(windows)
        return self.emit("forecast", {"wind_speed": ws.tolist(), "power_kw": self.pc(ws).tolist()})


class ScadaMonitoringAgent(Agent):
    """Rolling z-score anomaly detector on SCADA channels."""
    name = "scada_monitor"

    def __init__(self, bus, z=3.0):
        super().__init__(bus); self.z = z

    def run(self, history, current):     # history (T, N, C), current (N, C)
        mu, sd = history.mean(0), history.std(0) + 1e-6
        zs = np.abs((current - mu) / sd)
        return self.emit("anomalies", {"turbines": np.where((zs > self.z).any(1))[0].tolist(),
                                       "max_z": zs.max(1).round(2).tolist()})


class PredictiveMaintenanceAgent(Agent):
    """Threshold rules on vibration and gearbox temperature -> risk score and work orders (mock CMMS)."""
    name = "maintenance"

    def __init__(self, bus, retriever: Retriever, vib_limit=7.0, temp_limit=80.0):
        super().__init__(bus); self.r, self.vl, self.tl = retriever, vib_limit, temp_limit

    def run(self, vibration, gearbox_temp):
        risk = np.clip(np.maximum(vibration / self.vl, (gearbox_temp - 40) / (self.tl - 40)), 0, 1.5)
        orders = []
        for i in np.where(risk >= 0.9)[0]:
            q = "vibration stop inspection" if vibration[i] / self.vl >= (gearbox_temp[i] - 40) / (self.tl - 40) else "gearbox oil temperature"
            orders.append({"turbine": int(i), "risk": round(float(risk[i]), 3), "action": "derate_and_inspect" if risk[i] < 1 else "stop_and_inspect",
                           "evidence": self.r.query(q, k=1)})
        return self.emit("work_orders", {"risk": risk.round(3).tolist(), "orders": orders})


class GridTradingAgent(Agent):
    """Compares forecast fleet power with grid limit; proposes proportional curtailment."""
    name = "grid_trading"

    def run(self, forecast_kw, grid_limit_kw, unavailable):
        p = np.array(forecast_kw, float); p[list(unavailable)] = 0
        total = p.sum(); factor = min(1.0, grid_limit_kw / total) if total > 0 else 1.0
        return self.emit("dispatch", {"setpoints_kw": (p * factor).round(1).tolist(), "curtail_factor": round(factor, 4),
                                      "scheduled_kw": float((p * factor).sum())})


class ReportingHITLAgent(Agent):
    """Builds the action bundle and holds it for human approval (approver is a callable)."""
    name = "reporting_hitl"

    def __init__(self, bus, approver):
        super().__init__(bus); self.approver = approver

    def run(self, bundle):
        approved = bool(self.approver(bundle))
        return self.emit("approval", {"approved": approved, "bundle": bundle})
