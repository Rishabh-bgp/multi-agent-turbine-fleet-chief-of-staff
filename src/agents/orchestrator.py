from .core import ContextBus
from .specialists import (ForecastingAgent, ScadaMonitoringAgent, PredictiveMaintenanceAgent,
                          GridTradingAgent, ReportingHITLAgent)
from .retrieval import Retriever


class OrchestratorAgent:
    """Chief of staff: runs one planning cycle, resolves conflicts (maintenance vs. dispatch),
    and is the only component allowed to commit approved actions."""
    name = "orchestrator"

    def __init__(self, forecaster, power_curve, approver, knowledge=None):
        self.bus = ContextBus()
        r = Retriever(knowledge)
        self.fc = ForecastingAgent(self.bus, forecaster, power_curve)
        self.mon = ScadaMonitoringAgent(self.bus)
        self.pm = PredictiveMaintenanceAgent(self.bus, r)
        self.grid = GridTradingAgent(self.bus)
        self.hitl = ReportingHITLAgent(self.bus, approver)
        self.committed = []

    def cycle(self, windows, history, current, vibration, gearbox_temp, grid_limit_kw):
        f = self.fc.run(windows)
        an = self.mon.run(history, current)
        wo = self.pm.run(vibration, gearbox_temp)
        stopped = {o["turbine"] for o in wo["orders"] if o["action"] == "stop_and_inspect"}
        # conflict resolution: turbines under stop orders are unavailable to dispatch
        d = self.grid.run(f["power_kw"], grid_limit_kw, stopped)
        bundle = {"forecast": f, "anomalies": an, "work_orders": wo["orders"], "dispatch": d}
        ap = self.hitl.run(bundle)
        if ap["approved"]:
            self.committed.append(bundle)
        return bundle, ap["approved"]
