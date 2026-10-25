"""Plain-dataclass agent contracts (as in the paper's reference design: vendor-neutral, JSON-serialisable)."""
from dataclasses import dataclass, field, asdict
import json, time


@dataclass
class Message:
    sender: str
    kind: str
    payload: dict
    ts: float = field(default_factory=time.time)


class ContextBus:
    """Short-term blackboard for one planning cycle + append-only audit log."""
    def __init__(self):
        self.board, self.log = {}, []

    def post(self, msg: Message):
        self.board[msg.kind] = msg.payload; self.log.append(asdict(msg))

    def get(self, kind, default=None):
        return self.board.get(kind, default)

    def dump(self, path):
        json.dump(self.log, open(path, "w"), indent=1, default=str)


class Agent:
    name = "agent"

    def __init__(self, bus: ContextBus):
        self.bus = bus

    def emit(self, kind, payload):
        self.bus.post(Message(self.name, kind, payload)); return payload
