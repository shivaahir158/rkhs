from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class Task:
    id: int
    comp_costs: Dict[int, float]

@dataclass
class DAG:
    tasks: Dict[int, Task]
    edges: Dict[Tuple[int, int], float]
    successors: Dict[int, List[int]] = field(default_factory=dict)
    predecessors: Dict[int, List[int]] = field(default_factory=dict)

    def add_edge(self, u: int, v: int, cost: float):
        self.edges[(u, v)] = cost
        self.successors.setdefault(u, []).append(v)
        self.predecessors.setdefault(v, []).append(u)
        self.successors.setdefault(v, [])
        self.predecessors.setdefault(u, [])