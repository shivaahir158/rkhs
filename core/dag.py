"""
DAG representation for resource-constrained list scheduling (Section 1.2).

G = (V, E) where each node v has type tau(v) and duration d(v).
Each edge (u,v) encodes precedence u < v.
Resource capacities R_t limit how many type-t ops execute per cycle.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import networkx as nx


@dataclass
class Operation:
    """A single operation (node) in the DAG."""
    id: int
    op_type: str        # tau(v) in T
    duration: int = 1   # d(v), default 1


@dataclass
class DAG:
    """Resource-constrained DAG for HLS scheduling."""
    operations: Dict[int, Operation] = field(default_factory=dict)
    edges: Set[Tuple[int, int]] = field(default_factory=set)
    successors: Dict[int, List[int]] = field(default_factory=dict)
    predecessors: Dict[int, List[int]] = field(default_factory=dict)
    resource_limits: Dict[str, int] = field(default_factory=dict)

    def add_operation(self, op: Operation):
        self.operations[op.id] = op
        self.successors.setdefault(op.id, [])
        self.predecessors.setdefault(op.id, [])

    def add_edge(self, u: int, v: int):
        """Add precedence edge u -> v."""
        if (u, v) in self.edges:
            return
        self.edges.add((u, v))
        self.successors.setdefault(u, [])
        self.predecessors.setdefault(v, [])
        self.successors[u].append(v)
        self.predecessors[v].append(u)
        self.successors.setdefault(v, [])
        self.predecessors.setdefault(u, [])

    @property
    def nodes(self) -> List[int]:
        return list(self.operations.keys())

    @property
    def num_nodes(self) -> int:
        return len(self.operations)

    def to_networkx(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for v, op in self.operations.items():
            G.add_node(v, op_type=op.op_type, duration=op.duration)
        for u, v in self.edges:
            G.add_edge(u, v)
        return G

    def topological_order(self) -> List[int]:
        G = self.to_networkx()
        return list(nx.topological_sort(G))
