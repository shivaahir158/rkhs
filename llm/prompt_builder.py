"""
RAG-augmented prompt construction for RKHS (Steps 4-5 in Algorithm 1).

Constructs structured prompts containing:
- Target graph statistics
- Retrieved kernel motifs and heuristic templates
- Scheduling constraints
- Previous iteration feedback (if any)
"""


def format_kernel_info(retrieved_kernels):
    """Format retrieved kernels for the prompt."""
    lines = []
    for i, (kernel, sim) in enumerate(retrieved_kernels):
        lines.append(f"\n--- Retrieved Kernel {i+1} (similarity={sim:.3f}) ---")
        lines.append(f"Motif type: {kernel.motif_type}")
        lines.append(f"Heuristic template: {kernel.template_desc}")

        sig = kernel.signature
        lines.append(f"Structural signature:")
        lines.append(f"  avg_crit={sig['avg_crit']:.2f}, max_crit={sig['max_crit']:.2f}")
        lines.append(f"  avg_fanout={sig['avg_fanout']:.2f}, max_fanout={sig['max_fanout']:.2f}")
        lines.append(f"  high_fanout_ratio={sig['high_fanout_ratio']:.3f}")
        lines.append(f"  reconvergence_ratio={sig['reconv_ratio']:.3f}")
        lines.append(f"  chain_ratio={sig['chain_ratio']:.3f}")
        lines.append(f"  num_levels={sig['num_levels']:.1f}")
        lines.append(f"  edge_density={sig['edge_density']:.3f}")
    return "\n".join(lines)


def format_graph_stats(dag, node_features):
    """Format target graph statistics for the prompt."""
    n = dag.num_nodes
    crit_vals = [node_features[v]["crit"] for v in node_features]
    fanout_vals = [node_features[v]["fanout"] for v in node_features]
    level_vals = [node_features[v]["level"] for v in node_features]

    from collections import Counter
    type_counts = Counter(dag.operations[v].op_type for v in dag.operations)

    lines = [
        f"Number of operations: {n}",
        f"Number of edges: {len(dag.edges)}",
        f"Critical path range: [{min(crit_vals)}, {max(crit_vals)}]",
        f"Max fanout: {max(fanout_vals)}",
        f"Number of levels: {max(level_vals) + 1}",
        f"Operation type distribution: {dict(type_counts)}",
        f"Resource limits: {dag.resource_limits}",
    ]
    return "\n".join(lines)


def build_synthesis_prompt(
    dag,
    node_features,
    retrieved_kernels,
    feedback=None,
    iteration=0,
):
    """
    Build the RAG-augmented synthesis prompt (Step 4).

    The LLM generates executable Python code defining:
        def priority(v, features, dag) -> float
    """
    graph_stats = format_graph_stats(dag, node_features)
    kernel_info = format_kernel_info(retrieved_kernels)

    feedback_section = ""
    if feedback:
        feedback_section = f"""
## Previous Iteration Feedback
{feedback}
Use this feedback to improve your priority function. Fix any issues mentioned.
"""

    prompt = f"""You are an expert in high-level synthesis (HLS) scheduling.

## Task
Generate a compact, deterministic Python priority function for resource-constrained
list scheduling. The scheduler selects ready operations by descending priority score.

## Target Graph Statistics
{graph_stats}

## Retrieved Kernel Motifs (from similar training graphs)
{kernel_info}

## Scheduling Constraints
- Operations have types (ALU, MUL, MEM, CTRL) with per-type resource limits.
- Each operation has duration d(v) = 1.
- Precedence constraints: operation v cannot start before all predecessors finish.
- Resource constraint: at most R_t operations of type t can execute per cycle.
- Goal: minimize makespan (total schedule length).
{feedback_section}
## Available Features
Each node v has pre-computed features accessible via features[v]:
- "crit": remaining critical-path length to sink (float)
- "fanout": number of successors (int)
- "level": longest path from source (int)
- "slack": ALAP - ASAP scheduling flexibility (float)
- "reconvergence": reconvergence marker R(v) (int)
- "in_degree": number of predecessors (int)
- "out_degree": number of successors (int)
- "op_type": operation type string

## Output Format
Return ONLY a Python function definition. No explanation, no imports.
The function must be:
1. Deterministic
2. Interpretable (simple arithmetic on features)
3. Compatible with list scheduling

```python
def priority(v, features, dag):
    f = features[v]
    # Your priority computation here
    return score
```

Iteration {iteration + 1}: Generate the best priority function based on the
retrieved kernels and graph structure. Combine complementary signals like
criticality, fanout, reconvergence, and resource pressure.
"""
    return prompt
