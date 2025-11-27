"""Main graph for Qwen VL agent with logic and visual memory functionality.

This module defines the optimized memory-enabled evaluation workflow with parallel execution:

Optimized Memory Workflow (7 nodes):

Sequential Phase:
1. initialize_case - Extract image URLs (pure data transformation, no model calls)

Parallel Retrieve Phase (fan-out → fan-in):
2. retrieve_logic_memories (analyze problem + retrieve) ────┐
3. retrieve_visual_memories (retrieve + generate heatmap) ───┘ → (parallel execution)
   Both converge to → 4. call_model

Sequential Verification:
4. call_model - Model inference with both types of memories injected
5. verify_answer - Verify answer using math-verify

Parallel Error Analysis + Memory Generation Phase (fan-out → fan-in):
   (conditional: only if verified=False)
   6. generate_logic_memory (analyze + store) ──────────┐
   7. generate_visual_memory (analyze + store) ─────────┘
                                    ↓
                              8. save_output (waits for both branches)

Key Design Principles:
- initialize_case: Pure data transformation, sets up immutable Case information
- retrieve_logic_memories: Problem analysis (model call) + memory retrieval (logic-specific)
- retrieve_visual_memories: Visual memory retrieval + heatmap generation (vision-specific)
- Clear separation: Case metadata vs. dynamic state updates

This is the main graph shown in langgraph dev.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from vl_agent.context import Context
from vl_agent.nodes import (
    call_model,
    generate_logic_memory,
    generate_visual_memory,
    initialize_case,
    retrieve_logic_memories,
    retrieve_visual_memories,
    save_output,
    verify_answer,
)
from vl_agent.state import EvaluationState


def should_analyze_error(state: EvaluationState) -> str | list[str]:
    """Route based on verification result.

    If verified, go to save_output.
    If not verified, trigger parallel error analysis (fan-out to both generate nodes).
    """
    if state.verified:
        return "save_output"
    # Fan-out to both generate nodes for parallel execution
    return ["generate_logic_memory", "generate_visual_memory"]


# Build memory-enabled evaluation graph with logic and visual memory
builder = StateGraph(EvaluationState, context_schema=Context)

# Add all nodes (7 functional nodes + save_output = 8 nodes total)
builder.add_node("initialize_case", initialize_case)
builder.add_node("retrieve_logic_memories", retrieve_logic_memories)
builder.add_node("retrieve_visual_memories", retrieve_visual_memories)
builder.add_node("call_model", call_model)
builder.add_node("verify_answer", verify_answer)
builder.add_node("generate_logic_memory", generate_logic_memory)
builder.add_node("generate_visual_memory", generate_visual_memory)
builder.add_node("save_output", save_output)

# ========== Sequential Phase ==========
# Define edges: start → initialize_case
builder.add_edge("__start__", "initialize_case")

# ========== Parallel Retrieve Phase ==========
# Fan-out: initialize_case → both retrieve nodes (parallel execution)
builder.add_edge("initialize_case", "retrieve_logic_memories")
builder.add_edge("initialize_case", "retrieve_visual_memories")

# Fan-in: both retrieve nodes → call_model
# LangGraph automatically waits for both nodes before executing call_model
builder.add_edge("retrieve_logic_memories", "call_model")
builder.add_edge("retrieve_visual_memories", "call_model")

# Continue to verification
builder.add_edge("call_model", "verify_answer")

# ========== Conditional Error Analysis ==========
# Conditional routing based on verification result
# If verified: go to save_output
# If not verified: fan-out to both generate nodes (parallel execution)
builder.add_conditional_edges(
    "verify_answer",
    should_analyze_error,
    {
        "save_output": "save_output",
        "generate_logic_memory": "generate_logic_memory",
        "generate_visual_memory": "generate_visual_memory",
    },
)

# ========== Parallel Memory Generation Phase ==========
# Fan-in: Both generate nodes converge to save_output
# LangGraph's built-in fan-in mechanism ensures save_output executes
# only once after both parallel branches complete
builder.add_edge("generate_logic_memory", "save_output")
builder.add_edge("generate_visual_memory", "save_output")

# End
builder.add_edge("save_output", "__end__")

# Compile the main graph - this is what langgraph dev shows
graph = builder.compile(name="ViLoMem")
