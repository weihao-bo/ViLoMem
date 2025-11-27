"""Baseline graph for Qwen VL agent (no memory functionality).

This module defines a simplified evaluation workflow without memory:

Baseline Workflow (3 nodes):
1. call_model - Model inference with image+text inputs
2. verify_answer - Verify answer using math-verify
3. save_output - Save results to JSON (simplified, no memory tracking)

This is a minimal version for baseline comparisons without learning from errors.
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from vl_agent_baseline.context import Context
from vl_agent_baseline.nodes import call_model, save_output, verify_answer
from vl_agent_baseline.state import EvaluationState

# Build baseline evaluation graph (no memory)
builder = StateGraph(EvaluationState, context_schema=Context)

# Add nodes (reusing from full agent)
builder.add_node("call_model", call_model)
builder.add_node("verify_answer", verify_answer)
builder.add_node("save_output", save_output)

# Define simple sequential flow
builder.add_edge("__start__", "call_model")
builder.add_edge("call_model", "verify_answer")
builder.add_edge("verify_answer", "save_output")
builder.add_edge("save_output", "__end__")

# Compile the baseline graph
graph = builder.compile(name="vl-agent-baseline")


