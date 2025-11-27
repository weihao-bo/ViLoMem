"""Qwen VL Agent Baseline (vl-agent-baseline).

A simplified LangGraph agent for solving math problems using vision-language models
with step-by-step reasoning and answer verification. No memory functionality.

Key components:
- Context: Configuration for model and verification settings (no memory)
- State: Tracks messages, predictions, and verification results (reused from full agent)
- graph: The main baseline workflow (vl-agent-baseline) - 3 nodes

Baseline Workflow:
1. call_model - Model inference
2. verify_answer - Verify answer using math-verify
3. save_output - Save results to JSON

This is designed for baseline comparisons without learning from errors.

Example usage:
    >>> from vl_agent_baseline import graph, Context
    >>> from langchain_core.messages import HumanMessage
    >>>
    >>> # Create context (no memory)
    >>> context = Context(model="qwen:qwen3-vl-8b-instruct")
    >>>
    >>> # Run evaluation - simple inference without memory
    >>> result = await graph.ainvoke(
    ...     {
    ...         "messages": [HumanMessage(content="What is 2+2?")],
    ...         "question": "What is 2+2?",
    ...         "gold_answer": "4",
    ...         "example_id": "test-001",
    ...         "output_dir": "output/test",
    ...     },
    ...     context=context
    ... )
"""

from vl_agent.state import EvaluationState, InputState, State
from vl_agent_baseline.context import Context
from vl_agent_baseline.graph import builder, graph

__all__ = [
    "Context",
    "State",
    "InputState",
    "EvaluationState",
    "graph",
    "builder",
]


