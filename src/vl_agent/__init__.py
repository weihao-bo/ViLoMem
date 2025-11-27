"""ViLoMem Agent.

A LangGraph agent for solving math problems using vision-language models with
step-by-step reasoning, answer verification, and memory-based learning.

Key components:
- Context: Configuration for model, verification, and memory settings
- State: Tracks messages, predictions, verification results, and memories
- nodes: Reusable node functions for graph composition
- graph: The main ViLoMem workflow - 7 nodes
- builder: The graph builder for custom compilation

Memory System:
- Memories are stored in JSON files (default: memories.json in output_dir)
- Retrieval uses DashScope Rerank API (qwen3-rerank model)
- No need for embeddings or InMemoryStore setup

Example usage:
    >>> from vl_agent import graph, Context
    >>> from langchain_core.messages import HumanMessage
    >>>
    >>> # Create context with memory enabled
    >>> context = Context(
    ...     model="qwen:qwen3-vl-8b-instruct",
    ...     enable_memory=True,
    ...     memory_file_path="memories.json"
    ... )
    >>>
    >>> # Run evaluation - memories are automatically stored/retrieved from JSON
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

from vl_agent import nodes
from vl_agent.context import Context
from vl_agent.graph import builder, graph
from vl_agent.state import EvaluationState, InputState, State

__all__ = [
    "Context",
    "State",
    "InputState",
    "EvaluationState",
    "nodes",
    "graph",
    "builder",
]
