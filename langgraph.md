# LangGraph — Practical Guide

> Build stateful, multi-step AI workflows with cycles, conditional routing, and human-in-the-loop.

---

## Table of Contents

1. [What is LangGraph](#1-what-is-langgraph)
2. [Core Concepts](#2-core-concepts)
3. [Your First Graph](#3-your-first-graph)
4. [State Management](#4-state-management)
5. [Nodes](#5-nodes)
6. [Edges & Conditional Routing](#6-edges--conditional-routing)
7. [Cycles & Retries](#7-cycles--retries)
8. [Tool-Calling Agents](#8-tool-calling-agents)
9. [Human-in-the-Loop](#9-human-in-the-loop)
10. [Subgraphs & Multi-Agent](#10-subgraphs--multi-agent)
11. [Persistence & Checkpointing](#11-persistence--checkpointing)
12. [Streaming](#12-streaming)
13. [Error Handling](#13-error-handling)
14. [Production Patterns](#14-production-patterns)
15. [When to Use LangGraph](#15-when-to-use-langgraph)
16. [Common Pitfalls](#16-common-pitfalls)

---

## 1. What is LangGraph

LangGraph is a framework for building **stateful, graph-based workflows** for AI agents. Unlike simple chains (LangChain LCEL), LangGraph supports:

- **Cycles** — nodes can loop back (retry, iterate, refine)
- **Conditional edges** — route to different nodes based on state
- **Persistent state** — state is maintained and checkpointed across steps
- **Human-in-the-loop** — pause execution for human approval

**Install:**
```bash
pip install langgraph langchain-openai
```

**Mental model:** Think of it as a state machine where each node is a function that reads/writes state, and edges define the flow between nodes.

---

## 2. Core Concepts

```
                    ┌──────────────┐
    Input ────────► │  START node  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Node A     │ ◄── reads/writes State
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     conditional
                    │  Router      │────────────────┐
                    └──────┬───────┘                │
                           │                        │
                    ┌──────▼───────┐         ┌──────▼───────┐
                    │   Node B     │         │   Node C     │
                    └──────┬───────┘         └──────┬───────┘
                           │                        │
                           └───────────┬────────────┘
                                       │
                                ┌──────▼───────┐
                                │   END node   │
                                └──────────────┘
```

| Concept | What It Does |
|---|---|
| **State** | TypedDict or Pydantic model shared across all nodes |
| **Node** | A Python function that takes state, does work, returns state updates |
| **Edge** | Connection between nodes (normal or conditional) |
| **Conditional Edge** | Routes to different nodes based on state values |
| **START / END** | Special nodes marking entry and exit points |
| **Checkpointer** | Saves state after each node (enables persistence, time-travel) |
| **Interrupt** | Pauses execution for human review before continuing |

---

## 3. Your First Graph

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Step 1: Define state
class State(TypedDict):
    input: str
    output: str

# Step 2: Define nodes (functions)
def process(state: State) -> dict:
    return {"output": state["input"].upper()}

# Step 3: Build graph
graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

# Step 4: Compile and run
app = graph.compile()
result = app.invoke({"input": "hello world"})
print(result["output"])  # "HELLO WORLD"

# Visualize the graph (optional)
app.get_graph().print_ascii()
```

---

## 4. State Management

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

# ═══════════════════════════════
# Basic state with TypedDict
# ═══════════════════════════════
class AgentState(TypedDict):
    input: str
    messages: Annotated[list[BaseMessage], add_messages]  # append-only
    step_count: int
    result: str | None

# The `add_messages` annotation means:
# - When a node returns {"messages": [new_msg]}, it APPENDS to existing messages
# - Without annotation, it would OVERWRITE the entire list

# ═══════════════════════════════
# Custom reducers
# ═══════════════════════════════
import operator

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # append
    total: Annotated[int, operator.add]                    # sum
    items: Annotated[list[str], operator.add]               # concatenate lists
    latest: str                                             # overwrite (no annotation)

# When a node returns:
# {"total": 5, "items": ["new"], "latest": "updated"}
# Result: total += 5, items += ["new"], latest = "updated"

# ═══════════════════════════════
# Pydantic state (with validation)
# ═══════════════════════════════
from pydantic import BaseModel, Field

class PydanticState(BaseModel):
    query: str
    context: list[str] = Field(default_factory=list)
    answer: str | None = None
    confidence: float = 0.0

# Use with StateGraph:
graph = StateGraph(PydanticState)

# ═══════════════════════════════
# Accessing state in nodes
# ═══════════════════════════════
def my_node(state: AgentState) -> dict:
    # Read from state
    current_messages = state["messages"]
    step = state["step_count"]
    
    # Return ONLY the fields you want to update
    # Other fields remain unchanged
    return {
        "messages": [AIMessage(content="Done")],
        "step_count": step + 1,
    }
```

---

## 5. Nodes

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ═══════════════════════════════
# LLM node
# ═══════════════════════════════
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def call_llm(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# ═══════════════════════════════
# Tool execution node
# ═══════════════════════════════
from langgraph.prebuilt import ToolNode

tools = [search_tool, calculate_tool]
tool_node = ToolNode(tools)  # automatically executes tool calls from LLM

# ═══════════════════════════════
# Processing node
# ═══════════════════════════════
def analyze(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    # Do some processing
    analysis = perform_analysis(last_message.content)
    return {
        "messages": [AIMessage(content=analysis)],
        "step_count": state["step_count"] + 1,
    }

# ═══════════════════════════════
# Validation node
# ═══════════════════════════════
def validate(state: AgentState) -> dict:
    result = state["result"]
    try:
        validated = MySchema.model_validate_json(result)
        return {"validation_passed": True}
    except Exception as e:
        return {
            "validation_passed": False,
            "error": str(e),
        }

# ═══════════════════════════════
# Adding nodes to graph
# ═══════════════════════════════
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tools", tool_node)
graph.add_node("analyze", analyze)
graph.add_node("validate", validate)
```

---

## 6. Edges & Conditional Routing

```python
from langgraph.graph import END

# ═══════════════════════════════
# Normal edges (always follow this path)
# ═══════════════════════════════
graph.add_edge(START, "llm")
graph.add_edge("tools", "llm")          # after tools, go back to LLM

# ═══════════════════════════════
# Conditional edges (route based on state)
# ═══════════════════════════════
def should_use_tools(state: AgentState) -> str:
    """Route based on whether LLM wants to call tools."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"      # LLM wants to use tools
    return "end"            # LLM is done, go to END

graph.add_conditional_edges(
    "llm",                              # from this node
    should_use_tools,                   # routing function
    {
        "tools": "tools",              # if returns "tools" → go to tools node
        "end": END,                    # if returns "end" → finish
    },
)

# ═══════════════════════════════
# More complex routing
# ═══════════════════════════════
def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "search":
        return "search_node"
    elif intent == "generate":
        return "generate_node"
    elif intent == "error":
        return "error_handler"
    return "fallback"

graph.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "search_node": "search",
        "generate_node": "generate",
        "error_handler": "handle_error",
        "fallback": "default_response",
    },
)

# ═══════════════════════════════
# Multiple entry points
# ═══════════════════════════════
def route_entry(state: AgentState) -> str:
    if state.get("requires_context"):
        return "retrieve"
    return "generate"

graph.add_conditional_edges(START, route_entry, {
    "retrieve": "retrieval_node",
    "generate": "generation_node",
})
```

---

## 7. Cycles & Retries

This is LangGraph's superpower — nodes can loop back, enabling retry logic, iterative refinement, and self-correction.

```python
# ═══════════════════════════════
# Retry pattern with bounded attempts
# ═══════════════════════════════
class RetryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    attempt: int
    max_attempts: int
    result: str | None
    error: str | None

def generate(state: RetryState) -> dict:
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "result": response.content,
        "attempt": state["attempt"] + 1,
    }

def validate(state: RetryState) -> dict:
    try:
        parsed = MySchema.model_validate_json(state["result"])
        return {"error": None}
    except Exception as e:
        error_msg = f"Validation failed: {e}. Please fix and try again."
        return {
            "error": str(e),
            "messages": [HumanMessage(content=error_msg)],
        }

def should_retry(state: RetryState) -> str:
    if state["error"] is None:
        return "success"
    if state["attempt"] >= state["max_attempts"]:
        return "give_up"
    return "retry"

graph = StateGraph(RetryState)
graph.add_node("generate", generate)
graph.add_node("validate", validate)

graph.add_edge(START, "generate")
graph.add_edge("generate", "validate")
graph.add_conditional_edges("validate", should_retry, {
    "retry": "generate",        # loop back!
    "success": END,
    "give_up": END,
})

app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="Generate a valid JSON...")],
    "attempt": 0,
    "max_attempts": 3,
    "result": None,
    "error": None,
})

# ═══════════════════════════════
# Iterative refinement pattern
# ═══════════════════════════════
def refine(state):
    feedback = state.get("feedback", "")
    prompt = f"Previous output:\n{state['draft']}\n\nFeedback:\n{feedback}\n\nImprove it:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"draft": response.content, "iteration": state["iteration"] + 1}

def evaluate(state):
    score = evaluate_quality(state["draft"])
    return {"score": score}

def should_continue(state) -> str:
    if state["score"] >= 0.9 or state["iteration"] >= 5:
        return "done"
    return "refine"

# Wire: generate → evaluate → (refine → evaluate → ...) or done
```

---

## 8. Tool-Calling Agents

```python
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# ═══════════════════════════════
# Quick way: create_react_agent
# ═══════════════════════════════
@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[search, calculator],
)

result = agent.invoke({"messages": [("human", "What is 25 * 47?")]})

# ═══════════════════════════════
# Custom way: build the agent graph yourself
# ═══════════════════════════════
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

tools = [search, calculator]
llm_with_tools = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END,
})
graph.add_edge("tools", "agent")   # after tools, go back to agent

app = graph.compile()

# This creates the classic ReAct loop:
# agent → (tool calls?) → tools → agent → (tool calls?) → ... → end
```

---

## 9. Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

# ═══════════════════════════════
# Interrupt before a node
# ═══════════════════════════════
checkpointer = MemorySaver()

app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],    # pause before executing tools
)

# Run until interrupt
config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke(
    {"messages": [HumanMessage(content="Search for LangGraph docs")]},
    config=config,
)

# At this point, execution is paused before "tools" node
# Inspect what the agent wants to do:
state = app.get_state(config)
print(state.values["messages"][-1].tool_calls)
# [{"name": "search", "args": {"query": "LangGraph docs"}}]

# Option 1: Approve — continue execution
result = app.invoke(None, config=config)  # resumes from checkpoint

# Option 2: Reject — modify state and continue
app.update_state(config, {
    "messages": [AIMessage(content="I'll answer without searching.")]
})
result = app.invoke(None, config=config)

# ═══════════════════════════════
# Interrupt after a node (review output)
# ═══════════════════════════════
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_after=["generate"],    # pause after generation
)

# ═══════════════════════════════
# Dynamic approval based on state
# ═══════════════════════════════
def needs_approval(state) -> str:
    if state.get("confidence", 1.0) < 0.7:
        return "human_review"
    return "auto_approve"

graph.add_conditional_edges("generate", needs_approval, {
    "human_review": "wait_for_human",
    "auto_approve": "execute",
})
```

---

## 10. Subgraphs & Multi-Agent

```python
# ═══════════════════════════════
# Subgraph: a graph inside a graph
# ═══════════════════════════════

# Define inner graph
class InnerState(TypedDict):
    query: str
    results: list[str]

inner_graph = StateGraph(InnerState)
inner_graph.add_node("search", search_node)
inner_graph.add_node("rank", rank_node)
inner_graph.add_edge(START, "search")
inner_graph.add_edge("search", "rank")
inner_graph.add_edge("rank", END)
inner_app = inner_graph.compile()

# Use as a node in outer graph
class OuterState(TypedDict):
    input: str
    search_results: list[str]
    final_answer: str

def run_search_subgraph(state: OuterState) -> dict:
    result = inner_app.invoke({"query": state["input"]})
    return {"search_results": result["results"]}

outer_graph = StateGraph(OuterState)
outer_graph.add_node("search", run_search_subgraph)
outer_graph.add_node("answer", generate_answer)
outer_graph.add_edge(START, "search")
outer_graph.add_edge("search", "answer")
outer_graph.add_edge("answer", END)

# ═══════════════════════════════
# Multi-agent: supervisor pattern
# ═══════════════════════════════
class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str

def supervisor(state: MultiAgentState) -> dict:
    """Decide which agent should act next."""
    response = llm.with_structured_output(RouterSchema).invoke(
        state["messages"] + [SystemMessage(content="Route to: researcher, coder, or done")]
    )
    return {"next_agent": response.next}

def researcher(state: MultiAgentState) -> dict:
    response = research_llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=f"[Researcher] {response.content}")]}

def coder(state: MultiAgentState) -> dict:
    response = coding_llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=f"[Coder] {response.content}")]}

def route_agent(state: MultiAgentState) -> str:
    return state["next_agent"]

graph = StateGraph(MultiAgentState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_agent, {
    "researcher": "researcher",
    "coder": "coder",
    "done": END,
})
graph.add_edge("researcher", "supervisor")  # report back
graph.add_edge("coder", "supervisor")       # report back
```

---

## 11. Persistence & Checkpointing

```python
# ═══════════════════════════════
# In-memory (development)
# ═══════════════════════════════
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}

# First conversation turn
result = app.invoke({"messages": [HumanMessage(content="Hi")]}, config=config)

# Second turn — state is preserved!
result = app.invoke({"messages": [HumanMessage(content="What did I say?")]}, config=config)

# ═══════════════════════════════
# PostgreSQL (production)
# ═══════════════════════════════
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@host/db")
app = graph.compile(checkpointer=checkpointer)

# ═══════════════════════════════
# SQLite (lightweight production)
# ═══════════════════════════════
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# ═══════════════════════════════
# State inspection & time travel
# ═══════════════════════════════
config = {"configurable": {"thread_id": "session-1"}}

# Get current state
state = app.get_state(config)
print(state.values)          # current state dict
print(state.next)            # next node(s) to execute

# Get state history
for snapshot in app.get_state_history(config):
    print(snapshot.config)   # checkpoint config
    print(snapshot.values)   # state at that point
    print(snapshot.next)     # what would run next

# Time travel: resume from a previous checkpoint
old_config = snapshot.config  # pick a snapshot
result = app.invoke(None, config=old_config)
```

---

## 12. Streaming

```python
config = {"configurable": {"thread_id": "session-1"}}

# ═══════════════════════════════
# Stream node outputs
# ═══════════════════════════════
for event in app.stream({"messages": [HumanMessage(content="Hello")]}, config=config):
    for node_name, output in event.items():
        print(f"Node: {node_name}")
        print(f"Output: {output}")

# ═══════════════════════════════
# Stream LLM tokens (real-time)
# ═══════════════════════════════
for event in app.stream(
    {"messages": [HumanMessage(content="Write a poem")]},
    config=config,
    stream_mode="messages",     # stream individual message chunks
):
    msg, metadata = event
    if msg.content:
        print(msg.content, end="", flush=True)

# ═══════════════════════════════
# Stream events (most detailed)
# ═══════════════════════════════
async for event in app.astream_events(
    {"messages": [HumanMessage(content="Hello")]},
    config=config,
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
    elif kind == "on_tool_start":
        print(f"\nCalling tool: {event['name']}")
```

---

## 13. Error Handling

```python
# ═══════════════════════════════
# Try/except in nodes
# ═══════════════════════════════
def safe_tool_call(state: AgentState) -> dict:
    try:
        result = external_api.call(state["query"])
        return {"result": result, "error": None}
    except TimeoutError:
        return {"error": "API timeout", "result": None}
    except Exception as e:
        return {"error": str(e), "result": None}

# ═══════════════════════════════
# Route on errors
# ═══════════════════════════════
def handle_error(state: AgentState) -> str:
    if state.get("error"):
        if state["attempt"] < state["max_attempts"]:
            return "retry"
        return "fallback"
    return "continue"

graph.add_conditional_edges("process", handle_error, {
    "retry": "process",       # try again
    "fallback": "fallback",   # graceful degradation
    "continue": "next_step",
})

# ═══════════════════════════════
# Timeout for long-running nodes
# ═══════════════════════════════
import asyncio

async def with_timeout(state):
    try:
        result = await asyncio.wait_for(
            long_running_task(state),
            timeout=30.0,
        )
        return {"result": result}
    except asyncio.TimeoutError:
        return {"error": "Task timed out after 30s"}

# ═══════════════════════════════
# Graph-level error handling
# ═══════════════════════════════
try:
    result = app.invoke(input_data, config=config)
except GraphRecursionError:
    print("Graph exceeded maximum recursion — possible infinite loop")
```

---

## 14. Production Patterns

### Pattern 1: Plan → Execute → Validate

```python
# Common in code generation, document writing, data processing
graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", "validate")
graph.add_conditional_edges("validate", check_quality, {
    "pass": END,
    "fail": "plan",   # re-plan and try again
})
```

### Pattern 2: RAG with Fallback

```python
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade_documents")
graph.add_conditional_edges("grade_documents", has_good_docs, {
    "yes": "generate",
    "no": "web_search",    # fallback to web search
})
graph.add_edge("web_search", "generate")
graph.add_edge("generate", "check_hallucination")
graph.add_conditional_edges("check_hallucination", is_grounded, {
    "yes": END,
    "no": "generate",     # regenerate
})
```

### Pattern 3: Map-Reduce with Fan-out

```python
from langgraph.constants import Send

def fan_out(state):
    """Send each document to a separate processing node."""
    return [Send("process_doc", {"doc": doc}) for doc in state["documents"]]

graph.add_conditional_edges("split", fan_out)  # dynamic fan-out
graph.add_edge("process_doc", "merge")         # fan-in to merge
```

### Pattern 4: Configurable Graph

```python
from langgraph.graph import StateGraph

class Config(TypedDict):
    model: str
    temperature: float
    max_retries: int

def call_llm(state, config):
    """Node that reads from runtime config."""
    model = config["configurable"].get("model", "gpt-4o")
    llm = ChatOpenAI(model=model, temperature=config["configurable"]["temperature"])
    return {"messages": [llm.invoke(state["messages"])]}

# Pass config at runtime
result = app.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"model": "gpt-4o-mini", "temperature": 0.5, "max_retries": 3}},
)
```

---

## 15. When to Use LangGraph

| Scenario | Use LangGraph? |
|---|---|
| Simple prompt → response | ❌ Use LangChain LCEL |
| Linear chain of 2-3 steps | ❌ Use LangChain LCEL |
| Agent that calls tools in a loop | ✅ |
| Workflow with retry/self-correction | ✅ |
| Multi-step process with branching | ✅ |
| Human approval before actions | ✅ |
| Multi-agent collaboration | ✅ |
| Long-running workflow with checkpoints | ✅ |
| Stateful conversation across sessions | ✅ |
| Workflow you need to visualize/debug | ✅ |

**Rule of thumb:** If your workflow has **cycles, conditional branching, or needs to persist state**, use LangGraph.

---

## 16. Common Pitfalls

| Pitfall | Fix |
|---|---|
| **Infinite loops** | Always add a max iteration counter in state; check it in conditional edges |
| **State mutation instead of returning updates** | Nodes must return dicts, not modify state in-place |
| **Forgetting `add_messages` annotation** | Without it, returning messages overwrites instead of appending |
| **Not using a checkpointer** | Required for human-in-the-loop and persistence; even MemorySaver for dev |
| **Overly complex single graph** | Break into subgraphs; each subgraph handles one responsibility |
| **Not handling tool errors** | Wrap tool execution in try/except; return error as message for LLM to handle |
| **Conditional edge returning unlisted value** | All possible return values must be in the routing map |
| **Running sync in async context** | Use `ainvoke`, `astream` in async code; don't block the event loop |
| **Giant state objects** | Keep state minimal; store large data (docs, files) in external storage, reference by ID |

---

*For the basics of LLM chains and tools, see [`langchain.md`](./langchain.md). For retrieval pipelines, see [`rag.md`](./rag.md).*