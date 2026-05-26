# AI Agent Optimization — Architectures & Techniques

> Practical patterns for building faster, cheaper, more reliable AI agents with LangChain, LangGraph, and RAG.

---

## Table of Contents

1. [Why Optimize](#1-why-optimize)
2. [Architecture Patterns](#2-architecture-patterns)
3. [Prompt Optimization](#3-prompt-optimization)
4. [RAG Optimization](#4-rag-optimization)
5. [LangGraph Workflow Optimization](#5-langgraph-workflow-optimization)
6. [Cost Optimization](#6-cost-optimization)
7. [Latency Optimization](#7-latency-optimization)
8. [Reliability & Error Handling](#8-reliability--error-handling)
9. [Evaluation-Driven Development](#9-evaluation-driven-development)
10. [Production Architecture](#10-production-architecture)

---

## 1. Why Optimize

Naive AI agents are **slow, expensive, and unreliable**. A basic ReAct agent might:
- Make 10+ LLM calls per task ($$)
- Take 30+ seconds to respond
- Hallucinate or loop infinitely
- Fail silently on edge cases

Optimization targets three axes:

| Axis | Metric | Goal |
|---|---|---|
| **Cost** | $/task, tokens/task | Minimize LLM calls and token usage |
| **Latency** | Seconds to completion | Minimize time to first token and total time |
| **Reliability** | Success rate, error rate | Maximize correct completions |

---

## 2. Architecture Patterns

### 2.1 Router Architecture (Classify → Specialize)

Instead of one general-purpose agent, classify the task first and route to a specialized handler.

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class TaskClassification(BaseModel):
    category: str  # "simple_qa", "code_gen", "research", "data_analysis"
    complexity: str  # "low", "medium", "high"

def classify(state):
    result = llm.with_structured_output(TaskClassification).invoke(
        f"Classify this task: {state['input']}"
    )
    return {"category": result.category, "complexity": result.complexity}

def route(state) -> str:
    cat = state["category"]
    if cat == "simple_qa":
        return "simple_chain"       # cheap model, no tools
    elif cat == "code_gen":
        return "code_agent"         # specialized code agent
    elif cat == "research":
        return "research_agent"     # RAG + web search
    return "general_agent"

graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("simple_chain", simple_chain)     # GPT-4o-mini, no tools
graph.add_node("code_agent", code_agent)         # GPT-4o + code tools
graph.add_node("research_agent", research_agent) # GPT-4o + RAG + search
graph.add_node("general_agent", general_agent)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route, {
    "simple_chain": "simple_chain",
    "code_agent": "code_agent",
    "research_agent": "research_agent",
    "general_agent": "general_agent",
})
for node in ["simple_chain", "code_agent", "research_agent", "general_agent"]:
    graph.add_edge(node, END)
```

**Why:** 70%+ of queries might be simple and handleable by a cheap model. Only route complex tasks to expensive agents.

---

### 2.2 Plan-then-Execute Architecture

Separate planning from execution. The planner creates a step-by-step plan; the executor follows it.

```python
from pydantic import BaseModel

class Plan(BaseModel):
    steps: list[str]
    tools_needed: list[str]

class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    current_step: int
    results: list[str]
    final_answer: str

def planner(state):
    plan = llm.with_structured_output(Plan).invoke(
        f"Create a step-by-step plan to: {state['input']}\n"
        f"Available tools: {[t.name for t in tools]}"
    )
    return {"plan": plan.steps, "current_step": 0, "results": []}

def executor(state):
    step = state["plan"][state["current_step"]]
    # Execute with tools if needed
    result = agent_executor.invoke({
        "messages": [("human", f"Execute this step: {step}\n\nPrevious results: {state['results']}")]
    })
    return {
        "results": [result["messages"][-1].content],
        "current_step": state["current_step"] + 1,
    }

def should_continue(state) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

def synthesize(state):
    answer = llm.invoke(
        f"Question: {state['input']}\nResults from each step:\n" +
        "\n".join(f"Step {i+1}: {r}" for i, r in enumerate(state["results"]))
    )
    return {"final_answer": answer.content}

graph = StateGraph(PlanExecuteState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("synthesize", synthesize)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue, {
    "execute": "executor",
    "synthesize": "synthesize",
})
graph.add_edge("synthesize", END)
```

**Why:** Prevents the agent from going off-track. The plan constrains execution, reducing unnecessary tool calls and hallucination.

---

### 2.3 Reflection Architecture

Agent generates output, then a critic reviews it, and the agent refines based on feedback.

```python
class ReflectionState(TypedDict):
    input: str
    draft: str
    critique: str
    iteration: int
    final: bool

def generate(state):
    context = ""
    if state.get("critique"):
        context = f"\n\nPrevious attempt:\n{state['draft']}\n\nFeedback:\n{state['critique']}\n\nImprove based on feedback."
    
    response = llm.invoke(f"Task: {state['input']}{context}")
    return {"draft": response.content, "iteration": state["iteration"] + 1}

def reflect(state):
    critique = llm.invoke(
        f"Critically review this output. List specific issues and improvements:\n\n"
        f"Task: {state['input']}\nOutput:\n{state['draft']}"
    )
    # Check if good enough
    quality_check = llm.with_structured_output(QualityScore).invoke(
        f"Rate quality 1-10:\n{state['draft']}"
    )
    return {
        "critique": critique.content,
        "final": quality_check.score >= 8 or state["iteration"] >= 3,
    }

def should_continue(state) -> str:
    return "end" if state["final"] else "generate"

graph = StateGraph(ReflectionState)
graph.add_node("generate", generate)
graph.add_node("reflect", reflect)
graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {
    "generate": "generate",
    "end": END,
})
```

**Why:** First drafts are often 60-70% quality. One reflection pass typically pushes to 85-90%.

---

### 2.4 Supervisor Multi-Agent Architecture

A supervisor agent delegates to specialized sub-agents and synthesizes results.

```python
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    next: str
    task_complete: bool

def supervisor(state):
    decision = supervisor_llm.with_structured_output(RouterDecision).invoke(
        state["messages"] + [SystemMessage(content="""
        You are a supervisor. Delegate to:
        - 'researcher': for information gathering
        - 'coder': for writing/reviewing code
        - 'analyst': for data analysis
        - 'done': when the task is complete
        """)]
    )
    return {"next": decision.agent, "task_complete": decision.agent == "done"}

# Each sub-agent is its own compiled graph or chain
researcher = create_react_agent(model=llm, tools=[search, wiki])
coder = create_react_agent(model=llm, tools=[code_exec, file_read])
analyst = create_react_agent(model=llm, tools=[sql_query, chart])
```

**Why:** Specialized agents perform better than generalists. The supervisor prevents unnecessary delegation.

---

## 3. Prompt Optimization

### 3.1 Structured Output over Free Text

```python
# ❌ Bad: Parse free text
response = llm.invoke("Extract the name and age from: John is 30")
# "The name is John and the age is 30"  → now parse this somehow?

# ✅ Good: Structured output
class Person(BaseModel):
    name: str
    age: int

result = llm.with_structured_output(Person).invoke("Extract: John is 30")
# Person(name="John", age=30)  → directly usable
```

### 3.2 Few-Shot Examples for Consistency

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify support tickets into categories."),
    ("human", "My payment was declined"),
    ("ai", '{"category": "billing", "priority": "high"}'),
    ("human", "How do I change my password?"),
    ("ai", '{"category": "account", "priority": "low"}'),
    ("human", "{input}"),
])
```

### 3.3 System Prompt Engineering

```python
# ❌ Vague
system = "You are a helpful assistant."

# ✅ Specific and constrained
system = """You are a Python code reviewer. Your job:
1. Identify bugs and security issues
2. Suggest performance improvements
3. Check adherence to PEP 8

Rules:
- Only comment on actual issues, not style preferences
- Rate severity: critical, warning, info
- If the code is good, say so briefly
- Never suggest rewriting the entire function

Output format: JSON array of {line, severity, issue, suggestion}"""
```

### 3.4 Context Window Management

```python
from langchain_core.messages import trim_messages

# Trim to fit context window
trimmed = trim_messages(
    messages,
    max_tokens=8000,
    token_counter=llm,
    strategy="last",        # keep recent messages
    include_system=True,    # always keep system prompt
    start_on="human",       # don't start on an orphaned AI message
)

# For very long conversations: summarize older messages
def compress_history(messages, llm, keep_last=4):
    if len(messages) <= keep_last + 1:  # +1 for system
        return messages
    
    old = messages[1:-keep_last]  # skip system, skip recent
    summary = llm.invoke(f"Summarize this conversation:\n{format_messages(old)}")
    
    return (
        [messages[0]]  # system
        + [SystemMessage(content=f"Conversation summary: {summary.content}")]
        + messages[-keep_last:]
    )
```

---

## 4. RAG Optimization

### 4.1 Hybrid Search (Vector + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Keyword search catches exact terms; vector search catches semantic meaning
bm25 = BM25Retriever.from_documents(chunks, k=10)
vector = vectorstore.as_retriever(search_kwargs={"k": 10})

hybrid = EnsembleRetriever(
    retrievers=[bm25, vector],
    weights=[0.3, 0.7],  # tune based on your data
)
```

### 4.2 Two-Stage Retrieval (Retrieve → Re-rank)

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Stage 1: Retrieve many candidates (fast, cheap)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Stage 2: Re-rank with cross-encoder (slower, accurate)
reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever,
)
# Result: 5 highly relevant docs instead of 20 mediocre ones
```

### 4.3 Contextual Chunking

```python
# Add surrounding context to each chunk before indexing
def add_context_to_chunks(chunks, window=1):
    enriched = []
    for i, chunk in enumerate(chunks):
        context_parts = []
        if i > 0:
            context_parts.append(f"Previous section: {chunks[i-1].page_content[:100]}...")
        context_parts.append(chunk.page_content)
        if i < len(chunks) - 1:
            context_parts.append(f"Next section: {chunks[i+1].page_content[:100]}...")
        
        enriched_chunk = chunk.copy()
        enriched_chunk.page_content = "\n".join(context_parts)
        enriched.append(enriched_chunk)
    return enriched
```

### 4.4 Query Transformation

```python
# Rephrase query for better retrieval
def transform_query(query: str, llm) -> list[str]:
    result = llm.invoke(
        f"Generate 3 different search queries to find information about: {query}\n"
        "Return one query per line."
    )
    queries = [q.strip() for q in result.content.strip().split("\n") if q.strip()]
    return [query] + queries  # include original

# Retrieve for each query, deduplicate
all_docs = []
seen = set()
for q in transform_query(user_query, llm):
    for doc in retriever.invoke(q):
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            all_docs.append(doc)
```

### 4.5 Adaptive Retrieval

```python
# Don't retrieve if the question doesn't need context
def needs_retrieval(query: str, llm) -> bool:
    result = llm.with_structured_output(NeedsContext).invoke(
        f"Does this question require looking up specific information, "
        f"or can it be answered from general knowledge?\nQuestion: {query}"
    )
    return result.needs_context

# In your graph:
def route_retrieval(state) -> str:
    if needs_retrieval(state["input"], llm):
        return "retrieve"
    return "generate_directly"
```

---

## 5. LangGraph Workflow Optimization

### 5.1 Bounded Retries with Exponential Backoff

```python
import time

class RetryState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    attempt: int
    max_attempts: int
    last_error: str | None

def execute_with_retry(state):
    try:
        result = risky_operation(state)
        return {"result": result, "last_error": None}
    except Exception as e:
        wait = min(2 ** state["attempt"], 30)  # exponential backoff, max 30s
        time.sleep(wait)
        return {
            "last_error": str(e),
            "attempt": state["attempt"] + 1,
            "messages": [HumanMessage(
                content=f"Attempt {state['attempt']+1} failed: {e}. "
                f"Adjust your approach and try again."
            )],
        }

def should_retry(state) -> str:
    if state["last_error"] is None:
        return "success"
    if state["attempt"] >= state["max_attempts"]:
        return "fallback"
    return "retry"
```

### 5.2 Early Exit Conditions

```python
# Check at every node whether we can exit early
def check_early_exit(state) -> str:
    # If we already have a high-confidence answer, skip remaining steps
    if state.get("confidence", 0) > 0.95:
        return "end"
    # If token budget is exhausted
    if state.get("tokens_used", 0) > state.get("token_budget", 10000):
        return "end"
    return "continue"

# Add after every major node
for node in ["retrieve", "analyze", "generate"]:
    graph.add_conditional_edges(node, check_early_exit, {
        "end": END,
        "continue": next_node[node],
    })
```

### 5.3 Parallel Execution

```python
from langgraph.constants import Send

# Fan-out: process multiple items in parallel
def fan_out_research(state):
    topics = state["research_topics"]
    return [Send("research_single", {"topic": t}) for t in topics]

def research_single(state):
    docs = retriever.invoke(state["topic"])
    summary = llm.invoke(f"Summarize findings on: {state['topic']}\n\nDocs: {format_docs(docs)}")
    return {"findings": [summary.content]}

def merge_findings(state):
    combined = "\n\n".join(state["findings"])
    return {"research_summary": combined}

graph.add_conditional_edges("identify_topics", fan_out_research)
graph.add_edge("research_single", "merge_findings")
```

### 5.4 State Minimization

```python
# ❌ Bad: storing entire documents in state
class BadState(TypedDict):
    documents: list[Document]           # huge!
    all_embeddings: list[list[float]]   # enormous!

# ✅ Good: store references, fetch when needed
class GoodState(TypedDict):
    doc_ids: list[str]          # just IDs
    summary: str                # compressed representation
    current_step: int

# Fetch full data only when a node needs it
def process_node(state):
    docs = fetch_docs_by_ids(state["doc_ids"])  # fetch on demand
    # ... process ...
    return {"summary": summarize(docs)}
```

---

## 6. Cost Optimization

### 6.1 Model Tiering

```python
# Use cheap models for simple tasks, expensive for complex
cheap_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)   # ~$0.15/1M input
expensive_llm = ChatOpenAI(model="gpt-4o", temperature=0)     # ~$2.50/1M input

# Classification, routing, validation → cheap model
classifier = cheap_llm.with_structured_output(TaskCategory)

# Complex reasoning, code generation → expensive model
code_generator = expensive_llm

# Simple Q&A can often use the cheap model
simple_qa_chain = prompt | cheap_llm | StrOutputParser()
```

### 6.2 Caching

```python
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache

# Cache LLM responses (exact match)
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# Semantic cache (similar queries hit cache)
from langchain_community.cache import RedisSemanticCache
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.95,
))
```

### 6.3 Token Budget Tracking

```python
class TokenTracker:
    def __init__(self, budget: int = 50000):
        self.budget = budget
        self.used = 0
    
    def track(self, response):
        if hasattr(response, 'usage_metadata'):
            self.used += response.usage_metadata.get('total_tokens', 0)
    
    @property
    def remaining(self):
        return self.budget - self.used
    
    @property
    def exceeded(self):
        return self.used >= self.budget

# Use in nodes
tracker = TokenTracker(budget=50000)

def llm_node(state):
    response = llm.invoke(state["messages"])
    tracker.track(response)
    if tracker.exceeded:
        return {"messages": [response], "budget_exceeded": True}
    return {"messages": [response]}
```

---

## 7. Latency Optimization

### 7.1 Streaming

```python
# Stream tokens to user as they're generated
async for event in agent.astream_events(
    {"messages": [("human", query)]},
    config=config,
    version="v2",
):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        yield token  # send to frontend immediately
```

### 7.2 Parallel Tool Calls

```python
# LLMs can request multiple tool calls at once
# ToolNode handles them in parallel by default
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)  # executes parallel tool calls concurrently

# Or manually parallelize
import asyncio

async def parallel_retrieve(queries: list[str]):
    tasks = [retriever.ainvoke(q) for q in queries]
    results = await asyncio.gather(*tasks)
    return [doc for docs in results for doc in docs]
```

### 7.3 Prefetching

```python
# Start retrieval before the LLM finishes processing
async def prefetch_and_generate(query):
    # Start retrieval immediately
    retrieval_task = asyncio.create_task(retriever.ainvoke(query))
    
    # While retrieving, classify the query
    classification = await classifier.ainvoke(query)
    
    # By now, retrieval is likely done
    docs = await retrieval_task
    
    # Generate with both
    return await generate(query, docs, classification)
```

---

## 8. Reliability & Error Handling

### 8.1 Structured Validation at Every Step

```python
from pydantic import BaseModel, validator

class CodeOutput(BaseModel):
    language: str
    code: str
    explanation: str
    
    @validator("code")
    def code_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty")
        return v

# Validate in the graph node
def generate_code(state):
    for attempt in range(3):
        try:
            result = llm.with_structured_output(CodeOutput).invoke(state["messages"])
            return {"code_output": result}
        except Exception as e:
            state["messages"].append(
                HumanMessage(content=f"Output validation failed: {e}. Fix and retry.")
            )
    return {"error": "Failed to generate valid code after 3 attempts"}
```

### 8.2 Graceful Degradation

```python
def route_on_failure(state) -> str:
    error = state.get("error")
    if not error:
        return "success"
    
    error_type = classify_error(error)
    if error_type == "rate_limit":
        return "wait_and_retry"
    elif error_type == "context_too_long":
        return "summarize_and_retry"
    elif error_type == "tool_failure":
        return "skip_tool"
    return "return_partial"  # return what we have so far
```

### 8.3 Guardrails

```python
# Input guardrail: reject harmful/irrelevant inputs
def input_guard(state):
    check = llm.with_structured_output(SafetyCheck).invoke(
        f"Is this a valid, safe request for our system? '{state['input']}'"
    )
    if not check.is_safe:
        return {"error": check.reason, "blocked": True}
    return {"blocked": False}

# Output guardrail: validate before returning to user
def output_guard(state):
    check = llm.with_structured_output(OutputCheck).invoke(
        f"Does this response contain PII, hallucinations, or harmful content?\n{state['answer']}"
    )
    if not check.is_safe:
        return {"answer": "I cannot provide that information.", "flagged": True}
    return {"flagged": False}
```

---

## 9. Evaluation-Driven Development

### 9.1 Build an Eval Set First

```python
# Before optimizing, create test cases
eval_set = [
    {
        "input": "How do I deploy to ECS?",
        "expected_tools": ["search_docs"],
        "expected_contains": ["ECS", "task definition"],
        "max_steps": 3,
        "max_tokens": 5000,
    },
    {
        "input": "Create a Jira ticket for the login bug",
        "expected_tools": ["create_jira_ticket"],
        "expected_contains": ["PROJ-"],
        "max_steps": 2,
    },
]
```

### 9.2 Automated Regression Testing

```python
def run_eval(agent, eval_set):
    results = []
    for case in eval_set:
        start = time.time()
        output = agent.invoke({"messages": [("human", case["input"])]})
        elapsed = time.time() - start
        
        # Check criteria
        passed = all([
            elapsed < case.get("max_time", 30),
            any(kw in str(output) for kw in case.get("expected_contains", [])),
        ])
        
        results.append({
            "input": case["input"],
            "passed": passed,
            "time": elapsed,
            "steps": count_steps(output),
        })
    
    pass_rate = sum(r["passed"] for r in results) / len(results)
    avg_time = sum(r["time"] for r in results) / len(results)
    print(f"Pass rate: {pass_rate:.0%} | Avg time: {avg_time:.1f}s")
    return results
```

### 9.3 A/B Testing Architectures

```python
# Compare two agent architectures
def ab_test(eval_set, agent_a, agent_b):
    results_a = run_eval(agent_a, eval_set)
    results_b = run_eval(agent_b, eval_set)
    
    print(f"Agent A — Pass: {pass_rate(results_a):.0%}, Cost: ${total_cost(results_a):.2f}")
    print(f"Agent B — Pass: {pass_rate(results_b):.0%}, Cost: ${total_cost(results_b):.2f}")
```

---

## 10. Production Architecture

### Complete Optimized Agent Architecture

```
                            ┌────────────────┐
            User Query ────►│ Input Guard    │
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ Cache Check    │──── Hit ──── Return cached
                            └───────┬────────┘
                                    │ Miss
                            ┌───────▼────────┐
                            │ Classifier     │ (cheap model)
                            └───────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            ┌───────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
            │ Simple QA    │ │ RAG Agent  │ │ Multi-step  │
            │ (mini model) │ │ (retrieve  │ │ Agent       │
            │              │ │  + rerank  │ │ (plan +     │
            │              │ │  + generate│ │  execute)   │
            └───────┬──────┘ └─────┬──────┘ └──────┬──────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                            ┌───────▼────────┐
                            │ Output Guard   │
                            └───────┬────────┘
                                    │
                            ┌───────▼────────┐
                            │ Cache Store    │
                            └───────┬────────┘
                                    │
                                Response
```

### Key Optimization Checklist

```
□ Route simple queries to cheap models
□ Use structured output everywhere (no string parsing)
□ Implement hybrid search (vector + keyword)
□ Add re-ranking after retrieval
□ Bound all loops with max iterations
□ Cache LLM responses (exact or semantic)
□ Stream tokens for user-facing responses
□ Track token usage per task
□ Add input/output guardrails
□ Build eval set before optimizing
□ Log everything: latency, tokens, errors, tool calls
□ Set up alerts for error rate spikes
```

---

*For LangChain fundamentals, see [`langchain.md`](./langchain.md). For LangGraph workflows, see [`langgraph.md`](./langgraph.md). For RAG pipelines, see [`rag.md`](./rag.md).*