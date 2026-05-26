# LangChain — Practical Guide

> Build LLM-powered applications with composable chains, tools, and agents.

---

## Table of Contents

1. [What is LangChain](#1-what-is-langchain)
2. [Core Concepts](#2-core-concepts)
3. [Chat Models](#3-chat-models)
4. [Prompt Templates](#4-prompt-templates)
5. [Output Parsers & Structured Output](#5-output-parsers--structured-output)
6. [Chains (LCEL)](#6-chains-lcel)
7. [Tools](#7-tools)
8. [Agents](#8-agents)
9. [Memory & Chat History](#9-memory--chat-history)
10. [Callbacks & Tracing](#10-callbacks--tracing)
11. [Document Loaders](#11-document-loaders)
12. [Text Splitters](#12-text-splitters)
13. [Common Patterns](#13-common-patterns)
14. [When to Use LangChain vs LangGraph](#14-when-to-use-langchain-vs-langgraph)
15. [Common Pitfalls](#15-common-pitfalls)

---

## 1. What is LangChain

LangChain is a framework for building applications powered by language models. It provides composable abstractions for:
- Connecting LLMs to external data sources and tools
- Chaining multiple LLM calls together
- Structuring LLM output into usable formats
- Managing conversation memory

**Install:**
```bash
pip install langchain langchain-openai langchain-anthropic langchain-community
```

---

## 2. Core Concepts

```
Prompt Template → Chat Model → Output Parser
     ↑                              ↓
   Input                      Structured Output
```

| Concept | What It Does |
|---|---|
| **Chat Model** | Wrapper around LLM APIs (OpenAI, Anthropic, etc.) |
| **Prompt Template** | Parameterized prompt with variables |
| **Output Parser** | Converts LLM text output into structured data |
| **Chain (LCEL)** | Composable pipeline of operations using `|` operator |
| **Tool** | Function the LLM can call (search, calculator, API, etc.) |
| **Agent** | LLM that decides which tools to use and in what order |
| **Memory** | Maintains conversation state across turns |
| **Retriever** | Fetches relevant documents for context (used in RAG) |

---

## 3. Chat Models

```python
# ═══════════════════════════════
# OpenAI
# ═══════════════════════════════
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,          # 0 = deterministic, 1 = creative
    max_tokens=4096,
    api_key="sk-..."        # or set OPENAI_API_KEY env var
)

# ═══════════════════════════════
# Anthropic (Claude)
# ═══════════════════════════════
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=4096,
)

# ═══════════════════════════════
# Basic usage
# ═══════════════════════════════
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?"),
]

response = llm.invoke(messages)
print(response.content)       # string response
print(response.usage_metadata) # token usage

# ═══════════════════════════════
# Streaming
# ═══════════════════════════════
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)

# ═══════════════════════════════
# Async
# ═══════════════════════════════
response = await llm.ainvoke(messages)

# ═══════════════════════════════
# Batch (parallel calls)
# ═══════════════════════════════
responses = llm.batch([
    [HumanMessage(content="Summarize X")],
    [HumanMessage(content="Summarize Y")],
])
```

---

## 4. Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ═══════════════════════════════
# Basic template
# ═══════════════════════════════
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} expert."),
    ("human", "{question}"),
])

# Format with variables
messages = prompt.invoke({"role": "Python", "question": "What are decorators?"})

# ═══════════════════════════════
# With chat history placeholder
# ═══════════════════════════════
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# ═══════════════════════════════
# From a single string
# ═══════════════════════════════
prompt = ChatPromptTemplate.from_template(
    "Translate the following to {language}: {text}"
)

# ═══════════════════════════════
# Few-shot prompting
# ═══════════════════════════════
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Give the opposite of each word."),
    few_shot_prompt,
    ("human", "{input}"),
])
```

---

## 5. Output Parsers & Structured Output

```python
# ═══════════════════════════════
# Method 1: with_structured_output (RECOMMENDED)
# Works with OpenAI, Anthropic, and other providers
# ═══════════════════════════════
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    summary: str = Field(description="One-line summary")
    genres: list[str] = Field(description="List of genres")

structured_llm = llm.with_structured_output(MovieReview)
result = structured_llm.invoke("Review the movie Inception")
# result is a MovieReview object
print(result.title)    # "Inception"
print(result.rating)   # 9.2
print(result.genres)   # ["Sci-Fi", "Thriller"]

# ═══════════════════════════════
# Method 2: PydanticOutputParser (manual)
# ═══════════════════════════════
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract movie review info.\n{format_instructions}"),
    ("human", "{review}"),
])

chain = prompt | llm | parser
result = chain.invoke({
    "review": "Inception is a masterpiece...",
    "format_instructions": parser.get_format_instructions(),
})

# ═══════════════════════════════
# Method 3: Simple string parser
# ═══════════════════════════════
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "Hello"})  # returns plain string

# ═══════════════════════════════
# Method 4: JSON output parser
# ═══════════════════════════════
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=MovieReview)
chain = prompt | llm | parser
result = chain.invoke({...})  # returns dict
```

---

## 6. Chains (LCEL)

LCEL (LangChain Expression Language) lets you compose components with the `|` pipe operator. Every component implements the **Runnable** interface (`invoke`, `stream`, `batch`, `ainvoke`).

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# ═══════════════════════════════
# Basic chain: prompt → LLM → parser
# ═══════════════════════════════
prompt = ChatPromptTemplate.from_template("Explain {topic} in simple terms.")
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "quantum computing"})
# Returns a string explanation

# ═══════════════════════════════
# Chain with structured output
# ═══════════════════════════════
structured_chain = prompt | llm.with_structured_output(MySchema)

# ═══════════════════════════════
# Sequential chains (chain of chains)
# ═══════════════════════════════
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Step 1: Generate outline
outline_chain = (
    ChatPromptTemplate.from_template("Create an outline for: {topic}")
    | llm
    | StrOutputParser()
)

# Step 2: Write article from outline
article_chain = (
    ChatPromptTemplate.from_template("Write an article from this outline:\n{outline}")
    | llm
    | StrOutputParser()
)

# Combined: topic → outline → article
full_chain = (
    {"outline": outline_chain, "topic": RunnablePassthrough()}
    | RunnableLambda(lambda x: {"outline": x["outline"]})
    | article_chain
)

# ═══════════════════════════════
# Parallel chains (RunnableParallel)
# ═══════════════════════════════
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser(),
    keywords=ChatPromptTemplate.from_template("Extract keywords from: {text}") | llm | StrOutputParser(),
    sentiment=ChatPromptTemplate.from_template("What's the sentiment of: {text}") | llm | StrOutputParser(),
)

result = parallel.invoke({"text": "LangChain is amazing..."})
# result = {"summary": "...", "keywords": "...", "sentiment": "..."}

# ═══════════════════════════════
# Conditional routing
# ═══════════════════════════════
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: "code" in x["topic"].lower(), code_chain),
    (lambda x: "math" in x["topic"].lower(), math_chain),
    default_chain,  # fallback
)

# ═══════════════════════════════
# Adding fallbacks
# ═══════════════════════════════
chain_with_fallback = primary_chain.with_fallbacks([fallback_chain])

# ═══════════════════════════════
# Retry logic
# ═══════════════════════════════
chain_with_retry = chain.with_retry(stop_after_attempt=3)

# ═══════════════════════════════
# Streaming a chain
# ═══════════════════════════════
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

---

## 7. Tools

Tools let the LLM call functions. The LLM decides *when* to call a tool and *what arguments* to pass.

```python
from langchain_core.tools import tool

# ═══════════════════════════════
# Define a tool with @tool decorator
# ═══════════════════════════════
@tool
def search_database(query: str) -> str:
    """Search the product database for items matching the query."""
    # Your actual implementation
    results = db.search(query)
    return str(results)

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Example: '2 + 3 * 4'"""
    return str(eval(expression))  # simplified; use safe eval in production

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city.
    
    Args:
        city: City name
        unit: Temperature unit — 'celsius' or 'fahrenheit'
    """
    return f"Weather in {city}: 22°{unit[0].upper()}, Sunny"

# ═══════════════════════════════
# Define a tool with Pydantic schema
# ═══════════════════════════════
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results to return")

def search_fn(query: str, max_results: int = 5) -> str:
    return f"Found {max_results} results for '{query}'"

search_tool = StructuredTool.from_function(
    func=search_fn,
    name="search",
    description="Search the knowledge base",
    args_schema=SearchInput,
)

# ═══════════════════════════════
# Bind tools to a model
# ═══════════════════════════════
tools = [search_database, calculate, get_weather]
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("What's the weather in London?")
# response.tool_calls → [{"name": "get_weather", "args": {"city": "London"}}]

# ═══════════════════════════════
# Execute tool calls
# ═══════════════════════════════
from langchain_core.messages import ToolMessage

for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    # Find and execute the tool
    tool_map = {t.name: t for t in tools}
    result = tool_map[tool_name].invoke(tool_args)
    
    # Create a ToolMessage with the result
    tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])

# ═══════════════════════════════
# Built-in tools
# ═══════════════════════════════
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun

search = DuckDuckGoSearchRun()
result = search.invoke("LangChain latest version")
```

---

## 8. Agents

An agent uses an LLM to decide which tools to call, executes them, observes the results, and decides next steps. This is the **ReAct** pattern (Reasoning + Acting).

```python
# ═══════════════════════════════
# Simple agent with create_react_agent
# ═══════════════════════════════
from langgraph.prebuilt import create_react_agent

tools = [search_database, calculate, get_weather]

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=tools,
    prompt="You are a helpful assistant. Use tools when needed.",
)

# Run the agent
result = agent.invoke({"messages": [("human", "What's 25 * 47?")]})

# The agent will:
# 1. See the math question
# 2. Decide to use the calculate tool
# 3. Call calculate("25 * 47")
# 4. Get result "1175"
# 5. Respond with the answer

# ═══════════════════════════════
# Agent with system prompt
# ═══════════════════════════════
agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a data analyst. "
        "Always search the database before answering questions about products. "
        "Show your reasoning step by step."
    ),
)

# ═══════════════════════════════
# Stream agent steps
# ═══════════════════════════════
for event in agent.stream({"messages": [("human", "Find products under $50")]}):
    for key, value in event.items():
        print(f"\n--- {key} ---")
        print(value)
```

> **Note:** For complex agent workflows with multiple steps, conditional logic, or state management, use **LangGraph** instead. See [`langgraph.md`](./langgraph.md).

---

## 9. Memory & Chat History

```python
# ═══════════════════════════════
# Manual chat history (simplest approach)
# ═══════════════════════════════
from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

def chat(user_input: str) -> str:
    chat_history.append(HumanMessage(content=user_input))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"history": chat_history, "input": user_input})
    
    chat_history.append(AIMessage(content=response))
    return response

# ═══════════════════════════════
# Trim messages to fit context window
# ═══════════════════════════════
from langchain_core.messages import trim_messages

trimmed = trim_messages(
    chat_history,
    max_tokens=4000,
    token_counter=llm,         # uses the model's tokenizer
    strategy="last",           # keep the most recent messages
    include_system=True,       # always keep system message
)

# ═══════════════════════════════
# Summarize old messages (for long conversations)
# ═══════════════════════════════
def summarize_history(messages, llm):
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this conversation in 2-3 sentences:\n{conversation}"
    )
    conversation = "\n".join(f"{m.type}: {m.content}" for m in messages)
    summary = (summary_prompt | llm | StrOutputParser()).invoke(
        {"conversation": conversation}
    )
    return [SystemMessage(content=f"Previous conversation summary: {summary}")]

# ═══════════════════════════════
# ChatMessageHistory (in-memory store)
# ═══════════════════════════════
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Use with RunnableWithMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

result = chain_with_history.invoke(
    {"input": "Hello!"},
    config={"configurable": {"session_id": "user-123"}},
)
```

---

## 10. Callbacks & Tracing

```python
# ═══════════════════════════════
# LangSmith tracing (production observability)
# ═══════════════════════════════
# Set environment variables:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-key
# LANGCHAIN_PROJECT=my-project

# All chains automatically traced — no code changes needed!

# ═══════════════════════════════
# Custom callback handler
# ═══════════════════════════════
from langchain_core.callbacks import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with {len(prompts)} prompts")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished. Tokens: {response.llm_output}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool called: {serialized['name']} with input: {input_str}")
    
    def on_chain_error(self, error, **kwargs):
        print(f"Chain error: {error}")

# Use it
result = chain.invoke({"input": "Hello"}, config={"callbacks": [LoggingHandler()]})

# ═══════════════════════════════
# Token counting
# ═══════════════════════════════
from langchain_core.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = chain.invoke({"input": "Hello"})
    print(f"Total tokens: {cb.total_tokens}")
    print(f"Total cost: ${cb.total_cost:.4f}")
```

---

## 11. Document Loaders

```python
# ═══════════════════════════════
# Common loaders
# ═══════════════════════════════

# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()  # list of Document objects

# Web page
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com/article")
docs = loader.load()

# CSV
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader("data.csv")
docs = loader.load()

# Directory of files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader("./docs/", glob="**/*.md")
docs = loader.load()

# JSON
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader("data.json", jq_schema=".messages[].content")
docs = loader.load()

# Confluence
from langchain_community.document_loaders import ConfluenceLoader
loader = ConfluenceLoader(
    url="https://your-domain.atlassian.net/wiki",
    username="user@example.com",
    api_key="your-api-key",
    space_key="TEAM",
)
docs = loader.load()

# ═══════════════════════════════
# Document structure
# ═══════════════════════════════
doc = docs[0]
doc.page_content    # the text content
doc.metadata        # {"source": "file.pdf", "page": 0, ...}
```

---

## 12. Text Splitters

```python
# ═══════════════════════════════
# Recursive character splitter (RECOMMENDED default)
# ═══════════════════════════════
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,          # max characters per chunk
    chunk_overlap=200,        # overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # split priority
)

chunks = splitter.split_documents(docs)
# or
chunks = splitter.split_text("Your long text here...")

# ═══════════════════════════════
# Token-based splitter
# ═══════════════════════════════
from langchain_text_splitters import TokenTextSplitter

splitter = TokenTextSplitter(
    chunk_size=500,           # max tokens per chunk
    chunk_overlap=50,
)

# ═══════════════════════════════
# Markdown splitter (preserves headers as metadata)
# ═══════════════════════════════
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(markdown_text)
# Each chunk has metadata like {"Header 1": "Introduction", "Header 2": "Overview"}

# ═══════════════════════════════
# Code splitter (language-aware)
# ═══════════════════════════════
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)
```

> For more on how these fit into RAG pipelines, see [`rag.md`](./rag.md).

---

## 13. Common Patterns

### Pattern 1: Extraction Pipeline

```python
class Person(BaseModel):
    name: str
    age: int
    occupation: str

extraction_chain = (
    ChatPromptTemplate.from_template("Extract person info from: {text}")
    | llm.with_structured_output(Person)
)

result = extraction_chain.invoke({"text": "John is a 30-year-old engineer."})
# Person(name="John", age=30, occupation="engineer")
```

### Pattern 2: Map-Reduce over Documents

```python
from langchain_core.runnables import RunnableParallel

# Map: summarize each document
summarize = ChatPromptTemplate.from_template("Summarize: {text}") | llm | StrOutputParser()

# Reduce: combine summaries
combine = ChatPromptTemplate.from_template(
    "Combine these summaries into one:\n{summaries}"
) | llm | StrOutputParser()

def map_reduce(documents):
    # Map phase
    summaries = summarize.batch([{"text": doc.page_content} for doc in documents])
    # Reduce phase
    combined = combine.invoke({"summaries": "\n\n".join(summaries)})
    return combined
```

### Pattern 3: Self-Correcting Chain

```python
def self_correcting_chain(input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = chain.invoke(input_data)
            # Validate result
            validated = MySchema.model_validate(result)
            return validated
        except Exception as e:
            # Feed error back to LLM
            input_data["error_feedback"] = str(e)
    raise RuntimeError("Max retries exceeded")
```

### Pattern 4: Router Chain

```python
# Route to different chains based on input
from langchain_core.runnables import RunnableLambda

def route(input_dict):
    topic = input_dict.get("topic", "").lower()
    if "code" in topic:
        return code_chain
    elif "data" in topic:
        return data_chain
    return general_chain

router = RunnableLambda(route)
chain = {"topic": lambda x: x["input"]} | router
```

---

## 14. When to Use LangChain vs LangGraph

| Use Case | LangChain | LangGraph |
|---|---|---|
| Simple prompt → response | ✅ | Overkill |
| Chain of 2-3 LLM calls | ✅ | Optional |
| Tool-calling agent | ✅ (create_react_agent) | ✅ |
| Multi-step workflow with state | ❌ | ✅ |
| Conditional branching logic | Basic (RunnableBranch) | ✅ (conditional edges) |
| Human-in-the-loop approval | ❌ | ✅ |
| Retries with state modification | ❌ | ✅ |
| Multi-agent systems | ❌ | ✅ |
| Cycles/loops in workflow | ❌ | ✅ |
| Persistence across sessions | ❌ | ✅ |

**Rule of thumb:** Start with LangChain (LCEL chains). Move to LangGraph when you need state, cycles, or complex control flow.

---

## 15. Common Pitfalls

| Pitfall | Fix |
|---|---|
| **Not setting `temperature=0`** for deterministic tasks | Always use `temperature=0` for extraction, classification, structured output |
| **Huge prompts blowing context window** | Use `trim_messages` or summarize history; track token usage |
| **Not handling tool call errors** | Wrap tool execution in try/except; return error messages to the LLM |
| **Using string parsing instead of structured output** | Use `with_structured_output()` — it's more reliable than regex/JSON parsing |
| **Putting secrets in prompts** | Use environment variables; never pass API keys through prompt templates |
| **Not streaming for user-facing apps** | Use `.stream()` or `.astream()` — users perceive faster response |
| **Ignoring token costs in loops** | Add `max_iterations` to agents; log token usage with callbacks |
| **`[[0]*n]*m` bug in 2D lists** | Use `[[0]*n for _ in range(m)]` — same applies to LangChain state objects |

---

*For stateful workflows and multi-step agents, see [`langgraph.md`](./langgraph.md). For retrieval-augmented generation, see [`rag.md`](./rag.md).*