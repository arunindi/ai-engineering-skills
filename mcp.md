# MCP (Model Context Protocol) — Practical Guide

> Give AI agents standardized, pluggable access to external tools, data sources, and APIs.

---

## Table of Contents

1. [What is MCP](#1-what-is-mcp)
2. [Architecture](#2-architecture)
3. [Core Primitives](#3-core-primitives)
4. [Building an MCP Server](#4-building-an-mcp-server)
5. [Tools](#5-tools)
6. [Resources](#6-resources)
7. [Prompts](#7-prompts)
8. [Transport Mechanisms](#8-transport-mechanisms)
9. [MCP Client Integration](#9-mcp-client-integration)
10. [Using MCP with LangChain/LangGraph](#10-using-mcp-with-langchainlanggraph)
11. [Production Patterns](#11-production-patterns)
12. [Common Pitfalls](#12-common-pitfalls)

---

## 1. What is MCP

MCP (Model Context Protocol) is an **open standard** (by Anthropic) that defines how AI applications connect to external data sources and tools. Think of it as **USB-C for AI** — one standardized protocol instead of custom integrations for every tool.

**Without MCP:** Every tool needs custom API wrappers, auth handling, error mapping.
**With MCP:** Tools expose a standard interface; any MCP-compatible client can use them.

```bash
pip install mcp
```

---

## 2. Architecture

```
┌─────────────────┐     MCP Protocol      ┌─────────────────┐
│   MCP Client    │ ◄──────────────────► │   MCP Server    │
│ (your AI app,   │    JSON-RPC over      │ (exposes tools, │
│  Claude, IDE)   │    stdio / SSE        │  resources)     │
└─────────────────┘                       └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  External APIs   │
                                          │  (Jira, GitHub,  │
                                          │   databases, S3) │
                                          └─────────────────┘
```

**Host** — The application (Claude Desktop, IDE, your app)
**Client** — Maintains 1:1 connection with a server
**Server** — Exposes tools, resources, and prompts via MCP protocol

---

## 3. Core Primitives

| Primitive | Direction | Description | Example |
|---|---|---|---|
| **Tools** | Server → Client (LLM invokes) | Functions the LLM can call | `create_jira_ticket`, `search_github` |
| **Resources** | Server → Client (app reads) | Data the app can read | `file://config.json`, `db://users/123` |
| **Prompts** | Server → Client | Reusable prompt templates | `code_review`, `summarize_doc` |

---

## 4. Building an MCP Server

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

# ═══════════════════════════════
# Define a tool
# ═══════════════════════════════
@mcp.tool()
def search_database(query: str, limit: int = 10) -> str:
    """Search the product database.
    
    Args:
        query: Search query string
        limit: Maximum number of results
    """
    results = db.search(query, limit=limit)
    return json.dumps(results)

# ═══════════════════════════════
# Define a resource
# ═══════════════════════════════
@mcp.resource("config://app")
def get_app_config() -> str:
    """Return the current application configuration."""
    return json.dumps(load_config())

# Dynamic resource with URI template
@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """Get a user's profile by ID."""
    return json.dumps(fetch_user(user_id))

# ═══════════════════════════════
# Define a prompt template
# ═══════════════════════════════
@mcp.prompt()
def code_review(code: str, language: str = "python") -> str:
    """Generate a code review prompt."""
    return f"Review this {language} code for bugs, style, and performance:\n\n```{language}\n{code}\n```"

# ═══════════════════════════════
# Run the server
# ═══════════════════════════════
if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport
```

---

## 5. Tools

Tools are the primary way LLMs interact with external systems through MCP.

```python
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

mcp = FastMCP("tools-demo")

# Simple tool
@mcp.tool()
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    data = weather_api.get(city)
    return f"{data['temp']}°C, {data['condition']}"

# Tool with complex input
@mcp.tool()
def create_jira_ticket(
    project: str,
    summary: str,
    description: str,
    issue_type: str = "Task",
    priority: str = "Medium",
    labels: list[str] | None = None,
) -> str:
    """Create a new Jira ticket.
    
    Args:
        project: Jira project key (e.g., 'PROJ')
        summary: Ticket title
        description: Detailed description
        issue_type: Task, Bug, Story, Epic
        priority: Low, Medium, High, Critical
        labels: Optional list of labels
    """
    ticket = jira_client.create_issue(
        project=project,
        summary=summary,
        description=description,
        issuetype={"name": issue_type},
        priority={"name": priority},
        labels=labels or [],
    )
    return f"Created {ticket.key}: {ticket.fields.summary}"

# Tool that returns structured data
@mcp.tool()
def search_confluence(query: str, space: str = None, limit: int = 5) -> str:
    """Search Confluence wiki for documentation."""
    results = confluence.search(query, space=space, limit=limit)
    return json.dumps([{
        "title": r["title"],
        "url": r["url"],
        "excerpt": r["excerpt"][:200],
    } for r in results])

# Tool with error handling
@mcp.tool()
def run_sql_query(query: str) -> str:
    """Execute a read-only SQL query against the database."""
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    try:
        results = db.execute(query)
        return json.dumps(results[:100])  # limit rows
    except Exception as e:
        return f"Query error: {str(e)}"
```

---

## 6. Resources

Resources provide data that clients can read (like files, configs, API data).

```python
@mcp.resource("docs://api-reference")
def api_reference() -> str:
    """The complete API reference documentation."""
    return open("docs/api-reference.md").read()

@mcp.resource("metrics://dashboard")
def dashboard_metrics() -> str:
    """Current system metrics."""
    return json.dumps({
        "cpu": get_cpu_usage(),
        "memory": get_memory_usage(),
        "requests_per_second": get_rps(),
    })

# List resources dynamically
@mcp.resource("repos://{owner}/{repo}/readme")
def get_readme(owner: str, repo: str) -> str:
    """Get README from a GitHub repository."""
    return github.get_file_content(owner, repo, "README.md")
```

---

## 7. Prompts

Prompts are reusable templates that clients can use.

```python
@mcp.prompt()
def debug_error(error_message: str, stack_trace: str = "") -> str:
    """Help debug an error."""
    return f"""Analyze this error and suggest fixes:

Error: {error_message}
{"Stack trace:" + stack_trace if stack_trace else ""}

Provide:
1. Root cause analysis
2. Step-by-step fix
3. How to prevent in future"""

@mcp.prompt()
def pr_description(diff: str, ticket_id: str = "") -> str:
    """Generate a PR description from a diff."""
    return f"""Write a clear PR description for this change:
{"Ticket: " + ticket_id if ticket_id else ""}

Diff:
{diff}

Include: Summary, Changes Made, Testing Done, Screenshots (if UI)."""
```

---

## 8. Transport Mechanisms

```python
# ═══════════════════════════════
# stdio (default — for local tools, IDE integration)
# ═══════════════════════════════
if __name__ == "__main__":
    mcp.run()  # uses stdio

# Client config (e.g., Claude Desktop):
# {
#   "mcpServers": {
#     "my-server": {
#       "command": "python",
#       "args": ["server.py"]
#     }
#   }
# }

# ═══════════════════════════════
# SSE (Server-Sent Events — for remote/web servers)
# ═══════════════════════════════
if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8080)

# Client connects to: http://localhost:8080/sse

# ═══════════════════════════════
# Streamable HTTP (newest transport)
# ═══════════════════════════════
if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

---

## 9. MCP Client Integration

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to an MCP server
server_params = StdioServerParameters(
    command="python",
    args=["path/to/server.py"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        for tool in tools.tools:
            print(f"Tool: {tool.name} — {tool.description}")
        
        # Call a tool
        result = await session.call_tool("search_database", {"query": "python"})
        print(result.content)
        
        # List and read resources
        resources = await session.list_resources()
        content = await session.read_resource("config://app")
```

---

## 10. Using MCP with LangChain/LangGraph

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# Connect to multiple MCP servers
async with MultiServerMCPClient({
    "jira": {
        "command": "python",
        "args": ["servers/jira_server.py"],
    },
    "github": {
        "command": "python",
        "args": ["servers/github_server.py"],
    },
    "confluence": {
        "url": "http://localhost:8080/sse",  # remote server
    },
}) as client:
    # Get all tools from all servers
    tools = client.get_tools()
    
    # Create an agent with MCP tools
    agent = create_react_agent(
        model=ChatOpenAI(model="gpt-4o"),
        tools=tools,
    )
    
    result = await agent.ainvoke({
        "messages": [("human", "Create a Jira ticket for the login bug")]
    })
```

---

## 11. Production Patterns

### Pattern 1: Auth & Security
```python
import os

@mcp.tool()
def github_search(query: str) -> str:
    """Search GitHub repositories."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN not configured"
    headers = {"Authorization": f"token {token}"}
    # ... use token in API calls
```

### Pattern 2: Rate Limiting
```python
from functools import wraps
import time

def rate_limit(calls_per_minute=30):
    timestamps = []
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            timestamps[:] = [t for t in timestamps if now - t < 60]
            if len(timestamps) >= calls_per_minute:
                return f"Rate limited. Try again in {60 - (now - timestamps[0]):.0f}s"
            timestamps.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@mcp.tool()
@rate_limit(calls_per_minute=30)
def call_external_api(query: str) -> str:
    """Rate-limited external API call."""
    return api.search(query)
```

### Pattern 3: Caching
```python
from functools import lru_cache

@mcp.tool()
def get_documentation(topic: str) -> str:
    """Get cached documentation."""
    return _fetch_docs(topic)

@lru_cache(maxsize=100)
def _fetch_docs(topic: str) -> str:
    return confluence.get_page(topic)
```

---

## 12. Common Pitfalls

| Pitfall | Fix |
|---|---|
| **Tool descriptions too vague** | Write clear, specific docstrings — the LLM reads them to decide when to use tools |
| **No error handling in tools** | Always return error strings instead of raising exceptions |
| **Returning huge payloads** | Limit response size; paginate results; summarize data |
| **Secrets in tool arguments** | Use environment variables; never accept tokens as tool inputs |
| **No input validation** | Validate args before calling external APIs; reject dangerous inputs |
| **Missing type hints** | MCP uses type hints to generate schemas — always annotate args |
| **Blocking I/O in async servers** | Use `asyncio` for network calls; don't block the event loop |
| **No logging** | Log tool calls for debugging; track usage for monitoring |

---

*For building agents that use MCP tools, see [`langgraph.md`](./langgraph.md). For LLM chain basics, see [`langchain.md`](./langchain.md).*