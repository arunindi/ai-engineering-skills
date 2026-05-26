# AI Engineering Skills

Practical, code-first reference guides for building production AI systems. Each document covers core concepts, architecture patterns, and working code examples — designed to be useful whether you're starting a new project or need a quick refresher.

---

## 📂 Guides

| Guide | What You'll Learn |
|---|---|
| [`langchain.md`](./langchain.md) | Chains, tools, structured output, memory, agents, prompt templates |
| [`langgraph.md`](./langgraph.md) | Stateful workflows, conditional routing, retries, human-in-the-loop, persistence |
| [`rag.md`](./rag.md) | Document loading, chunking, embeddings, vector stores, retrieval, re-ranking, evaluation |
| [`mcp.md`](./mcp.md) | Model Context Protocol — servers, tools, resources, transports, building integrations |
| [`ai-agent-optimization.md`](./ai-agent-optimization.md) | Architecture patterns, cost/latency/reliability optimization, production patterns |

---

## 🎯 Who This Is For

- Developers building **LLM-powered applications** who want practical patterns, not just API docs
- Engineers evaluating **LangChain vs LangGraph** or figuring out when to use what
- Anyone implementing **RAG pipelines** and wanting to go beyond basic tutorials
- Teams integrating **MCP** to give AI agents access to external tools and data
- Backend/AI engineers deploying on **AWS** with production-grade infrastructure

---

## 💡 How to Use

Each guide follows the same structure:
1. **What it is** — core concept in 2-3 sentences
2. **Key concepts** — the mental model you need
3. **Code examples** — working Python code you can copy and adapt
4. **Patterns & best practices** — what to do in production
5. **Common pitfalls** — mistakes to avoid
6. **When to use / when not to** — decision framework

---

## 🛠 Tech Stack Covered

```
LangChain ─── chains, tools, agents, memory, callbacks
LangGraph ─── stateful multi-step workflows, conditional edges
RAG ───────── chunking, embeddings, vector DBs, retrieval, re-ranking
MCP ───────── Model Context Protocol servers, tool/resource integration
AWS ───────── ECS, S3, SQS, SNS, EventBridge, RDS, ALB, ACM, IaC
```

---

## 📜 License

MIT — use freely, star if helpful.