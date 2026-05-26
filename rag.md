# RAG (Retrieval-Augmented Generation) — Practical Guide

> Ground LLM responses in real data by retrieving relevant context before generating answers.

---

## Table of Contents

1. [What is RAG](#1-what-is-rag)
2. [RAG Architecture](#2-rag-architecture)
3. [Document Loading](#3-document-loading)
4. [Chunking Strategies](#4-chunking-strategies)
5. [Embeddings](#5-embeddings)
6. [Vector Stores](#6-vector-stores)
7. [Retrieval](#7-retrieval)
8. [Re-ranking](#8-re-ranking)
9. [Generation](#9-generation)
10. [Advanced RAG Patterns](#10-advanced-rag-patterns)
11. [Evaluation](#11-evaluation)
12. [Production Considerations](#12-production-considerations)
13. [Common Pitfalls](#13-common-pitfalls)

---

## 1. What is RAG

RAG = **Retrieve** relevant documents + **Augment** the prompt with them + **Generate** a grounded answer.

Instead of relying solely on the LLM's training data (which can be outdated or hallucinated), RAG fetches real documents and includes them as context.

```
User Query → Retrieve relevant docs → Stuff into prompt → LLM generates answer
```

**When to use RAG:**
- Q&A over your own documents (docs, wikis, code)
- Customer support with knowledge base
- Any task where the LLM needs access to private/recent data

---

## 2. RAG Architecture

```
┌─────────────────── INDEXING (offline) ───────────────────┐
│                                                          │
│  Documents → Chunking → Embeddings → Vector Store        │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── RETRIEVAL (runtime) ──────────────────┐
│                                                          │
│  Query → Embed query → Search vector store → Top-K docs  │
│       → (optional) Re-rank → Final context               │
│                                                          │
└──────────────────────────────────────────────────────────┘

┌─────────────────── GENERATION (runtime) ─────────────────┐
│                                                          │
│  System prompt + Retrieved context + User query → LLM    │
│       → Answer (grounded in retrieved docs)              │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Document Loading

```python
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader, CSVLoader,
    DirectoryLoader, TextLoader, JSONLoader,
    ConfluenceLoader, GitLoader,
)

# PDF
docs = PyPDFLoader("report.pdf").load()

# Web pages
docs = WebBaseLoader(["https://example.com/page1", "https://example.com/page2"]).load()

# Directory of files
docs = DirectoryLoader("./knowledge_base/", glob="**/*.md", loader_cls=TextLoader).load()

# Git repository
docs = GitLoader(repo_path="./my-repo", branch="main", file_filter=lambda f: f.endswith(".py")).load()

# Each document has:
# doc.page_content  → the text
# doc.metadata      → {"source": "file.pdf", "page": 0, ...}

# Add custom metadata
for doc in docs:
    doc.metadata["project"] = "my-project"
    doc.metadata["indexed_at"] = "2024-01-01"
```

---

## 4. Chunking Strategies

Chunking is **the most impactful decision** in a RAG pipeline. Bad chunks = bad retrieval = bad answers.

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)

# ═══════════════════════════════
# Strategy 1: Recursive character (default choice)
# ═══════════════════════════════
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # characters per chunk
    chunk_overlap=200,     # overlap to preserve context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)

# ═══════════════════════════════
# Strategy 2: Token-based (matches LLM context window)
# ═══════════════════════════════
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)

# ═══════════════════════════════
# Strategy 3: Markdown-aware (preserves document structure)
# ═══════════════════════════════
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
)
md_chunks = md_splitter.split_text(markdown_text)
# Headers become metadata: {"h1": "Introduction", "h2": "Setup"}

# Then apply size-based splitting on each section:
final_chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(md_chunks)

# ═══════════════════════════════
# Strategy 4: Code-aware splitting
# ═══════════════════════════════
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100,
)

# ═══════════════════════════════
# Strategy 5: Semantic chunking (split by meaning)
# ═══════════════════════════════
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

semantic_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation"
)
chunks = semantic_splitter.split_documents(docs)
```

### Chunking Decision Guide

| Content Type | Strategy | chunk_size | chunk_overlap |
|---|---|---|---|
| General text | RecursiveCharacter | 500-1000 | 100-200 |
| Technical docs | Markdown-aware + Recursive | 500-1000 | 100-200 |
| Code | Language-aware | 1000-2000 | 100 |
| Short entries (FAQ) | No splitting needed | — | — |
| Long narratives | Semantic chunking | varies | — |

---

## 5. Embeddings

Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts → similar vectors.

```python
# ═══════════════════════════════
# OpenAI embeddings (most common)
# ═══════════════════════════════
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536 dimensions
# or text-embedding-3-large for 3072 dimensions (more accurate, costs more)

# Embed a single text
vector = embeddings.embed_query("What is RAG?")  # returns list[float]

# Embed multiple texts (batch)
vectors = embeddings.embed_documents(["text 1", "text 2", "text 3"])

# ═══════════════════════════════
# Open-source alternatives
# ═══════════════════════════════
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
# Runs locally — no API costs, but slower

# ═══════════════════════════════
# Cohere embeddings (with input_type)
# ═══════════════════════════════
from langchain_cohere import CohereEmbeddings

embeddings = CohereEmbeddings(model="embed-english-v3.0")
```

---

## 6. Vector Stores

Vector stores index embeddings for fast similarity search.

```python
# ═══════════════════════════════
# Chroma (local, great for development)
# ═══════════════════════════════
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",   # persists to disk
    collection_name="my_docs",
)

# ═══════════════════════════════
# FAISS (local, fast, Facebook AI)
# ═══════════════════════════════
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ═══════════════════════════════
# pgvector (PostgreSQL — production)
# ═══════════════════════════════
from langchain_postgres import PGVector

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection="postgresql+psycopg://user:pass@host:5432/db",
    collection_name="my_docs",
)

# ═══════════════════════════════
# Pinecone (managed, scalable)
# ═══════════════════════════════
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    chunks, embeddings, index_name="my-index"
)

# ═══════════════════════════════
# Common operations
# ═══════════════════════════════

# Similarity search
results = vectorstore.similarity_search("How to deploy?", k=5)

# Similarity search with scores
results = vectorstore.similarity_search_with_score("How to deploy?", k=5)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")

# Filter by metadata
results = vectorstore.similarity_search(
    "deployment", k=5, filter={"project": "my-project"}
)

# Add more documents
vectorstore.add_documents(new_chunks)

# Delete documents
vectorstore.delete(ids=["doc-id-1", "doc-id-2"])

# Convert to retriever (for use in chains)
retriever = vectorstore.as_retriever(
    search_type="similarity",     # or "mmr" (maximal marginal relevance)
    search_kwargs={"k": 5},
)
```

---

## 7. Retrieval

```python
# ═══════════════════════════════
# Basic retriever
# ═══════════════════════════════
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("What is RAG?")

# ═══════════════════════════════
# MMR (Maximal Marginal Relevance) — diversity + relevance
# ═══════════════════════════════
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    # fetch_k: retrieve 20 candidates, then pick 5 most diverse
    # lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
)

# ═══════════════════════════════
# Multi-query retriever (generates multiple search queries)
# ═══════════════════════════════
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)
# LLM generates 3+ rephrased queries, retrieves for each, deduplicates

# ═══════════════════════════════
# Contextual compression (filter irrelevant chunks)
# ═══════════════════════════════
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
# Retrieves 10 docs, then LLM extracts only relevant parts

# ═══════════════════════════════
# Ensemble retriever (combine multiple strategies)
# ═══════════════════════════════
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25 = BM25Retriever.from_documents(chunks, k=5)       # keyword-based
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # semantic

ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6],  # 40% keyword, 60% semantic
)

# ═══════════════════════════════
# Parent document retriever (small chunks for search, full docs for context)
# ═══════════════════════════════
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
# Searches on small chunks, returns the full parent chunk
```

---

## 8. Re-ranking

Re-ranking improves precision by scoring retrieved documents with a cross-encoder model.

```python
# ═══════════════════════════════
# Cohere re-ranker
# ═══════════════════════════════
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(model="rerank-english-v3.0", top_n=5)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
# Retrieves 20 docs, re-ranks them, returns top 5

# ═══════════════════════════════
# Cross-encoder re-ranker (local, free)
# ═══════════════════════════════
from langchain_community.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=model, top_n=5)

# ═══════════════════════════════
# LLM-based re-ranking (expensive but flexible)
# ═══════════════════════════════
def llm_rerank(query, docs, llm, top_n=5):
    scored = []
    for doc in docs:
        prompt = f"Rate 1-10 how relevant this doc is to '{query}':\n{doc.page_content[:500]}"
        score = llm.invoke(prompt)
        scored.append((doc, float(score.content)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_n]]
```

---

## 9. Generation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ═══════════════════════════════
# Basic RAG chain
# ═══════════════════════════════
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the context doesn't contain enough information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How do I deploy to production?")

# ═══════════════════════════════
# RAG with source citations
# ═══════════════════════════════
from pydantic import BaseModel

class AnswerWithSources(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

prompt_with_sources = ChatPromptTemplate.from_template("""
Answer based on context. Include which sources you used.

Context:
{context}

Question: {question}
""")

rag_with_sources = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_with_sources
    | llm.with_structured_output(AnswerWithSources)
)

# ═══════════════════════════════
# Conversational RAG (with chat history)
# ═══════════════════════════════
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Step 1: Contextualize the question using chat history
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given chat history and a question, reformulate it as a standalone question."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

# Step 2: Answer using retrieved docs
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the following context:\n\n{context}"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Use it
result = rag_chain.invoke({
    "input": "How do I configure the database?",
    "chat_history": [],
})
print(result["answer"])
print(result["context"])  # retrieved documents
```

---

## 10. Advanced RAG Patterns

### Corrective RAG (CRAG)
```
Query → Retrieve → Grade relevance → If good: generate
                                    → If bad: web search → generate
```

### Self-RAG
```
Query → Retrieve → Generate → Check for hallucination
                             → If hallucinated: re-retrieve or regenerate
                             → If grounded: return answer
```

### Agentic RAG
```
Query → Agent decides: retrieve from DB? search web? use tool? ask clarification?
      → Executes chosen action → Generates answer
```

### Hypothetical Document Embedding (HyDE)
```python
# Generate a hypothetical answer, embed THAT, then search
hypothetical = llm.invoke(f"Write a short answer to: {query}")
docs = vectorstore.similarity_search(hypothetical.content, k=5)
# The hypothetical answer is closer to real docs in embedding space
```

### Query Decomposition
```python
# Break complex queries into sub-queries
sub_queries = llm.invoke(
    f"Break this into 2-3 simpler sub-questions: {complex_query}"
)
# Retrieve for each sub-query, combine results
```

---

## 11. Evaluation

```python
# ═══════════════════════════════
# Key metrics
# ═══════════════════════════════
# 1. Retrieval quality: Are the right documents retrieved?
#    - Precision@K: % of retrieved docs that are relevant
#    - Recall@K: % of relevant docs that were retrieved
#    - MRR: How high is the first relevant doc ranked?

# 2. Generation quality: Is the answer correct and grounded?
#    - Faithfulness: Does the answer only use info from context?
#    - Answer relevance: Does it actually answer the question?
#    - Correctness: Is it factually right?

# ═══════════════════════════════
# RAGAS evaluation framework
# ═══════════════════════════════
# pip install ragas
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset=eval_dataset,  # questions, ground truth, retrieved contexts, answers
    metrics=[faithfulness, answer_relevancy, context_precision],
)

# ═══════════════════════════════
# LLM-as-judge (simple but effective)
# ═══════════════════════════════
judge_prompt = """
Given:
- Question: {question}
- Context: {context}
- Answer: {answer}

Rate the answer on:
1. Faithfulness (1-5): Does it only use info from context?
2. Relevance (1-5): Does it answer the question?
3. Completeness (1-5): Does it cover all aspects?
"""
```

---

## 12. Production Considerations

| Concern | Solution |
|---|---|
| **Indexing speed** | Batch embed documents; use async; parallelize |
| **Storage costs** | Use dimensionality reduction; choose smaller embedding models |
| **Stale data** | Implement incremental indexing; track document versions |
| **Latency** | Cache frequent queries; use smaller models for re-ranking |
| **Scale** | Use managed vector DBs (Pinecone, Weaviate); shard large indices |
| **Security** | Filter by user permissions in metadata; never expose raw vectors |
| **Monitoring** | Log retrieval scores; track answer quality; alert on low confidence |
| **Context window** | Limit retrieved docs to fit; summarize if needed |

---

## 13. Common Pitfalls

| Pitfall | Fix |
|---|---|
| **Chunks too large** | LLM gets diluted context; use 500-1000 char chunks |
| **Chunks too small** | Lose context; increase overlap or use parent-document retriever |
| **No overlap between chunks** | Boundary information lost; use 10-20% overlap |
| **Ignoring metadata** | Add source, date, section info; filter during retrieval |
| **Only using vector search** | Add keyword search (BM25) via ensemble retriever |
| **Not re-ranking** | Top-K from vector search isn't always best; add a re-ranker |
| **Stuffing all docs in prompt** | Hit token limits; use map-reduce or select top-3 |
| **Not evaluating** | Build an eval set of 50+ QA pairs; measure before optimizing |
| **Using wrong embedding model** | Match embedding model to your content language/domain |
| **Embedding queries same as docs** | Some models need different prefixes for queries vs documents |

---

*For LLM chain setup, see [`langchain.md`](./langchain.md). For stateful RAG workflows, see [`langgraph.md`](./langgraph.md).*