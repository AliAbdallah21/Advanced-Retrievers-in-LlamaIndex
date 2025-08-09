# 📚 LlamaIndex Retrieval Strategies Guide

## 1️⃣ Project Overview

This project is a **practical guide** to LlamaIndex’s retrieval capabilities.  
It demonstrates how to implement different retrieval strategies to improve the **accuracy**, **context quality**, and **relevance** of responses in Large Language Model (LLM) applications.

### Included Retrievers
- **Vector Index Retriever** – Standard semantic search using vector similarity.
- **BM25 Retriever** – Keyword-based search with advanced scoring.
- **Document Summary Index Retrievers** – Retrieve via LLM-generated summaries (LLM-based or embedding-based).
- **Auto Merging Retriever** – Retrieve small relevant chunks, then merge them with their larger parent chunks for context.
- **Recursive Retriever** – Follow explicit links/references between documents for broader context.
- **QueryFusionRetriever** – Generate multiple sub-queries and combine results for improved recall.

---

## 2️⃣ Setup & Usage

### Installation
Run the following command in your environment:
```bash
!pip install llama-index-core \
    llama-index-llms-openai \
    llama-index-embeddings-openai \
    llama-index-retrievers-bm25 \
    sentence-transformers \
    rank-bm25 \
    pystemmer \
    llama-index-embeddings-huggingface
```
### API Key Setup
Before running the notebook, set your OpenAI API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

## 3️⃣ Retriever Explanations

### **BM25 Retriever**
- **Purpose:** Improves on TF-IDF by using:
  - **Saturation function (k1):** Prevents high term frequency from dominating.
  - **Length normalization (b):** Avoids favoring long documents.
- **Best for:** Keyword-heavy queries and exact match search.

---

### **Document Summary Index Retrievers**
1. **LLM-based:**  
   - LLM generates summaries for each document.  
   - LLM selects the most relevant summaries for the query.  
   - Retrieves full documents from selected summaries.
2. **Embedding-based:**  
   - Uses vector similarity between the query and summary embeddings to retrieve.

**Best for:** When full documents are long but a concise summary can guide retrieval.

---

### **Auto Merging Retriever**
- Creates a hierarchy of chunks: small → medium → large.
- Retrieves small chunks first.
- If multiple chunks share a parent, retrieves the full parent for added context.

**Best for:** Long-form content summarization or detailed answers requiring more context.

---

### **Recursive Retriever**
- Initial search retrieves documents.
- Follows links/references in metadata to fetch related documents.

**Best for:** Multi-step reasoning and reference-heavy datasets.

---

### **QueryFusionRetriever**
Generates multiple sub-queries from the original query and fuses results.  
Supports three **fusion modes**:

| Fusion Mode | How It Works | Best For |
|-------------|--------------|----------|
| **RRF (Reciprocal Rank Fusion)** | Scores = 1 / (rank + k). Combines based on rank positions. | Stable ranking, different score scales. |
| **RSF (Relative Score Fusion)** | Normalizes scores by max per query, then sums. | When retriever scores are meaningful and comparable. |
| **DSF (Distribution-Based Score Fusion)** | Uses z-score normalization to combine. | When scores vary greatly between sub-queries. |

---

## 4️⃣ Recommended Retrievers by Use Case

| Use Case | Primary Retriever | Secondary/Hybrid |
|----------|------------------|------------------|
| Simple Q&A | Vector | RRF or RSF Fusion |
| Complex Queries (Multi-step reasoning) | Recursive | Auto-merging |
| Long-form Content Summarization | Auto-merging | LLM-based Summary |
| Structured Data Extraction | BM25 or Vector | Recursive |
| Domain-specific Q&A | Vector | RRF Fusion |
| Chatbots & Conversational Agents | Vector with Memory | Recursive |

---

## 📂 Repository Structure
├── notebooks/
│ └── retrieval_demo.ipynb # Main walkthrough notebook
├── README.md # Project documentation


---

## 🛠 Requirements
- Python 3.8+
- Jupyter Notebook
- OpenAI API Key
- Internet connection for model downloads

---

## 📜 License
This project is released under the MIT License.
