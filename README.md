# Mortgage Document RAG Chatbot

A locally-run Retrieval-Augmented Generation system for answering questions over mortgage-related PDFs — built as part of a data engineering externship at Outmation, then extended well beyond the assignment spec.

## What It Does
Most RAG demos send everything to the LLM and hope for the best. The problem with mortgage documents specifically is that LLMs hallucinate financial numbers when reading tables. This system separates the problem: deterministic structured extraction handles numeric questions, LLM reasoning handles everything else. Every answer comes back with sources, confidence scores, and latency.

## Architecture
PDF Upload
│
▼
PyMuPDF extraction → Tesseract OCR fallback (scanned pages)
│
▼
Clean → Chunk → Metadata tag (document type, page range, source)
│
▼
SentenceTransformers (all-MiniLM-L6-v2) → FAISS vector index
│
▼
Query Router
├── Numeric/financial question → Structured extraction (deterministic)
└── Open-ended question → Top-K retrieval → Mistral 7B → Answer + citations


## Key Design Decisions
* **Structured extraction layer:** Not in the original assignment spec. Added after discovering LLMs were unreliable on financial tables. Numeric answers bypass the model entirely and return exact extracted values. LLM time on these queries is 0.0 seconds.
* **Document-type routing:** Prevents a question about a lender fee sheet from accidentally retrieving contract text. Queries are scoped by document category before hitting the index.
* **Model selection:** Evaluated Phi-2 (inconsistent instruction-following on financial text) and LLaMA (too much memory overhead for the environment) before landing on **Mistral 7B Instruct** quantized to `Q4_K_M` via `llama.cpp`. Runs fully locally — no API keys, no quota limits, and no sensitive data leaving the machine.
* **PyMuPDF + Tesseract:** The assignment said "use OCR where needed" without specifying a library. PyMuPDF handles digital PDFs; Tesseract kicks in automatically for scanned or low-quality pages.

## Tech Stack
| Layer | Technology |
| :--- | :--- |
| **PDF Extraction** | PyMuPDF (fitz) |
| **OCR** | Tesseract |
| **Embeddings** | sentence-transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS |
| **LLM** | Mistral 7B Instruct (`Q4_K_M` GGUF via `llama.cpp`) |
| **UI** | Gradio |
| **Environment** | Google Colab / Local Jupyter |

## Evaluation Results
Tested on a hand-labeled set of 10 questions covering numeric extraction, sub-component queries, conceptual reasoning, and negative/missing information queries:
* **Recall@4:** 1.0
* **End-to-end accuracy:** 80%
* **Avg latency (GPU):** 2.1 seconds
* **Hallucinations on financial data:** 0

> **Note:** The 20% miss rate is on broad conceptual queries — a known limitation of the current chunk retrieval strategy without a cross-encoder reranker.

## How to Run
1. Download the Mistral 7B GGUF model:
```bash
   wget [https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
Install the required dependencies:

Bash
   pip install pymupdf pytesseract sentence-transformers faiss-cpu gradio llama-cpp-python transformers torch
Execute the cells inside final_philliplee_full_RAG.ipynb to launch the Gradio interface.

Known Limitations & Future Roadmap
Scrambled table text can break row alignment in structured extraction.

Broad queries without document-type filtering can retrieve unrelated chunks.

What I'd Do Differently: Integrate a persistent vector storage layer (ChromaDB or Pinecone), a cross-encoder reranker on top of FAISS retrieval, an expanded evaluation framework, and a user feedback loop for flagging inaccurate responses.
