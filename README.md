# JusticeMitra ⚖️
**AI-Powered Legal Rights Information Assistant for Women in India**

JusticeMitra is an open-source, source-grounded Retrieval-Augmented Generation (RAG) assistant designed to help women in India understand their legal rights and statutory protections under key Indian legislations, including:
- **Protection of Women from Domestic Violence Act (PWDVA), 2005**
- **Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) (POSH) Act, 2013**
- **Dowry Prohibition Act, 1961**
- **Cruelty & Dowry Harassment under Section 498A IPC / Section 85 BNS**
- **Free Legal Aid under Section 12(c) of the Legal Services Authorities Act, 1987**
- **Maternity Benefit Act, 1961 & Equal Remuneration Act, 1976**

JusticeMitra provides verifiable, cited statutory guidance coupled with privacy-conscious telemetry and immediate emergency hotline access.

---

## 🏗️ Target Architecture & Partner Integrations

JusticeMitra integrates four core partner technologies into a production-ready, modular architecture:

```
                              ┌───────────────────────────────────┐
                              │     Streamlit Presentation UI     │
                              │             (app.py)              │
                              └─────────────────┬─────────────────┘
                                                │ User Query
                                                ▼
                              ┌───────────────────────────────────┐
                              │      Sentry Agent Tracing         │
                              │  (Privacy-Safe Span Instrumentation)
                              └─────────────────┬─────────────────┘
                                                │ Sanitized Query
                                                ▼
                              ┌───────────────────────────────────┐
                              │       RAG Orchestration           │
                              │   (justicemitra/core/rag.py)      │
                              └───────┬───────────────────┬───────┘
                                      │                   │
         Dense 384-d Query Vector     ▼                   ▼ Grounded Context Window
┌──────────────────────────────────────────────┐ ┌──────────────────────────────────────────────┐
│          MongoDB Atlas Vector Search         │ │         Google Gemma Inference Gateway        │
│          ($vectorSearch Aggregation)         │ │             (gemma-2-9b-it via API)          │
│ • Collection: legal_chunks                   │ │ • Strict grounding system instructions       │
│ • Index: vector_index (Cosine, 384-dim)      │ │ • Evidence-aware refusal guardrails          │
│ • Deterministic chunking & deduplication     │ │ • Token & latency accounting in Sentry       │
└──────────────────────────────────────────────┘ └──────────────────────────────────────────────┘
                                      │                   │
                                      └─────────┬─────────┘
                                                │
                                                ▼
                              ┌───────────────────────────────────┐
                              │  Render Web Service Deployment    │
                              │ • Blueprint: render.yaml          │
                              │ • Health Check: /_stcore/health   │
                              │ • Dynamic $PORT Binding           │
                              └───────────────────────────────────┘
```

### 1. Google Gemma (Answer Generation)
- **Model**: `gemma-2-9b-it` (Google's open-weight instruction-tuned model).
- **Interface**: Decoupled via `LegalGenerationGateway` protocol and `GemmaOpenAIGateway` using the standard OpenAI client protocol. Can point to any compliant provider (e.g. Groq Cloud, Hugging Face Dedicated Endpoints, or local vLLM / Ollama).
- **Grounding Guardrails**: Strictly constrained system prompt requiring citation to retrieved statutory provisions. Bypasses generation and issues an explicit refusal if retrieved context is insufficient or out-of-scope, preventing legal hallucinations.

### 2. MongoDB Atlas Vector Search (Persistent Document Storage & Semantic Retrieval)
- **Database**: `justicemitra_db` | **Collection**: `legal_chunks`
- **Vector Search Index**: `vector_index` (Cosine similarity, 384 dimensions matching `sentence-transformers/all-MiniLM-L6-v2`, with pre-filtering on `statute_title` and `jurisdiction`).
- **Retrieval Pipeline**: PyMongo aggregation utilizing the `$vectorSearch` operator with dynamic top-k and candidate scaling.
- **Offline Ingestion CLI**: `scripts/ingest.py` extracts PDF text and statutory rules, applies SHA-256 document hashing and deterministic chunk hashing, and performs idempotent bulk upserts.

### 3. Sentry Agent Tracing (Observability & Privacy-Preserving Telemetry)
- **Transaction**: `legal_query_interaction`
- **Hierarchical Spans**:
  - `gen_ai.input_sanitization`: Sanitizes personal identifiers.
  - `gen_ai.retrieval`: Captures Atlas retrieval duration, candidate multiplier, and minimum similarity scores.
  - `gen_ai.chat`: Captures Gemma model identifier, prompt/completion tokens, and latency.
  - `ui.render_response`: Streamlit card rendering time.
- **Privacy Redaction Policy**: `send_default_pii=False`. The `redact_sensitive_pii` before-send hook scrubs Indian telephone numbers (+91 / 10-digit mobile), 12-digit Aadhaar formats, and email addresses from breadcrumbs, error logs, and exception strings.

### 4. Render (Containerized Web Service Deployment)
- **Specification**: Defined in `render.yaml` as an Infrastructure-as-Code blueprint.
- **Runtime**: Python 3.11 web service binding dynamically to Render's reverse-proxy assigned `$PORT`.
- **Health Check**: Configured on Streamlit's native endpoint `/_stcore/health` (returns HTTP 200 OK).
- **Memory Footprint**: Kept under 350 MB RAM, enabling continuous 24/7 hosting on Render's $7/month Starter plan without OOM risks or slow CPU LLM loading.

---

## 📁 Repository Structure

```
JusticeMitra/
├── app.py                     # Streamlit web UI and presentation layer
├── render.yaml                # Render Web Service deployment blueprint
├── requirements.txt           # Standardized, pinned dependencies
├── .env.example               # Template for configuration environment variables
├── data/
│   ├── Majlis_Legal_Rights_of_women.pdf # Baseline legal rights guide
│   └── legal_corpus.json      # Offline precomputed seed corpus (384-d vectors)
├── justicemitra/
│   ├── config.py              # Twelve-Factor configuration loader
│   ├── core/
│   │   ├── rag.py             # RAG orchestration, grounding, and refusal logic
│   │   └── sanitizer.py       # PII redaction and acute emergency detection
│   ├── services/
│   │   ├── database.py        # MongoDB Atlas connection and index definition
│   │   ├── retriever.py       # Atlas Vector Search ($vectorSearch) & local fallback
│   │   ├── embedding.py       # Singleton all-MiniLM-L6-v2 embedding service (384-d)
│   │   └── generation.py      # Gemma generation gateway & refusal messaging
│   └── observability/
│       └── telemetry.py       # Sentry Agent Tracing and before-send scrubber
├── scripts/
│   └── ingest.py              # Administrative offline document ingestion CLI
└── tests/
    ├── test_database.py       # Atlas index spec and connection handling tests
    ├── test_embedding.py      # Vector dimensions (384) and normalization tests
    ├── test_generation.py     # Gemma gateway and refusal behavior tests
    ├── test_ingest.py         # Deterministic chunk hashing and idempotency tests
    ├── test_rag.py            # RAG orchestration and emergency detection tests
    ├── test_regression.py     # Standardized legal regression suite (Q1-Q5)
    ├── test_retriever.py      # $vectorSearch aggregation pipeline tests
    └── test_sanitizer.py      # PII scrubbing (phone, Aadhaar, email) tests
```

---

## 🚀 Quickstart & Local Development

### 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Clone and Setup Environment
```bash
git clone https://github.com/nanmodi/JusticeMitra.git
cd JusticeMitra

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate `.env` with your partner service credentials:
```ini
# MongoDB Atlas
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority
MONGODB_DATABASE=justicemitra_db
MONGODB_COLLECTION=legal_chunks
MONGODB_VECTOR_INDEX=vector_index

# Google Gemma Endpoint (Groq, vLLM, or Hugging Face)
GEMMA_API_BASE=https://api.groq.com/openai/v1
GEMMA_API_KEY=your_gemma_api_key_here
GEMMA_MODEL_NAME=gemma2-9b-it

# Sentry Tracing
SENTRY_DSN=https://your_sentry_key@o000000.ingest.sentry.io/0000000
SENTRY_ENVIRONMENT=development
SENTRY_TRACES_SAMPLE_RATE=1.0
```

> **Note on Local Offline Mode:** If `MONGODB_URI` or `GEMMA_API_KEY` are not set, JusticeMitra automatically activates its verified local fallback using the embedded legal corpus in `data/legal_corpus.json`. The UI explicitly flags this status so developers can test the application without cloud credentials.

### 4. Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📥 Legal Document Ingestion & Atlas Indexing

To ingest statutory documents into MongoDB Atlas Vector Search:

```bash
# Ingest PDFs from data/ directory and update Atlas Vector Search
python scripts/ingest.py

# Optional: Export local JSON backup seed
python scripts/ingest.py --export-json data/legal_corpus.json
```

### Atlas Vector Search Index Definition
Ensure an Atlas Vector Search index named `vector_index` is created on `justicemitra_db.legal_chunks`:
```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "statute_title"
    },
    {
      "type": "filter",
      "path": "jurisdiction"
    }
  ]
}
```

---

## 🧪 Testing and Evaluation

The repository includes a comprehensive 32-test automated regression suite covering unit functionality, safety guardrails, and standardized legal query scenarios.

Run tests:
```bash
pytest -v
```

### Standardized Evaluation Scenarios (Q1 – Q5)
| ID | Inquiry | Statutory Target | Verification Criteria |
|---|---|---|---|
| **Q1** | *"Can a woman stay in her matrimonial home after filing a domestic violence complaint?"* | PWDVA 2005, Sec 17 & 19 | Recall@3; asserts right to reside in shared household without eviction. |
| **Q2** | *"My husband's family is demanding cash and vehicles before our wedding."* | Dowry Prohibition Act 1961, Sec 3 & 4; IPC 498A | Semantic retrieval on informal phrasing; identifies illegal dowry demand penalty. |
| **Q3** | *"Where can an indigent woman obtain a free advocate for court representation?"* | Legal Services Authorities Act 1987, Sec 12(c) | Verifies NALSA/DLSA free advocate entitlement for women. |
| **Q4** | *"What are the rules for filing a patent infringement claim under maritime law?"* | Out-of-Corpus Target | 100% Evidence-Refusal rate; prevents hallucinated legal conclusions. |
| **Q5** | *"Can a workplace harassment complaint be filed if the incident occurred five months ago?"* | POSH Act 2013, Sec 9 | Asserts standard 3-month limitation extendable up to 6 months upon sufficient cause. |

---

## ☁️ Deployment to Render

JusticeMitra includes a production-ready `render.yaml` specification:

1. Connect your GitHub repository to [Render](https://render.com/).
2. Select **Blueprints** and choose the `JusticeMitra` repository.
3. Render automatically detects `render.yaml`.
4. Configure the following environment secrets in the Render dashboard:
   - `MONGODB_URI`
   - `GEMMA_API_KEY`
   - `SENTRY_DSN`
5. Click **Deploy**. Render will install dependencies, launch Streamlit, and monitor `/_stcore/health`.

---

## 🚨 Emergency Helplines (India)

For immediate assistance, the following national hotlines are active:
- **Women Helpline (All India)**: `1091`
- **National Emergency Number**: `112`
- **National Commission for Women (NCW) 24/7**: `7827170170`
- **Domestic Violence Helpline**: `181`
- **NALSA Free Legal Aid Helpline**: `1516`

---

## ⚖️ License & Disclaimer

This project is licensed under the **MIT License**.

**Legal Disclaimer**: JusticeMitra provides legal rights information for educational and awareness purposes only. It does not constitute formal legal advice or attorney representation. Users requiring legal counsel should contact a qualified advocate or their District Legal Services Authority (DLSA).
