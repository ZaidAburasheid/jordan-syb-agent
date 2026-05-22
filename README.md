---
title: Jordan Statistical Yearbook Agent
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Jordan Statistical Yearbook Agent
### مساعد الكتاب الإحصائي السنوي الأردني

An AI-powered chat agent that answers questions about Jordan's official statistics in **Arabic and English**. Ask anything — population, GDP, births, crime, education, health — and get an instant answer with interactive charts and downloadable data.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange) ![LangChain](https://img.shields.io/badge/LangChain-latest-green) ![Gemini](https://img.shields.io/badge/Gemini-Flash-purple)

---

## What it does

- Understands questions in Arabic or English
- Searches across **53 statistical tables** covering demographics, economy, health, education, transport, crime, tourism, and more
- Generates and executes SQL automatically
- Builds up to **3 interactive Plotly charts** per answer
- Streams the response word-by-word
- Suggests follow-up questions after each answer
- Lets you download results as **CSV** and charts as **HTML**

---

## Demo

| Ask in Arabic | Ask in English |
|---|---|
| كم عدد المواليد في عمان عام 2020؟ | What is the GDP of Jordan in 2023? |
| اعطني اتجاه الطلاق على مر السنين | Compare births by governorate in 2022 |
| قارن الوفيات والمواليد في الأردن | Show employment trends by sex |

---

## Architecture

```
User question
      │
      ▼
[Input checks]  ←── length limit, injection detection, rate limit
      │
      ▼
[Semantic search]  ←── embed question → cosine similarity vs 53 table embeddings
      │
      ▼
[LangChain Agent]  ←── Gemini function calling
   ├── find_table   → semantic search (score threshold: 0.65)
   ├── get_schema   → column names + sample values
   ├── run_sql      → SELECT on read-only SQLite
   ├── filter_data  → pandas transformations (sandboxed)
   └── create_chart → Plotly figures (sandboxed)
      │
      ▼
[Streaming response]  ←── astream_events v2, word-by-word
      │
      ▼
[Gradio UI]  ←── chatbot + 3 charts + suggestions + downloads
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini (Flash / Flash Lite) via `langchain-google-genai` |
| Embeddings | `gemini-embedding-001` |
| Agent framework | LangChain `create_agent` + LangGraph `MemorySaver` |
| Database | SQLite (read-only, 53 tables) |
| UI | Gradio 4.44+ |
| Charts | Plotly |
| Observability | LangSmith |
| Session isolation | Python `contextvars` |

---

## Project Structure

```
├── app.ipynb              # Main notebook (9 code cells + markdown docs)
├── app.py                 # Flat Python script for deployment (HF Spaces / CLI)
├── requirements.txt       # Python dependencies
├── data/
│   ├── syb_database.db    # SQLite database (53 statistical tables + metadata)
│   ├── 3/                 # Excel source files — demographics
│   ├── 13/                # Excel source files — education
│   ├── 14/                # Excel source files — health
│   ├── 23/                # Excel source files — GDP/economy
│   └── ...                # Other chapters
├── data_preparation.ipynb # Builds syb_database.db from the Excel files
└── SYB_books/             # Original Jordan Statistical Yearbook PDFs/docs
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/jordan-syb-agent.git
cd jordan-syb-agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
LANGCHAIN_API_KEY=your_langsmith_key_here   # optional
```

Get your Gemini API key free at [aistudio.google.com](https://aistudio.google.com).

### 4. Run

**Notebook:**
```
Open app.ipynb → Kernel → Restart & Run All
```

**Script:**
```bash
python app.py
```

The Gradio UI opens at `http://localhost:7860`.

---

## Deployment on Hugging Face Spaces

### Via GitHub (recommended)

1. Push this repo to GitHub
2. Create a new Space at [huggingface.co](https://huggingface.co) → **Gradio** SDK
3. In Space Settings → Repository → link your GitHub repo
4. In Space Settings → Variables and Secrets → add:
   - `GOOGLE_API_KEY`
   - `LANGCHAIN_API_KEY` (optional)
5. Every `git push` auto-redeploys

> **Note:** Make sure `data/syb_database.db` is committed to the repo. GitHub handles files up to 100 MB natively; use Git LFS for anything larger.

---

## API Limits & Capacity

The agent uses **multiple Gemini models** and auto-switches on quota errors (429):

```
gemini-2.0-flash-lite → gemini-2.0-flash → gemini-2.5-flash → gemini-3.1-flash-lite
```

**Free tier capacity** (approximately):
- ~4 LLM calls per question
- ~580 total daily requests across all models
- **~23–38 users/day** unoptimized
- **~150 users/day** with question caching enabled

For higher traffic, upgrade to Gemini Tier 1 (paid) which gives ~10,000 RPD on Flash.

---

## Security

| Measure | Implementation |
|---------|---------------|
| Read-only database | SQLite `?mode=ro` URI — writes rejected at driver level |
| SQL injection | Comment stripping + keyword allowlist (SELECT only) |
| Code sandbox | Python code runs with restricted builtins — no file I/O, no OS access |
| Prompt injection | 12 bilingual regex patterns (Arabic + English) |
| Rate limiting | 10 requests per 60 seconds per server process |
| Input length | 500 character maximum |
| Session isolation | `contextvars` — each browser tab has isolated data |
| Error sanitization | Full traceback stays in console; users see a generic message |

---

## Data Coverage

53 statistical tables across 9 chapters of the Jordan Statistical Yearbook (2015–2023):

| Chapter | Topics |
|---------|--------|
| Chapter 3 | Births, deaths, marriages, divorces |
| Chapter 4 | Migration |
| Chapter 11 | Roads and transport |
| Chapter 13 | Education |
| Chapter 14 | Health |
| Chapter 15 | Tourism |
| Chapter 16 | Agriculture |
| Chapter 17 | Crime |
| Chapter 18 | Employment and population |
| Chapter 19 | Housing |
| Chapter 20 | Energy |
| Chapter 23 | GDP and national accounts |

All tables share universal Arabic column names (`قيمة المؤشر`, `سنة فترة القياس`, `المحافظة`, `الجنس`) making cross-table queries straightforward.

---

## How the Agent Thinks

For a question like *"مقارنة المواليد بين المحافظات عام 2022"*:

```
1. Check DATA KNOWLEDGE → births = SYB_3_3 (skip find_table, save 1 API call)
2. run_sql → SELECT "المحافظة", SUM("قيمة المؤشر") FROM SYB_3_3
             WHERE "سنة فترة القياس" = 2022 GROUP BY "المحافظة"
3. create_chart → bar chart, governorates on x-axis, sorted descending
4. Stream final answer in Arabic with insight (highest/lowest governorate)
5. Generate 3 follow-up suggestions
```

Total: ~4 LLM API calls, ~3 seconds end-to-end.

---

## Monitoring & Observability

This project uses **LangSmith** to trace every agent run end-to-end.

### What gets traced automatically

| Signal | Details |
|--------|---------|
| LLM calls | Model used, prompt, response, token count, latency |
| Tool calls | `find_table`, `run_sql`, `filter_data`, `create_chart` — inputs and outputs |
| Agent steps | Full reasoning chain per question |
| Errors | Exceptions captured with full context |

### How to enable

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Get your API key from Settings → API Keys
3. Add to your `.env` (local) or Space secrets (HF):
   ```env
   LANGCHAIN_API_KEY=your_langsmith_key_here
   LANGCHAIN_TRACING_V2=true
   ```

That's it — every question the agent answers will appear in your LangSmith dashboard with a full trace. No code changes needed.

### What to watch in production

- **Latency per tool** — if `run_sql` is slow, the query is the bottleneck
- **Model fallbacks** — frequent fallbacks from Flash Lite → Flash means you're hitting quota
- **Tool error rate** — repeated `find_table` misses (score < 0.65) mean a gap in data coverage
- **Token usage** — helps forecast cost if you upgrade to a paid Gemini tier

> LangSmith is optional. The agent works without it — tracing is simply skipped if the env vars are not set.

---

## License

This project uses data from Jordan's Department of Statistics. The statistical data belongs to the Hashemite Kingdom of Jordan. Code is MIT licensed.
