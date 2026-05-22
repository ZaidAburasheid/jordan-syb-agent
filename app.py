# Jordan Statistical Yearbook Agent — HF Spaces / standalone entry point
# Run locally:  python app.py
# HF Spaces:    set GOOGLE_API_KEY (and LANGCHAIN_API_KEY) in Space secrets

from dotenv import load_dotenv
load_dotenv()

import os, re, uuid, time, asyncio, tempfile, traceback, pathlib, builtins as _builtins_mod
import contextvars as _cv, sqlite3
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import gradio as gr

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver

# ── Constants ─────────────────────────────────────────────────────────────────
DB_PATH = "data/syb_database.db"
BLUE    = "#2563eb"
PALETTE = ["#2563eb","#16a34a","#f59e0b","#dc2626","#7c3aed",
           "#0891b2","#db2777","#65a30d","#ea580c","#4f46e5","#0d9488","#b45309"]
FONT    = "Arial, Tahoma, sans-serif"

# ── Per-session isolated state (contextvars, async-safe) ──────────────────────
_active_sid = _cv.ContextVar("active_sid", default="__default__")
_sessions: dict = {}

def _sess() -> dict:
    sid = _active_sid.get()
    if sid not in _sessions:
        _sessions[sid] = {"df_history": deque(maxlen=3), "figs": [], "prev_temps": []}
    return _sessions[sid]

# ── Plotly theme ──────────────────────────────────────────────────────────────
pio.templates["syb"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, size=13, color="#1e293b"),
        plot_bgcolor="white", paper_bgcolor="white", colorway=PALETTE,
        title=dict(x=0.5, xanchor="center", font=dict(size=15, color="#0f172a", family=FONT), pad=dict(b=10)),
        xaxis=dict(showgrid=False, showline=True, linecolor="#e2e8f0", linewidth=1,
                   tickfont=dict(size=11, color="#64748b"), title_font=dict(size=12, color="#475569"),
                   zeroline=False, automargin=True),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, showline=False,
                   tickfont=dict(size=11, color="#64748b"), title_font=dict(size=12, color="#475569"),
                   zeroline=False, automargin=True, tickformat=",.0f"),
        hoverlabel=dict(bgcolor="white", bordercolor="#cbd5e1", font=dict(size=13, family=FONT, color="#1e293b")),
        margin=dict(l=60, r=50, t=70, b=60),
    )
)
pio.templates.default = "syb"

# ── Semantic table index ──────────────────────────────────────────────────────
def _make_embed_model():
    for name in ["models/gemini-embedding-001", "models/text-embedding-004", "text-embedding-004"]:
        try:
            m = GoogleGenerativeAIEmbeddings(model=name, google_api_key=os.getenv("GOOGLE_API_KEY"))
            m.embed_query("test")
            print(f"Embedding model: {name}")
            return m
        except Exception as e:
            print(f"  {name} failed: {e}")
    raise RuntimeError("No embedding model available.")

def _build_table_index(embed_model):
    import time
    conn = sqlite3.connect(DB_PATH)
    df_meta = pd.read_sql("SELECT sql_table, name_en, name_ar FROM metadata ORDER BY sql_table", conn)
    conn.close()
    texts = [f"{r.sql_table}: {r.name_en} | {r.name_ar}" for _, r in df_meta.iterrows()]
    print("Embedding table descriptions (once)...")
    for attempt in range(5):
        try:
            embs = embed_model.embed_documents(texts)
            return texts, np.array(embs, dtype="float32")
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"Rate limit hit, retrying in {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding quota exhausted after retries.")

_embed_model              = _make_embed_model()
_table_texts, _table_embs = _build_table_index(_embed_model)
print(f"Table index ready — {len(_table_texts)} tables.")

# ── LangSmith tracing ─────────────────────────────────────────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = "jordan-syb-agent"
    print("LangSmith tracing active.")

# ── Security helpers ──────────────────────────────────────────────────────────
_DB_URI = pathlib.Path(DB_PATH).resolve().as_uri() + "?mode=ro"

_SAFE_BUILTINS = {
    name: getattr(_builtins_mod, name)
    for name in [
        'abs','all','any','bool','dict','divmod','enumerate','filter','float','frozenset',
        'getattr','hasattr','hash','int','isinstance','issubclass','iter','len','list',
        'map','max','min','next','object','pow','print','range','repr','reversed','round',
        'set','slice','sorted','str','sum','tuple','type','zip',
        'True','False','None',
        'Exception','ValueError','TypeError','KeyError','IndexError','AttributeError',
    ]
    if hasattr(_builtins_mod, name)
}

def _validate_sql(query: str):
    s = re.sub(r"--[^\n]*", "", query)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    first = s.strip().split()[0].upper() if s.strip() else ""
    if first != "SELECT":
        return f"Error: only plain SELECT queries are allowed. Got '{first}'."
    return None

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def find_table(query: str) -> str:
    """Search for relevant statistical tables by topic. Returns top 3. Call ONLY when topic is NOT in DATA KNOWLEDGE."""
    q_emb = np.array(_embed_model.embed_query(query), dtype="float32")
    norms = (np.linalg.norm(_table_embs, axis=1) * np.linalg.norm(q_emb)).clip(1e-9)
    scores = (_table_embs @ q_emb) / norms
    top = np.argsort(scores)[-3:][::-1]
    if scores[top[0]] < 0.65:
        return "No relevant table found. This agent covers Jordan statistical data only."
    return "\n".join(f"- {_table_texts[i]}" for i in top)

@tool
def get_schema(sql_table: str) -> str:
    """Returns columns and sample values. Call ONLY to verify exact filter values."""
    conn = sqlite3.connect(_DB_URI, uri=True)
    df = pd.read_sql("SELECT columns_info, sample_values FROM metadata WHERE sql_table = ?",
                     conn, params=(sql_table,))
    conn.close()
    if df.empty:
        return f"Table '{sql_table}' not found."
    return f"Columns: {df.iloc[0]['columns_info']}\n\nSample values:\n{df.iloc[0]['sample_values']}"

@tool
def run_sql(query: str) -> str:
    """Executes a SELECT query. Result stored for charting and filtering."""
    err = _validate_sql(query)
    if err:
        return err
    try:
        conn = sqlite3.connect(_DB_URI, uri=True)
        df = pd.read_sql(query, conn)
        conn.close()
        if df.empty:
            return "Query returned no results. Try different filter values or a different table."
        _sess()["df_history"].append(df.copy())
        if len(df) > 20:
            return df.head(20).to_string(index=False) + f"\n\n(Showing first 20 of {len(df)} rows)"
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

@tool
def filter_data(pandas_code: str) -> str:
    """Filter/transform the most recent SQL result. Available: df, pd. Assign result to result_df."""
    s = _sess()
    if not s["df_history"]:
        return "No data. Call run_sql first."
    try:
        local = {"__builtins__": _SAFE_BUILTINS, "df": s["df_history"][-1].copy(), "pd": pd}
        exec(pandas_code, local)  # noqa: S102 — sandboxed via _SAFE_BUILTINS
        result_df = local.get("result_df")
        if result_df is None:
            return "Error: assign result to `result_df`."
        if result_df.empty:
            return "Filter returned no rows."
        s["df_history"].append(result_df.copy())
        return result_df.to_string(index=False)
    except Exception as e:
        return f"Filter error: {e}"

@tool
def create_chart(plotly_code: str) -> str:
    """Build a Plotly chart. Available: df, df_prev, df_old, go, pd, BLUE, PALETTE, FONT. Assign to fig."""
    s = _sess()
    if not s["df_history"]:
        return "No data. Call run_sql first."
    if len(s["figs"]) >= 3:
        return "Maximum 3 charts per answer reached."
    try:
        dh = s["df_history"]
        local = {
            "__builtins__": _SAFE_BUILTINS,
            "df":      dh[-1].copy(),
            "df_prev": dh[-2].copy() if len(dh) >= 2 else None,
            "df_old":  dh[-3].copy() if len(dh) >= 3 else None,
            "go": go, "pd": pd, "BLUE": BLUE, "PALETTE": PALETTE, "FONT": FONT,
        }
        exec(plotly_code, local)  # noqa: S102 — sandboxed via _SAFE_BUILTINS
        fig = local.get("fig")
        if fig is None:
            return "Error: no `fig` variable was created."
        s["figs"].append(fig)
        return f"Chart {len(s['figs'])} created."
    except Exception as e:
        return f"Chart error: {e}"

tools = [find_table, get_schema, run_sql, filter_data, create_chart]

# ── System prompt ─────────────────────────────────────────────────────────────
system_prompt = """You are an expert data analyst for the Jordan Statistical Yearbook (الكتاب الإحصائي السنوي الأردني).
You have 53 statistical tables covering demographics, economy, health, education, transport, crime, and more.

## UNIVERSAL COLUMN NAMES (same in ALL 53 tables)
- Value column  → "قيمة المؤشر"
- Year column   → "سنة فترة القياس"
- Measure name  → "اسم المؤشر" / "Measure Name"
- Governorate   → "المحافظة" / "Governorate"
- Sex           → "الجنس" / "Sex"  (ذكر/Male, أنثى/Female)

## TOOL USAGE
  find_table  → SKIP if topic matches DATA KNOWLEDGE.
  get_schema  → SKIP if columns are already known.
  run_sql     → Fetch data. Call multiple times for multi-table questions.
  filter_data → For follow-ups on existing data.
  create_chart → After run_sql/filter_data. Up to 3 charts per answer.

## AGGREGATION RULES
- Jordan total   → SUM("قيمة المؤشر") GROUP BY "سنة فترة القياس"
- By governorate → GROUP BY "المحافظة"

## DATA KNOWLEDGE
- births/مواليد → SYB_3_3  |  marriages/زواج → SYB_3_11
- divorce/طلاق  → SYB_3_17, SYB_3_19v2, SYB_3_20
- GDP/الناتج المحلي → SYB_23_1, SYB_23_3, SYB_23_6, SYB_23_10
- education/تعليم → SYB_13_x  |  health/صحة → SYB_14_x
- employment/توظيف → SYB_18_x  |  crime/جرائم → SYB_17_x
- roads/transport → SYB_11_x   |  tourism/سياحة → SYB_15_x
- population/سكان → SYB_18_3, SYB_3_3

## VISUALIZATION
After run_sql or filter_data, build a chart if data warrants it.
Always: fig.update_layout(template="syb", height=440, title="...")

## ANSWER FORMAT
- Reply in the SAME language the user used.
- Lead with the direct answer. Add insight: trend, highest, lowest.
"""

# ── Model manager ─────────────────────────────────────────────────────────────
MODELS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"]
memory = MemorySaver()

class ModelManager:
    def __init__(self, models, tools, system_prompt, checkpointer):
        self._models, self._tools = models, tools
        self._prompt, self._checkpointer = system_prompt, checkpointer
        self._idx = 0
        self._activate(0)

    def _activate(self, idx):
        name = self._models[idx]
        self.llm = ChatGoogleGenerativeAI(model=name, google_api_key=os.getenv("GOOGLE_API_KEY"),
                                          temperature=0, streaming=True)
        self.agent = create_react_agent(model=self.llm, tools=self._tools,
                                       state_modifier=self._prompt, checkpointer=self._checkpointer)
        self._idx = idx
        print(f"Active model: {name}")

    def switch_next(self):
        if self._idx < len(self._models) - 1:
            self._activate(self._idx + 1)
            return True
        return False

    @property
    def current_name(self): return self._models[self._idx]

model_mgr = ModelManager(MODELS, tools, system_prompt, memory)
print("Agent ready.")

# ── Gradio helpers ────────────────────────────────────────────────────────────
_MAX_RETRIES = 3
RATE_LIMIT_MAX, RATE_LIMIT_WINDOW, MAX_INPUT_LEN = 10, 60, 500
_request_times = deque()

_INJECTION_RE = re.compile("|".join([
    r"ignore\s+(all\s+)?(previous|prior|above|your)?\s*(instructions?|rules?|prompt|system)",
    r"forget\s+(all\s+)?(previous|prior|above|your)?\s*(instructions?|rules?|prompt|system)",
    r"disregard\s+(all\s+)?(previous|prior|above)?\s*(instructions?|rules?|prompt|system)",
    r"you\s+are\s+now\s+(a\s+)?(different|new|another)",
    r"act\s+as\s+(a\s+)?(?!data|analyst|assistant\s+for\s+jordan)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"your\s+(new\s+)?(system\s+prompt|instructions?\s+are|role\s+is)",
    r"jailbreak|dan\s+mode|developer\s+mode",
    r"تجاهل.{0,20}(التعليمات|النظام|الأوامر)",
    r"انسَ?.{0,20}(التعليمات|السابقة|الأوامر)",
    r"أنت\s+الآن\s+مساعد\s+مختلف", r"دورك\s+الجديد",
]), re.IGNORECASE)

def _check_injection(text): return bool(_INJECTION_RE.search(text))

def extract_text(content):
    if isinstance(content, str): return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return ""

def _check_rate_limit():
    now = time.time()
    while _request_times and now - _request_times[0] > RATE_LIMIT_WINDOW:
        _request_times.popleft()
    if len(_request_times) >= RATE_LIMIT_MAX:
        return False, int(RATE_LIMIT_WINDOW - (now - _request_times[0])) + 1
    _request_times.append(now)
    return True, 0

def _save_downloads():
    s = _sess()
    for path in s["prev_temps"]:
        try: os.unlink(path)
        except Exception: pass
    s["prev_temps"].clear()
    data_path, chart_paths = None, [None, None, None]
    if s["df_history"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="syb_data_")
        s["df_history"][-1].to_csv(tmp.name, index=False, encoding="utf-8-sig")
        data_path = tmp.name; s["prev_temps"].append(tmp.name)
    for i, fig in enumerate(s["figs"][:3]):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix=f"syb_chart{i+1}_")
        fig.write_html(tmp.name)
        chart_paths[i] = tmp.name; s["prev_temps"].append(tmp.name)
    return data_path, chart_paths

async def _generate_suggestions(user_msg, ai_response):
    try:
        prompt = (f'User asked: "{user_msg}"\nResponse: "{ai_response[:300]}"\n\n'
                  "Suggest 3 short follow-up questions about Jordan statistics in the SAME language. "
                  "One per line. No bullets. Under 12 words each.")
        result = await model_mgr.llm.ainvoke(prompt)
        lines = [l.strip().lstrip("•-–0123456789.) ") for l in result.content.strip().split("\n") if l.strip()]
        return (lines + ["", "", ""])[:3]
    except Exception:
        return ["", "", ""]

def _interim(cv):
    return (cv, gr.update(), gr.update(), gr.update(), "", "", gr.update(visible=False, value=""),
            ["", "", ""], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

async def respond(user_msg, history, session_id):
    if not user_msg.strip(): yield _interim(history); return
    if len(user_msg) > MAX_INPUT_LEN:
        msg = f"⚠️ Message too long ({len(user_msg)} chars). Max is {MAX_INPUT_LEN}."
        yield _interim(history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": msg}]); return
    if _check_injection(user_msg):
        msg = "⛔ Message rejected. Please ask a question about Jordan statistics."
        yield _interim(history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": msg}]); return
    allowed, wait = _check_rate_limit()
    if not allowed:
        msg = f"⏳ Too many requests. Please wait {wait} seconds."
        yield _interim(history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": msg}]); return

    _active_sid.set(session_id)
    _sess()["figs"].clear()
    history = history + [{"role": "user", "content": user_msg}]
    yield _interim(history)

    config, partial, retries = {"configurable": {"thread_id": session_id}}, "", 0
    try:
        while True:
            try:
                async for event in model_mgr.agent.astream_events(
                    {"messages": [("human", user_msg)]}, config=config, version="v2"
                ):
                    if event["event"] == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk and chunk.content:
                            text = extract_text(chunk.content)
                            if text:
                                partial += text
                                yield _interim(history + [{"role": "assistant", "content": partial}])
                break
            except Exception as e:
                err, etype = str(e), type(e).__name__
                if ("429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()) and model_mgr.switch_next():
                    partial = ""; retries = 0; continue
                if ("RemoteProtocolError" in etype or "Server disconnected" in err) and retries < _MAX_RETRIES:
                    retries += 1; await asyncio.sleep(2 * retries); partial = ""; continue
                raise

        if not partial: partial = "لم يتم الحصول على إجابة."
        sugs = await _generate_suggestions(user_msg, partial)
        s = _sess()
        fig1 = s["figs"][0] if len(s["figs"]) > 0 else None
        fig2 = s["figs"][1] if len(s["figs"]) > 1 else None
        fig3 = s["figs"][2] if len(s["figs"]) > 2 else None
        data_path, chart_paths = _save_downloads()
        yield (
            history + [{"role": "assistant", "content": partial}],
            gr.update(value=fig1, visible=fig1 is not None),
            gr.update(value=fig2, visible=fig2 is not None),
            gr.update(value=fig3, visible=fig3 is not None),
            f"[{model_mgr.current_name}]", "", gr.update(visible=False, value=""), sugs,
            gr.update(value=sugs[0], visible=bool(sugs[0])),
            gr.update(value=sugs[1], visible=bool(sugs[1])),
            gr.update(value=sugs[2], visible=bool(sugs[2])),
            gr.update(value=data_path,      visible=data_path is not None),
            gr.update(value=chart_paths[0], visible=chart_paths[0] is not None),
            gr.update(value=chart_paths[1], visible=chart_paths[1] is not None),
            gr.update(value=chart_paths[2], visible=chart_paths[2] is not None),
        )
    except Exception:
        print(traceback.format_exc())
        user_err = "⚠️ An unexpected error occurred. Please try again."
        yield (history, gr.update(), gr.update(), gr.update(), "", "",
               gr.update(visible=True, value=user_err), ["", "", ""],
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

def _use_suggestion(idx):
    async def handler(sugs, history, session_id):
        q = sugs[idx] if sugs and len(sugs) > idx else ""
        if not q: yield _interim(history); return
        async for chunk in respond(q, history, session_id): yield chunk
    return handler

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="مساعد الكتاب الإحصائي السنوي") as demo:
    gr.Markdown("# مساعد الكتاب الإحصائي السنوي\n"
                "اسأل عن أي إحصائية من 53 جدول | Ask about any statistic from 53 tables")
    _sugs = gr.State(["", "", ""])
    session_id = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="المحادثة", type="messages")
            msg_box = gr.Textbox(placeholder="اكتب سؤالك هنا...", label="سؤالك", lines=2, max_lines=6)
            with gr.Row():
                send_btn  = gr.Button("إرسال", variant="primary")
                clear_btn = gr.Button("مسح")
            gr.Markdown("**اقتراحات للمتابعة | Follow-up suggestions**")
            with gr.Row():
                sug1 = gr.Button("", visible=False, size="sm", variant="secondary")
                sug2 = gr.Button("", visible=False, size="sm", variant="secondary")
                sug3 = gr.Button("", visible=False, size="sm", variant="secondary")
            gr.Markdown("**تحميل النتائج | Download**")
            with gr.Row():
                dl_data   = gr.DownloadButton("⬇ البيانات CSV",  visible=False, size="sm")
                dl_chart1 = gr.DownloadButton("⬇ الرسم 1 HTML", visible=False, size="sm")
                dl_chart2 = gr.DownloadButton("⬇ الرسم 2 HTML", visible=False, size="sm")
                dl_chart3 = gr.DownloadButton("⬇ الرسم 3 HTML", visible=False, size="sm")
        with gr.Column(scale=1):
            plot1     = gr.Plot(label="الرسم البياني 1")
            plot2     = gr.Plot(label="الرسم البياني 2", visible=False)
            plot3     = gr.Plot(label="الرسم البياني 3", visible=False)
            token_out = gr.Textbox(label="النموذج المستخدم", interactive=False)

    error_box = gr.Textbox(label="Error Details", visible=False, interactive=False, lines=4)
    gr.Examples(examples=[
        "كم عدد المواليد الذكور في عمان عام 2020؟",
        "قارن عدد المواليد في كل محافظة عام 2022",
        "What is the GDP of Jordan in 2023?",
        "اعطني اتجاه الطلاق في الاردن على مر السنين",
    ], inputs=msg_box)

    _out = [chatbot, plot1, plot2, plot3, token_out, msg_box, error_box, _sugs,
            sug1, sug2, sug3, dl_data, dl_chart1, dl_chart2, dl_chart3]

    demo.load(lambda: str(uuid.uuid4()), outputs=[session_id])
    send_btn.click(respond, [msg_box, chatbot, session_id], _out)
    msg_box.submit(respond, [msg_box, chatbot, session_id], _out)
    clear_btn.click(
        lambda: ([], None, None, None, "", "", gr.update(visible=False, value=""), ["", "", ""],
                 gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                 gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                 gr.update(visible=False), str(uuid.uuid4())),
        outputs=_out + [session_id],
    )
    sug1.click(_use_suggestion(0), [_sugs, chatbot, session_id], _out)
    sug2.click(_use_suggestion(1), [_sugs, chatbot, session_id], _out)
    sug3.click(_use_suggestion(2), [_sugs, chatbot, session_id], _out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
