"""FastAPI service for the AI JEE Study Agent."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.rag import JEEAgent

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["agent"] = JEEAgent()  # load model + index once at startup
    yield
    state.clear()


app = FastAPI(
    title="AI JEE Study Agent",
    description="Bilingual (Hindi/English) RAG study assistant for JEE preparation.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000, examples=["JEE Main का पैटर्न क्या है?"])


class Source(BaseModel):
    source: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    mode: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "llm_enabled": state["agent"].llm is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    return state["agent"].answer(question)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AI JEE Study Agent</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.4rem}textarea{width:100%;height:80px;padding:.6rem;font-size:1rem;box-sizing:border-box}
button{margin-top:.6rem;padding:.6rem 1.2rem;font-size:1rem;cursor:pointer}
#out{white-space:pre-wrap;margin-top:1.2rem;padding:1rem;background:#f4f4f6;border-radius:8px;display:none}
small{color:#666;display:block;margin-top:.8rem}
</style></head><body>
<h1>AI JEE Study Agent</h1>
<p>Ask in Hindi or English about JEE syllabus, formulas, exam pattern or strategy.</p>
<textarea id="q" placeholder="e.g. What is the marking scheme of JEE Main? / JEE Main में नेगेटिव मार्किंग कितनी है?"></textarea>
<button onclick="ask()">Ask</button>
<div id="out"></div>
<small>API docs: <a href="/docs">/docs</a></small>
<script>
async function ask(){
  const out=document.getElementById('out'); out.style.display='block'; out.textContent='Thinking...';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:document.getElementById('q').value})});
    const d=await r.json();
    if(!r.ok){out.textContent=JSON.stringify(d);return;}
    out.textContent=d.answer+(d.sources.length?'\\n\\nSources: '+[...new Set(d.sources.map(s=>s.source))].join(', '):'');
  }catch(e){out.textContent='Error: '+e;}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return PAGE
