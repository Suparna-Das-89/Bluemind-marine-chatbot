# BlueMind: The Marine Chatbot — Groq JSON edition (no persona, no refine box)

from typing import Dict, List, Optional
import json, textwrap, re

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

import os

def get_groq_key() -> Optional[str]:
    # Try secrets and env, multiple casings, to avoid typos
    return (
        (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
        or os.getenv("GROQ_API_KEY")
        or os.getenv("groq_api_key")
        or (st.secrets.get("groq_api_key") if hasattr(st, "secrets") else None)
    )

def mask(s: Optional[str]) -> str:
    if not s:
        return "False"
    s = s.strip()
    return f"True (len={len(s)}, startswith={s[:4]}•••)"


# Optional local fallback model (no key). If not present, we just skip it.
try:
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# ---------------------------
# HTTP helper with User-Agent
# ---------------------------
@st.cache_data(show_spinner=False)
def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    try:
        if headers is None:
            headers = {"User-Agent": "BlueMind/0.3 (+https://github.com/your-username/bluemind-marine-chatbot)"}
        else:
            headers.setdefault("User-Agent", "BlueMind/0.3 (+https://github.com/your-username/bluemind-marine-chatbot)")
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Request failed: {e}")
        return None

# ---------------------------
# Wikipedia wrappers
# ---------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"

@st.cache_data(show_spinner=False)
def wiki_search(query: str, limit: int = 5) -> List[Dict]:
    params = {
        "action": "query", "list": "search", "srsearch": query,
        "format": "json", "srlimit": limit, "utf8": 1,
    }
    data = http_get_json(WIKI_API, params)
    if not data:
        return []
    return data.get("query", {}).get("search", [])

@st.cache_data(show_spinner=False)
def wiki_extract(title: str, sentences: int = 6) -> str:
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1, "exsentences": sentences,
        "titles": title, "format": "json", "utf8": 1,
    }
    data = http_get_json(WIKI_API, params)
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""

@st.cache_data(show_spinner=False)
def wiki_page_image(title: str) -> Optional[str]:
    params = {
        "action": "query", "prop": "pageimages", "format": "json",
        "piprop": "original", "titles": title, "utf8": 1,
    }
    data = http_get_json(WIKI_API, params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    original = page.get("original")
    return original.get("source") if original else None

# ---------------------------
# Ocean data (Open-Meteo) — fixed JSON array for hourly 
# ---------------------------

# Marine API (waves only)
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"
# Weather API (for wind speed)
OPEN_METEO_WEATHER = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(show_spinner=False)
def fetch_marine_timeseries(lat: float, lon: float, start: str, end: str) -> Optional[pd.DataFrame]:
    try:
        # Validate dates
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        if end_dt < start_dt:
            end_dt = start_dt + pd.Timedelta(days=1)

        # --- 1) Marine API request (wave data only) ---
        marine_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join([
                "wave_height",
                "wave_direction",
                "wind_wave_height",
                "wind_wave_direction",
                "swell_wave_height",
                "swell_wave_direction",
            ]),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "timezone": "UTC",
        }

        marine_data = http_get_json(OPEN_METEO_MARINE, marine_params)

        if not marine_data or "hourly" not in marine_data:
            st.warning("No marine data returned. Try different dates or coordinates.")
            return None

        df_marine = pd.DataFrame(marine_data["hourly"])
        df_marine["time"] = pd.to_datetime(df_marine["time"])

        # --- 2) Weather API request (wind speed only) ---
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "windspeed_10m",  # <-- correct variable name here
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "timezone": "UTC",
        }

        weather_data = http_get_json(OPEN_METEO_WEATHER, weather_params)

        df_weather = pd.DataFrame(weather_data["hourly"])
        df_weather["time"] = pd.to_datetime(df_weather["time"])

        # --- 3) Merge both DataFrames ---
        df = pd.merge(df_marine, df_weather, on="time", how="outer")

        return df

    except Exception as e:
        st.warning(f"Request failed: {e}")
        return None





# ---------------------------
# Groq (primary) + HF fallback
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_local_llm():
    if not _HF_AVAILABLE:
        return None
    try:
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception:
        return None

import os

def get_groq_key() -> Optional[str]:
    # Try multiple places/names to avoid simple typos/misconfig
    return (
        (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
        or os.getenv("GROQ_API_KEY")
        or os.getenv("groq_api_key")
        or (st.secrets.get("groq_api_key") if hasattr(st, "secrets") else None)
    )

def ask_llm(prompt: str) -> str:
    """Prefer Groq; fallback to local tiny model if no key."""
    groq_key = get_groq_key()
    if groq_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_key.strip()}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "llama-3.3-70b-versatile",  # updated model id
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            r = requests.post(url, json=body, headers=headers, timeout=45)
            if r.status_code == 401:
                # Surface a clear hint
                raise RuntimeError("Groq 401 Unauthorized — key missing/invalid or wrong app secrets.")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            st.warning(f"Groq call failed, using local fallback. ({e})")

    # Local fallback (optional)
    llm = load_local_llm()
    if llm is None:
        return ""
    out = llm(prompt, max_new_tokens=420)
    return out[0]["generated_text"].strip()


    # Local fallback (optional)
    llm = load_local_llm()
    if llm is None:
        return ""
    out = llm(prompt, max_new_tokens=420)
    return out[0]["generated_text"].strip()


# ---------------------------
# Intent routing
# ---------------------------
THREAT_PAGES = [
    "Overfishing","Bycatch","Marine pollution","Plastic pollution",
    "Ocean acidification","Climate change and oceans","Coral bleaching",
    "Habitat destruction","Noise pollution","Invasive species",
]
MARINE_ENTITY_WORDS = (
    "fish","shark","whale","dolphin","porpoise","seal","sea lion","manatee","dugong",
    "turtle","ray","skate","octopus","squid","cuttlefish","jellyfish","crab","lobster",
    "shrimp","krill","eel","anchovy","tuna","salmon","cod","mola","sunfish","basking"
)
DANGER_KEYWORDS = (
    "danger","threat","endangered","risk","dying","decline",
    "why are ocean animals","why are marine animals","why are sea animals",
)

def classify_intent(question: str) -> str:
    q = (question or "").lower().strip()
    if any(k in q for k in DANGER_KEYWORDS):
        return "threats"
    if any(w in q for w in ("largest","biggest","fastest","deepest","heaviest","longest")):
        return "entity"
    if any(w in q for w in MARINE_ENTITY_WORDS):
        return "entity"
    return "topic"

def route_titles(question: str, intent: str) -> List[str]:
    q = (question or "").strip()
    ql = q.lower()

    if intent == "threats":
        return THREAT_PAGES[:5]

    if intent == "entity":
        if "largest fish" in ql or "biggest fish" in ql:
            return ["Whale shark", "List of largest fish"]
        if "largest shark" in ql or "biggest shark" in ql:
            return ["Whale shark", "List of largest sharks"]
        if "largest mammal" in ql or "biggest mammal" in ql:
            return ["Blue whale"]
        hits = wiki_search(q, limit=6)
        titles = [h.get("title") for h in hits if h.get("title")]
        keep = []
        for t in titles:
            tl = t.lower()
            if any(w in tl for w in MARINE_ENTITY_WORDS) or "list of" in tl:
                keep.append(t)
        return keep[:4] or titles[:3]

    # topic default
    hits = wiki_search(q, limit=6)
    titles = [h.get("title") for h in hits if h.get("title")]
    words = [w for w in ql.split() if len(w) >= 4]
    def good(t):
        tl = t.lower()
        return any(w in tl for w in words) or len(words) == 0
    filtered = [t for t in titles if good(t)]
    return filtered[:3] or titles[:3]

# ---------------------------
# Build context from titles
# ---------------------------
def build_context(titles: List[str], sents: int = 8) -> str:
    extracts = []
    for t in titles:
        if not t:
            continue
        extracts.append(f"# {t}\n" + (wiki_extract(t, sentences=sents) or ""))
    return "\n\n".join(extracts) if extracts else "(No external context retrieved.)"

# ---------------------------
# JSON prompts per intent (fixed professional tone)
# ---------------------------
def json_prompt_threats(question: str, context: str) -> str:
    schema = textwrap.dedent("""
    {
      "type":"object",
      "properties":{
        "direct_answer":{"type":"string"},
        "causes":{"type":"array","items":{"type":"object","properties":{
          "cause":{"type":"string"},
          "mechanism":{"type":"string"},
          "example":{"type":"string"}
        },"required":["cause","mechanism"]}},
        "actions":{"type":"array","items":{"type":"string"}},
        "sources":{"type":"array","items":{"type":"string"}}
      },
      "required":["direct_answer","causes","actions","sources"]
    }
    """).strip()
    return f"""
You are BlueMind, an ocean expert. Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Return a JSON object that matches this JSON Schema exactly (no extra fields, no prose outside JSON).
Schema:
{schema}

Guidelines:
- direct_answer: 1–2 sentences.
- causes: up to 5; each one line mechanism; include example if context supports it.
- actions: 3–5 concise actions.
- sources: page titles used.
"""

def json_prompt_entity(question: str, context: str) -> str:
    schema = textwrap.dedent("""
    {
      "type":"object",
      "properties":{
        "direct_answer":{"type":"string"},
        "facts":{"type":"array","items":{"type":"string"}},
        "note":{"type":"string"},
        "sources":{"type":"array","items":{"type":"string"}}
      },
      "required":["direct_answer","facts","sources"]
    }
    """).strip()
    return f"""
You are BlueMind, a precise marine naturalist. Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Return a JSON object per this Schema (no prose outside JSON):
{schema}

Guidelines:
- direct_answer: 1 sentence answering the superlative (e.g., "The whale shark ...").
- facts: 3–4 bullets (size range, distribution/habitat, diet/behavior).
- note: optional comparison/uncertainty if applicable.
- sources: page titles used.
"""

def json_prompt_topic(question: str, context: str) -> str:
    schema = textwrap.dedent("""
    {
      "type":"object",
      "properties":{
        "what":{"type":"string"},
        "how":{"type":"array","items":{"type":"string"}},
        "why":{"type":"array","items":{"type":"string"}},
        "watch":{"type":"array","items":{"type":"string"}},
        "sources":{"type":"array","items":{"type":"string"}}
      },
      "required":["what","how","why","watch","sources"]
    }
    """).strip()
    return f"""
You are BlueMind, an ocean explainer. Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

TASK:
Return a JSON object per this Schema (no prose outside JSON):
{schema}

Guidelines:
- what: 1–2 sentences definition.
- how: 3 bullets (process → effect).
- why: 2 bullets (ecosystems/people).
- watch: 2 bullets (indicators/events).
- sources: page titles used.
"""

# ---------------------------
# JSON parsing & fallbacks
# ---------------------------
def try_parse_json(s: str) -> Optional[dict]:
    if not s:
        return None
    m = re.search(r"\{.*\}\s*$", s, flags=re.S)
    candidate = m.group(0) if m else s.strip()
    try:
        return json.loads(candidate)
    except Exception:
        return None

def fallback_json(intent: str, titles: List[str]) -> dict:
    if intent == "threats":
        mech = {
            "Overfishing": "Stocks depleted; food-web shifts; cod collapse (1990s).",
            "Bycatch": "Non-target mortality in nets/longlines; turtles & seabirds.",
            "Marine pollution": "Nutrients/oil/sewage → hypoxia & disease.",
            "Plastic pollution": "Ingestion/entanglement → injury & starvation.",
            "Ocean acidification": "Lower pH → fewer carbonate ions → weaker shells.",
            "Climate change and oceans": "Warming/stratification → heatwaves & shifts.",
            "Coral bleaching": "Heat stress → symbionts expelled → mass bleaching.",
        }
        picked = [t for t in titles if t in mech] or titles[:5]
        return {
            "direct_answer": "Marine animals face multiple human-driven threats that reduce habitat quality, alter food webs, and increase mortality.",
            "causes": [{"cause": t, "mechanism": mech.get(t, "See context.")} for t in picked[:5]],
            "actions": ["Sustainable catch limits", "Reduce bycatch", "Cut nutrient/plastic pollution", "Protect habitats", "Lower GHG emissions"],
            "sources": picked
        }
    if intent == "entity":
        if "Whale shark" in titles or "List of largest fish" in titles:
            return {
                "direct_answer": "The whale shark (Rhincodon typus) is the largest living fish.",
                "facts": [
                    "Typical size ~10–12 m; occasionally reported larger.",
                    "Tropical & warm-temperate oceans; migratory.",
                    "Filter-feeding on plankton and small fishes."
                ],
                "note": "The basking shark is also very large but typically smaller.",
                "sources": ["Whale shark","List of largest fish"]
            }
        return {
            "direct_answer": "Based on context, the superlative likely refers to a well-known marine species.",
            "facts": ["See sources for top candidates."],
            "note": "Provide a more specific phrase if possible.",
            "sources": titles[:3]
        }
    # topic
    return {
        "what": "See sources for definition.",
        "how": ["Mechanism 1","Mechanism 2","Mechanism 3"],
        "why": ["Ecosystem impact","Human impact"],
        "watch": ["Indicator 1","Indicator 2"],
        "sources": titles[:3]
    }

def render_answer(intent: str, data: dict) -> str:
    if intent == "threats":
        lines = []
        lines.append(f"1) Direct answer: {data.get('direct_answer','')}")
        causes = data.get("causes", [])
        if causes:
            lines.append("\n2) Top causes:")
            for c in causes[:5]:
                cause = c.get("cause","")
                mech = c.get("mechanism","")
                ex = c.get("example")
                item = f"- {cause} → {mech}" + (f" → {ex}" if ex else "")
                lines.append(item)
        actions = data.get("actions", [])
        if actions:
            lines.append("\n3) What can help: " + "; ".join(actions))
        sources = data.get("sources", [])
        lines.append("\n4) Sources: " + ", ".join(sources))
        return "\n".join(lines)

    if intent == "entity":
        lines = []
        lines.append(f"1) Direct answer: {data.get('direct_answer','')}")
        facts = data.get("facts", [])
        if facts:
            lines.append("\n2) Key facts:")
            for f in facts[:5]:
                lines.append(f"- {f}")
        note = data.get("note")
        if note:
            lines.append(f"\n3) Context note: {note}")
        sources = data.get("sources", [])
        lines.append("\n4) Sources: " + ", ".join(sources))
        return "\n".join(lines)

    # topic
    lines = []
    lines.append(f"1) What it is: {data.get('what','')}")
    how = data.get("how", [])
    if how:
        lines.append("\n2) How it works:")
        for h in how[:5]:
            lines.append(f"- {h}")
    why = data.get("why", [])
    if why:
        lines.append("\n3) Why it matters:")
        for w in why[:4]:
            lines.append(f"- {w}")
    watch = data.get("watch", [])
    if watch:
        lines.append("\n4) What scientists watch:")
        for w in watch[:4]:
            lines.append(f"- {w}")
    sources = data.get("sources", [])
    lines.append("\n5) Sources: " + ", ".join(sources))
    return "\n".join(lines)

# ---------------------------
# Orchestrator
# ---------------------------
def build_json_prompt(intent: str, question: str, context: str) -> str:
    if intent == "threats":
        return json_prompt_threats(question, context)
    if intent == "entity":
        return json_prompt_entity(question, context)
    return json_prompt_topic(question, context)

def generate_answer(question: str) -> Dict:
    intent = classify_intent(question)
    titles = route_titles(question, intent)
    context = build_context(titles, sents=8)

    prompt = build_json_prompt(intent, question, context)
    raw = ask_llm(prompt)
    data = try_parse_json(raw)

    if data is None:
        fixed = ask_llm(f"Return VALID JSON only. Fix this to valid JSON without adding content:\n{raw}")
        data = try_parse_json(fixed)

    if data is None:
        data = fallback_json(intent, titles)

    answer_md = render_answer(intent, data)
    return {"answer": answer_md, "sources": titles, "raw_context": context}

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="BlueMind – Marine Chatbot", page_icon="🌊", layout="wide")
st.title("🌊 BlueMind – The Marine Chatbot")
st.caption("Ask about oceans, climate, species, currents, exploration, and conservation.")

with st.sidebar:
    st.header("Controls")
    # Persona removed; fixed professional tone
    st.markdown("---")
    st.subheader("Ocean Data Query")
    lat = st.number_input("Latitude", value=0.0, step=0.5, format="%.3f")
    lon = st.number_input("Longitude", value=0.0, step=0.5, format="%.3f")
    start_date = st.text_input("Start date (YYYY-MM-DD)", value=pd.Timestamp.utcnow().date().isoformat())
    end_date = st.text_input("End date (YYYY-MM-DD)", value=pd.Timestamp.utcnow().date().isoformat())
    st.markdown("---")
    st.info("Groq enabled if GROQ_API_KEY is set in Secrets. Otherwise tries a tiny local model.")

chat_tab, data_tab, species_tab, map_tab = st.tabs(["💬 Chat", "🌡 Ocean Data", "🐠 Species", "🗺 Map"])

with chat_tab:
    st.subheader("Ask the Ocean")
    with st.form("chat_form"):
        q = st.text_input("Your question about the ocean:", placeholder="e.g., biggest fish / coral bleaching / why are ocean animals in danger?")
        submitted = st.form_submit_button("Answer")

    if submitted and q.strip():
        with st.spinner("Thinking with tides..."):
            result = generate_answer(q)
        st.markdown("### Answer")
        st.write(result["answer"])
        if result.get("sources"):
            st.markdown("### Sources (Wikipedia)")
            st.write(", ".join(result["sources"]))
        with st.expander("Show retrieved context"):
            st.code(result.get("raw_context") or "(none)")

with data_tab:
    st.subheader("Marine Conditions (Hourly)")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Fetch timeseries"):
            df = fetch_marine_timeseries(lat, lon, start_date, end_date)
            if df is None or df.empty:
                st.warning("No data returned. Try different dates or coordinates.")
            else:
                st.success(f"Fetched {len(df)} hourly points.")
                st.dataframe(df.head(24))
                if "wave_height" in df.columns:
                    fig = px.line(df, x="time", y="wave_height", title="Significant Wave Height (m)")
                    st.plotly_chart(fig, use_container_width=True)
                if "wind_speed_10m" in df.columns:
                    fig2 = px.line(df, x="time", y="wind_speed_10m", title="Wind Speed 10m (m/s)")
                    st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.info("Data source: Open-Meteo Marine (free). Add NOAA ERDDAP/NDBC later for richer data.")

with species_tab:
    st.subheader("Marine Species Guide")
    species_query = st.text_input("Species or topic (e.g., 'Blue whale', 'Giant squid')")
    if st.button("Lookup species"):
        if not species_query.strip():
            st.warning("Enter a species or topic.")
        else:
            pages = wiki_search(species_query, limit=1)
            if not pages:
                st.warning("No result found.")
            else:
                title = pages[0].get("title")
                if title:
                    img = wiki_page_image(title)
                    if img:
                        st.image(img, caption=title, use_column_width=True)
                    extract = wiki_extract(title, sentences=10)
                    st.markdown(f"### {title}")
                    st.write(extract or "(no extract)")

with map_tab:
    st.subheader("Global Ocean Map (scaffold)")
    st.write("This is a placeholder world map; add layers like SST, chlorophyll, or buoys.")
    df_ocean = pd.DataFrame({
        "lat": [0, 30, -30, 15, -15],
        "lon": [-140, 20, 60, -10, 120],
        "label": ["Pacific", "Atlantic", "Indian", "Atlantic N", "Pacific W"],
    })
    figm = px.scatter_geo(df_ocean, lat="lat", lon="lon", hover_name="label", projection="natural earth")
    figm.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(figm, use_container_width=True)

st.markdown("---")
st.caption("BlueMind • Groq LLaMA-3.1-70B JSON answers • Wikipedia + Open-Meteo.")
