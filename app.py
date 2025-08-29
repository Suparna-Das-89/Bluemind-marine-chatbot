# BlueMind: The Marine Chatbot (single-file Streamlit app)

from typing import Dict, List, Optional
import textwrap

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# Try to load a small local model via transformers (no API key needed)
try:
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# ---------------------------
# Utility: Caching web fetches
# ---------------------------
@st.cache_data(show_spinner=False)
def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    """GET JSON with a proper User-Agent so Wikipedia/Open APIs don't 403."""
    try:
        if headers is None:
            headers = {"User-Agent": "BlueMind/0.1 (+https://github.com/your-username/bluemind-marine-chatbot)"}
        else:
            headers.setdefault("User-Agent", "BlueMind/0.1 (+https://github.com/your-username/bluemind-marine-chatbot)")
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Request failed: {e}")
        return None

# -----------------------------------
# Retrieval: Simple Wikipedia wrappers
# -----------------------------------
WIKI_API = "https://en.wikipedia.org/w/api.php"

@st.cache_data(show_spinner=False)
def wiki_search(query: str, limit: int = 5) -> List[Dict]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
        "utf8": 1,
    }
    data = http_get_json(WIKI_API, params)
    if not data:
        return []
    return data.get("query", {}).get("search", [])

@st.cache_data(show_spinner=False)
def wiki_extract(title: str, sentences: int = 6) -> str:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsentences": sentences,
        "titles": title,
        "format": "json",
        "utf8": 1,
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
        "action": "query",
        "prop": "pageimages",
        "format": "json",
        "piprop": "original",
        "titles": title,
        "utf8": 1,
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

# -------------------------------------
# Ocean Data: Open-Meteo marine (key-free)
# -------------------------------------
OPEN_METEO_MARINE = "https://marine-api.open-meteo.com/v1/marine"

@st.cache_data(show_spinner=False)
def fetch_marine_timeseries(lat: float, lon: float, start: str, end: str) -> Optional[pd.DataFrame]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wave_height",
            "wave_direction",
            "wind_wave_height",
            "wind_wave_direction",
            "swell_wave_height",
            "swell_wave_direction",
            "wind_speed_10m",
        ],
        "start_date": start,  # YYYY-MM-DD
        "end_date": end,
        "timezone": "UTC",
    }
    data = http_get_json(OPEN_METEO_MARINE, params)
    if not data or "hourly" not in data:
        return None
    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    return df

# -----------------------------
# LLM: Explain & answer builder
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_llm():
    """Load a small local text2text model; optional and key-free."""
    if not _HF_AVAILABLE:
        return None
    try:
        # Flan-T5 is lightweight and okay for demos (quality is modest)
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception as e:
        st.warning(f"Could not load local HF model: {e}. Tip: add 'torch' to requirements.txt or switch to an API LLM later.")
        return None

# -----------------------------
# Intent routing & topic banks
# -----------------------------
THREAT_PAGES = [
    "Overfishing", "Bycatch", "Marine pollution", "Plastic pollution",
    "Ocean acidification", "Climate change and oceans", "Coral bleaching",
    "Habitat destruction", "Noise pollution", "Invasive species",
]
MARINE_ENTITY_WORDS = (
    "fish","shark","whale","dolphin","porpoise","seal","sea lion","manatee","dugong",
    "turtle","ray","skate","octopus","squid","cuttlefish","jellyfish","crab","lobster",
    "shrimp","krill","eel","anchovy","tuna","salmon","cod","mola","sunfish"
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
    if intent == "threats":
        return THREAT_PAGES[:5]

    # Heuristics for common superlative questions
    ql = q.lower()
    if intent == "entity":
        if "largest fish" in ql or "biggest fish" in ql:
            return ["Whale shark", "List of largest fish"]
        if "largest shark" in ql or "biggest shark" in ql:
            return ["Whale shark", "List of largest sharks"]
        if "largest mammal" in ql or "biggest mammal" in ql:
            return ["Blue whale"]
        # Otherwise, use search and filter to marine-y titles
        hits = wiki_search(q, limit=5)
        titles = [h.get("title") for h in hits if h.get("title")]
        keep = []
        for t in titles:
            tl = t.lower()
            if any(w in tl for w in MARINE_ENTITY_WORDS):
                keep.append(t)
        return keep[:3] or titles[:3]

    # topic (default): use search; prefer exact-ish matches
    hits = wiki_search(q, limit=5)
    titles = [h.get("title") for h in hits if h.get("title")]
    # Prefer exact title or those that contain the query words
    words = [w for w in ql.split() if len(w) >= 4]
    def good(t):
        tl = t.lower()
        return any(w in tl for w in words) or len(words) == 0
    filtered = [t for t in titles if good(t)]
    return filtered[:3] or titles[:3]

# -----------------------------
# Prompts per intent
# -----------------------------
def prompt_threats(question: str, context: str, persona: str) -> str:
    role = {
        "Scientist": "Explain with clear steps, definitions, and short paragraphs.",
        "Naturalist": "Explain with ecology focus and accessible language.",
        "Policy": "Explain practical implications for conservation and people.",
        "Poetic": "Explain accurately with gentle, metaphorical tone.",
    }.get(persona, "Explain clearly and concisely.")

    exemplar = """
EXAMPLE ANSWER FORMAT:
1) Direct answer: One or two sentences summarizing the key threats.
2) Top causes:
- Overfishing → Stocks depleted → Atlantic cod collapse (1990s).
- Plastic pollution → Ingestion/entanglement → Turtles ingesting bags.
- Ocean acidification → Weaker shells/skeletons → Pteropod shell dissolution.
- Marine heatwaves → Heat stress → 2016 Great Barrier Reef bleaching.
- Bycatch → Non-target mortality → Turtle/seabird deaths in longlines.
3) What can help: Concrete actions in 1 line.
4) Sources: Comma-separated page titles.
""".strip()

    tpl = f"""
You are BlueMind, a careful ocean expert. {role}
Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Follow the EXAMPLE ANSWER FORMAT exactly.
- Keep bullets to one line: cause → mechanism → example.
- If context is missing something, say so briefly in Direct answer.

{exemplar}

NOW WRITE YOUR ANSWER:
"""
    return textwrap.dedent(tpl).strip()

def prompt_entity(question: str, context: str, persona: str) -> str:
    role = {
        "Scientist": "Be precise with sizes, taxonomy, and qualifiers.",
        "Naturalist": "Be clear and friendly; highlight key traits and habitat.",
        "Policy": "Note conservation status and human impacts briefly.",
        "Poetic": "Accurate, but with a hint of wonder.",
    }.get(persona, "Be precise and concise.")

    exemplar = """
EXAMPLE ANSWER FORMAT:
1) Direct answer: “The whale shark (Rhincodon typus) is the largest living fish.”
2) Key facts:
- Typical & maximum size (ranges, sources vary; note uncertainty).
- Where it lives (oceans/latitudes/habitats).
- Diet/behavior basics.
3) Context note: If multiple contenders exist (e.g., basking shark), mention them briefly.
4) Sources: Page titles used.
""".strip()

    tpl = f"""
You are BlueMind, a marine naturalist. {role}
Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Start with a one-sentence direct answer to the superlative question.
- Then give 3–4 short bullets of key facts.
- Avoid generic threats unless the question asks about threats.
- If context is thin, state the most likely answer and note uncertainty.
- End with 'Sources:' and the page titles.

{exemplar}

NOW WRITE YOUR ANSWER:
"""
    return textwrap.dedent(tpl).strip()

def prompt_topic(question: str, context: str, persona: str) -> str:
    role = {
        "Scientist": "Explain mechanisms and use crisp structure.",
        "Naturalist": "Keep it accessible; link process to ecosystems.",
        "Policy": "Highlight impacts and mitigation briefly.",
        "Poetic": "Stay accurate; add gentle imagery.",
    }.get(persona, "Explain clearly.")

    exemplar = """
EXAMPLE ANSWER FORMAT:
1) What it is: 1–2 lines definition.
2) How it works: 3 bullets (process → effect).
3) Why it matters: 2 bullets (ecology/people).
4) What scientists watch: 2 bullets (indicators/events).
5) Sources: Titles.
""".strip()

    tpl = f"""
You are BlueMind, an ocean explainer. {role}
Use ONLY the context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Follow the EXAMPLE ANSWER FORMAT.
- Be concise; no filler.
- End with 'Sources:' and the page titles.

{exemplar}

NOW WRITE YOUR ANSWER:
"""
    return textwrap.dedent(tpl).strip()

# -----------------------------
# Validators & fallbacks
# -----------------------------
def validate_has_sections(text: str, required: List[str]) -> bool:
    if not text or len(text) < 60:
        return False
    return all(h in text for h in required)

def fallback_threats(titles: List[str]) -> str:
    mech = {
        "Overfishing": "Stocks depleted → food-web shifts → Cod collapse (1990s).",
        "Bycatch": "Non-target mortality in nets/longlines → Turtles & seabirds.",
        "Marine pollution": "Nutrients/oil/sewage → Hypoxia & disease.",
        "Plastic pollution": "Ingestion/entanglement → Injury & starvation.",
        "Ocean acidification": "Lower pH → Fewer carbonate ions → Weaker shells.",
        "Climate change and oceans": "Warming/stratification → Heatwaves & habitat shifts.",
        "Coral bleaching": "Heat stress → Symbionts expelled → Mass bleaching.",
        "Habitat destruction": "Trawling/coastal build → Seafloor & nursery loss.",
        "Noise pollution": "Ship/seismic noise → Communication stress.",
        "Invasive species": "Ballast transfer → Native displacement → Lionfish.",
    }
    picked = [t for t in titles if t in mech] or titles[:5]
    bullets = [f"- {t} → {mech.get(t, 'See context excerpt above.')}" for t in picked[:5]]
    direct = (
        "Marine animals face multiple human-driven threats that reduce habitat quality, alter food webs, "
        "and increase mortality."
    )
    helpm = "Cut overfishing/bycatch, reduce pollution, protect habitats, and curb greenhouse-gas emissions."
    return (
        f"1) Direct answer: {direct}\n\n"
        f"2) Top causes:\n" + "\n".join(bullets) + "\n\n"
        f"3) What can help: {helpm}\n\n"
        f"4) Sources: " + ", ".join(picked)
    )

def fallback_entity(titles: List[str]) -> str:
    if "Whale shark" in titles or "List of largest fish" in titles:
        return (
            "1) Direct answer: The whale shark (*Rhincodon typus*) is the largest living fish.\n\n"
            "2) Key facts:\n"
            "- Typical size ~10–12 m; occasionally reported larger; filter-feeding shark.\n"
            "- Found in tropical & warm-temperate oceans; migratory.\n"
            "- Gentle filter feeder on plankton/small fishes; often near surface.\n\n"
            "3) Context note: The basking shark is also very large but typically smaller.\n\n"
            "4) Sources: Whale shark, List of largest fish"
        )
    # Generic entity fallback
    return (
        "1) Direct answer: Based on context, the requested marine superlative likely refers to a well-known species.\n\n"
        "2) Key facts:\n- See sources for the most relevant candidate(s).\n\n"
        "3) Context note: Provide a more specific phrase (e.g., 'largest fish in the ocean').\n\n"
        "4) Sources: " + ", ".join(titles[:3])
    )

def fallback_topic(titles: List[str]) -> str:
    return (
        "1) What it is: See the sources for a concise definition of the topic.\n\n"
        "2) How it works:\n- Mechanism 1\n- Mechanism 2\n- Mechanism 3\n\n"
        "3) Why it matters:\n- Ecosystem impact\n- Human impact\n\n"
        "4) What scientists watch:\n- Key indicator 1\n- Key indicator 2\n\n"
        "5) Sources: " + ", ".join(titles[:3])
    )

# -----------------------------
# Orchestrator
# -----------------------------
def build_context(titles: List[str], sents: int = 8) -> str:
    extracts = []
    for t in titles:
        if not t:
            continue
        extracts.append(f"# {t}\n" + (wiki_extract(t, sentences=sents) or ""))
    return "\n\n".join(extracts) if extracts else "(No external context retrieved.)"

def generate_answer(question: str, persona: str, search_terms: Optional[str] = None) -> Dict:
    intent = classify_intent(search_terms or question)
    titles = route_titles(search_terms or question, intent)
    context = build_context(titles, sents=8)

    # Choose prompt per intent
    if intent == "threats":
        prompt = prompt_threats(question, context, persona)
        required = ["1) Direct answer:", "2) Top causes:", "3) What can help:", "4) Sources:"]
    elif intent == "entity":
        prompt = prompt_entity(question, context, persona)
        required = ["1) Direct answer:", "2) Key facts:", "3) Context note:", "4) Sources:"]
    else:
        prompt = prompt_topic(question, context, persona)
        required = ["1) What it is:", "2) How it works:", "3) Why it matters:", "4) What scientists watch:", "5) Sources:"]

    # Run model (optional) then validate; else fall back deterministically
    answer = ""
    llm = load_llm()
    if llm is not None:
        try:
            out = llm(prompt, max_new_tokens=360)
            answer = out[0]["generated_text"].strip()
        except Exception as e:
            answer = f"(Model error: {e})"

    if not validate_has_sections(answer, required):
        if intent == "threats":
            answer = fallback_threats(titles)
        elif intent == "entity":
            answer = fallback_entity(titles)
        else:
            answer = fallback_topic(titles)

    return {"answer": answer, "sources": titles, "raw_context": context}

# ---------------------
# Streamlit UI assembly
# ---------------------
st.set_page_config(page_title="BlueMind – Marine Chatbot", page_icon="🌊", layout="wide")
st.title("🌊 BlueMind – The Marine Chatbot")
st.caption("Ask about oceans, climate, species, currents, exploration, and conservation.")

with st.sidebar:
    st.header("Controls")
    persona = st.selectbox("Answer style", ["Scientist", "Naturalist", "Policy", "Poetic"], index=0)
    st.markdown("---")
    st.subheader("Ocean Data Query")
    lat = st.number_input("Latitude", value=0.0, step=0.5, format="%.3f")
    lon = st.number_input("Longitude", value=0.0, step=0.5, format="%.3f")
    start_date = st.text_input("Start date (YYYY-MM-DD)", value=pd.Timestamp.utcnow().date().isoformat())
    end_date = st.text_input("End date (YYYY-MM-DD)", value=pd.Timestamp.utcnow().date().isoformat())
    st.markdown("---")
    st.info("Tip: Change persona for different answer styles. Use Ocean Data tab to fetch marine conditions for a point.")

# Use a form so Enter submits
chat_tab, data_tab, species_tab, map_tab = st.tabs(["💬 Chat", "🌡 Ocean Data", "🐠 Species", "🗺 Map"])

with chat_tab:
    st.subheader("Ask the Ocean")
    with st.form("chat_form"):
        q = st.text_input("Your question about the ocean:", placeholder="e.g., Why are ocean animals in danger? / biggest fish / coral bleaching")
        search_terms = st.text_input("Optional: refine retrieval keywords", placeholder="e.g., bycatch plastic pollution")
        submitted = st.form_submit_button("Answer")
    if submitted and q.strip():
        with st.spinner("Thinking with tides..."):
            result = generate_answer(q, persona, search_terms)
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
        st.info(
            "Data source: Open-Meteo Marine (free). "
            "Swap in NOAA ERDDAP/NDBC for production. "
            "Enhancements: add bathymetry & SST layers, anomalies vs. climatology, buoy metadata."
        )

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
st.caption("BlueMind • Wikipedia + Open-Meteo • Intent-aware answers (threats / entity / topic).")
