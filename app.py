# BlueMind: The Marine Chatbot (single-file Streamlit app)

import textwrap
from typing import Dict, List, Optional

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

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
def wiki_search(query: str, limit: int = 3) -> List[Dict]:
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

# Curated pages for generic "why are ocean animals in danger?" queries
THREAT_PAGES = [
    "Overfishing",
    "Bycatch",
    "Marine pollution",
    "Plastic pollution",
    "Ocean acidification",
    "Climate change and oceans",
    "Coral bleaching",
    "Habitat destruction",
    "Noise pollution",
    "Invasive species",
]

DANGER_KEYWORDS = (
    "danger", "threat", "endangered", "risk", "dying", "decline",
    "why are ocean animals", "why are marine animals", "why are sea animals",
)

def route_topics(question: str) -> List[str]:
    """Router: for 'danger/threat' style queries, use curated pages; else use wiki search."""
    q = (question or "").lower()
    if any(k in q for k in DANGER_KEYWORDS):
        return THREAT_PAGES[:5]  # concise top 5
    pages = wiki_search(question, limit=3)
    titles = [p.get("title") for p in pages if p.get("title")]
    return titles

def build_prompt(question: str, retrieved_notes: str, persona: str = "Scientist") -> str:
    role = {
        "Scientist": "Explain with clear steps, definitions, and short paragraphs.",
        "Naturalist": "Explain with biology/ecology focus and accessible language.",
        "Policy": "Explain practical implications for conservation policy and humans.",
        "Poetic": "Explain accurately but with a gentle, metaphorical tone.",
    }.get(persona, "Explain clearly and concisely.")

    exemplar = """
EXAMPLE QUESTION:
Why are coral reefs bleaching?

EXAMPLE CONTEXT (snippets):
# Coral bleaching
Coral bleaching occurs when corals expel symbiotic algae (zooxanthellae) due to heat stress...
# Ocean acidification
Decreasing pH reduces carbonate ion availability, impairing calcification...

EXAMPLE ANSWER (follow this exact structure):
1) Direct answer: Coral reefs bleach mainly when unusually warm water stresses corals, causing them to expel symbiotic algae. Pollution and ocean acidification make recovery harder.

2) Top causes:
- Marine heatwaves → Heat stress → 2016 Great Barrier Reef mass bleaching.
- Ocean acidification → Weaker calcification → Slower growth and fragility.
- Pollution & sediments → Light reduction & disease risk → Nearshore reef decline.
- Overfishing → Food-web imbalance (algae overgrowth) → Lower reef resilience.
- Destructive practices → Physical damage → Blast/anchor damage on coral heads.

3) What can help: Cut greenhouse gases, reduce local pollution, protect herbivores, expand marine protected areas, and monitor heat alerts.

4) Sources: Coral bleaching, Ocean acidification, Overfishing
""".strip()

    template = f"""
You are BlueMind, a careful ocean expert. {role}
Use ONLY the context provided to answer.

CONTEXT:
{retrieved_notes}

QUESTION:
{question}

INSTRUCTIONS:
- Follow the EXAMPLE ANSWER structure exactly.
- Do not repeat headings without content.
- Keep each bullet to one line: cause → mechanism → example.
- If context is missing something, say so briefly in the Direct answer.

{exemplar}

NOW WRITE YOUR ANSWER:
"""
    return textwrap.dedent(template).strip()

def validate_answer(text: str) -> bool:
    """Check if the model filled the template with real content."""
    if not text or len(text) < 100:
        return False
    must_have = ["1) Direct answer:", "2) Top causes:", "3) What can help:", "4) Sources:"]
    if not all(m in text for m in must_have):
        return False
    bad_frag = "Top causes (bulleted, cause"
    if bad_frag.lower() in text.lower():
        return False
    if "- " not in text:
        return False
    return True

def compose_fallback_answer(titles: List[str]) -> str:
    """Deterministic answer if the LLM outputs junk."""
    mechanisms = {
        "Overfishing": "Stocks depleted → food-web shifts → Atlantic cod collapse (1990s).",
        "Bycatch": "Non-target species killed in nets/longlines → turtle & seabird losses.",
        "Marine pollution": "Nutrients/oil/sewage → hypoxia & disease → Gulf of Mexico dead zone.",
        "Plastic pollution": "Ingestion/entanglement → injury & starvation → turtles ingesting bags.",
        "Ocean acidification": "Lower pH → fewer carbonate ions → weaker shells/skeletons.",
        "Climate change and oceans": "Warming & stratification → habitat shifts & heatwaves.",
        "Coral bleaching": "Heat stress → algae expelled → GBR mass bleaching 2016/2020.",
        "Habitat destruction": "Trawling/coastal build → seafloor & nursery loss.",
        "Noise pollution": "Ship/seismic noise → communication stress → whale navigation issues.",
        "Invasive species": "Ballast transport → native displacement → lionfish in W. Atlantic.",
    }
    picked = [t for t in titles if t in mechanisms] or titles[:5]
    bullets = [f"- {t} → {mechanisms.get(t, 'See context excerpt above.')}" for t in picked[:5]]
    direct = (
        "Marine animals are under pressure from multiple human-driven threats that reduce habitat quality, "
        "alter food webs, and increase mortality. The biggest drivers are overfishing/bycatch, pollution "
        "(including plastics), and climate-related changes such as warming and acidification."
    )
    helpm = (
        "Tighten sustainable catch limits, cut bycatch, reduce nutrient/plastic pollution, protect and restore "
        "critical habitats, and reduce greenhouse-gas emissions."
    )
    return (
        f"1) Direct answer: {direct}\n\n"
        f"2) Top causes:\n" + "\n".join(bullets) + "\n\n"
        f"3) What can help: {helpm}\n\n"
        f"4) Sources: " + ", ".join(picked)
    )

def generate_answer(question: str, persona: str, search_terms: Optional[str] = None) -> Dict:
    # Decide which pages to use (router → curated topics for danger-style questions)
    titles = route_topics(search_terms or question)

    # Build context from Wikipedia extracts
    extracts = []
    for t in titles:
        if not t:
            continue
        extracts.append(f"# {t}\n" + (wiki_extract(t, sentences=8) or ""))
    context = "\n\n".join(extracts) if extracts else "(No external context retrieved.)"

    # Build prompt
    prompt = build_prompt(question, context, persona)

    # Try HF local model; if unavailable or garbage, fall back to deterministic synthesis
    answer = ""
    llm = load_llm()
    if llm is not None:
        try:
            out = llm(prompt, max_new_tokens=360)
            answer = out[0]["generated_text"].strip()
        except Exception as e:
            answer = f"Model error, falling back to summary. Error: {e}"

    if not validate_answer(answer):
        answer = compose_fallback_answer(titles)

    return {
        "answer": answer,
        "sources": titles,
        "raw_context": context,
    }

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

chat_tab, data_tab, species_tab, map_tab = st.tabs(["💬 Chat", "🌡 Ocean Data", "🐠 Species", "🗺 Map"])

with chat_tab:
    st.subheader("Ask the Ocean")

    with st.form("chat_form"):
        q = st.text_input("Your question about the ocean:", placeholder="e.g., Why are ocean animals in danger?")
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
st.caption("BlueMind starter • RAG via Wikipedia • Marine data via Open-Meteo fallback • Swap in NOAA/ESA for richer insights.")
