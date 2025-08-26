# BlueMind: The Marine Chatbot (single-file Streamlit app)

import os
import io
import json
import time
import textwrap
from typing import Dict, List, Optional

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px

try:
    # Lightweight, open model for text2text generation
    from transformers import pipeline
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# ---------------------------
# Utility: Caching web fetches
# ---------------------------
@st.cache_data(show_spinner=False)
def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
    try:
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
    return page.get("extract", "")

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
# Ocean Data: NOAA / Open-Meteo fallback
# -------------------------------------
# NOTE: NOAA has many datasets & endpoints (ERDDAP, NDBC, etc.). To keep this starter
# self-contained and key-free, we use Open-Meteo's free marine endpoint as a fallback.
# You can later swap/extend with NOAA/SST datasets and bathymetry layers.

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
    df["time"] = pd.to_datetime(df["time"])  # ensure datetime
    return df

# -----------------------------
# LLM: Explain & answer builder
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_llm():
    if not _HF_AVAILABLE:
        return None
    try:
        # Flan-T5 is lightweight and good for zero-cost demos (text2text)
        return pipeline("text2text-generation", model="google/flan-t5-base")
    except Exception as e:
        st.warning(f"Could not load local HF model: {e}")
        return None


def build_prompt(question: str, retrieved_notes: str, persona: str = "Scientist") -> str:
    role = {
        "Scientist": "Explain with clear steps, definitions, and short paragraphs.",
        "Naturalist": "Explain with biology/ecology focus and accessible language.",
        "Policy": "Explain practical implications for conservation policy and humans.",
        "Poetic": "Explain accurately but with a gentle, metaphorical tone.",
    }.get(persona, "Explain clearly and concisely.")

    template = f"""
You are BlueMind, a helpful ocean expert. {role}
Use the context to answer the user's question. If context is missing something, say what is uncertain.
Always keep answers grounded to marine science.

Context:
{retrieved_notes}

Question: {question}

Answer:
"""
    return textwrap.dedent(template).strip()


def generate_answer(question: str, persona: str, search_terms: Optional[str] = None) -> Dict:
    # Retrieve context via Wikipedia (simple, free)
    pages = wiki_search(search_terms or question, limit=3)
    extracts = []
    titles = []
    for p in pages:
        t = p.get("title")
        if not t:
            continue
        titles.append(t)
        extracts.append(f"# {t}\n" + wiki_extract(t, sentences=6))
    context = "\n\n".join(extracts) if extracts else "(No external context retrieved.)"

    # Build prompt
    prompt = build_prompt(question, context, persona)

    # Try HF local model; if unavailable, fall back to a simple stitched summary
    llm = load_llm()
    if llm is not None:
        try:
            out = llm(prompt, max_new_tokens=256)
            answer = out[0]["generated_text"].strip()
        except Exception as e:
            answer = f"Model error, falling back to summary. Error: {e}"
    else:
        # Extremely simple fallback: return context + heuristic tail
        answer = ("Context-based summary (fallback):\n\n" + context[:1200] +
                  "\n\n(Answer synthesis omitted: add a local HF model for richer outputs.)")

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

# Tabs
chat_tab, data_tab, species_tab, map_tab = st.tabs(["💬 Chat", "🌡 Ocean Data", "🐠 Species", "🗺 Map"])

with chat_tab:
    st.subheader("Ask the Ocean")
    q = st.text_input("Your question about the ocean:", placeholder="e.g., What causes coral bleaching?")
    search_terms = st.text_input("Optional: refine retrieval keywords", placeholder="e.g., coral bleaching symbiosis")
    if st.button("Answer"):
        with st.spinner("Thinking with tides..."):
            result = generate_answer(q, persona, search_terms)
        st.markdown("### Answer")
        st.write(result["answer"])  # model output
        if result["sources"]:
            st.markdown("### Sources (Wikipedia)")
            st.write(", ".join(result["sources"]))
        with st.expander("Show retrieved context"):
            st.code(result["raw_context"] or "(none)")

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
                # Plot wave height if available
                if "wave_height" in df.columns:
                    fig = px.line(df, x="time", y="wave_height", title="Significant Wave Height (m)")
                    st.plotly_chart(fig, use_container_width=True)
                if "wind_speed_10m" in df.columns:
                    fig2 = px.line(df, x="time", y="wind_speed_10m", title="Wind Speed 10m (m/s)")
                    st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.info("""
        Data source: Open-Meteo Marine (free). Swap in NOAA ERDDAP/NDBC for production.
        Examples of enhancements you can add:
        • Add bathymetry and SST layers
        • Compute anomalies vs. climatology
        • Pull buoy metadata and recent observations
        """)

with species_tab:
    st.subheader("Marine Species Guide")
    species_query = st.text_input("Species or topic (e.g., 'Blue whale', 'Giant squid')")
    if st.button("Lookup species"):
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
    # Simple scattergeo centered on oceans
    df_ocean = pd.DataFrame({
        "lat": [0, 30, -30, 15, -15],
        "lon": [-140, 20, 60, -10, 120],
        "label": ["Pacific", "Atlantic", "Indian", "Atlantic N", "Pacific W"],
    })
    figm = px.scatter_geo(df_ocean, lat="lat", lon="lon", hover_name="label", projection="natural earth")
    figm.update_layout(height=500, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(figm, use_container_width=True)

# -------------
# Footer credits
# -------------
st.markdown("---")
st.caption("BlueMind starter • RAG via Wikipedia • Marine data via Open-Meteo fallback • Swap in NOAA/ESA for richer insights.")
