# 🌊 BlueMind – The Marine Chatbot

BlueMind is an **AI-powered marine science assistant** that answers questions about oceans, climate, species, and conservation.  
It combines **Groq-hosted LLaMA models**, **Wikipedia knowledge retrieval**, and **Open-Meteo APIs** for real-time marine data.  
Built with **Streamlit** for an interactive user experience.  

---

## 🚀 Features

- **Intelligent Chat**
  - Structured JSON answers powered by Groq LLaMA (with HuggingFace fallback)
  - Supports different intents: threats, entities, and general topics
  - Always returns sourced, contextual answers from Wikipedia

- **Ocean Data Explorer**
  - Fetches real-time wave height, swell, and wind data via Open-Meteo Marine & Weather APIs
  - Interactive **Plotly** charts for timeseries visualization

- **Marine Species Guide**
  - Wikipedia-powered lookup for species info and images
  - Quick reference on marine animals and related topics

- **Interactive Map**
  - Global ocean map built with **Folium**
  - NASA GIBS sea surface temperature (SST) overlays
  - Click to get latitude/longitude + live mouse coordinates

---

## Tech Stack

- **Core:** Python, Streamlit, Pandas  
- **LLM:** Groq API (LLaMA-3.3-70B), HuggingFace Transformers (fallback)  
- **Data Sources:** Wikipedia API, Open-Meteo Marine & Weather APIs  
- **Visualization:** Plotly, Folium, Streamlit UI  
- **Other:** Requests, JSON, Regex helpers  

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bluemind-marine-chatbot.git
   cd bluemind-marine-chatbot
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your **Groq API key**:

   * In Streamlit secrets (`.streamlit/secrets.toml`):

     ```toml
     GROQ_API_KEY = "api_key_here"
     ```
   * Or export as an environment variable:

     ```bash
     export GROQ_API_KEY="api_key_here"
     ```

---

## ▶️ Usage

Run the app with:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📸 Screenshots

*(Add screenshots of chat, ocean data chart, species lookup, and map here)*

---

## 📚 Sources & APIs

* [Groq API](https://groq.com/) – Fast inference with LLaMA models
* [HuggingFace Transformers](https://huggingface.co/transformers/) – Local fallback model
* [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page) – Knowledge retrieval
* [Open-Meteo Marine API](https://open-meteo.com/) – Ocean data (waves, swell, wind)
* [NASA GIBS](https://gibs.earthdata.nasa.gov/) – Global imagery tiles

---

## License

MIT License.
Feel free to fork, modify, and share.

---

## About

BlueMind was built as an educational project to explore:

* **Marine science communication**
* **LLM-powered structured answers**
* **Open data visualization**

> “Ask the Ocean, get structured answers.” 

```

---

```


