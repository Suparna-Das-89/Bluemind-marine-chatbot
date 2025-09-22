# 🌊 [BlueMind: Ask the Ocean!](https://bluemind-marine-chatbot.streamlit.app) 

👉 [Click here to try the live chatbot](https://bluemind-marine-chatbot.streamlit.app)  

Ask marine-related questions and get real-time, LLM-powered answers from the deep blue!  
Ever wondered about coral bleaching, whale migration, or today’s ocean wave height?  
This interactive Streamlit chatbot connects you to **marine knowledge from Wikipedia, Open-Meteo, and NASA** — all powered by state-of-the-art LLMs.  

---

🚀 Features  

- **LLM-Powered Responses** using Groq’s hosted LLaMA 3 model (with HuggingFace fallback)  
- **Wikipedia Summaries & Images** for species and ocean-related facts  
- **Real-Time Marine Data** (waves, swell, wind) via Open-Meteo Marine API  
- **Interactive Ocean Map** with NASA GIBS SST (sea surface temperature) overlays  
- **Species Explorer** with images and extracts from Wikipedia  
- **Structured JSON Outputs** for clear, schema-driven answers  

---

🧑‍💻 Tech Stack  

**Tools and Purposes:**  

- **Streamlit** : Web interface  
- **Groq LLaMA 3 LLM** : Large Language Model (via API)  
- **HuggingFace Transformers** : Local fallback model  
- **Wikipedia API** : Factual extracts & species info  
- **Open-Meteo Marine & Weather APIs** : Ocean wave, swell, wind data  
- **NASA GIBS** : Global imagery tiles (SST overlays)  
- **Plotly & Folium** : Timeseries and map visualizations  
- **Pandas** : Data processing  

---

⚙️ Installation & Setup  

1. **Clone this repository**
2. **Install required packages**
   
   pip install -r requirements.txt
3. **API Keys**
   
   Create a .streamlit/secrets.toml file and add:
   
   GROQ_API_KEY = "groq_api_key"
4. **Run the app**
   
   streamlit run app.py

---

🌊 Example Questions

"What are the biggest threats to coral reefs?"

"Show me today’s wave height in the Pacific Ocean"

"Tell me about the blue whale"

"What’s the sea surface temperature near Australia?"

---


🙏 Acknowledgements

- Groq API
- HuggingFace Transformers
- Wikipedia REST API
- Open-Meteo Marine & Weather APIs
- NASA GIBS

