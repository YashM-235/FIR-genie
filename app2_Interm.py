import streamlit as st
import pandas as pd
import sqlite3
import datetime
import base64
import requests
from streamlit_lottie import st_lottie
from streamlit_js_eval import get_geolocation
from geopy.geocoders import Nominatim
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
#enter your api key
genai.configure(api_key="")
model = genai.GenerativeModel("models/gemini-1.5-flash")
geolocator = Nominatim(user_agent="fir_app")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_left = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
lottie_right = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_zrqthn6o.json")

left_col, spacer, right_col = st.columns([1, 0.45, 1])
with left_col:
    st_lottie(lottie_left, speed=3, loop=True, quality="high", height=600, key="left-lottie")
with right_col:
    st_lottie(lottie_right, speed=5, loop=True, quality="high", height=600, key="right-lottie")

ipc_df = pd.read_csv("ipc_sections.csv")
ipc_df.fillna('', inplace=True)
ipc_df['Offense'] = ipc_df['Offense'].astype(str).str.lower()
encoder = SentenceTransformer('all-MiniLM-L6-v2')
ipc_embeddings = encoder.encode(ipc_df['Offense'].tolist())

conn = sqlite3.connect("fir_records.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS fir_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    ipc_section TEXT,
    offense TEXT,
    punishment TEXT,
    timestamp TEXT,
    user_name TEXT,
    phone_number TEXT,
    address TEXT,
    fir_summary TEXT,
    latitude REAL,
    longitude REAL,
    location_address TEXT
)
""")
cursor.execute("PRAGMA table_info(fir_logs);")
columns = [column[1] for column in cursor.fetchall()]
if 'latitude' not in columns:
    cursor.execute("ALTER TABLE fir_logs ADD COLUMN latitude REAL;")
if 'longitude' not in columns:
    cursor.execute("ALTER TABLE fir_logs ADD COLUMN longitude REAL;")
if 'location_address' not in columns:
    cursor.execute("ALTER TABLE fir_logs ADD COLUMN location_address TEXT;")
conn.commit()

def classify_severity(offense, punishment):
    offense = offense.lower()
    punishment = punishment.lower()
    if any(word in offense for word in ["murder", "rape", "terrorism", "kidnapping"]) or "death" in punishment:
        return "Severe"
    elif any(word in offense for word in ["assault", "theft", "robbery", "bribery", "smuggling"]) or "imprisonment" in punishment:
        return "Moderate"
    else:
        return "Mild"

def get_fallback_location():
    try:
        response = requests.get("http://ip-api.com/json/").json()
        if response['status'] == 'success':
            return response['lat'], response['lon'], f"{response['city']}, {response['regionName']}, {response['country']}"
    except:
        pass
    return None, None, "Unknown"

def analyze_input(user_name, phone_number, address, text, latitude, longitude, location_address):
    user_embedding = encoder.encode([text])
    sims = cosine_similarity(user_embedding, ipc_embeddings)
    best_idx = sims.argmax()
    best_row = ipc_df.iloc[best_idx]
    severity = classify_severity(best_row['Offense'], best_row['Punishment'])
    prompt = f"""
    Generate a formal FIR summary based on the details:
    Name: {user_name}
    Phone: {phone_number}
    Address: {address}
    Complaint: {text}
    IPC Section: {best_row['Section']}
    Offense: {best_row['Offense']}
    Punishment: {best_row['Punishment']}
    Severity: {severity}
    """
    response = model.generate_content(prompt)
    fir_summary = response.text.strip()
    cursor.execute("""INSERT INTO fir_logs 
    (user_input, ipc_section, offense, punishment, user_name, phone_number, address, timestamp, fir_summary, latitude, longitude, location_address) 
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
    (text, best_row['Section'], best_row['Offense'], best_row['Punishment'],
     user_name, phone_number, address, datetime.datetime.now().isoformat(),
     fir_summary, latitude, longitude, location_address))
    conn.commit()
    return {
        "Predicted IPC Section": best_row['Section'],
        "Offense": best_row['Offense'],
        "Punishment": best_row['Punishment'],
        "Severity Level": severity,
        "FIR Summary": fir_summary
    }

def get_legal_advice(query, ipc_section=None, offense=None):
    if ipc_section and offense:
        prompt = f"""You are an AI legal advisor. Provide practical legal advice for the following situation:
        
        Query: {query}
        Related IPC Section: {ipc_section}
        Related Offense: {offense}
        
        Please provide:
        1. Immediate steps the person should take
        2. Legal rights they should be aware of
        3. Recommended course of action
        4. Any precautions or warnings
        5. Suggested legal professionals or resources if applicable
        
        Structure your response clearly with headings for each section.
        """
    else:
        prompt = f"""You are an AI legal advisor. Provide general legal advice for the following query:
        
        {query}
        
        Please provide:
        1. Analysis of the legal situation
        2. Potential legal options
        3. Recommended first steps
        4. Any relevant laws or precedents
        5. When to consult a human lawyer
        
        Structure your response clearly with headings for each section.
        """
    
    response = model.generate_content(prompt)
    return response.text

st.title("📝 FIR Generator & Legal Advisor")

tab1, tab2 = st.tabs(["📄 FIR Generator", "⚖️ Legal Advisor"])

with tab1:
    st.header("File a First Information Report (FIR)")
    user_name = st.text_input("Name:")
    phone_number = st.text_input("Phone No.:")
    address = st.text_area("Address:")
    user_input = st.text_area("Enter Complaint:")

    if st.button("Generate FIR"):
        if user_name.strip() and phone_number.strip() and address.strip() and user_input.strip():
            location_data = get_geolocation()
            if location_data and location_data["latitude"] and location_data["longitude"]:
                lat = location_data["latitude"]
                lon = location_data["longitude"]
                try:
                    location_address = geolocator.reverse(f"{lat}, {lon}").address
                except:
                    location_address = "Unknown"
            else:
                lat, lon, location_address = get_fallback_location()

            result = analyze_input(user_name, phone_number, address, user_input, lat, lon, location_address)
            for key, value in result.items():
                if key == "Severity Level":
                    color = {"Severe": "red", "Moderate": "orange", "Mild": "green"}[value]
                    st.markdown(f"<p style='color:{color}; font-weight:bold; font-size:16px;'>{key}: {value}</p>", unsafe_allow_html=True)
                elif key == "FIR Summary":
                    st.markdown(f"<p style='color:darkblue; font-size:15px;'><b>{key}:</b><br>{value}</p>", unsafe_allow_html=True)
                elif key == "Punishment":
                    st.markdown(f"<p style='color:crimson; font-size:16px;'><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
                elif key == "Predicted IPC Section":
                    st.markdown(f"<p style='color:navy; font-size:16px;'><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p style='font-size:15px;'><b>{key}:</b> {value}</p>", unsafe_allow_html=True)
        else:
            st.warning("Please fill all the fields including name, phone number, address, and complaint.")

    with st.expander("📋 View All FIR Records"):
        logs_df = pd.read_sql_query("SELECT * FROM fir_logs ORDER BY timestamp ASC", conn)
        st.dataframe(logs_df)

    if st.button("📍 Show FIR Locations"):
        map_data = pd.read_sql_query(
            "SELECT latitude AS lat, longitude AS lon FROM fir_logs WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
            conn
        )
        if not map_data.empty:
            st.map(map_data)
        else:
            st.info("No FIR records with valid geolocation found.")

    with st.expander("🌍 Map View of FIRs"):
        map_data = pd.read_sql_query("SELECT latitude, longitude FROM fir_logs WHERE latitude IS NOT NULL", conn)
        if not map_data.empty:
            st.map(map_data)

with tab2:
    st.header("AI Legal Advisor")
    st.write("Get legal advice for your situation. You can ask general questions or get specific advice based on an existing FIR.")
    
    use_existing_fir = st.checkbox("Connect to an existing FIR record")
    
    if use_existing_fir:
        if user_name and user_name.strip():
            user_firs = pd.read_sql_query(
                f"SELECT id, timestamp, offense, ipc_section FROM fir_logs WHERE user_name = ? ORDER BY timestamp DESC",
                conn, params=(user_name,)
            )
            
            if not user_firs.empty:
                fir_options = [
                    f"FIR #{row['id']} - {row['offense']} (Filed: {row['timestamp'][:10]})" 
                    for _, row in user_firs.iterrows()
                ]
                
                selected_fir = st.selectbox(
                    "Select your FIR record:",
                    fir_options
                )
                
                selected_index = fir_options.index(selected_fir)
                
                ipc_section = user_firs.iloc[selected_index]['ipc_section']
                offense = user_firs.iloc[selected_index]['offense']
            else:
                st.info("No previous FIR records found for this name.")
                ipc_section = None
                offense = None
        else:
            st.warning("Please enter your name in the FIR Generator tab first.")
            ipc_section = None
            offense = None
    else:
        ipc_section = None
        offense = None
    
    legal_query = st.text_area("Describe your legal situation or question:", height=200)
    
    if st.button("Get Legal Advice"):
        if legal_query.strip():
            with st.spinner("Analyzing your legal situation..."):
                advice = get_legal_advice(legal_query, ipc_section, offense)
                st.markdown("### Legal Advice")
                st.markdown(advice)
                
                st.markdown("""
                <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 20px;">
                <strong>⚠️ Important Disclaimer:</strong> This AI-generated legal advice is for informational purposes only 
                and should not be considered as a substitute for professional legal counsel from a qualified attorney. 
                Laws vary by jurisdiction and may change over time. For serious legal matters, please consult with a 
                licensed legal professional in your area.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please describe your legal situation or question.")
conn.close()
