import streamlit as st
import pandas as pd
import sqlite3
import datetime
import base64
import requests
from streamlit_lottie import st_lottie
from streamlit_js_eval import get_geolocation
from geopy.geocoders import Nominatim
import google.generativeai as genai
import re
from difflib import SequenceMatcher

"Configure Gemini AI"
#enter your api key
genai.configure(api_key="")
model = genai.GenerativeModel("models/gemini-1.5-flash")
geolocator = Nominatim(user_agent="fir_app")

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

"Load animations with error handling"
lottie_left = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_jtbfg2nb.json")
lottie_right = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_zrqthn6o.json")

"Layout with animations"
left_col, spacer, right_col = st.columns([1, 0.45, 1])
with left_col:
    if lottie_left:
        st_lottie(lottie_left, speed=3, loop=True, quality="high", height=600, key="left-lottie")
with right_col:
    if lottie_right:
        st_lottie(lottie_right, speed=5, loop=True, quality="high", height=600, key="right-lottie")

"Load IPC data"
try:
    ipc_df = pd.read_csv("ipc_sections.csv")
    ipc_df.fillna('', inplace=True)
    ipc_df['Offense'] = ipc_df['Offense'].astype(str).str.lower()
except FileNotFoundError:
    st.error("IPC sections CSV file not found. Please ensure 'ipc_sections.csv' exists in the same directory.")
    st.stop()

"Simple text similarity function (replacement for sentence transformers)"
def simple_text_similarity(text1, text2):
    """Calculate similarity between two texts using SequenceMatcher"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def find_best_matching_offense(user_input, ipc_df):
    """Find the best matching offense using simple text similarity"""
    user_input_lower = user_input.lower()
    best_match = None
    best_score = 0
    
    "Keywords for common offenses"
    offense_keywords = {
        'theft': ['theft', 'steal', 'stolen', 'robbery', 'burglary', 'loot'],
        'assault': ['assault', 'attack', 'hit', 'beaten', 'violence', 'hurt'],
        'murder': ['murder', 'kill', 'death', 'homicide', 'manslaughter'],
        'fraud': ['fraud', 'cheat', 'scam', 'deception', 'forgery'],
        'kidnapping': ['kidnap', 'abduct', 'missing', 'taken'],
        'rape': ['rape', 'sexual assault', 'molestation', 'sexual harassment'],
        'dowry': ['dowry', 'dowry death', 'bride burning'],
        'domestic violence': ['domestic violence', 'wife beating', 'marital abuse'],
        'bribery': ['bribe', 'corruption', 'illegal gratification'],
        'cybercrime': ['cyber', 'hacking', 'online fraud', 'identity theft', 'phishing']
    }
    
    "Check for keyword matches first"
    for offense_type, keywords in offense_keywords.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                " Find matching row in IPC data"
                for idx, row in ipc_df.iterrows():
                    offense_text = str(row['Offense']).lower()
                    if keyword in offense_text or offense_type in offense_text:
                        score = simple_text_similarity(user_input_lower, offense_text)
                        if score > best_score:
                            best_score = score
                            best_match = idx
    
    "If no keyword match, use general similarity"
    if best_match is None:
        for idx, row in ipc_df.iterrows():
            offense_text = str(row['Offense']).lower()
            score = simple_text_similarity(user_input_lower, offense_text)
            if score > best_score:
                best_score = score
                best_match = idx
    
    return best_match if best_match is not None else 0

"Database setup"
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

"Add missing columns if they don't exist"
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
        response = requests.get("http://ip-api.com/json/", timeout=5).json()
        if response['status'] == 'success':
            return response['lat'], response['lon'], f"{response['city']}, {response['regionName']}, {response['country']}"
    except:
        pass
    return None, None, "Unknown"

def analyze_input(user_name, phone_number, address, text, latitude, longitude, location_address):
    "Use simple text matching instead of sentence transformers"
    best_idx = find_best_matching_offense(text, ipc_df)
    best_row = ipc_df.iloc[best_idx]
    severity = classify_severity(best_row['Offense'], best_row['Punishment'])
    
    prompt = f"""
    Generate a formal FIR summary based on the following details:
    
    Name: {user_name}
    Phone: {phone_number}
    Address: {address}
    Complaint: {text}
    IPC Section: {best_row['Section']}
    Offense: {best_row['Offense']}
    Punishment: {best_row['Punishment']}
    Severity: {severity}
    
    Please create a professional FIR summary that includes:
    1. Complainant details
    2. Nature of the complaint
    3. Relevant IPC section
    4. Brief description of the incident
    5. Any immediate actions recommended
    """
    
    try:
        response = model.generate_content(prompt)
        fir_summary = response.text.strip()
    except Exception as e:
        fir_summary = f"Error generating FIR summary: {str(e)}"
    
    "Insert into database"
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
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating legal advice: {str(e)}"

st.title("📝 FIR Generator & Legal Advisor")

"Create tabs for different functionalities"
tab1, tab2 = st.tabs(["📄 FIR Generator", "⚖️ Legal Advisor"])

with tab1:
    st.header("File a First Information Report (FIR)")
    user_name = st.text_input("Name:")
    phone_number = st.text_input("Phone No.:")
    address = st.text_area("Address:")
    user_input = st.text_area("Enter Complaint:")

    if st.button("Generate FIR"):
        if user_name.strip() and phone_number.strip() and address.strip() and user_input.strip():
            with st.spinner("Generating FIR..."):
                "Get location"
                location_data = get_geolocation()
                if location_data and location_data.get("latitude") and location_data.get("longitude"):
                    lat = location_data["latitude"]
                    lon = location_data["longitude"]
                    try:
                        location_address = geolocator.reverse(f"{lat}, {lon}", timeout=10).address
                    except:
                        location_address = "Unknown"
                else:
                    lat, lon, location_address = get_fallback_location()

                result = analyze_input(user_name, phone_number, address, user_input, lat, lon, location_address)
                
                "Display results"
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
        try:
            logs_df = pd.read_sql_query("SELECT * FROM fir_logs ORDER BY timestamp DESC", conn)
            st.dataframe(logs_df)
        except Exception as e:
            st.error(f"Error loading FIR records: {str(e)}")

    if st.button("📍 Show FIR Locations"):
        try:
            map_data = pd.read_sql_query(
                "SELECT latitude AS lat, longitude AS lon FROM fir_logs WHERE latitude IS NOT NULL AND longitude IS NOT NULL",
                conn
            )
            if not map_data.empty:
                st.map(map_data)
            else:
                st.info("No FIR records with valid geolocation found.")
        except Exception as e:
            st.error(f"Error loading map data: {str(e)}")

with tab2:
    st.header("AI Legal Advisor")
    st.write("Get legal advice for your situation. You can ask general questions or get specific advice based on an existing FIR.")
    
    "Option to link with existing FIR"
    use_existing_fir = st.checkbox("Connect to an existing FIR record")
    
    if use_existing_fir:
        "Get user's previous FIRs"
        if user_name and user_name.strip():
            try:
                user_firs = pd.read_sql_query(
                    "SELECT id, timestamp, offense, ipc_section FROM fir_logs WHERE user_name = ? ORDER BY timestamp DESC",
                    conn, params=(user_name,)
                )
                
                if not user_firs.empty:
                    "Create a list of options with FIR details"
                    fir_options = [
                        f"FIR #{row['id']} - {row['offense']} (Filed: {row['timestamp'][:10]})" 
                        for _, row in user_firs.iterrows()
                    ]
                    
                    selected_fir = st.selectbox(
                        "Select your FIR record:",
                        fir_options
                    )
                    
                    "Get the index of the selected FIR"
                    selected_index = fir_options.index(selected_fir)
                    
                    "Get the corresponding FIR details"
                    ipc_section = user_firs.iloc[selected_index]['ipc_section']
                    offense = user_firs.iloc[selected_index]['offense']
                else:
                    st.info("No previous FIR records found for this name.")
                    ipc_section = None
                    offense = None
            except Exception as e:
                st.error(f"Error loading FIR records: {str(e)}")
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
                
                "Add disclaimer"
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

"Close database connection at the end"
@st.cache_resource
def get_connection():
    return sqlite3.connect("fir_records.db", check_same_thread=False)

"Use session state to manage connection"
if 'conn' not in st.session_state:
    st.session_state.conn = get_connection()
