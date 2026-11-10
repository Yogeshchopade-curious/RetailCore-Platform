import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import pickle
import numpy as np
import lightgbm as lgb

# -----------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM CSS
# -----------------------------------------
st.set_page_config(page_title="ForecastAI | Retail Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Clean Metric Cards with Soft Shadow */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
    }
    
    /* Primary Blue Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    
    /* Table Styling */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    /* --------------------------------- */
    /* BOLT.AI EXACT SIDEBAR MENU HACKS  */
    /* --------------------------------- */
    
    /* 1. Hide the default Streamlit radio button circles */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* 2. Style the menu items */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 0.6rem 1rem !important;
        margin-bottom: 0.2rem !important;
        border-radius: 6px !important;
        color: #475569 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out;
        border-left: 4px solid transparent; 
    }
    
    /* 3. Hover state */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
    }
    
    /* 4. ACTIVE SELECTED STATE */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #eff6ff !important;
        color: #2563eb !important;
        border-left: 4px solid #2563eb !important;
        border-top-left-radius: 0px !important;
        border-bottom-left-radius: 0px !important;
    }
    
    /* Align icon and text */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
        font-size: 0.95rem !important;
        margin: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# 2. BACKEND: LOAD MODEL & DATABASE
# -----------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        with open("lightgbm_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return None

@st.cache_data
def load_database_data():
    try:
        conn = sqlite3.connect("retail.db")
        df = pd.read_sql_query("SELECT * FROM daily_inventory", conn)
        conn.close()
        return df
    except:
        dates = pd.date_range(start="2026-10-06", periods=30)
        return pd.DataFrame({
            "date": dates,
            "revenue": np.random.randint(5000, 12000, size=30),
            "units_sold": np.random.randint(50, 150, size=30)
        })

model = load_ml_model()
df = load_database_data()

# -----------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------
st.sidebar.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px; padding-top: 10px;">
        <div style="background-color: #2563eb; padding: 6px 10px; border-radius: 8px;">
            <span style="color: white; font-size: 22px; font-weight: bold;">📈</span>
        </div>
        <div>
            <h2 style="margin: 0; font-size: 1.3rem; color: #0f172a; font-weight: 700;">ForecastAI</h2>
            <p style="margin: 0; font-size: 0.8rem; color: #64748b;">Retail Analytics</p>
        </div>
    </div>
    <p style="font-size: 0.75rem; color: #94a3b8; font-weight: 700; margin-bottom: 5px; letter-spacing: 1px;">MENU</p>
""", unsafe_allow_html=True)

menu = [
    "⊞  Dashboard",
    "📈  Sales Analysis",
    "🧠  Sales Forecast",
    "📦  Inventory",
    "⚠️  Low Stock Alerts",
    "🛍️  Products",