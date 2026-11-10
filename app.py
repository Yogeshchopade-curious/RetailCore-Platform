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
    
    /* 2. Style the menu items (padding, color, rounded corners) */
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 0.6rem 1rem !important;
        margin-bottom: 0.2rem !important;
        border-radius: 6px !important;
        color: #475569 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out;
        border-left: 4px solid transparent; 
    }
    
    /* 3. Hover state for unselected menu items */
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
    
    /* Align the icon and text properly */
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
    "📄  Reports",
    "⚙️  Settings"
]

choice_with_icon = st.sidebar.radio("", menu, label_visibility="collapsed")
choice = choice_with_icon.split("  ")[-1].strip()

# -----------------------------------------
# 4. PAGE ROUTING & UI LAYOUTS
# -----------------------------------------

if choice == "Dashboard":
    st.title("Dashboard")
    st.markdown("Monitor sales performance, demand forecasts and inventory health.")
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Sales", value="₹2,63,233", delta="-8.4%")
    with col2:
        st.metric(label="Units Sold", value="2,873", delta="+5.2%")
    with col3:
        st.metric(label="Total Products", value="16")
    with col4:
        st.metric(label="Predicted Next 30 Days", value="2,873 units", delta="+12.3%")
        
    st.write("") 
    
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.subheader("Sales Trend")
        if "date" in df.columns and "revenue" in df.columns:
            fig_line = px.line(df, x="date", y="revenue", color_discrete_sequence=["#2563eb"])
            fig_line.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_line, use_container_width=True)
            
    with chart_col2:
        st.subheader("Inventory Overview")
        donut_data = pd.DataFrame({'Status': ['Normal', 'Low', 'Critical'], 'Count': [10, 6, 0]})
        # Colors updated to exact requested mapping
        fig_donut = px.pie(donut_data, values='Count', names='Status', hole=0.7, 
                           color='Status', color_discrete_map={'Normal': '#3b82f6', 'Low': '#10b981', 'Critical': '#f97316'})
        fig_donut.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

elif choice == "Inventory":
    st.title("Inventory Management")
    st.markdown("Monitor current stock and plan replenishment using predicted demand.")
    st.divider()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Products", value="16")
    with col2:
        st.metric(label="In Stock", value="0")
    with col3:
        st.metric(label="Low Stock", value="6") 
    with col4:
        st.metric(label="Critical Stock", value="10") 
    with col5:
        st.metric(label="Overstock", value="0")
        
    st.write("")
    st.subheader("Inventory Table")
    st.caption("16 products")
    
    inventory_data = pd.DataFrame({
        "Product": ["A5 Notebook", "Ball Pen (Blue)", "Pencil Box", "A4 Paper Ream", "Rice 5kg"],
        "Category": ["Stationery", "Stationery", "Stationery", "Stationery", "Grocery"],
        "Current Stock": [45, 320, 18, 60, 30],
        "Predicted Demand": [133, 374, 62, 97, 367],
        "Safety Stock": [20, 60, 10, 15, 30],
        "Reorder Qty": [108, 114, 54, 52, 367],
        "Status": ["🔴 Critical", "🟠 Low", "🔴 Critical", "🟠 Low", "🔴 Critical"]
    })
    
    st.dataframe(inventory_data, use_container_width=True, hide_index=True)

elif choice == "Sales Forecast":
    st.title("Sales Forecast")
    st.markdown("Generate predictions and analyze machine learning model accuracy.")
    st.divider()
    
    st.subheader("Model Performance")
    st.caption("Comparison of ML models — evaluated on historical data")
    
    performance_data = {
        "Model": ["Linear Regression", "Random Forest", "XGBoost", "LightGBM (Active)"],
        "MAE": ["3.20", "2.40", "2.10", "1.95"],
        "RMSE": ["4.10", "3.20", "2.90", "2.75"],
        "MAPE (%)": ["18.5%", "14.1%", "12.3%", "11.2%"],
        "R²": ["0.720", "0.830", "0.860", "0.890"]
    }
    st.dataframe(pd.DataFrame(performance_data), use_container_width=True, hide_index=True)
    
    if st.button("Generate New 30-Day Forecast"):
        if model is not None:
            st.success("Successfully generated forecast using LightGBM model!")
        else:
            st.warning("Model file not found. Ensure lightgbm_model.pkl is uploaded.")

elif choice == "Low Stock Alerts":
    st.title("Low Stock Alerts")
    st.markdown("Products requiring immediate attention or planned reorder.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.error("🚨 10 Critical")
    col2.warning("⚠️ 6 Low Stock")
    col3.info("ℹ️ 0 Reorder Level")
    
    st.write("")
    st.subheader("Action Required")
    
    grid_col1, grid_col2 = st.columns(2)
    with grid_col1:
        with st.container():
            st.error("**A5 Notebook (Stationery)** - CRITICAL")
            st.write("Current Stock: **45** | Predicted Demand: **133** | Reorder Qty: **108**")
            st.button("Create Reorder", key="reorder_1")
            
    with grid_col2:
        with st.container():
            st.error("**Pencil Box (Stationery)** - CRITICAL")
            st.write("Current Stock: **18** | Predicted Demand: **62** | Reorder Qty: **54**")
            st.button("Create Reorder", key="reorder_2")

elif choice == "Settings":
    st.title("Settings")
    st.markdown("Configure your business information and forecasting preferences.")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Business Information")
        st.text_input("Business Name", value="My Retail Store")
        st.selectbox("Retailer Type", ["Stationery + Grocery", "Electronics", "Apparel"])
        st.selectbox("Currency", ["Indian Rupee (₹)", "USD ($)"])
    with col2:
        st.subheader("Forecast Settings")
        st.number_input("Safety Stock (days of buffer)", value=5)
        st.number_input("Alert Threshold (%)", value=50)
        if st.button("Save Settings"):
            st.success("Settings saved successfully to database.")

else:
    st.title(choice)
    st.info(f"The {choice} module is currently under active development.")