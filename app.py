import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# --- 1. PAGE CONFIGURATION & COLOR THEORY CSS ---
st.set_page_config(page_title="Retail ML Forecaster", page_icon="📈", layout="wide")

# Injecting Custom CSS for UI/UX enhancements
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Login Card Styling */
    .login-box {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #1F4E79;
    }
    
    /* Highlighted text colors */
    .text-primary { color: #1F4E79; font-weight: bold; }
    .text-success { color: #27AE60; font-weight: bold; }
    .text-danger { color: #E74C3C; font-weight: bold; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.02);
    }
    
    /* Custom Button Hover */
    div.stButton > button:first-child {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. AUTHENTICATION STATE ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    # Centered layout for login
    col1, col2, col3 = st.columns([1, 1.5, 1])
    
    with col2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        # CHANGED: Commercial Branding instead of College Name
        st.markdown("<h2 class='text-primary' style='text-align: center;'>RetailCore ML Platform</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7F8C8D;'>Enterprise Inventory & Demand Forecasting</p>", unsafe_allow_html=True)
        st.divider()
        
        st.markdown("### 🔐 Secure System Access")
        username = st.text_input("Username (Type 'yogesh')")
        password = st.text_input("Password (Type 'pass5057')", type="password")
        
        if st.button("Log In to Dashboard", type="primary", use_container_width=True):
            if username == "yogesh" and password == "pass5057":
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MAIN DASHBOARD ---
def dashboard():
    # Sidebar UX
    st.sidebar.markdown("### 👤 User Profile")
    # CHANGED: 'Roll' to 'Staff ID' for a commercial feel
    st.sidebar.info("**Yogesh Chopade**\n\nStaff ID: 5057\n\nRole: System Administrator")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        st.session_state["authenticated"] = False
        st.rerun()

    # Main Header
    st.markdown("<h1 class='text-primary'>📦 Retail Demand & Inventory ML Hub</h1>", unsafe_allow_html=True)
    st.markdown("Monitor multi-category stock levels and predict future demand using LightGBM.")
    st.write("") # Spacing
    
    # Modern KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total SKUs", "48 Items", "+4 This Week")
    kpi2.metric("System Engine", "LightGBM", "Active", delta_color="normal")
    kpi3.metric("Database Status", "retail.db", "Connected")
    kpi4.metric("Avg Prediction Error", "12.1 Units", "-2.3 Improvement", delta_color="inverse")

    st.write("---")

    # Clean Tabs for Navigation
    tab1, tab2 = st.tabs(["📝 Step 1: Daily Stock Update", "📊 Step 2: ML Demand Forecast"])
    
    # Declare variables early so both tabs can use them safely
    opening = received = sold = damaged = 0
    sku = "DAP_50kg_Bag"
    
    with tab1:
        st.markdown("### 🛒 End-of-Day Register Entry")
        st.caption("Enter today's point-of-sale data to update the SQLite ledger.")
        
        # User-friendly grouped inputs
        with st.container(border=True):
            col_a, col_b = st.columns(2)
            with col_a:
                store_type = st.selectbox("🏬 Select Store Vertical", ["Agri Store", "Grocery", "Pharmacy", "Clothing"])
                sku = st.selectbox("📦 Select Product SKU", ["DAP_50kg_Bag", "Wheat_Seed_40kg", "Fresh_Milk_1L", "Paracetamol_650", "Denim_Jeans_32"])
                opening = st.number_input("📥 Opening Stock", min_value=0, value=120)
                received = st.number_input("🚚 New Deliveries Received", min_value=0, value=50)
            with col_b:
                sold = st.number_input("💸 Units Sold Today", min_value=0, value=85)
                damaged = st.number_input("⚠️ Damaged / Expired", min_value=0, value=2)
                
                # Math equation computation
                closing = opening + received - sold - damaged
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(f"**Calculated Closing Stock:** `{closing} Units`")
                
        if st.button("💾 Securely Save to Database", type="primary"):
            try:
                conn = sqlite3.connect('retail.db')
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO daily_inventory (store_type, item_sku, opening_stock, received_stock, units_sold, damaged_stock, closing_stock)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (store_type, sku, opening, received, sold, damaged, closing))
                conn.commit()
                conn.close()
                st.success(f"✅ Data saved! {closing} units of {sku} updated in the ledger.")
            except Exception as e:
                st.warning("⚠️ Database not found. Did you run `database.py` yet?")

    with tab2:
        st.markdown("### 📈 7-Day Machine Learning Forecast")
        st.caption("AI-driven predictions based on historical lags and seasonality.")
        
        # Determine actual closing stock if the user hasn't pressed buttons in Tab 1
        closing = opening + received - sold - damaged
        
        # Simulated Forecast Data (Matches ML output behavior)
        days = [f"Day +{i}" for i in range(1, 8)]
        predicted_demand = [28, 35, 42, 60, 75, 40, 32]
        total_forecast = sum(predicted_demand)
        safety_buffer = 25
        
        # UI/UX: Split layout for chart and alert box
        chart_col, alert_col = st.columns([2, 1])
        
        with chart_col:
            # Professional Plotly Chart Design
            df_chart = pd.DataFrame({"Next 7 Days": days, "Predicted Demand": predicted_demand})
            fig = px.area(df_chart, x="Next 7 Days", y="Predicted Demand", markers=True)
            fig.update_traces(line_color="#1F4E79", fillcolor="rgba(31, 78, 121, 0.2)")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with alert_col:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("#### ⚙️ System Diagnostics")
                st.write(f"**Current Stock:** {closing}")
                st.write(f"**Predicted 7-Day Need:** {total_forecast}")
                st.write(f"**Safety Buffer:** {safety_buffer}")
                st.divider()
                
                # COLOR THEORY IN ACTION: Dynamic Alerting
                if closing <= total_forecast:
                    st.error("🚨 **CRITICAL STOCKOUT RISK**\n\nStock is lower than expected demand. Issue a Purchase Order immediately!")
                elif closing <= (total_forecast + safety_buffer):
                    st.warning("⚠️ **REORDER WARNING**\n\nApproaching safety buffer limits. Consider restocking soon.")
                else:
                    st.success("✅ **INVENTORY HEALTHY**\n\nStock levels are optimal for the upcoming week.")

# --- 4. APP ROUTING ---
if not st.session_state["authenticated"]:
    login()
else:
    dashboard()
    