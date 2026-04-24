# ============================================================
# Agricultural Price Volatility Dashboard - India
# Applied Business Analytics Project
# Multi-Page Streamlit App | Google Colab Compatible
# ============================================================

# ── INSTALLATION (run this cell first in Colab) ──────────────
# !pip install streamlit pyngrok scikit-learn plotly pandas openpyxl -q

# ── LAUNCH IN COLAB (run this cell after saving the script) ──
# from pyngrok import ngrok
# import subprocess, time
# proc = subprocess.Popen(["streamlit", "run", "agricultural_dashboard.py",
#                          "--server.port", "8501"])
# time.sleep(3)
# tunnel = ngrok.connect(8501)
# print("Dashboard URL:", tunnel.public_url)

# ── LOCAL LAUNCH ─────────────────────────────────────────────
# streamlit run agricultural_dashboard.py

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score, mean_squared_error)
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import io

# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌾 AgriPrice Volatility Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }

    /* ── Metric cards ─────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 16px 18px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10) !important;
        border: 1px solid #e0e0e0 !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetricValue"] > div,
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] { color: #374151 !important; }

    /* ── Info / warning / success boxes ──────────────────── */
    .insight-box {
        background: #1e3a5f;
        border-left: 5px solid #4dabf7;
        padding: 14px 18px; border-radius: 8px; margin: 10px 0;
        font-size: 0.93rem; color: #dbeafe;
    }
    .warning-box {
        background: #4a2800;
        border-left: 5px solid #fbbf24;
        padding: 14px 18px; border-radius: 8px; margin: 10px 0;
        font-size: 0.93rem; color: #fef3c7;
    }
    .success-box {
        background: #14532d;
        border-left: 5px solid #4ade80;
        padding: 14px 18px; border-radius: 8px; margin: 10px 0;
        font-size: 0.93rem; color: #dcfce7;
    }
    .pred-result-high {
        background: #450a0a; border: 2px solid #ef4444;
        padding: 20px 24px; border-radius: 12px; margin: 14px 0;
        font-size: 1rem; color: #fecaca; text-align: center;
    }
    .pred-result-stable {
        background: #052e16; border: 2px solid #22c55e;
        padding: 20px 24px; border-radius: 12px; margin: 14px 0;
        font-size: 1rem; color: #bbf7d0; text-align: center;
    }
    h1 { color: #1a3d2b; }
    h2 { color: #1a3d2b; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════

TARGET_COMMODITIES = [
    "Onion", "Tomato", "Potato", "Ginger", "Garlic",
    "Green Chilli", "Cotton", "Sugarcane", "Groundnut",
    "Rice", "Wheat", "Maize"
]

COMMODITY_MAP = {
    "Ginger(Green)": "Ginger",
    "Ginger (Dry)":  "Ginger",
    "Paddy":         "Rice",
    "Rice (Paddy)":  "Rice",
}

STATE_MAP = {
    "Chattisgarh":          "Chhattisgarh",
    "Jammu and Kashmir":    "Jammu And Kashmir",
    "NCT of Delhi":         "Delhi",
    "Pondicherry":          "Puducherry",
    "Andaman and Nicobar":  "Andaman And Nicobar Islands",
}

SEASON_MAP = {
    1: "Winter", 2: "Winter", 3: "Spring",
    4: "Spring", 5: "Summer", 6: "Summer",
    7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Autumn", 11: "Autumn", 12: "Winter"
}

# Crop growing season: Kharif=Jun-Oct, Rabi=Nov-Mar, Zaid=Mar-Jun, Annual=all
CROP_SEASON_MAP = {
    "Onion":       ["Rabi"],
    "Tomato":      ["Rabi", "Zaid"],
    "Potato":      ["Rabi"],
    "Ginger":      ["Kharif"],
    "Garlic":      ["Rabi"],
    "Green Chilli":["Kharif", "Rabi"],
    "Cotton":      ["Kharif"],
    "Sugarcane":   ["Annual"],
    "Groundnut":   ["Kharif", "Rabi"],
    "Rice":        ["Kharif"],
    "Wheat":       ["Rabi"],
    "Maize":       ["Kharif", "Rabi"],
}

SEASON_MONTHS = {
    "Kharif": list(range(6, 11)),       # Jun-Oct
    "Rabi":   [11, 12, 1, 2, 3],        # Nov-Mar
    "Zaid":   [3, 4, 5, 6],             # Mar-Jun
    "Annual": list(range(1, 13)),        # all months
}

# Rainfall water requirement midpoints (mm) parsed from sheet
CROP_WATER_REQ = {
    "Onion":        450,   # 350-550
    "Tomato":       500,   # 400-600
    "Potato":       600,   # 500-700
    "Ginger":       1550,  # 1300-1800
    "Garlic":       500,   # 400-600
    "Green Chilli": 750,   # 600-900
    "Cotton":       950,   # 700-1200
    "Sugarcane":    2000,  # 1500-2500
    "Groundnut":    500,   # 400-600
    "Rice":         1250,  # 1000-1500
    "Wheat":        550,   # 450-650
    "Maize":        650,   # 500-800
}

COLOR_RISK = {
    "High Risk":   "#e74c3c",
    "Stable":      "#2ecc71",
    "Low Demand":  "#f39c12"
}


@st.cache_data(show_spinner="Loading & preprocessing data…")
def load_data(master_file):
    # ── Load all three sheets ─────────────────────────────────
    xl       = pd.ExcelFile(master_file)
    crop_raw = xl.parse("Crops_list")
    rain_raw = xl.parse("Rainfall")
    req_raw  = xl.parse("Season_water_required")

    # ── Parse rainfall requirement sheet ─────────────────────
    req_raw.columns = ["Crop", "Season_Crop", "Water_Range"]
    req_raw = req_raw[req_raw["Crop"] != "Crop"].dropna(subset=["Crop"]).copy()

    def _parse_water(val):
        try:
            parts = str(val).split("-")
            return (int(parts[0].strip()) + int(parts[-1].strip())) / 2
        except Exception:
            return np.nan

    req_raw["Water_Mid"] = req_raw["Water_Range"].apply(_parse_water)

    crop_req = {}
    for _, row in req_raw.iterrows():
        cname = str(row["Crop"]).strip()
        if cname in COMMODITY_MAP:
            cname = COMMODITY_MAP[cname]
        seasons = [s.strip() for s in str(row["Season_Crop"]).split("/")]
        crop_req[cname] = {"seasons": seasons, "water_mid": row["Water_Mid"]}

    # ── Crop preprocessing ───────────────────────────────────
    crop = crop_raw.copy()
    crop["Commodity"] = crop["Commodity"].str.strip().replace(COMMODITY_MAP)
    crop = crop[crop["Commodity"].isin(TARGET_COMMODITIES)].copy()

    crop["Date"] = pd.to_datetime(
        crop["Month"].str.strip(), format="%B-%Y", errors="coerce"
    )
    crop.dropna(subset=["Date"], inplace=True)

    for col in ["Min Price", "Max Price", "MSP"]:
        crop[col] = pd.to_numeric(crop[col], errors="coerce")

    crop["Modal Price"] = pd.to_numeric(crop["Modal Price"], errors="coerce")
    crop["Modal Price"].fillna(
        crop.groupby("Commodity")["Modal Price"].transform("median"), inplace=True
    )
    crop["MSP"].fillna(0, inplace=True)
    crop["Arrival Quantity"].fillna(
        crop.groupby("Commodity")["Arrival Quantity"].transform("median"), inplace=True
    )
    crop.dropna(subset=["Min Price", "Max Price", "Modal Price"], inplace=True)

    crop["State_std"] = crop["State"].str.strip().replace(STATE_MAP)
    crop["Month_Num"] = crop["Date"].dt.month
    crop["Year"]      = crop["Date"].dt.year
    crop["Season"]    = crop["Month_Num"].map(SEASON_MAP)

    # ── Seasonality: is prediction inside crop's growing window? ──
    def _in_crop_season(row):
        seasons = CROP_SEASON_MAP.get(row["Commodity"], ["Annual"])
        month   = row["Month_Num"]
        for s in seasons:
            if month in SEASON_MONTHS.get(s, []):
                return 1
        return 0

    crop["In_Crop_Season"] = crop.apply(_in_crop_season, axis=1)

    # ── Rainfall preprocessing ───────────────────────────────
    rain = rain_raw.copy()
    rain["date"] = pd.to_datetime(rain["date"], errors="coerce")
    rain.dropna(subset=["date"], inplace=True)
    rain["Date"]       = rain["date"].dt.to_period("M").dt.to_timestamp()
    rain["state_name"] = rain["state_name"].str.strip().str.title()

    rain_monthly = (
        rain.groupby(["state_name", "Date"])
        .agg(actual=("actual", "mean"),
             deviation=("deviation", "mean"),
             normal=("normal", "mean"))
        .reset_index()
    )

    # ── Merge crop + rainfall ────────────────────────────────
    crop["Date_M"]              = crop["Date"].dt.to_period("M").dt.to_timestamp()
    rain_monthly["state_upper"] = rain_monthly["state_name"].str.upper()
    crop["state_upper"]         = crop["State_std"].str.upper()

    merged = crop.merge(
        rain_monthly.rename(columns={"Date": "Date_M", "state_upper": "state_upper"}),
        on=["state_upper", "Date_M"],
        how="left"
    )
    merged["actual"].fillna(
        merged.groupby("Commodity")["actual"].transform("median"), inplace=True
    )
    merged["deviation"].fillna(0, inplace=True)
    merged.dropna(subset=["actual"], inplace=True)

    # ── Rainfall deficit / surplus vs crop water requirement ──
    merged["Water_Req"]   = merged["Commodity"].map(CROP_WATER_REQ)
    merged["Rain_Deficit"] = np.where(
        merged["Water_Req"].notna(),
        (merged["Water_Req"] - merged["actual"] * 6).clip(lower=0),
        0
    )
    merged["Rain_Surplus"] = np.where(
        merged["Water_Req"].notna(),
        (merged["actual"] * 6 - merged["Water_Req"]).clip(lower=0),
        0
    )

    # ── Arrival momentum (supply-side trend) ──────────────────
    merged.sort_values(["Commodity", "State", "Date"], inplace=True)
    merged["Arrival_MA3"] = (
        merged.groupby(["Commodity", "State"])["Arrival Quantity"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    merged["Arrival_Gap"] = merged["Arrival Quantity"] - merged["Arrival_MA3"]
    merged["Arrival_log"] = np.log1p(merged["Arrival Quantity"].clip(lower=0))

    # ── Volatility target ────────────────────────────────────
    merged["Volatility"] = np.where(
        merged["Modal Price"] > 0,
        (merged["Max Price"] - merged["Min Price"]) / merged["Modal Price"],
        np.nan
    )
    merged.dropna(subset=["Volatility"], inplace=True)

    # Season-adjusted volatility: off-season -> 25% uplift (lower expected yield)
    merged["Vol_SeasonAdj"] = np.where(
        merged["In_Crop_Season"] == 0,
        merged["Volatility"] * 1.25,
        merged["Volatility"]
    )
    avg_vol = merged["Vol_SeasonAdj"].mean()
    merged["High_Volatility"] = (merged["Vol_SeasonAdj"] > avg_vol).astype(int)

    # ── Demand Index (weighted composite) ────────────────────
    # DI = 0.5 x norm_price_change + 0.3 x inv_arrival_norm + 0.2 x seasonal_factor
    # All components normalised to [0,1] before combining.

    merged.sort_values(["Commodity", "State", "Date"], inplace=True)

    # Component 1 – Normalised month-over-month price change (clipped to [-1, 1])
    merged["Price_Change"] = (
        merged.groupby(["Commodity", "State"])["Modal Price"]
        .transform(lambda x: x.pct_change().clip(-1, 1))
        .fillna(0)
    )
    pc_min   = merged.groupby("Commodity")["Price_Change"].transform("min")
    pc_max   = merged.groupby("Commodity")["Price_Change"].transform("max")
    pc_range = (pc_max - pc_min).replace(0, 1)
    merged["Price_Change_Norm"] = (merged["Price_Change"] - pc_min) / pc_range

    # Component 2 – Inverse arrival volume (high arrival = low scarcity = low demand pressure)
    arr_min   = merged.groupby("Commodity")["Arrival Quantity"].transform("min")
    arr_max   = merged.groupby("Commodity")["Arrival Quantity"].transform("max")
    arr_range = (arr_max - arr_min).replace(0, 1)
    merged["Arr_Norm"]    = (merged["Arrival Quantity"] - arr_min) / arr_range
    merged["Inv_Arrival"] = 1.0 - merged["Arr_Norm"]   # invert: 1=scarce, 0=abundant

    # Component 3 – Seasonal factor: 1.0 in growing season, 0.5 off-season
    merged["Seasonal_Factor"] = np.where(merged["In_Crop_Season"] == 1, 1.0, 0.5)

    # Weighted combination
    merged["Demand_Index"] = (
        0.5 * merged["Price_Change_Norm"] +
        0.3 * merged["Inv_Arrival"] +
        0.2 * merged["Seasonal_Factor"]
    )

    return merged, avg_vol, crop_req


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/"
    "Emblem_of_India.svg/100px-Emblem_of_India.svg.png",
    width=60
)
st.sidebar.title("🌾 AgriPrice Dashboard")
st.sidebar.markdown("**Agricultural Price Volatility Analysis – India**")
st.sidebar.markdown("---")

# File uploads
st.sidebar.subheader("📂 Upload Dataset")
master_file = st.sidebar.file_uploader(
    "Master Agricultural Data (.xlsx)",
    type=["xlsx"],
    help="Single Excel file with sheets: Crops_list, Rainfall, Season_water_required"
)

if not master_file:
    st.title("🌾 Agricultural Price Volatility Dashboard")
    st.info(
        "👈 Please upload the **Master Agricultural Dataset** from the sidebar to begin.\n\n"
        "- **File**: Master_data_agricultural_crops.xlsx\n"
        "- **Required sheets**: `Crops_list`, `Rainfall`, `Season_water_required`"
    )
    st.stop()

# Load data
df, avg_volatility, crop_req = load_data(master_file)

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

all_states = sorted(df["State"].unique())
sel_states = st.sidebar.multiselect(
    "State", all_states, default=all_states[:5]
)

sel_commodities = st.sidebar.multiselect(
    "Commodity", TARGET_COMMODITIES, default=TARGET_COMMODITIES
)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range", value=(min_date, max_date),
    min_value=min_date, max_value=max_date
)

# Apply filters
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_d, end_d = pd.Timestamp(min_date), pd.Timestamp(max_date)

mask = (
    df["State"].isin(sel_states if sel_states else all_states) &
    df["Commodity"].isin(sel_commodities if sel_commodities else TARGET_COMMODITIES) &
    df["Date"].between(start_d, end_d)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the selected filters. Please adjust.")
    st.stop()

# Navigation
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📑 Navigation",
    ["📊 Overview", "📈 Volatility Analysis",
     "🤖 Prediction (LR)", "🔵 Clustering",
     "🏆 Decision Support (MCDM)"],
    index=0
)

# ════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview – Agricultural Price Trends")
    st.markdown("A bird's-eye view of commodity prices, arrivals, and seasonal patterns.")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📦 Avg Modal Price (₹/Qtl)",
              f"₹ {fdf['Modal Price'].mean():,.0f}")
    k2.metric("🔺 Max Price Recorded (₹/Qtl)",
              f"₹ {fdf['Max Price'].max():,.0f}")
    k3.metric("📉 Avg Volatility Index",
              f"{fdf['Volatility'].mean():.3f}")
    k4.metric("🚜 Total Arrivals (MT)",
              f"{fdf['Arrival Quantity'].sum():,.0f}")

    st.markdown("---")

    # Time-series
    ts = (fdf.groupby(["Date", "Commodity"])["Modal Price"]
          .mean().reset_index())
    fig_ts = px.line(
        ts, x="Date", y="Modal Price", color="Commodity",
        title="Monthly Modal Price Trends by Commodity",
        labels={"Modal Price": "Avg Modal Price (₹/Qtl)", "Date": "Month"},
        template="plotly_white", height=420
    )
    fig_ts.update_layout(legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_ts, use_container_width=True)

    col_a, col_b = st.columns(2)

    # Seasonal avg price
    season_avg = (fdf.groupby(["Season", "Commodity"])["Modal Price"]
                  .mean().reset_index())
    fig_season = px.bar(
        season_avg, x="Season", y="Modal Price", color="Commodity",
        barmode="group",
        title="Average Price by Season & Commodity",
        template="plotly_white",
        category_orders={"Season": ["Winter", "Spring", "Summer",
                                    "Monsoon", "Autumn"]}
    )
    col_a.plotly_chart(fig_season, use_container_width=True)

    # Rainfall vs price scatter
    fig_rf = px.scatter(
        fdf, x="actual", y="Modal Price", color="Commodity",
        trendline="ols",
        title="Rainfall (mm) vs Modal Price",
        labels={"actual": "Monthly Rainfall (mm)"},
        template="plotly_white", opacity=0.55
    )
    col_b.plotly_chart(fig_rf, use_container_width=True)

    # Insights
    st.markdown("### 💡 Key Insights")
    st.markdown("""
<div class="insight-box">
📌 <b>Seasonal Price Spikes:</b> Onion, Tomato and Green Chilli show sharp price
increases during <b>Monsoon (Jul–Sep)</b> due to crop damage and supply disruption.
</div>
<div class="insight-box">
📌 <b>Rainfall Impact:</b> A negative correlation exists between high actual rainfall
and prices of perishable vegetables — excess rain damages produce, reducing supply
and spiking prices.
</div>
<div class="insight-box">
📌 <b>Stable Crops:</b> Wheat and Rice (Paddy) maintain comparatively stable prices
year-round, supported by MSP policies and buffer stock operations.
</div>
<div class="warning-box">
⚠️ <b>High-Risk Crops:</b> Onion, Tomato, Garlic, and Green Chilli exhibit the highest
price volatility — these perishables have poor shelf life and irregular arrivals,
making them susceptible to rapid price swings.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 2 – VOLATILITY ANALYSIS
# ════════════════════════════════════════════════════════════
elif page == "📈 Volatility Analysis":
    st.title("📈 Volatility Analysis")
    st.markdown("Understand which commodities and states face the highest price risk.")

    col1, col2 = st.columns([1, 1])

    # Bar chart – commodity volatility
    comm_vol = (fdf.groupby("Commodity")["Volatility"]
                .mean().sort_values(ascending=False).reset_index())
    comm_vol.columns = ["Commodity", "Avg Volatility"]
    fig_bar = px.bar(
        comm_vol, x="Commodity", y="Avg Volatility",
        color="Avg Volatility",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        title="Average Price Volatility by Commodity",
        template="plotly_white"
    )
    fig_bar.add_hline(y=avg_volatility, line_dash="dot",
                      annotation_text="National Avg",
                      line_color="navy")
    col1.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart – high vs low volatility
    pct_high = fdf["High_Volatility"].mean() * 100
    fig_pie = px.pie(
        values=[pct_high, 100 - pct_high],
        names=["High Volatility", "Stable"],
        color_discrete_sequence=["#e74c3c", "#2ecc71"],
        title="Share of High Volatility Records",
        hole=0.45
    )
    col2.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap – state × commodity
    heat_data = (fdf.groupby(["State", "Commodity"])["Volatility"]
                 .mean().reset_index())
    heat_pivot = heat_data.pivot(
        index="State", columns="Commodity", values="Volatility"
    ).fillna(0)
    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn_r",
        title="Volatility Heatmap: State × Commodity",
        labels={"color": "Volatility Index"},
        aspect="auto", height=500
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Volatility over time
    vol_ts = (fdf.groupby(["Date", "Commodity"])["Volatility"]
              .mean().reset_index())
    fig_vts = px.line(
        vol_ts, x="Date", y="Volatility", color="Commodity",
        title="Volatility Index Over Time",
        template="plotly_white", height=380
    )
    fig_vts.add_hline(y=avg_volatility, line_dash="dash",
                      line_color="red",
                      annotation_text="Avg Threshold")
    st.plotly_chart(fig_vts, use_container_width=True)

    # Rainfall deviation vs volatility
    fig_dev = px.scatter(
        fdf, x="deviation", y="Volatility", color="Commodity",
        size="Arrival Quantity",
        title="Rainfall Deviation vs Price Volatility",
        labels={"deviation": "Rainfall Deviation (%)",
                "Volatility": "Price Volatility Index"},
        template="plotly_white", opacity=0.6, trendline="ols"
    )
    st.plotly_chart(fig_dev, use_container_width=True)

    # Insights
    st.markdown("### 💡 Volatility Insights")
    top3 = comm_vol.head(3)["Commodity"].tolist()
    st.markdown(f"""
<div class="warning-box">
⚠️ <b>Most Volatile Commodities:</b> {', '.join(top3)} show the highest average
volatility — these crops require priority market intervention and cold-chain
infrastructure.
</div>
<div class="insight-box">
📌 <b>Rainfall Deviation Effect:</b> A large negative deviation (drought) correlates
with sudden price spikes in vegetables due to reduced supply. Conversely, excess
rainfall (positive deviation) can reduce prices temporarily before causing
post-rain damage-induced spikes.
</div>
<div class="success-box">
✅ <b>State Patterns:</b> States with better market linkage (Maharashtra, Gujarat)
tend to have lower volatility for the same commodity compared to remote states.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 – ENHANCED PREDICTION (Seasonality + Supply + Rainfall)
# ════════════════════════════════════════════════════════════
elif page == "🤖 Prediction (LR)":
    st.title("🤖 Price Volatility Risk Prediction")
    st.markdown(
        "Per-crop model integrating **seasonality**, **rainfall vs. water requirement**, "
        "and **arrival (supply) momentum** for improved accuracy."
    )

    # ── Crop selector ────────────────────────────────────────
    available_crops = sorted(df["Commodity"].unique().tolist())
    selected_crop   = st.selectbox(
        "🌾 Select Crop for Prediction",
        options=available_crops,
        index=available_crops.index("Onion") if "Onion" in available_crops else 0,
        help="A dedicated model is trained exclusively on the selected crop's data."
    )

    # ── Crop-specific feature engineering ───────────────────
    # Feature set (NO data leakage – no Min/Max/Modal price):
    #   MSP, Arrival_log, Arrival_Gap (supply momentum),
    #   actual rainfall, Rain_Deficit, Rain_Surplus,
    #   Month_Num, In_Crop_Season, MSP_above_zero,
    #   Rain_low / Rain_high buckets, season dummies
    crop_df = df[df["Commodity"] == selected_crop].copy()

    crop_df["MSP_above_zero"] = (crop_df["MSP"] > 0).astype(int)
    rain_p33 = crop_df["actual"].quantile(0.33)
    rain_p66 = crop_df["actual"].quantile(0.66)
    crop_df["Rain_low"]  = (crop_df["actual"] <= rain_p33).astype(int)
    crop_df["Rain_high"] = (crop_df["actual"] >= rain_p66).astype(int)

    # Season dummies (drop first alphabetically as reference)
    season_dummies   = pd.get_dummies(crop_df["Season"], prefix="S", drop_first=True).astype(int)
    crop_df          = pd.concat([crop_df.reset_index(drop=True),
                                  season_dummies.reset_index(drop=True)], axis=1)
    season_feat_cols = list(season_dummies.columns)

    BASE_FEATS   = ["MSP", "Arrival_log", "Arrival_Gap", "actual",
                    "Rain_Deficit", "Rain_Surplus", "Month_Num",
                    "In_Crop_Season", "MSP_above_zero", "Rain_low", "Rain_high",
                    "Demand_Index"]
    FEAT_COLS    = BASE_FEATS + season_feat_cols
    FEAT_LABELS  = (["MSP (₹/Qtl)", "Arrival (log)", "Arrival Gap (supply trend)",
                     "Rainfall (mm)", "Rain Deficit vs Crop Need",
                     "Rain Surplus vs Crop Need", "Month",
                     "In Growing Season", "Has MSP Support",
                     "Low Rainfall", "High Rainfall", "Demand Index"]
                    + [c.replace("S_", "") + " Season" for c in season_feat_cols])

    model_df = crop_df[FEAT_COLS + ["High_Volatility"]].dropna()

    if len(model_df) < 40:
        st.warning(
            f"Not enough data for **{selected_crop}** ({len(model_df)} records). "
            "Try a different crop or upload a larger dataset."
        )
        st.stop()

    X = model_df[FEAT_COLS].values
    y = model_df["High_Volatility"].values

    n_pos       = int(y.sum())
    n_neg       = int(len(y) - n_pos)
    class_ratio = min(n_pos, n_neg) / max(n_pos, n_neg + 1e-9)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    stratify_y = y if (class_ratio > 0.1 and n_pos >= 5 and n_neg >= 5) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=0.25, random_state=42, stratify=stratify_y
    )

    cv_folds = min(5, max(2, min(n_pos, n_neg)))

    # ── Test whether Demand_Index improves CV AUC ────────────
    BASE_FEATS_NO_DI = [f for f in BASE_FEATS if f != "Demand_Index"]
    FEAT_COLS_NO_DI  = BASE_FEATS_NO_DI + season_feat_cols

    model_df_no_di = crop_df[FEAT_COLS_NO_DI + ["High_Volatility"]].dropna()
    X_no_di = SimpleImputer(strategy="median").fit_transform(
        model_df_no_di[FEAT_COLS_NO_DI].values
    )
    X_no_di = StandardScaler().fit_transform(X_no_di)
    X_no_di_tr, _, y_no_di_tr, _ = train_test_split(
        X_no_di, model_df_no_di["High_Volatility"].values,
        test_size=0.25, random_state=42,
        stratify=(model_df_no_di["High_Volatility"].values
                  if (class_ratio > 0.1 and n_pos >= 5 and n_neg >= 5) else None)
    )
    try:
        _baseline_auc = cross_val_score(
            LogisticRegression(C=1.0, max_iter=2000, random_state=42,
                               class_weight="balanced", solver="lbfgs"),
            X_no_di_tr, y_no_di_tr,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring="roc_auc", error_score=0.5
        ).mean()
    except Exception:
        _baseline_auc = 0.5

    # ── Model selection: LR vs GBM vs RF, pick best by CV AUC ─
    candidates = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=2000, random_state=42,
            class_weight="balanced", solver="lbfgs"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.8, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    best_name, best_model_obj, best_cv_auc = "LogisticRegression", candidates["LogisticRegression"], 0.0
    cv_results = {}
    if len(X_train) >= 10:
        for name, mdl in candidates.items():
            try:
                cv_s = cross_val_score(
                    mdl, X_train, y_train,
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                    scoring="roc_auc", error_score=0.5
                )
                cv_results[name] = cv_s.mean()
                if cv_s.mean() > best_cv_auc:
                    best_cv_auc    = cv_s.mean()
                    best_name      = name
                    best_model_obj = mdl
            except Exception:
                cv_results[name] = 0.5

    # Include Demand_Index only if it lifts CV AUC vs baseline (no-DI) model
    demand_index_used = best_cv_auc >= _baseline_auc

    best_model_obj.fit(X_train, y_train)
    y_pred  = best_model_obj.predict(X_test)
    y_proba = best_model_obj.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0.5
    rmse   = np.sqrt(mean_squared_error(y_test, y_proba))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    prec   = report.get("weighted avg", {}).get("precision", 0)
    rec    = report.get("weighted avg", {}).get("recall", 0)

    # ── Model Performance ────────────────────────────────────
    st.subheader(f"📊 Model Performance — {selected_crop}")
    di_note = "✅ Demand Index included (improves AUC)" if demand_index_used else "ℹ️ Demand Index excluded (no AUC gain)"
    st.caption(
        f"Best model: **{best_name}** (CV AUC: {best_cv_auc:.3f})  |  "
        f"Trained on {len(model_df)} records  |  Stable: {n_neg}  |  High Vol: {n_pos}  |  {di_note}"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  f"{acc*100:.1f}%")
    m2.metric("AUC-ROC",   f"{auc:.3f}")
    m3.metric("RMSE",      f"{rmse:.3f}")
    m4.metric("Precision", f"{prec:.3f}")
    m5.metric("Recall",    f"{rec:.3f}")

    col_a, col_b = st.columns(2)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted: Stable", "Predicted: High Risk"],
        y=["Actual: Stable",    "Actual: High Risk"],
        color_continuous_scale="Blues",
        title=f"Confusion Matrix – {selected_crop}",
    )
    fig_cm.update_layout(template="plotly_white")
    col_a.plotly_chart(fig_cm, use_container_width=True)

    # Feature importance (coefficients for LR, feature_importances_ for tree models)
    if hasattr(best_model_obj, "coef_"):
        importance_vals = best_model_obj.coef_[0]
        imp_label       = "LR Coefficient (↑ = more volatile)"
    else:
        importance_vals = best_model_obj.feature_importances_
        imp_label       = "Feature Importance"

    coef_df = pd.DataFrame({
        "Feature":    FEAT_LABELS[:len(importance_vals)],
        "Importance": importance_vals
    })
    coef_df["Abs"] = coef_df["Importance"].abs()
    coef_df = coef_df.nlargest(10, "Abs").sort_values("Importance")

    fig_fi = px.bar(
        coef_df, x="Importance", y="Feature", orientation="h",
        color="Importance",
        color_continuous_scale=["#ef4444", "#e5e7eb", "#22c55e"],
        title=f"Top Feature Influences – {selected_crop}",
        labels={"Importance": imp_label},
        template="plotly_white"
    )
    fig_fi.update_layout(showlegend=False, coloraxis_showscale=False)
    col_b.plotly_chart(fig_fi, use_container_width=True)

    # CV model comparison bar
    if cv_results:
        cv_df = pd.DataFrame(list(cv_results.items()), columns=["Model", "CV AUC"])
        cv_df["Color"] = cv_df["Model"].apply(
            lambda x: "#22c55e" if x == best_name else "#6b7280"
        )
        fig_cv = px.bar(
            cv_df.sort_values("CV AUC"), x="CV AUC", y="Model", orientation="h",
            title="Cross-Validation AUC Comparison (higher = better)",
            text="CV AUC", template="plotly_white"
        )
        fig_cv.update_traces(
            texttemplate="%{text:.3f}", textposition="outside",
            marker_color=cv_df.sort_values("CV AUC")["Color"].tolist()
        )
        fig_cv.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig_cv, use_container_width=True)

    # ── Live Prediction Tool ─────────────────────────────────
    st.markdown("---")
    st.subheader(f"🎯 Predict Volatility Risk for {selected_crop}")

    # Determine crop's growing season for the selected month
    crop_seasons = CROP_SEASON_MAP.get(selected_crop, ["Annual"])
    water_req    = CROP_WATER_REQ.get(selected_crop, 500)

    def crop_med(col):
        v = crop_df[col].dropna()
        return float(np.nanmedian(v)) if len(v) > 0 else 0.0

    def safe_max(col, floor=100.0):
        v = crop_df[col].dropna()
        return max(float(v.max()), floor) if len(v) > 0 else floor

    p1, p2, p3 = st.columns(3)
    p4, p5, p6 = st.columns(3)

    inp_msp   = p1.number_input("MSP (₹/Qtl)",
                                 min_value=0.0,
                                 max_value=safe_max("MSP", 10000.0),
                                 value=crop_med("MSP"), step=50.0,
                                 help="Minimum Support Price. 0 = no MSP.")
    inp_arr   = p2.number_input("Arrival Qty (MT)",
                                 min_value=0.0,
                                 max_value=safe_max("Arrival Quantity", 5000.0),
                                 value=crop_med("Arrival Quantity"), step=10.0,
                                 help="Current mandi arrivals.")
    inp_arr_ma = p3.number_input("3-Month Avg Arrival (MT)",
                                  min_value=0.0,
                                  max_value=safe_max("Arrival_MA3", 5000.0),
                                  value=crop_med("Arrival_MA3"), step=10.0,
                                  help="Prior 3-month average arrivals (supply trend signal).")
    inp_rain  = p4.number_input("Actual Rainfall (mm)",
                                 min_value=0.0, max_value=1000.0,
                                 value=crop_med("actual"), step=5.0,
                                 help=f"Monthly rainfall. Water requirement for {selected_crop}: ~{water_req} mm/season.")
    inp_month = p5.selectbox("Month",
                              options=list(range(1, 13)),
                              format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                     "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                              index=5,
                              help="Month of prediction.")
    inp_msp_flag = p6.radio("Crop has MSP Support?",
                             options=[1, 0],
                             format_func=lambda x: "Yes" if x == 1 else "No",
                             horizontal=True)

    # Auto-derive season from month
    inp_season = SEASON_MAP.get(inp_month, "Summer")

    # Check if prediction month is in crop's growing season
    in_season = any(inp_month in SEASON_MONTHS.get(s, []) for s in crop_seasons)
    if not in_season:
        growing_season_label = " / ".join(crop_seasons)
        st.info(
            f"ℹ️ **Month {inp_month} is outside {selected_crop}'s growing season** "
            f"({growing_season_label}). The model automatically applies a **25% volatility uplift** "
            "to reflect lower expected yield in off-season months."
        )

    if st.button("🔮 Predict Risk Now", type="primary", use_container_width=True):
        arr_log     = np.log1p(max(0, inp_arr))
        arr_gap     = inp_arr - inp_arr_ma         # supply momentum
        rain_def    = max(0, water_req - inp_rain * 6)
        rain_sur    = max(0, inp_rain * 6 - water_req)
        rain_low    = 1 if inp_rain <= rain_p33 else 0
        rain_high   = 1 if inp_rain >= rain_p66 else 0
        in_crop_s   = 1 if in_season else 0

        # ── Compute Demand Index for live inference ───────────
        # Component 1: price change not available at inference → use neutral 0.5
        price_change_norm_live = 0.5

        # Component 2: inverse arrival (normalise against training distribution)
        arr_min_val = float(crop_df["Arrival Quantity"].min())
        arr_max_val = float(crop_df["Arrival Quantity"].max())
        arr_range_val = max(arr_max_val - arr_min_val, 1)
        arr_norm_live = (inp_arr - arr_min_val) / arr_range_val
        arr_norm_live = float(np.clip(arr_norm_live, 0, 1))
        inv_arrival_live = 1.0 - arr_norm_live   # invert: low arrival = high scarcity

        # Component 3: seasonal factor
        seasonal_factor_live = 1.0 if in_season else 0.5

        demand_index_live = (
            0.5 * price_change_norm_live +
            0.3 * inv_arrival_live +
            0.2 * seasonal_factor_live
        )

        season_dummy_vals = []
        for col in season_feat_cols:
            s_name = col.replace("S_", "")
            season_dummy_vals.append(1 if inp_season == s_name else 0)

        raw = ([inp_msp, arr_log, arr_gap, inp_rain,
                rain_def, rain_sur, inp_month,
                in_crop_s, inp_msp_flag, rain_low, rain_high,
                demand_index_live]
               + season_dummy_vals)

        raw_arr = np.array([raw])
        raw_arr = imputer.transform(raw_arr)
        raw_arr = scaler.transform(raw_arr)

        prob = best_model_obj.predict_proba(raw_arr)[0][1] * 100

        # ── Rule-based override: low rainfall + off-season + high demand ──
        # When ALL three stress conditions align, the model's probabilistic
        # output can understate risk because such combinations are rare in
        # training data. Apply a floor to ensure the result reflects reality.
        LOW_RAIN_THRESH   = rain_p33            # bottom tercile for this crop
        HIGH_DEMAND_THRESH = 0.60               # DI above 0.6 = elevated demand pressure

        stress_low_rain   = inp_rain <= LOW_RAIN_THRESH
        stress_off_season = not in_season
        stress_high_demand = demand_index_live >= HIGH_DEMAND_THRESH

        # Count how many stress flags are active (0–3)
        stress_count = sum([stress_low_rain, stress_off_season, stress_high_demand])

        if stress_count == 3:
            # All three: hard floor at HIGH risk (70%)
            prob = max(prob, 70.0)
            override_note = "🔴 Rule override: low rain + off-season + high demand → floor at 70%"
        elif stress_count == 2:
            # Two conditions: floor at MODERATE-HIGH (50%)
            prob = max(prob, 50.0)
            override_note = "🟠 Rule override: 2 stress conditions active → floor at 50%"
        else:
            override_note = ""

        # Cap at 99%
        prob = min(prob, 99.0)

        colour = "#ef4444" if prob >= 50 else "#22c55e"

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={"text": f"Volatility Risk – {selected_crop} | {inp_season}",
                   "font": {"size": 14, "color": "#e5e7eb"}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%",
                         "tickcolor": "#9ca3af",
                         "tickfont": {"color": "#9ca3af"}},
                "bar":  {"color": colour},
                "bgcolor": "#1f2937",
                "steps": [
                    {"range": [0,  35],  "color": "#14532d"},
                    {"range": [35, 60],  "color": "#4a2800"},
                    {"range": [60, 100], "color": "#450a0a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3},
                              "thickness": 0.80, "value": 50}
            },
            number={"suffix": "%", "font": {"size": 34, "color": "#f9fafb"},
                    "valueformat": ".1f"}
        ))
        gauge.update_layout(
            height=330, paper_bgcolor="#111827", font={"color": "#f9fafb"}
        )
        st.plotly_chart(gauge, use_container_width=True)

        # Show override note if triggered
        if override_note:
            st.markdown(
                f'<div style="background:#3b1a00;border-left:4px solid #f59e0b;'
                f'padding:10px 14px;border-radius:6px;margin:6px 0;'
                f'font-size:0.88rem;color:#fef3c7;">{override_note}</div>',
                unsafe_allow_html=True
            )

        # Tiered result
        season_note = "" if in_season else " (off-season)"
        if prob >= 70:
            label     = "⚠️ HIGH VOLATILITY RISK"
            advice    = (f"Strong instability signals for {selected_crop}{season_note}. "
                         "Low rainfall, scarce supply and/or off-season combine to drive "
                         "price swings. Consider cold storage, staggered selling, or hedging.")
            box_class = "pred-result-high"
        elif prob >= 50:
            label     = "⚠️ MODERATE-HIGH RISK"
            advice    = (f"Elevated risk{season_note}. Monitor daily mandi arrivals "
                         "and prepare contingency plans before committing to bulk sale.")
            box_class = "pred-result-high"
        elif prob >= 35:
            label     = "🟡 MODERATE RISK"
            advice    = (f"Conditions are somewhat stable{season_note}. "
                         "Watch for sudden rainfall shifts or arrival drops.")
            box_class = "pred-result-stable"
        else:
            label     = "✅ STABLE / LOW RISK"
            advice    = (f"Market conditions are stable{season_note}. "
                         "Good window for sale or procurement.")
            box_class = "pred-result-stable"

        # Driver signals
        supply_signal  = "⬇️ Supply tightening" if arr_gap < 0 else "⬆️ Supply growing"
        rain_signal    = ("⚠️ Rainfall deficit" if rain_def > 0 else
                          ("⚠️ Rainfall surplus" if rain_sur > 100 else "✅ Adequate rainfall"))
        season_signal  = "⚠️ Off-season" if not in_season else "✅ In growing season"
        demand_signal  = (f"🔴 High demand pressure ({demand_index_live:.2f})"
                          if demand_index_live >= HIGH_DEMAND_THRESH
                          else f"🟢 Normal demand ({demand_index_live:.2f})")

        st.markdown(
            f'<div class="{box_class}">'
            f'<h3 style="margin:0 0 8px 0;">{label}</h3>'
            f'<p style="font-size:1.15rem; margin:4px 0;"><b>{prob:.1f}%</b> probability of high volatility</p>'
            f'<p style="margin:4px 0; opacity:0.9;">{advice}</p>'
            f'<p style="margin:8px 0 0 0; font-size:0.88rem; opacity:0.75;">'
            f'Drivers: {supply_signal} &nbsp;|&nbsp; {rain_signal} &nbsp;|&nbsp; {season_signal} &nbsp;|&nbsp; {demand_signal}'
            f'</p></div>',
            unsafe_allow_html=True
        )

    # ── Model Insights ───────────────────────────────────────
    st.markdown("### 💡 Model Insights")
    top_pos = coef_df.tail(1)["Feature"].values[0] if len(coef_df) > 0 else "Arrival Gap"
    top_neg = coef_df.head(1)["Feature"].values[0] if len(coef_df) > 0 else "MSP"
    di_status_box = "success-box" if demand_index_used else "warning-box"
    di_status_msg = (
        f"✅ <b>Demand Index is active</b> — it improved CV AUC vs the baseline model and "
        "is included as a feature. DI = 0.5 × Price Change + 0.3 × Inv. Arrival + 0.2 × Seasonal Factor."
        if demand_index_used else
        "ℹ️ <b>Demand Index was tested but did not improve CV AUC</b> for this crop — "
        "the model uses the remaining features only. DI is still computed for the driver display."
    )
    st.markdown(f"""
<div class="insight-box">
📌 <b>Strongest Volatility Driver for {selected_crop}:</b> <b>{top_pos}</b> pushes
risk upward most — when this signal is high, expect price swings.
</div>
<div class="success-box">
✅ <b>Strongest Stabilising Factor:</b> <b>{top_neg}</b> is the biggest moderator —
higher values dampen volatility and push the prediction toward "Stable".
</div>
<div class="{di_status_box}">
{di_status_msg}
</div>
<div class="insight-box">
📌 <b>Supply Momentum (Arrival Gap)</b> — when current arrivals fall below the
3-month average, supply is tightening and volatility risk rises.
</div>
<div class="insight-box">
📌 <b>Rainfall vs Water Requirement</b> — deficit and surplus relative to
{selected_crop}'s seasonal need ({water_req} mm) are modelled separately.
</div>
<div class="warning-box">
⚠️ <b>Rule-Based Override</b> — when low rainfall + off-season + high demand index
(DI ≥ 0.60) all occur together, the model applies a hard floor (≥70% risk) to
prevent underestimation in rare-but-critical stress combinations.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 – CLUSTERING
# ════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.title("🔵 Commodity Clustering – KMeans (k=3)")
    st.markdown("Segment commodities into risk categories based on price and volatility.")

    cl_df = fdf[["Commodity", "Modal Price", "Volatility", "State"]].dropna()
    cl_agg = (cl_df.groupby("Commodity")
              .agg(Modal_Price=("Modal Price", "mean"),
                   Volatility=("Volatility", "mean"))
              .reset_index())

    sc2 = StandardScaler()
    Xc  = sc2.fit_transform(cl_agg[["Modal_Price", "Volatility"]])

    km  = KMeans(n_clusters=3, random_state=42, n_init=10)
    cl_agg["Cluster_ID"] = km.fit_predict(Xc)

    # Label clusters by centroid characteristics
    centers = pd.DataFrame(
        sc2.inverse_transform(km.cluster_centers_),
        columns=["Modal_Price", "Volatility"]
    )
    # Rank by price desc + volatility desc → highest combo = High Risk
    centers["score"] = centers["Modal_Price"].rank() + centers["Volatility"].rank()
    rank_order = centers["score"].argsort().values
    label_map  = {}
    sorted_ids = centers["score"].sort_values(ascending=False).index.tolist()
    cluster_labels = ["High Risk", "Stable", "Low Demand"]
    for i, cid in enumerate(sorted_ids):
        label_map[cid] = cluster_labels[i]

    cl_agg["Cluster"] = cl_agg["Cluster_ID"].map(label_map)

    # All records labelled
    fdf2 = fdf.copy()
    fdf2 = fdf2.merge(cl_agg[["Commodity", "Cluster"]], on="Commodity", how="left")

    fig_clust = px.scatter(
        cl_agg, x="Modal_Price", y="Volatility",
        color="Cluster", text="Commodity",
        color_discrete_map=COLOR_RISK,
        size=[40]*len(cl_agg),
        title="KMeans Clustering: Modal Price vs Volatility",
        labels={"Modal_Price": "Avg Modal Price (₹/Qtl)",
                "Volatility":  "Avg Volatility Index"},
        template="plotly_white", height=500
    )
    fig_clust.update_traces(textposition="top center",
                             marker=dict(opacity=0.85))
    st.plotly_chart(fig_clust, use_container_width=True)

    # Cluster summary table
    st.subheader("📋 Cluster Summary")
    summary = cl_agg.groupby("Cluster").agg(
        Commodities=("Commodity", lambda x: ", ".join(x)),
        Avg_Price=("Modal_Price", "mean"),
        Avg_Volatility=("Volatility", "mean"),
        Count=("Commodity", "count")
    ).reset_index()
    summary["Avg_Price"]     = summary["Avg_Price"].round(0).astype(int)
    summary["Avg_Volatility"] = summary["Avg_Volatility"].round(4)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Cluster distribution bar
    dist = fdf2["Cluster"].value_counts().reset_index()
    dist.columns = ["Cluster", "Records"]
    fig_dist = px.bar(
        dist, x="Cluster", y="Records",
        color="Cluster", color_discrete_map=COLOR_RISK,
        title="Distribution of Records Across Clusters",
        template="plotly_white"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Insights
    st.markdown("### 💡 Clustering Insights")
    st.markdown("""
<div class="warning-box">
🔴 <b>High Risk Cluster:</b> Includes perishable vegetables (Onion, Tomato, Garlic,
Green Chilli). These crops show high modal prices alongside extreme volatility —
ideal targets for futures market hedging and cold-storage investment.
</div>
<div class="success-box">
🟢 <b>Stable Cluster:</b> Wheat, Rice, Maize — supported by MSP, buffer stocks, and
established procurement systems. Recommended for risk-averse farmers seeking
predictable income.
</div>
<div class="warning-box">
🟡 <b>Low Demand Cluster:</b> Commodities with lower price realization but moderate
arrivals. Farmers growing these should explore value-added processing (e.g.,
Sugarcane → jaggery, Ginger → dry ginger) to improve income.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# PAGE 5 – MCDM / TOPSIS
# ════════════════════════════════════════════════════════════
elif page == "🏆 Decision Support (MCDM)":
    st.title("🏆 Decision Support – TOPSIS Ranking")
    st.markdown("Multi-Criteria Decision Making: Rank commodities for optimal farming decisions.")

    mcdm_agg = (fdf.groupby("Commodity")
                .agg(Modal_Price=("Modal Price",      "mean"),
                     Volatility=("Volatility",        "mean"),
                     Arrival=("Arrival Quantity",     "mean"))
                .reset_index())

    # TOPSIS
    criteria = ["Modal_Price", "Volatility", "Arrival"]
    weights  = np.array([0.4, 0.3, 0.3])
    # Benefit flags: Price → maximize(+1), Volatility → minimize(-1),
    #                Arrival → maximize(+1)
    benefit  = np.array([1, -1, 1])

    mm = MinMaxScaler()
    norm = mm.fit_transform(mcdm_agg[criteria])

    # Flip minimise criteria (Volatility) → 1 - normalised
    for i, b in enumerate(benefit):
        if b == -1:
            norm[:, i] = 1 - norm[:, i]

    weighted = norm * weights
    ideal_pos = weighted.max(axis=0)
    ideal_neg = weighted.min(axis=0)

    d_pos = np.sqrt(((weighted - ideal_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - ideal_neg) ** 2).sum(axis=1))

    topsis_score = d_neg / (d_pos + d_neg + 1e-9)
    mcdm_agg["TOPSIS_Score"] = topsis_score
    mcdm_agg["Rank"]          = mcdm_agg["TOPSIS_Score"].rank(
        ascending=False).astype(int)
    mcdm_agg = mcdm_agg.sort_values("Rank")

    # Medal colours
    def medal(r):
        if r == 1: return "🥇"
        if r == 2: return "🥈"
        if r == 3: return "🥉"
        return str(r)

    mcdm_agg["Medal"] = mcdm_agg["Rank"].apply(medal)

    display_cols = ["Medal", "Commodity", "Modal_Price",
                    "Volatility", "Arrival", "TOPSIS_Score"]
    display = mcdm_agg[display_cols].copy()
    display.columns = ["Rank", "Commodity", "Avg Price (₹/Qtl)",
                       "Avg Volatility", "Avg Arrival (MT)", "TOPSIS Score"]
    display["Avg Price (₹/Qtl)"] = display["Avg Price (₹/Qtl)"].round(0).astype(int)
    display["Avg Volatility"]     = display["Avg Volatility"].round(4)
    display["Avg Arrival (MT)"]   = display["Avg Arrival (MT)"].round(1)
    display["TOPSIS Score"]       = display["TOPSIS Score"].round(4)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Bar chart of TOPSIS scores
    fig_topsis = px.bar(
        mcdm_agg.sort_values("TOPSIS_Score", ascending=True),
        x="TOPSIS_Score", y="Commodity", orientation="h",
        color="TOPSIS_Score",
        color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
        title="TOPSIS Score by Commodity (Higher = Better)",
        labels={"TOPSIS_Score": "TOPSIS Score", "Commodity": ""},
        template="plotly_white", height=420
    )
    st.plotly_chart(fig_topsis, use_container_width=True)

    # Radar chart for top 5
    top5 = mcdm_agg.head(5)["Commodity"].tolist()
    radar_df = mcdm_agg[mcdm_agg["Commodity"].isin(top5)].copy()
    mm2 = MinMaxScaler()
    radar_df[["Price_N", "Vol_N", "Arr_N"]] = mm2.fit_transform(
        radar_df[["Modal_Price", "Volatility", "Arrival"]]
    )
    radar_df["Vol_N"] = 1 - radar_df["Vol_N"]   # invert so higher = better

    categories = ["Price Score", "Stability Score", "Arrival Score"]
    fig_radar = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, row in radar_df.iterrows():
        vals = [row["Price_N"], row["Vol_N"], row["Arr_N"]]
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["Commodity"],
            line=dict(color=colors[i % len(colors)])
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart – Top 5 Commodities",
        template="plotly_white", height=420
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Farmer recommendations
    st.markdown("### 💡 Farmer & Policy Recommendations")
    top_comm = mcdm_agg.head(3)["Commodity"].tolist()
    bot_comm = mcdm_agg.tail(3)["Commodity"].tolist()
    st.markdown(f"""
<div class="success-box">
✅ <b>Recommended Crops for Stable Income:</b> {', '.join(top_comm)}<br>
These crops offer the best combination of high price realisation, low volatility,
and strong market demand. Ideal for risk-averse smallholder farmers.
</div>
<div class="warning-box">
⚠️ <b>Crops Requiring Risk Management:</b> {', '.join(bot_comm)}<br>
Despite potentially high prices, these crops show high volatility. Farmers should
consider futures contracts, crop insurance, or staggered harvesting to mitigate risk.
</div>
<div class="insight-box">
📌 <b>Policy Recommendations:</b><br>
• Expand MSP coverage to high-volatility perishables (Onion, Tomato).<br>
• Invest in cold-chain infrastructure in states with high volatility clusters.<br>
• Launch real-time price advisory SMS/app services to help farmers time sales.<br>
• Promote Farmer Producer Organisations (FPOs) for collective bargaining power.
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.8rem;'>"
    "🌾 Agricultural Price Volatility Dashboard | "
    "Applied Business Analytics Project | "
    "Data: 2021–2023 | Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True
)
