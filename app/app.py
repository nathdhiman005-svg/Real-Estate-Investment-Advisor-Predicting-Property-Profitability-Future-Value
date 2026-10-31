# -*- coding: utf-8 -*-
"""
Real Estate Investment Advisor - Web Application.

An interactive Streamlit platform for predicting property profitability,
5-year future valuations, comprehensive exploratory market analytics,
and real-time MLflow operational monitoring.
"""

from typing import Dict, Any, List, Optional, cast
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Reconfigure stdout for utf-8
reconfig = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig):
    reconfig(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Automatically bridge Streamlit Cloud Secrets into environment variables for MLflow
try:
    if hasattr(st, "secrets"):
        for sec_k, sec_v in st.secrets.items():
            if isinstance(sec_v, str):
                os.environ.setdefault(sec_k, sec_v)
except Exception:
    pass

from src.mlflow_service import (
    predict_property_investment,
    get_configured_tracking_uri,
)

# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for cards, badges, and clean layout
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    .badge-good {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-caution {
        background-color: #f8d7da;
        color: #842029;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .telemetry-pill {
        background-color: #e2e3e5;
        color: #41464b;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Static Domain Constants & Lookups
# -----------------------------------------------------------------------------

CITY_TO_STATE: Dict[str, str] = {
    "Ahmedabad": "Gujarat", "Amritsar": "Punjab", "Bangalore": "Karnataka", "Bhopal": "Madhya Pradesh",
    "Bhubaneswar": "Odisha", "Bilaspur": "Chhattisgarh", "Chennai": "Tamil Nadu", "Coimbatore": "Tamil Nadu",
    "Cuttack": "Odisha", "Dehradun": "Uttarakhand", "Durgapur": "West Bengal", "Dwarka": "Delhi",
    "Faridabad": "Haryana", "Gaya": "Bihar", "Gurgaon": "Haryana", "Guwahati": "Assam",
    "Haridwar": "Uttarakhand", "Hyderabad": "Telangana", "Indore": "Madhya Pradesh", "Jaipur": "Rajasthan",
    "Jamshedpur": "Jharkhand", "Jodhpur": "Rajasthan", "Kochi": "Kerala", "Kolkata": "West Bengal",
    "Lucknow": "Uttar Pradesh", "Ludhiana": "Punjab", "Mangalore": "Karnataka", "Mumbai": "Maharashtra",
    "Mysore": "Karnataka", "Nagpur": "Maharashtra", "New Delhi": "Delhi", "Noida": "Uttar Pradesh",
    "Patna": "Bihar", "Pune": "Maharashtra", "Raipur": "Chhattisgarh", "Ranchi": "Jharkhand",
    "Silchar": "Assam", "Surat": "Gujarat", "Trivandrum": "Kerala", "Vijayawada": "Andhra Pradesh",
    "Vishakhapatnam": "Andhra Pradesh", "Warangal": "Telangana"
}

ALL_CITIES = sorted(list(CITY_TO_STATE.keys()))
ALL_STATES = sorted(list(set(CITY_TO_STATE.values())))
PROPERTY_TYPES = ["Apartment", "Independent House", "Villa"]
FACING_DIRECTIONS = ["East", "North", "West", "South"]
OWNER_TYPES = ["Owner", "First Owner", "Second Owner", "Builder", "Broker"]
AVAILABILITY_STATUSES = ["Ready to Move", "Under Construction"]
FURNISHED_STATUSES = ["Furnished", "Semi-Furnished", "Unfurnished"]
TRANSPORT_OPTIONS = ["High", "Medium", "Low"]
AMENITY_OPTIONS = ["Gym", "Pool", "Clubhouse", "Security", "Garden", "Playground"]

CITY_SAMPLE_LOCALITY: Dict[str, str] = {
    "Ahmedabad": "SG Highway",
    "Amritsar": "Ranjit Avenue",
    "Bangalore": "Whitefield",
    "Bhopal": "Arera Colony",
    "Bhubaneswar": "Patia",
    "Bilaspur": "Vyapar Vihar",
    "Chennai": "OMR",
    "Coimbatore": "RS Puram",
    "Cuttack": "CDA Sector",
    "Dehradun": "Rajpur Road",
    "Durgapur": "City Centre",
    "Dwarka": "Sector 10",
    "Faridabad": "Sector 15",
    "Gaya": "Civil Lines",
    "Gurgaon": "Cyber City",
    "Guwahati": "GS Road",
    "Haridwar": "Ranipur",
    "Hyderabad": "Gachibowli",
    "Indore": "Vijay Nagar",
    "Jaipur": "Malviya Nagar",
    "Jamshedpur": "Bistupur",
    "Jodhpur": "Shastri Nagar",
    "Kochi": "Kakkanad",
    "Kolkata": "Salt Lake",
    "Lucknow": "Gomti Nagar",
    "Ludhiana": "Model Town",
    "Mangalore": "Kadri",
    "Mumbai": "Andheri West",
    "Mysore": "Jayalakshmipuram",
    "Nagpur": "Dharampeth",
    "New Delhi": "Dwarka",
    "Noida": "Sector 62",
    "Patna": "Boring Road",
    "Pune": "Hinjewadi",
    "Raipur": "Civil Lines",
    "Ranchi": "Morabadi",
    "Silchar": "Tarapur",
    "Surat": "Vesu",
    "Trivandrum": "Kazhakkoottam",
    "Vijayawada": "Benz Circle",
    "Vishakhapatnam": "Madhurawada",
    "Warangal": "Hanamkonda"
}


# -----------------------------------------------------------------------------
# Sidebar: System Status & Presets
# -----------------------------------------------------------------------------

def render_sidebar() -> Optional[Dict[str, Any]]:
    """Render the application sidebar with quick demo presets and user guidance."""
    st.sidebar.title("🏡 Advisor Portal")
    st.sidebar.caption("AI-Powered Real Estate Investment Analytics")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Demo Presets")
    st.sidebar.write("Select a pre-configured listing scenario to test the advisor:")

    preset_choice = st.sidebar.selectbox(
        "Load sample scenario:",
        [
            "Custom Input",
            "Prime Metro 3BHK (Bangalore)",
            "Emerging Tech Suburb (Pune)",
            "Overpriced Luxury 1BHK (Mumbai)"
        ]
    )

    preset_data: Optional[Dict[str, Any]] = None
    if preset_choice == "Prime Metro 3BHK (Bangalore)":
        preset_data = {
            "city": "Bangalore",
            "locality": "Whitefield",
            "property_type": "Apartment",
            "bhk": 3,
            "size_sqft": 1750.0,
            "price_lakhs": 150.0,
            "floor_no": 7,
            "total_floors": 18,
            "year_built": 2022,
            "furnished": "Semi-Furnished",
            "facing": "North-East",
            "owner": "First Owner",
            "avail": "Ready to Move",
            "transport": "High",
            "parking": True,
            "security": True,
            "schools": 5,
            "hospitals": 4,
            "amenities": ["Gym", "Pool", "Clubhouse", "Security", "Garden"],
        }
        st.sidebar.success("🌟 **Prime Metro Asset:** High-demand 3BHK in IT hub with strong civic infrastructure and solid appreciation potential.")
    elif preset_choice == "Emerging Tech Suburb (Pune)":
        preset_data = {
            "city": "Pune",
            "locality": "Hinjewadi",
            "property_type": "Apartment",
            "bhk": 2,
            "size_sqft": 1050.0,
            "price_lakhs": 75.0,
            "floor_no": 4,
            "total_floors": 12,
            "year_built": 2023,
            "furnished": "Semi-Furnished",
            "facing": "East",
            "owner": "Owner",
            "avail": "Under Construction",
            "transport": "Medium",
            "parking": True,
            "security": True,
            "schools": 3,
            "hospitals": 2,
            "amenities": ["Gym", "Clubhouse", "Garden", "Security"],
        }
        st.sidebar.info("🚀 **Growth Corridor:** 2BHK in expanding suburb with balanced pricing and promising rental yields.")
    elif preset_choice == "Overpriced Luxury 1BHK (Mumbai)":
        preset_data = {
            "city": "Mumbai",
            "locality": "Andheri West",
            "property_type": "Apartment",
            "bhk": 1,
            "size_sqft": 420.0,
            "price_lakhs": 220.0,
            "floor_no": 2,
            "total_floors": 4,
            "year_built": 1998,
            "furnished": "Unfurnished",
            "facing": "South",
            "owner": "Second Owner",
            "avail": "Ready to Move",
            "transport": "Low",
            "parking": False,
            "security": False,
            "schools": 1,
            "hospitals": 1,
            "amenities": [],
        }
        st.sidebar.warning("⚠️ **High-Risk Profile:** Small 1BHK at excessive price per sq.ft. with weak civic infrastructure and low upside.")
    else:
        st.sidebar.info("💡 **Custom Mode:** Fill in property attributes and pricing in the main panel to evaluate any custom listing.")

    if "last_preset_choice" not in st.session_state:
        st.session_state["last_preset_choice"] = preset_choice
    elif st.session_state["last_preset_choice"] != preset_choice:
        st.session_state["last_preset_choice"] = preset_choice
        st.session_state["eval_result"] = None

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### 📌 Evaluation Guide")
    st.sidebar.markdown(
        """
        1. **Input Attributes:** Configure location, dimensions, and amenities.
        2. **Run AI Evaluation:** Receive instant investment verdict & 5-year valuation curve.
        3. **Analyze Scorecard:** Review civic infrastructure scores and key value drivers.
        """
    )

    tracking_uri = get_configured_tracking_uri()
    st.sidebar.markdown("---")
    st.sidebar.link_button("📊 View MLflow Tracking UI ↗", tracking_uri, use_container_width=True)

    return preset_data


# -----------------------------------------------------------------------------
# Tab 1: Property Valuation & Investment Recommendation (Step 5.1)
# -----------------------------------------------------------------------------

def render_valuation_tab(preset: Optional[Dict[str, Any]] = None) -> None:
    """Render Tab 1: Interactive Valuation & Investment Recommendation Engine."""
    st.subheader("🎯 Real Estate Investment Valuation & Profitability Advisor")
    st.write("Evaluate any Indian residential listing for instant investment verdicts and 5-year valuation forecasts.")

    # 1. User Input Form
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)

        # Column 1: Location & Property Identity
        with col1:
            st.markdown("##### 📍 Location & Type")
            city_default_idx = ALL_CITIES.index(preset["city"]) if preset and preset.get("city") in ALL_CITIES else ALL_CITIES.index("Bangalore")
            city = st.selectbox("City", ALL_CITIES, index=city_default_idx)

            # Auto-detect state based on city, while allowing user flexibility
            detected_state = CITY_TO_STATE.get(city, "Karnataka")
            state_default_idx = ALL_STATES.index(detected_state) if detected_state in ALL_STATES else 0
            state = st.selectbox(
                "State",
                ALL_STATES,
                index=state_default_idx,
                help="Auto-synced with the selected city. Can also be manually adjusted."
            )

            # Context-aware locality default
            sample_loc = preset.get("locality") if preset else CITY_SAMPLE_LOCALITY.get(city, "Locality_1")
            locality = st.text_input(
                "Locality / Neighborhood",
                value=sample_loc,
                help="Enter locality name or number (e.g. Whitefield, Patia, Koramangala, Locality_87)."
            )

            prop_type_default = PROPERTY_TYPES.index(preset["property_type"]) if preset and preset.get("property_type") in PROPERTY_TYPES else 0
            property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=prop_type_default)

            facing_default = FACING_DIRECTIONS.index(preset["facing"]) if preset and preset.get("facing") in FACING_DIRECTIONS else 0
            facing = st.selectbox("Facing Direction", FACING_DIRECTIONS, index=facing_default)

        # Column 2: Physical Attributes & Pricing
        with col2:
            st.markdown("##### 📐 Physical Dimensions & Price")
            bhk_default = preset.get("bhk", 3) if preset else 3
            bhk = st.slider("BHK Configuration", min_value=1, max_value=6, value=bhk_default, step=1)

            size_default = preset.get("size_sqft", 1500.0) if preset else 1500.0
            size_in_sqft = st.number_input("Property Size (Sq. Ft.)", min_value=250.0, max_value=12000.0, value=float(size_default), step=50.0)

            price_default = preset.get("price_lakhs", 125.0) if preset else 125.0
            price_in_lakhs = st.number_input("Listing Price (₹ in Lakhs)", min_value=10.0, max_value=5000.0, value=float(price_default), step=5.0)

            sub_col_f1, sub_col_f2 = st.columns(2)
            with sub_col_f1:
                floor_no_default = preset.get("floor_no", 5) if preset else 5
                floor_no = st.number_input("Floor No", min_value=0, max_value=60, value=int(floor_no_default), step=1)
            with sub_col_f2:
                total_floors_default = preset.get("total_floors", 14) if preset else 14
                total_floors = st.number_input("Total Floors", min_value=1, max_value=60, value=int(max(floor_no, total_floors_default)), step=1)

            year_default = preset.get("year_built", 2021) if preset else 2021
            year_built = st.number_input("Year Built", min_value=1980, max_value=2026, value=int(year_default), step=1)

        # Column 3: Civic Infrastructure & Amenities
        with col3:
            st.markdown("##### 🏛️ Civic Infrastructure & Specs")
            furnished_default = FURNISHED_STATUSES.index(preset["furnished"]) if preset and preset.get("furnished") in FURNISHED_STATUSES else 1
            furnished_status = st.selectbox("Furnished Status", FURNISHED_STATUSES, index=furnished_default)

            avail_default = AVAILABILITY_STATUSES.index(preset["avail"]) if preset and preset.get("avail") in AVAILABILITY_STATUSES else 0
            availability_status = st.selectbox("Availability Status", AVAILABILITY_STATUSES, index=avail_default)

            transport_default = TRANSPORT_OPTIONS.index(preset["transport"]) if preset and preset.get("transport") in TRANSPORT_OPTIONS else 0
            public_transport = st.selectbox("Public Transport Accessibility", TRANSPORT_OPTIONS, index=transport_default)

            sub_infra_1, sub_infra_2 = st.columns(2)
            with sub_infra_1:
                schools_default = preset.get("schools", 4) if preset else 4
                nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=15, value=int(schools_default), step=1)
            with sub_infra_2:
                hospitals_default = preset.get("hospitals", 3) if preset else 3
                nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=15, value=int(hospitals_default), step=1)

            owner_default = OWNER_TYPES.index(preset["owner"]) if preset and preset.get("owner") in OWNER_TYPES else 1
            owner_type = st.selectbox("Owner Type", OWNER_TYPES, index=owner_default)

            sub_sec_1, sub_sec_2 = st.columns(2)
            with sub_sec_1:
                parking_default = preset.get("parking", True) if preset else True
                parking_space = st.checkbox("Parking Space", value=bool(parking_default))
            with sub_sec_2:
                sec_default = preset.get("security", True) if preset else True
                security_gated = st.checkbox("Gated Security", value=bool(sec_default))

        # Full-width Amenities Selection
        amenities_default = preset.get("amenities", ["Gym", "Pool", "Security", "Garden"]) if preset else ["Gym", "Pool", "Security", "Garden"]
        selected_amenities = st.multiselect(
            "Society Amenities Available",
            AMENITY_OPTIONS,
            default=[a for a in amenities_default if a in AMENITY_OPTIONS],
            help="Select all active amenities provided by the residential community."
        )

        # Evaluation Trigger Button
        submitted = st.button("⚡ Evaluate Property Investment Potential", type="primary", use_container_width=True)

    # 2. Evaluation & Results Display
    if "eval_result" not in st.session_state:
        st.session_state["eval_result"] = None

    if submitted:
        with st.spinner("Calling production MLflow models and evaluating investment potential..."):
            amenities_joined = ", ".join(selected_amenities) if selected_amenities else "None"
            payload: Dict[str, Any] = {
                "Property_Type": property_type,
                "City": city,
                "Locality": locality,
                "State": state,
                "BHK": bhk,
                "Size_in_SqFt": size_in_sqft,
                "Price_in_Lakhs": price_in_lakhs,
                "Floor_No": floor_no,
                "Total_Floors": max(floor_no, total_floors),
                "Year_Built": year_built,
                "Furnished_Status": furnished_status,
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability_status,
                "Amenities": amenities_joined,
                "Public_Transport_Accessibility": public_transport,
                "Parking_Space": "Yes" if parking_space else "No",
                "Security": "Yes" if security_gated else "No",
                "Nearby_Schools": nearby_schools,
                "Nearby_Hospitals": nearby_hospitals,
            }

            st.session_state["eval_result"] = predict_property_investment(payload, log_to_mlflow=True)

    if st.session_state["eval_result"] is not None:
        result = cast(Dict[str, Any], st.session_state["eval_result"])

        st.markdown("---")
        st.subheader("📊 Executive Investment Assessment")

        is_good = bool(result["prediction"] == 1)
        verdict_badge = "badge-good" if is_good else "badge-caution"
        verdict_symbol = "✅" if is_good else "⚠️"

        # Highlight Banner
        st.markdown(
            f"""
            <div style="background-color: {'#e8f5e9' if is_good else '#ffebee'}; padding: 18px 24px; border-radius: 10px; border-left: 6px solid {'#2e7d32' if is_good else '#c62828'};">
                <span style="font-size: 1.4rem; font-weight: 700; color: {'#1b5e20' if is_good else '#b71c1c'};">
                    {verdict_symbol} Verdict: {result['verdict']} — {result['recommendation_level']}
                </span>
                <p style="margin: 6px 0 0 0; color: #424242; font-size: 1rem;">
                    Model Confidence: <b>{result['confidence_percent']}%</b> &nbsp;|&nbsp; 
                    Projected 5-Year Gain: <b>₹{result['appreciation_lakhs']} Lakhs ({result['appreciation_multiplier']}x)</b> &nbsp;|&nbsp;
                    Annualized CAGR: <b>{result['cagr_percent']}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # 4 Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        price_per_sqft = round((price_in_lakhs * 100000.0) / max(size_in_sqft, 1.0), 2)
        with m_col1:
            st.metric(
                label="Current Listing Valuation",
                value=f"₹{price_in_lakhs:.2f} L",
                delta=f"₹{price_per_sqft:,.0f} / sq.ft",
                delta_color="off"
            )

        with m_col2:
            st.metric(
                label="5-Year Projected Valuation",
                value=f"₹{result['predicted_5y_price_lakhs']:.2f} L",
                delta=f"+₹{result['appreciation_lakhs']:.2f} L (+{round((result['appreciation_multiplier']-1)*100, 1)}%)",
                delta_color="normal"
            )

        with m_col3:
            st.metric(
                label="Projected Annual CAGR",
                value=f"{result['cagr_percent']:.2f}%",
                delta="5-Year Horizon",
                delta_color="off"
            )

        with m_col4:
            st.metric(
                label="AI Model Confidence",
                value=f"{result['confidence_percent']:.1f}%",
                delta=f"Latency: {result['latency_ms']:.1f} ms",
                delta_color="off"
            )

        st.markdown("---")

        # Visualizations & Deep Dive Drivers
        chart_col, driver_col = st.columns([3, 2])

        with chart_col:
            st.markdown("##### 📈 Projected 5-Year Valuation Trajectory")
            years = [0, 1, 2, 3, 4, 5]
            cur_p = float(price_in_lakhs)
            cagr_ratio = 1.0 + (float(result["cagr_percent"]) / 100.0)
            projected_curve = [round(cur_p * (cagr_ratio ** yr), 2) for yr in years]

            fig_growth = go.Figure()
            fig_growth.add_trace(go.Scatter(
                x=[f"Year {y}" if y > 0 else "Present" for y in years],
                y=projected_curve,
                mode="lines+markers+text",
                text=[f"₹{v}L" for v in projected_curve],
                textposition="top center",
                line=dict(color="#198754" if is_good else "#dc3545", width=3),
                marker=dict(size=8, color="#0d6efd"),
                fill="tozeroy",
                fillcolor="rgba(25, 135, 84, 0.1)" if is_good else "rgba(220, 53, 69, 0.1)",
                name="Valuation (₹ Lakhs)"
            ))

            fig_growth.update_layout(
                yaxis_title="Property Value (₹ Lakhs)",
                xaxis_title="Investment Horizon",
                margin=dict(l=20, r=20, t=30, b=20),
                height=340,
                hovermode="x unified",
            )
            st.plotly_chart(fig_growth, use_container_width=True)

        with driver_col:
            st.markdown("##### 🔍 Key Value Drivers & Scorecard")

            # Civic Infrastructure Metric Calculation
            t_score = 3 if public_transport == "High" else (2 if public_transport == "Medium" else 1)
            sh_density = nearby_schools + nearby_hospitals
            am_count = len(selected_amenities)
            infra_score = round((
                0.25 * ((t_score - 1) / 2.0) +
                0.25 * (min(sh_density, 20) / 20.0) +
                0.20 * (1 if security_gated else 0) +
                0.15 * (1 if parking_space else 0) +
                0.15 * (min(am_count, 5) / 5.0)
            ) * 100.0, 1)

            st.write(f"**Civic Infrastructure Score:** `{infra_score}/100`")
            st.progress(min(int(infra_score), 100))

            st.markdown(f"""
            - **Local Amenities:** `{am_count}` selected ({', '.join(selected_amenities) if selected_amenities else 'None'})
            - **Civic Density:** `{nearby_schools}` Schools & `{nearby_hospitals}` Hospitals nearby
            - **Transport Accessibility:** `{public_transport}` access tier
            - **Security & Parking:** Gated security `{security_gated}` | Parking `{parking_space}`
            - **Floor Positioning:** Floor `{floor_no}` of `{total_floors}` floors
            """)

            # MLflow MLOps telemetry link
            if result.get("mlflow_run_id"):
                tracking_uri_active = get_configured_tracking_uri()
                run_id_val = result["mlflow_run_id"]
                st.markdown(
                    f"""
                    <div style="background-color: #f1f3f5; padding: 10px 14px; border-radius: 8px; margin-top: 12px; font-size: 0.85rem;">
                        <b>⚡ MLflow Live Audit Run:</b> <code>{run_id_val[:8]}</code><br>
                        <a href="{tracking_uri_active}" target="_blank" style="text-decoration: none; color: #0d6efd; font-weight: 600;">
                            View live query parameters on DagsHub MLflow &rarr;
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# -----------------------------------------------------------------------------
# Main Application Entrypoint
# -----------------------------------------------------------------------------

def main() -> None:
    preset_data = render_sidebar()
    render_valuation_tab(preset=preset_data)


if __name__ == "__main__":
    main()
