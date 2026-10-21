import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Customer Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #070B14;
    --bg-surface: #0E1626;
    --bg-surface-elevated: #131E33;
    --border-subtle: #1C2D4A;
    --border-highlight: #2A436D;
    --text-primary: #F1F5F9;
    --text-secondary: #8EA3BF;
    --text-muted: #566B88;
    --teal-primary: #00E5C3;
    --teal-glow: rgba(0, 229, 195, 0.15);
    --blue-accent: #0099FF;
    --risk-high: #FF4560;
    --risk-high-bg: rgba(255, 69, 96, 0.12);
    --risk-med: #FFB800;
    --risk-med-bg: rgba(255, 184, 0, 0.12);
    --risk-low: #00E5A0;
    --risk-low-bg: rgba(0, 229, 160, 0.12);
}

/* Global Reset & Streamlit Header Dark Overrides (removes top white strip) */
html, body, [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section {
    background-color: var(--bg-base) !important;
}

header[data-testid="stHeader"],
[data-testid="stHeader"],
.stAppHeader {
    background-color: var(--bg-base) !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border-subtle) !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
    color: var(--text-primary) !important;
}

/* Reduce excessive top whitespace */
.main .block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1400px !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1120 0%, #060913 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
    min-width: 330px !important;
    max-width: 390px !important;
    overflow-x: hidden !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #080D18;
}
::-webkit-scrollbar-thumb {
    background: #1C2D4A;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #2E456E;
}





/* Expander in Sidebar & Main */
div[data-testid="stExpander"] {
    background-color: #0E1626 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    overflow: hidden !important;
    min-width: 0 !important;
}

div[data-testid="stExpander"] details > summary {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 10px !important;
    min-height: 44px !important;
    padding: 10px 14px !important;
    box-sizing: border-box !important;
    background: #0E1626 !important;
    color: #F1F5F9 !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    line-height: 1.4 !important;
}

/* Hide Streamlit's broken expander icon */
div[data-testid="stExpander"] details > summary svg,
div[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"] {
    display: none !important;
}

/* Create a clean replacement arrow */
div[data-testid="stExpander"] details > summary::before {
    content: "›";
    display: inline-block;
    flex: 0 0 12px;
    color: var(--teal-primary);
    font-size: 22px;
    line-height: 1;
    text-align: center;
}

/* Rotate the arrow when expanded */
div[data-testid="stExpander"] details[open] > summary::before {
    transform: rotate(90deg);
}

/* Preserve heading text */
div[data-testid="stExpander"] details > summary p {
    margin: 0 !important;
    color: #F1F5F9 !important;
    line-height: 1.4 !important;
}

div[data-testid="stExpander"] details > summary:hover {
    color: var(--teal-primary) !important;
}

/* Expanded content */
div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
    padding: 12px 14px 14px 14px !important;
    border-top: 1px solid #15223A !important;
    background-color: #0A1120 !important;
    box-sizing: border-box !important;
}




/* Form Controls & Dropdown Customization */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-bottom: 3px !important;
}

div[data-baseweb="select"] {
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #111C30 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    min-height: 38px !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: var(--border-highlight) !important;
}

div[data-baseweb="select"] svg {
    color: var(--teal-primary) !important;
    fill: var(--teal-primary) !important;
}

div[data-baseweb="popover"] {
    background-color: #0E1626 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    box-shadow: 0 12px 32px rgba(0,0,0,0.6) !important;
}

div[data-baseweb="popover"] ul {
    background-color: #0E1626 !important;
    padding: 4px !important;
}

div[data-baseweb="popover"] li {
    color: #E2E8F0 !important;
    background-color: transparent !important;
    padding: 8px 12px !important;
    font-size: 0.84rem !important;
    border-radius: 6px !important;
}

div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: rgba(0, 229, 195, 0.15) !important;
    color: var(--teal-primary) !important;
}

div[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #111C30 !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}

div[data-testid="stNumberInput"] input {
    color: #F1F5F9 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}

div[data-testid="stNumberInput"] button {
    color: var(--teal-primary) !important;
    background: #15223A !important;
    border: none !important;
}

/* Header & Brand Banner */
.cg-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 24px;
    background: linear-gradient(135deg, rgba(14, 22, 38, 0.85) 0%, rgba(10, 16, 29, 0.95) 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

.cg-brand-group {
    display: flex;
    align-items: center;
    gap: 16px;
}

.cg-brand-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(0, 229, 195, 0.2) 0%, rgba(0, 153, 255, 0.2) 100%);
    border: 1px solid rgba(0, 229, 195, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
}

.cg-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #00E5C3 0%, #00B4D8 50%, #0099FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.15;
}

.cg-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 3px 0 0 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.cg-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    background: rgba(0, 229, 160, 0.08);
    border: 1px solid rgba(0, 229, 160, 0.28);
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--risk-low);
    letter-spacing: 0.03em;
}

.cg-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: var(--risk-low);
    box-shadow: 0 0 8px var(--risk-low);
    animation: pulse-dot 2s infinite ease-in-out;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

/* KPI Banner */
.cg-kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 14px;
}

@media (max-width: 992px) {
    .cg-kpi-grid { grid-template-columns: repeat(2, 1fr); }
    .cg-header { flex-direction: column; align-items: flex-start; gap: 14px; }
}

.cg-kpi-card {
    background: linear-gradient(145deg, #0E1726 0%, #0B1220 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.cg-kpi-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-highlight);
}

.cg-kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00E5C3, #0099FF);
    opacity: 0.7;
}

.cg-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.cg-kpi-val {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.1;
}

.cg-kpi-sub {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 5px;
}

/* Notice Banner */
.cg-notice-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    background: rgba(0, 229, 195, 0.05);
    border: 1px solid rgba(0, 229, 195, 0.18);
    border-radius: 10px;
    font-size: 0.78rem;
    color: #94A3B8;
    margin-bottom: 22px;
}

/* Result Cards */
.cg-result-card {
    border-radius: 18px;
    padding: 28px 24px;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}

.cg-result-card.churn {
    background: linear-gradient(145deg, #1A0D14 0%, #120A10 100%);
    border-color: rgba(255, 69, 96, 0.5);
    box-shadow: 0 8px 32px rgba(255, 69, 96, 0.12);
}

.cg-result-card.medium {
    background: linear-gradient(145deg, #1C150A 0%, #130F08 100%);
    border-color: rgba(255, 184, 0, 0.5);
    box-shadow: 0 8px 32px rgba(255, 184, 0, 0.12);
}

.cg-result-card.safe {
    background: linear-gradient(145deg, #091A14 0%, #07130F 100%);
    border-color: rgba(0, 229, 160, 0.5);
    box-shadow: 0 8px 32px rgba(0, 229, 160, 0.12);
}

.cg-result-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 30px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.cg-result-badge.churn { background: var(--risk-high-bg); color: var(--risk-high); border: 1px solid var(--risk-high); }
.cg-result-badge.medium { background: var(--risk-med-bg); color: var(--risk-med); border: 1px solid var(--risk-med); }
.cg-result-badge.safe { background: var(--risk-low-bg); color: var(--risk-low); border: 1px solid var(--risk-low); }

.cg-prob-display {
    font-size: 3.6rem;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0;
    letter-spacing: -0.02em;
}

.cg-prob-display.churn { color: var(--risk-high); }
.cg-prob-display.medium { color: var(--risk-med); }
.cg-prob-display.safe { color: var(--risk-low); }

.cg-verdict-text {
    font-size: 0.88rem;
    line-height: 1.5;
    margin-top: 14px;
    padding: 12px 18px;
    border-radius: 10px;
    font-weight: 500;
}

.cg-verdict-text.churn { background: var(--risk-high-bg); color: #FFA3B2; border-left: 3px solid var(--risk-high); }
.cg-verdict-text.medium { background: var(--risk-med-bg); color: #FFE399; border-left: 3px solid var(--risk-med); }
.cg-verdict-text.safe { background: var(--risk-low-bg); color: #A3FFD8; border-left: 3px solid var(--risk-low); }

/* Dual Mini Metrics in Result */
.cg-dual-metrics {
    display: flex;
    gap: 12px;
    margin-top: 18px;
}

.cg-dual-metric-box {
    flex: 1;
    background: rgba(14, 22, 38, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 10px 14px;
    text-align: center;
}

.cg-dual-metric-box .label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
}

.cg-dual-metric-box .value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F1F5F9;
    margin-top: 2px;
}

/* Modern Card Surfaces */
.cg-card {
    background: #0E1626;
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
}

.cg-card-title {
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #F1F5F9;
    margin: 0 0 16px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.cg-card-title-badge {
    font-size: 0.7rem;
    color: var(--teal-primary);
    background: rgba(0, 229, 195, 0.1);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
}

/* SHAP Horizontal Bars */
.cg-shap-item {
    margin-bottom: 14px;
}

.cg-shap-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    margin-bottom: 5px;
}

.cg-shap-name {
    color: #E2E8F0;
    font-weight: 500;
}

.cg-shap-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--teal-primary);
    font-weight: 600;
}

.cg-shap-bar-bg {
    width: 100%;
    height: 8px;
    background: #162238;
    border-radius: 4px;
    overflow: hidden;
}

.cg-shap-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #00E5C3 0%, #0099FF 100%);
    border-radius: 4px;
}

/* Recommendation Cards */
.cg-rec-card {
    background: linear-gradient(145deg, #101B2E 0%, #0C1524 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.cg-rec-card:hover {
    transform: translateX(3px);
    border-color: var(--border-highlight);
}

.cg-rec-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.cg-rec-pill {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(0, 229, 195, 0.1);
    color: var(--teal-primary);
    border: 1px solid rgba(0, 229, 195, 0.25);
}

.cg-rec-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0;
}

.cg-rec-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 4px 0 0 0;
    line-height: 1.45;
}

/* Profile Key-Value Grid */
.cg-attr-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px 14px;
}

.cg-attr-row {
    display: flex;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #131E33;
    font-size: 0.8rem;
}

.cg-attr-k { color: var(--text-muted); }
.cg-attr-v { color: #E2E8F0; font-weight: 600; }

/* Empty State */
.cg-empty-state {
    border: 1px dashed var(--border-subtle);
    border-radius: 16px;
    padding: 50px 30px;
    text-align: center;
    background: rgba(14, 22, 38, 0.3);
}

.cg-empty-icon {
    font-size: 2.8rem;
    margin-bottom: 12px;
    opacity: 0.8;
}

.cg-empty-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #CBD5E1;
    margin: 0 0 8px 0;
}

.cg-empty-desc {
    font-size: 0.85rem;
    color: var(--text-muted);
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.5;
}

/* Button Overrides */
div[data-testid="stButton"] > button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00E5C3 0%, #0099FF 100%) !important;
    color: #070B14 !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(0, 229, 195, 0.25) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    box-shadow: 0 6px 20px rgba(0, 229, 195, 0.38) !important;
    transform: translateY(-1px) !important;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background: #0E1626 !important;
    color: #CBD5E1 !important;
    border: 1px solid var(--border-subtle) !important;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: var(--teal-primary) !important;
    color: #FFFFFF !important;
}

/* Custom Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 4px;
    margin-bottom: 22px;
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    background-color: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0 16px !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    color: var(--teal-primary) !important;
    border-bottom: 2px solid var(--teal-primary) !important;
}

/* Image Card Container */
.cg-img-container {
    background: #0E1626;
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 18px;
}

/* Footer */
.cg-footer {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    margin-top: 50px;
    padding-top: 24px;
    border-top: 1px solid var(--border-subtle);
    line-height: 1.8;
}

.cg-footer a {
    color: var(--teal-primary);
    text-decoration: none;
    font-weight: 600;
    transition: opacity 0.2s ease;
}

.cg-footer a:hover {
    text-decoration: underline;
    opacity: 0.85;
}

.cg-badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}

.cg-schema-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    background: #111C30;
    color: #94A3B8;
    border: 1px solid #1C2D4A;
    border-radius: 6px;
    padding: 2px 8px;
}

/* Restore Streamlit Material icon rendering */
[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                 "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    direction: ltr !important;
    font-feature-settings: "liga" !important;
    -webkit-font-feature-settings: "liga" !important;
}

</style>
""", unsafe_allow_html=True)


# ── Load Model Pipeline ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("churn_model_pipeline.pkl")

try:
    pipeline = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

TRAINING_COLS = [
    "age", "gender", "region_category", "membership_category",
    "joined_through_referral", "preferred_offer_types", "medium_of_operation",
    "internet_option", "days_since_last_login", "avg_time_spent",
    "avg_transaction_value", "avg_frequency_login_days", "points_in_wallet",
    "used_special_discount", "offer_application_preference",
    "past_complaint", "complaint_status", "feedback",
]


# ── Helper for Image Padding (prevents label clipping) ─────────────────────────
@st.cache_data
def get_padded_image(img_path, pad_left=35, pad_right=25, pad_top=20, pad_bottom=20):
    if not os.path.exists(img_path):
        return None
    im = Image.open(img_path)
    bg_color = im.getpixel((0, 0))
    new_w = im.width + pad_left + pad_right
    new_h = im.height + pad_top + pad_bottom
    new_im = Image.new(im.mode, (new_w, new_h), bg_color)
    new_im.paste(im, (pad_left, pad_top))
    return new_im


# ── Profile Preset Management ──────────────────────────────────────────────────
PRESETS = {
    "default": {
        "age": 35, "gender": "F", "region_category": "City",
        "membership_category": "Basic Membership",
        "joined_through_referral": "No", "preferred_offer_types": "Gift Vouchers/Coupons",
        "medium_of_operation": "Desktop", "internet_option": "Wi-Fi",
        "days_since_last_login": 5, "avg_frequency_login_days": 10.0,
        "avg_time_spent": 300.0, "avg_transaction_value": 25000.0,
        "points_in_wallet": 500.0, "used_special_discount": "Yes",
        "offer_application_preference": "Yes", "past_complaint": "No",
        "complaint_status": "Not Applicable", "feedback": "Products always in Stock"
    },
    "high_risk": {
        "age": 48, "gender": "M", "region_category": "Village",
        "membership_category": "No Membership",
        "joined_through_referral": "No", "preferred_offer_types": "Without Offers",
        "medium_of_operation": "Smartphone", "internet_option": "Mobile_Data",
        "days_since_last_login": 24, "avg_frequency_login_days": 26.0,
        "avg_time_spent": 42.0, "avg_transaction_value": 8500.0,
        "points_in_wallet": 115.0, "used_special_discount": "No",
        "offer_application_preference": "No", "past_complaint": "Yes",
        "complaint_status": "Unsolved", "feedback": "Poor Product Quality"
    },
    "low_risk": {
        "age": 34, "gender": "F", "region_category": "City",
        "membership_category": "Platinum Membership",
        "joined_through_referral": "Yes", "preferred_offer_types": "Gift Vouchers/Coupons",
        "medium_of_operation": "Desktop", "internet_option": "Wi-Fi",
        "days_since_last_login": 2, "avg_frequency_login_days": 4.0,
        "avg_time_spent": 480.0, "avg_transaction_value": 48000.0,
        "points_in_wallet": 880.0, "used_special_discount": "Yes",
        "offer_application_preference": "Yes", "past_complaint": "No",
        "complaint_status": "Not Applicable", "feedback": "Quality Customer Care"
    }
}

def apply_preset(name):
    target = PRESETS.get(name, PRESETS["default"])
    for k, v in target.items():
        st.session_state[f"inp_{k}"] = v

# Initialize session state for all inputs if missing
if "inp_age" not in st.session_state:
    apply_preset("default")


# ── Sidebar: Customer Profiler with Collapsible Sections ───────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 2px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.25rem;">👤</span>
            <span style="font-size: 1.05rem; font-weight: 800; color: #F1F5F9; letter-spacing: -0.01em;">Customer Profiler</span>
        </div>
        <p style="font-size: 0.74rem; color: #64748B; margin: 3px 0 0 0;">
            Configure individual parameters for inference
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick Loaders
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #00E5C3;">Quick Scenarios</span>
        <span style="font-size: 0.68rem; color: #64748B;">1-Click Pre-fill</span>
    </div>
    """, unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.button("🚨 High", on_click=apply_preset, args=("high_risk",), use_container_width=True, help="Load customer with high churn signals")
    with p_col2:
        st.button("🛡️ Low", on_click=apply_preset, args=("low_risk",), use_container_width=True, help="Load loyal high-engagement customer")
    with p_col3:
        st.button("🔄 Reset", on_click=apply_preset, args=("default",), use_container_width=True, help="Reset to baseline default profile")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Group 1: Demographics & Region (Collapsible)
    with st.expander("👤 1. Demographics & Region", expanded=True):
        age = st.slider("Customer Age", min_value=10, max_value=90, step=1, key="inp_age")
        gender = st.selectbox("Gender", ["F", "M"], key="inp_gender")
        region_category = st.selectbox("Region Category", ["City", "Town", "Village"], key="inp_region_category")

    # Group 2: Membership & Loyalty (Collapsible)
    with st.expander("💳 2. Membership & Loyalty", expanded=True):
        membership_category = st.selectbox(
            "Membership Tier",
            ["No Membership", "Basic Membership", "Silver Membership",
             "Gold Membership", "Premium Membership", "Platinum Membership"],
            key="inp_membership_category"
        )
        joined_through_referral = st.selectbox("Joined via Referral?", ["Yes", "No"], key="inp_joined_through_referral")
        preferred_offer_types = st.selectbox(
            "Preferred Offer Type",
            ["Gift Vouchers/Coupons", "Credit/Debit Card Offers", "Without Offers"],
            key="inp_preferred_offer_types"
        )

    # Group 3: Engagement & Platform (Collapsible)
    with st.expander("📱 3. Engagement & Platform", expanded=False):
        medium_of_operation = st.selectbox("Primary Platform", ["Desktop", "Smartphone", "Both"], key="inp_medium_of_operation")
        internet_option = st.selectbox("Internet Connectivity", ["Wi-Fi", "Mobile_Data", "Fiber_Optic"], key="inp_internet_option")
        days_since_last_login = st.slider("Days Since Last Login", 0, 30, key="inp_days_since_last_login")
        avg_frequency_login_days = st.slider("Avg Login Frequency (Days)", 0.0, 30.0, step=0.5, key="inp_avg_frequency_login_days")
        avg_time_spent = st.slider("Avg Session Time (Mins)", 0.0, 3000.0, step=10.0, key="inp_avg_time_spent")

    # Group 4: Transactional & Wallet (Collapsible)
    with st.expander("💰 4. Transactional & Wallet", expanded=False):
        avg_transaction_value = st.number_input(
            "Avg Transaction Value (₦)", min_value=0.0, max_value=100000.0, step=500.0, key="inp_avg_transaction_value"
        )
        points_in_wallet = st.number_input(
            "Points in Wallet", min_value=0.0, max_value=2000.0, step=10.0, key="inp_points_in_wallet"
        )
        used_special_discount = st.selectbox("Used Special Discount?", ["Yes", "No"], key="inp_used_special_discount")
        offer_application_preference = st.selectbox("Applies for Offers?", ["Yes", "No"], key="inp_offer_application_preference")

    # Group 5: Service & Feedback (Collapsible)
    with st.expander("💬 5. Service & Feedback", expanded=False):
        past_complaint = st.selectbox("Has Past Complaint?", ["Yes", "No"], key="inp_past_complaint")
        complaint_status = st.selectbox(
            "Complaint Resolution Status",
            ["Not Applicable", "Solved", "Solved in Follow-up", "Unsolved"],
            key="inp_complaint_status"
        )
        feedback = st.selectbox(
            "Customer Feedback Sentiment",
            ["Products always in Stock", "Quality Customer Care", "User Friendly Website",
             "No reason specified", "Poor Website", "Poor Customer Support",
             "Poor Product Quality", "Too many ads"],
            key="inp_feedback"
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Run Churn Analysis", type="primary", use_container_width=True)


# ── Top Enterprise Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="cg-header">
    <div class="cg-brand-group">
        <div class="cg-brand-icon">🛡️</div>
        <div>
            <h1 class="cg-title">ChurnGuard AI</h1>
            <p class="cg-subtitle">Customer Churn Prediction & Retention Analytics Platform · XGBoost Pipeline</p>
        </div>
    </div>
    <div>
        <div class="cg-status-pill">
            <span class="cg-status-dot"></span>
            Production Model Online
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Overview Performance Metrics (Verified Model Benchmarks) ──────────────────
st.markdown("""
<div class="cg-kpi-grid">
    <div class="cg-kpi-card">
        <div class="cg-kpi-label">F1-Score</div>
        <div class="cg-kpi-val">94.01%</div>
        <div class="cg-kpi-sub">Precision-Recall Harmonic Mean</div>
    </div>
    <div class="cg-kpi-card">
        <div class="cg-kpi-label">ROC-AUC</div>
        <div class="cg-kpi-val">97.60%</div>
        <div class="cg-kpi-sub">Separability Across Thresholds</div>
    </div>
    <div class="cg-kpi-card">
        <div class="cg-kpi-label">Accuracy</div>
        <div class="cg-kpi-val">93.47%</div>
        <div class="cg-kpi-sub">Holdout Test Set Classification</div>
    </div>
    <div class="cg-kpi-card">
        <div class="cg-kpi-label">Training Cohort</div>
        <div class="cg-kpi-val">36,992</div>
        <div class="cg-kpi-sub">Enterprise Customer Records</div>
    </div>
    <div class="cg-kpi-card">
        <div class="cg-kpi-label">Features</div>
        <div class="cg-kpi-val">18</div>
        <div class="cg-kpi-sub">Multi-Modal Behavioral Predictors</div>
    </div>
</div>
<div class="cg-notice-bar">
    <span>💡 <strong>Model Benchmark Overview:</strong> Evaluated on 7,399 unseen holdout test samples with 5-fold cross validation. These metrics evaluate the trained model pipeline and are distinct from individual customer prediction outputs.</span>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model file `churn_model_pipeline.pkl` could not be loaded. Please ensure it is located in the working directory.")
    st.stop()


# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮  Single Customer Prediction",
    "📂  Batch Portfolio Prediction",
    "📊  Model Intelligence & SHAP Governance"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Customer Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if predict_btn:
        input_data = pd.DataFrame([{
            "age": age, "gender": gender,
            "region_category": region_category,
            "membership_category": membership_category,
            "joined_through_referral": joined_through_referral,
            "preferred_offer_types": preferred_offer_types,
            "medium_of_operation": medium_of_operation,
            "internet_option": internet_option,
            "days_since_last_login": days_since_last_login,
            "avg_time_spent": avg_time_spent,
            "avg_transaction_value": avg_transaction_value,
            "avg_frequency_login_days": avg_frequency_login_days,
            "points_in_wallet": points_in_wallet,
            "used_special_discount": used_special_discount,
            "offer_application_preference": offer_application_preference,
            "past_complaint": past_complaint,
            "complaint_status": complaint_status,
            "feedback": feedback,
        }])

        churn_pred = pipeline.predict(input_data)[0]
        churn_prob = pipeline.predict_proba(input_data)[0][1]
        prob_pct   = churn_prob * 100
        safe_pct   = 100.0 - prob_pct

        if prob_pct >= 70:
            risk_class = "churn"
            badge_icon = "🚨"
            badge_text = "HIGH CHURN RISK"
            verdict_msg = "Immediate retention intervention recommended. Customer exhibits critical flight risk signals across wallet balance, loyalty tier, or recent service friction."
            posture = "Urgent Intervention"
        elif prob_pct >= 40:
            risk_class = "medium"
            badge_icon = "⚡"
            badge_text = "MODERATE CHURN RISK"
            verdict_msg = "Monitor closely and initiate targeted engagement. Customer displays moderate vulnerability and could benefit from value-add incentives."
            posture = "Proactive Retention"
        else:
            risk_class = "safe"
            badge_icon = "✅"
            badge_text = "LOW CHURN RISK"
            verdict_msg = "Customer profile indicates high stability and healthy loyalty indicators. Continue standard lifecycle marketing and relationship nurturing."
            posture = "Stable - Maintain Strategy"

        col_left, col_right = st.columns([1.05, 0.95], gap="large")

        with col_left:
            # Executive Verdict Card
            st.markdown(f"""
            <div class="cg-result-card {risk_class}">
                <div class="cg-result-badge {risk_class}">
                    <span>{badge_icon}</span>
                    <span>{badge_text}</span>
                </div>
                <div class="cg-prob-display {risk_class}">{prob_pct:.1f}%</div>
                <p style="font-size: 0.8rem; color: #8EA3BF; text-transform: uppercase; letter-spacing: 0.08em; margin: 0;">
                    Predicted Churn Probability
                </p>
                <div class="cg-dual-metrics">
                    <div class="cg-dual-metric-box">
                        <div class="label">Churn Likelihood</div>
                        <div class="value" style="color: {'#FF4560' if risk_class=='churn' else '#FFB800' if risk_class=='medium' else '#00E5A0'};">
                            {prob_pct:.1f}%
                        </div>
                    </div>
                    <div class="cg-dual-metric-box">
                        <div class="label">Retention Probability</div>
                        <div class="value" style="color: #00E5C3;">{safe_pct:.1f}%</div>
                    </div>
                    <div class="cg-dual-metric-box">
                        <div class="label">Recommended Posture</div>
                        <div class="value" style="font-size: 0.88rem; padding-top: 4px;">{posture}</div>
                    </div>
                </div>
                <div class="cg-verdict-text {risk_class}">
                    {verdict_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk Gauge Meter
            st.markdown("<div style='margin-top: 18px;'>", unsafe_allow_html=True)
            st.markdown("""
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #64748B; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">
                <span>0% Safe Zone</span>
                <span>40% Moderate</span>
                <span>70% Critical Threshold</span>
                <span>100%</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(int(prob_pct), 0), 100))
            st.markdown("</div>", unsafe_allow_html=True)

            # Submitted Profile Audit
            st.markdown("""
            <div class="cg-card" style="margin-top: 20px;">
                <div class="cg-card-title">
                    <span>Evaluated Profile Snapshot</span>
                    <span class="cg-card-title-badge">Real-Time Payload</span>
                </div>
                <div class="cg-attr-grid">
                    <div class="cg-attr-row"><span class="cg-attr-k">Customer Age / Gender</span><span class="cg-attr-v">""" + f"{age} / {gender}" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Region</span><span class="cg-attr-v">""" + f"{region_category}" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Membership Tier</span><span class="cg-attr-v">""" + f"{membership_category}" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Wallet Points</span><span class="cg-attr-v">""" + f"{points_in_wallet:,.0f}" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Avg Transaction Value</span><span class="cg-attr-v">""" + f"₦{avg_transaction_value:,.0f}" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Days Since Last Login</span><span class="cg-attr-v">""" + f"{days_since_last_login} days" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Past Complaint</span><span class="cg-attr-v">""" + f"{past_complaint} ({complaint_status})" + """</span></div>
                    <div class="cg-attr-row"><span class="cg-attr-k">Feedback Sentiment</span><span class="cg-attr-v">""" + f"{feedback}" + """</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Top Churn Drivers (SHAP)
            st.markdown("""
            <div class="cg-card">
                <div class="cg-card-title">
                    <span>Key Churn Drivers (SHAP Attribution)</span>
                    <span class="cg-card-title-badge">XGBoost Impact</span>
                </div>
                <div class="cg-shap-item">
                    <div class="cg-shap-header">
                        <span class="cg-shap-name">🏆 Points in Wallet</span>
                        <span class="cg-shap-score">+0.4607</span>
                    </div>
                    <div class="cg-shap-bar-bg"><div class="cg-shap-bar-fill" style="width: 100%;"></div></div>
                    <p style="font-size: 0.72rem; color: #64748B; margin: 3px 0 0 0;">Primary risk determinant. Balances below 500 sharply elevate churn likelihood.</p>
                </div>

                <div class="cg-shap-item">
                    <div class="cg-shap-header">
                        <span class="cg-shap-name">💳 Membership Category</span>
                        <span class="cg-shap-score">+0.3552</span>
                    </div>
                    <div class="cg-shap-bar-bg"><div class="cg-shap-bar-fill" style="width: 77%;"></div></div>
                    <p style="font-size: 0.72rem; color: #64748B; margin: 3px 0 0 0;">Basic and No-Membership tiers display significantly higher flight velocity.</p>
                </div>

                <div class="cg-shap-item">
                    <div class="cg-shap-header">
                        <span class="cg-shap-name">💬 Customer Feedback</span>
                        <span class="cg-shap-score">+0.0401</span>
                    </div>
                    <div class="cg-shap-bar-bg"><div class="cg-shap-bar-fill" style="width: 18%;"></div></div>
                    <p style="font-size: 0.72rem; color: #64748B; margin: 3px 0 0 0;">Negative service or product remarks act as direct dissatisfaction indicators.</p>
                </div>

                <div class="cg-shap-item">
                    <div class="cg-shap-header">
                        <span class="cg-shap-name">💰 Avg Transaction Value</span>
                        <span class="cg-shap-score">+0.0301</span>
                    </div>
                    <div class="cg-shap-bar-bg"><div class="cg-shap-bar-fill" style="width: 14%;"></div></div>
                    <p style="font-size: 0.72rem; color: #64748B; margin: 3px 0 0 0;">Transaction spending volume reflects ongoing customer account commitment.</p>
                </div>

                <div class="cg-shap-item">
                    <div class="cg-shap-header">
                        <span class="cg-shap-name">📱 Medium of Operation</span>
                        <span class="cg-shap-score">+0.0091</span>
                    </div>
                    <div class="cg-shap-bar-bg"><div class="cg-shap-bar-fill" style="width: 8%;"></div></div>
                    <p style="font-size: 0.72rem; color: #64748B; margin: 3px 0 0 0;">Platform access medium and multi-device adoption correlate with stickiness.</p>
                </div>

                <div style="margin-top: 14px; padding: 10px 12px; background: rgba(0, 153, 255, 0.06); border: 1px solid rgba(0, 153, 255, 0.15); border-radius: 8px;">
                    <p style="font-size: 0.72rem; color: #8EA3BF; margin: 0; line-height: 1.45;">
                        <strong style="color: #00E5C3;">Governance Note:</strong> SHAP values indicate statistical feature attribution within the trained XGBoost model and do not constitute direct causal relationships.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Business Retention Recommendations
            st.markdown("""
            <div class="cg-card">
                <div class="cg-card-title">
                    <span>Prescriptive Retention Playbook</span>
                    <span class="cg-card-title-badge">Action Matrix</span>
                </div>

                <div class="cg-rec-card">
                    <div class="cg-rec-top">
                        <p class="cg-rec-title">🎯 Targeted Wallet Re-engagement</p>
                        <span class="cg-rec-pill">Loyalty Incentive</span>
                    </div>
                    <p class="cg-rec-desc">
                        Trigger automated promotional bonus points or cashback offers for customers falling below the critical wallet point threshold.
                    </p>
                </div>

                <div class="cg-rec-card">
                    <div class="cg-rec-top">
                        <p class="cg-rec-title">⬆️ Membership Upgrade Pathway</p>
                        <span class="cg-rec-pill">Tier Migration</span>
                    </div>
                    <p class="cg-rec-desc">
                        Target Basic & No-Membership users with trial access to Silver or Gold tiers, reducing churn propensities observed in lower tiers.
                    </p>
                </div>

                <div class="cg-rec-card">
                    <div class="cg-rec-top">
                        <p class="cg-rec-title">🛠️ Priority Service Resolution</p>
                        <span class="cg-rec-pill">Concierge Care</span>
                    </div>
                    <p class="cg-rec-desc">
                        Fast-track unresolved tickets and initiate dedicated outreach for accounts indicating dissatisfaction or poor product quality feedback.
                    </p>
                </div>

                <div class="cg-rec-card" style="margin-bottom: 0;">
                    <div class="cg-rec-top">
                        <p class="cg-rec-title">📡 Real-Time Surveillance & Alerting</p>
                        <span class="cg-rec-pill">Automated Pipeline</span>
                    </div>
                    <p class="cg-rec-desc">
                        Flag elevated-risk accounts daily using this automated inference pipeline to trigger timely proactive account manager intervention.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Refined Empty State
        st.markdown("""
        <div class="cg-empty-state">
            <div class="cg-empty-icon">🔮</div>
            <h3 class="cg-empty-title">Awaiting Customer Profile Evaluation</h3>
            <p class="cg-empty-desc">
                Select customer parameters in the left sidebar or load a quick test profile, then click <strong>Run Churn Analysis</strong> to generate real-time risk scores, SHAP attributions, and retention strategies.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Portfolio Prediction
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">
            <span>Portfolio Batch Inference</span>
            <span class="cg-card-title-badge">CSV Pipeline</span>
        </div>
        <p style="font-size: 0.84rem; color: #8EA3BF; margin: 0 0 14px 0; line-height: 1.6;">
            Upload a CSV containing multiple customer records for high-throughput batch scoring. The pipeline validates required features, executes model inference, appends probability scores and risk categorizations, and exports enriched datasets.
        </p>
        <div style="font-size: 0.74rem; color: #64748B; font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">
            Required Feature Schema (18 Columns):
        </div>
        <div class="cg-badge-row">
            <span class="cg-schema-pill">age</span>
            <span class="cg-schema-pill">gender</span>
            <span class="cg-schema-pill">region_category</span>
            <span class="cg-schema-pill">membership_category</span>
            <span class="cg-schema-pill">joined_through_referral</span>
            <span class="cg-schema-pill">preferred_offer_types</span>
            <span class="cg-schema-pill">medium_of_operation</span>
            <span class="cg-schema-pill">internet_option</span>
            <span class="cg-schema-pill">days_since_last_login</span>
            <span class="cg-schema-pill">avg_time_spent</span>
            <span class="cg-schema-pill">avg_transaction_value</span>
            <span class="cg-schema-pill">avg_frequency_login_days</span>
            <span class="cg-schema-pill">points_in_wallet</span>
            <span class="cg-schema-pill">used_special_discount</span>
            <span class="cg-schema-pill">offer_application_preference</span>
            <span class="cg-schema-pill">past_complaint</span>
            <span class="cg-schema-pill">complaint_status</span>
            <span class="cg-schema-pill">feedback</span>
        </div>
        <p style="font-size: 0.74rem; color: #64748B; margin: 8px 0 0 0;">
            <em>Note: Identifier columns such as <code>customer_id</code> are automatically preserved and mapped through to output files.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV Generator
    sample_batch_df = pd.DataFrame([
        {'customer_id': 'CUST-1001', 'age': 32, 'gender': 'F', 'region_category': 'City', 'membership_category': 'Platinum Membership', 'joined_through_referral': 'Yes', 'preferred_offer_types': 'Gift Vouchers/Coupons', 'medium_of_operation': 'Desktop', 'internet_option': 'Wi-Fi', 'days_since_last_login': 2, 'avg_time_spent': 450.0, 'avg_transaction_value': 45000.0, 'avg_frequency_login_days': 5.0, 'points_in_wallet': 850.0, 'used_special_discount': 'Yes', 'offer_application_preference': 'Yes', 'past_complaint': 'No', 'complaint_status': 'Not Applicable', 'feedback': 'Quality Customer Care'},
        {'customer_id': 'CUST-1002', 'age': 45, 'gender': 'M', 'region_category': 'Village', 'membership_category': 'No Membership', 'joined_through_referral': 'No', 'preferred_offer_types': 'Without Offers', 'medium_of_operation': 'Smartphone', 'internet_option': 'Mobile_Data', 'days_since_last_login': 25, 'avg_time_spent': 45.0, 'avg_transaction_value': 8000.0, 'avg_frequency_login_days': 28.0, 'points_in_wallet': 120.0, 'used_special_discount': 'No', 'offer_application_preference': 'No', 'past_complaint': 'Yes', 'complaint_status': 'Unsolved', 'feedback': 'Poor Product Quality'},
        {'customer_id': 'CUST-1003', 'age': 28, 'gender': 'F', 'region_category': 'Town', 'membership_category': 'Gold Membership', 'joined_through_referral': 'Yes', 'preferred_offer_types': 'Credit/Debit Card Offers', 'medium_of_operation': 'Both', 'internet_option': 'Fiber_Optic', 'days_since_last_login': 12, 'avg_time_spent': 180.0, 'avg_transaction_value': 28000.0, 'avg_frequency_login_days': 14.0, 'points_in_wallet': 520.0, 'used_special_discount': 'Yes', 'offer_application_preference': 'No', 'past_complaint': 'No', 'complaint_status': 'Not Applicable', 'feedback': 'User Friendly Website'},
        {'customer_id': 'CUST-1004', 'age': 52, 'gender': 'M', 'region_category': 'City', 'membership_category': 'Basic Membership', 'joined_through_referral': 'No', 'preferred_offer_types': 'Without Offers', 'medium_of_operation': 'Desktop', 'internet_option': 'Wi-Fi', 'days_since_last_login': 22, 'avg_time_spent': 75.0, 'avg_transaction_value': 12000.0, 'avg_frequency_login_days': 20.0, 'points_in_wallet': 210.0, 'used_special_discount': 'No', 'offer_application_preference': 'No', 'past_complaint': 'Yes', 'complaint_status': 'Unsolved', 'feedback': 'Poor Customer Support'},
        {'customer_id': 'CUST-1005', 'age': 39, 'gender': 'F', 'region_category': 'City', 'membership_category': 'Silver Membership', 'joined_through_referral': 'Yes', 'preferred_offer_types': 'Gift Vouchers/Coupons', 'medium_of_operation': 'Smartphone', 'internet_option': 'Wi-Fi', 'days_since_last_login': 4, 'avg_time_spent': 320.0, 'avg_transaction_value': 31000.0, 'avg_frequency_login_days': 8.0, 'points_in_wallet': 710.0, 'used_special_discount': 'Yes', 'offer_application_preference': 'Yes', 'past_complaint': 'No', 'complaint_status': 'Not Applicable', 'feedback': 'Products always in Stock'},
        {'customer_id': 'CUST-1006', 'age': 41, 'gender': 'M', 'region_category': 'Town', 'membership_category': 'Premium Membership', 'joined_through_referral': 'Yes', 'preferred_offer_types': 'Credit/Debit Card Offers', 'medium_of_operation': 'Both', 'internet_option': 'Fiber_Optic', 'days_since_last_login': 3, 'avg_time_spent': 410.0, 'avg_transaction_value': 48000.0, 'avg_frequency_login_days': 4.0, 'points_in_wallet': 890.0, 'used_special_discount': 'Yes', 'offer_application_preference': 'Yes', 'past_complaint': 'No', 'complaint_status': 'Not Applicable', 'feedback': 'Quality Customer Care'}
    ])
    sample_csv_data = sample_batch_df.to_csv(index=False).encode('utf-8')

    upload_col1, upload_col2 = st.columns([1.2, 0.8], gap="medium")
    with upload_col1:
        uploaded_file = st.file_uploader("Upload Customer Portfolio CSV", type=["csv"], help="Upload CSV containing required customer attributes")
    with upload_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📄  Download Verified Sample CSV Template",
            data=sample_csv_data,
            file_name="sample_churnguard_batch.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download a pre-formatted 6-customer CSV template ready for testing"
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = [c for c in TRAINING_COLS if c not in df.columns]

            if missing_cols:
                st.error(f"❌ Schema validation failed: The uploaded file is missing {len(missing_cols)} required columns: `{', '.join(missing_cols)}`")
            else:
                X_batch = df[TRAINING_COLS].copy()
                probs   = pipeline.predict_proba(X_batch)[:, 1]
                preds   = pipeline.predict(X_batch)

                def get_risk_label(p):
                    if p >= 0.70:   return "🚨 High Risk"
                    elif p >= 0.40: return "⚡ Medium Risk"
                    else:           return "✅ Low Risk"

                results_df = df.copy()
                results_df["Churn Probability (%)"] = (probs * 100).round(2)
                results_df["Prediction"]            = ["Churn" if p == 1 else "No Churn" for p in preds]
                results_df["Risk Level"]            = [get_risk_label(p) for p in probs]

                total_records = len(results_df)
                n_churn       = int((preds == 1).sum())
                n_safe        = int((preds == 0).sum())
                n_high        = int((probs >= 0.70).sum())
                n_med         = int(((probs >= 0.40) & (probs < 0.70)).sum())
                n_low         = int((probs < 0.40).sum())
                avg_prob      = probs.mean() * 100

                # Revenue at risk calculation if transaction value is present
                at_risk_revenue = 0.0
                if "avg_transaction_value" in results_df.columns:
                    at_risk_revenue = results_df[results_df["Prediction"] == "Churn"]["avg_transaction_value"].sum()

                # Batch Overview Cards
                st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
                bc1, bc2, bc3, bc4, bc5 = st.columns(5)
                bc1.metric("Customers Processed", f"{total_records:,}")
                bc2.metric("Predicted Churn", f"{n_churn:,}", f"{(n_churn/total_records*100):.1f}%", delta_color="inverse")
                bc3.metric("Predicted Retained", f"{n_safe:,}", f"{(n_safe/total_records*100):.1f}%")
                bc4.metric("Mean Churn Risk", f"{avg_prob:.1f}%")
                bc5.metric("Revenue at Risk", f"₦{at_risk_revenue:,.0f}" if at_risk_revenue > 0 else "N/A")

                st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

                # Segment distribution summary
                st.markdown(f"""
                <div style="background: #0E1626; border: 1px solid #1C2D4A; border-radius: 12px; padding: 14px 18px; margin-bottom: 18px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                    <div style="font-size: 0.82rem; color: #8EA3BF; font-weight: 600;">Portfolio Risk Cohort Breakdown:</div>
                    <div style="display: flex; gap: 16px; font-size: 0.82rem;">
                        <span style="color: #FF4560; font-weight: 700;">🚨 High Risk: {n_high} ({n_high/total_records*100:.1f}%)</span>
                        <span style="color: #FFB800; font-weight: 700;">⚡ Moderate Risk: {n_med} ({n_med/total_records*100:.1f}%)</span>
                        <span style="color: #00E5A0; font-weight: 700;">✅ Low Risk: {n_low} ({n_low/total_records*100:.1f}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Display Results Table
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download Enriched Batch Predictions (CSV)",
                    data=csv_out,
                    file_name="churnguard_predictions_output.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"❌ Error processing batch file: {e}")

    else:
        st.markdown("""
        <div class="cg-empty-state">
            <div class="cg-empty-icon">📂</div>
            <h3 class="cg-empty-title">Ready for Batch File Ingestion</h3>
            <p class="cg-empty-desc">
                Drag and drop a customer portfolio CSV above, or download our pre-formatted sample template to test automated predictions across multiple accounts simultaneously.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Intelligence & SHAP Governance
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="cg-card">
        <div class="cg-card-title">
            <span>Model Specifications & Operational Architecture</span>
            <span class="cg-card-title-badge">Production Registry</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
            <div style="background: #111C30; padding: 12px 14px; border-radius: 8px; border: 1px solid #1C2D4A;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase;">Classifier Engine</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-top: 2px;">XGBoost Classifier</div>
            </div>
            <div style="background: #111C30; padding: 12px 14px; border-radius: 8px; border: 1px solid #1C2D4A;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase;">Hyperparameters</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-top: 2px;">200 Trees · Depth 3 · LR 0.05</div>
            </div>
            <div style="background: #111C30; padding: 12px 14px; border-radius: 8px; border: 1px solid #1C2D4A;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase;">Preprocessing</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-top: 2px;">ColumnTransformer + Scaling</div>
            </div>
            <div style="background: #111C30; padding: 12px 14px; border-radius: 8px; border: 1px solid #1C2D4A;">
                <div style="font-size: 0.7rem; color: #64748B; text-transform: uppercase;">Evaluation Benchmark</div>
                <div style="font-size: 0.95rem; font-weight: 700; color: #F1F5F9; margin-top: 2px;">ROC-AUC 97.60% · F1 94.01%</div>
            </div>
        </div>
        <p style="font-size: 0.8rem; color: #8EA3BF; margin: 0; line-height: 1.55;">
            The ChurnGuard pipeline combines rigorous scikit-learn preprocessing (median/most-frequent imputation, standard scaling, and ordinal encoding) with an optimized Extreme Gradient Boosting classifier. The pipeline handles missing values gracefully and provides probabilistic calibration across 18 behavioral signals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sub-tabs for Visual Diagnostics
    diag_tab1, diag_tab2, diag_tab3 = st.tabs([
        "🔬  SHAP Global Explainability",
        "📈  ROC Curves & Evaluation",
        "⚖️  Responsible AI & Causation Guidelines"
    ])

    # ── Sub-tab 1: SHAP Global Explainability ──
    with diag_tab1:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0 0 4px 0; font-weight: 600;">
                Cooperative Game Theory Explainability (SHAP TreeExplainer)
            </p>
            <p style="font-size: 0.8rem; color: #8EA3BF; margin: 0; line-height: 1.55;">
                SHAP values quantify the marginal contribution of each behavioral signal to the predicted churn probability across all 36,992 customer records. Explore individual views below with unclipped, high-contrast visual fidelity.
            </p>
        </div>
        """, unsafe_allow_html=True)

        shap_view = st.radio(
            "Select SHAP Diagnostic View:",
            ["📊 Global Feature Importance (Bar Plot)",
             "🐝 Directional Distribution (Beeswarm)",
             "📉 Wallet Balance Dependence Curve",
             "⚖️ Side-by-Side Global Analysis"],
            horizontal=True
        )

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        if shap_view == "📊 Global Feature Importance (Bar Plot)":
            col_sb1, col_sb2 = st.columns([1.1, 0.9], gap="large")
            with col_sb1:
                im_bar = get_padded_image("visuals/shap_summary_bar.png", pad_left=40, pad_right=30)
                if im_bar:
                    st.image(im_bar, caption="SHAP Global Mean Absolute Impact Across All Features", use_container_width=True)
                else:
                    st.info("Visual asset `shap_summary_bar.png` not found.")
            with col_sb2:
                st.markdown("""
                <div class="cg-card">
                    <div class="cg-card-title">
                        <span>Feature Importance Ranking</span>
                        <span class="cg-card-title-badge">Mean |SHAP|</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #8EA3BF; line-height: 1.6;">
                        The bar plot measures overall magnitude of influence. Key observations:
                    </p>
                    <ul style="font-size: 0.82rem; color: #CBD5E1; padding-left: 18px; line-height: 1.7;">
                        <li><strong style="color: #00E5C3;">Points in Wallet (+0.4607):</strong> The single most dominant factor by a wide margin, accounting for nearly half of model attribution.</li>
                        <li><strong style="color: #00E5C3;">Membership Category (+0.3552):</strong> Tier classification represents the second most critical anchor. Basic/No-membership tiers exhibit elevated vulnerability.</li>
                        <li><strong style="color: #00E5C3;">Customer Feedback (+0.0401):</strong> Qualitative satisfaction remarks provide immediate early warning signals of dissatisfaction.</li>
                        <li><strong style="color: #00E5C3;">Avg Transaction Value (+0.0301):</strong> Spending frequency correlates with account commitment.</li>
                        <li><strong style="color: #00E5C3;">Medium of Operation (+0.0091):</strong> Omnichannel platform usage correlates with higher customer retention.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        elif shap_view == "🐝 Directional Distribution (Beeswarm)":
            col_sw1, col_sw2 = st.columns([1.1, 0.9], gap="large")
            with col_sw1:
                im_bee = get_padded_image("visuals/shap_summary_beeswarm.png", pad_left=40, pad_right=30)
                if im_bee:
                    st.image(im_bee, caption="SHAP Summary Beeswarm Plot — Feature Value vs. Directional Impact", use_container_width=True)
                else:
                    st.info("Visual asset `shap_summary_beeswarm.png` not found.")
            with col_sw2:
                st.markdown("""
                <div class="cg-card">
                    <div class="cg-card-title">
                        <span>How to Interpret Beeswarm Plots</span>
                        <span class="cg-card-title-badge">Directionality</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #8EA3BF; line-height: 1.6;">
                        Each dot represents a customer record from the test set:
                    </p>
                    <ul style="font-size: 0.82rem; color: #CBD5E1; padding-left: 18px; line-height: 1.7;">
                        <li><strong style="color: #FF4560;">Red Dots:</strong> High value of the feature (e.g. high wallet points, high transaction value).</li>
                        <li><strong style="color: #0099FF;">Blue Dots:</strong> Low value of the feature (e.g. 0 wallet points, basic membership).</li>
                        <li><strong style="color: #F1F5F9;">Right of Zero (Positive SHAP):</strong> Pushes the prediction <em>toward churn</em>. Notice how blue dots (low wallet points) extend far to the right!</li>
                        <li><strong style="color: #F1F5F9;">Left of Zero (Negative SHAP):</strong> Pushes the prediction <em>toward retention</em>. Red dots (high wallet points) cluster heavily on the left.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        elif shap_view == "📉 Wallet Balance Dependence Curve":
            col_dp1, col_dp2 = st.columns([1.15, 0.85], gap="large")
            with col_dp1:
                im_dep = get_padded_image("visuals/shap_dependence_wallet.png", pad_left=30, pad_right=25)
                if im_dep:
                    st.image(im_dep, caption="SHAP Dependence Plot: Points in Wallet vs. Attribution to Churn", use_container_width=True)
                else:
                    st.info("Visual asset `shap_dependence_wallet.png` not found.")
            with col_dp2:
                st.markdown("""
                <div class="cg-card">
                    <div class="cg-card-title">
                        <span>Wallet Dependence Analysis</span>
                        <span class="cg-card-title-badge">Threshold Effect</span>
                    </div>
                    <p style="font-size: 0.82rem; color: #8EA3BF; line-height: 1.6;">
                        This dependence plot uncovers a non-linear relationship between customer loyalty points and churn risk:
                    </p>
                    <ul style="font-size: 0.82rem; color: #CBD5E1; padding-left: 18px; line-height: 1.7;">
                        <li><strong style="color: #FF4560;">Critical Cliff (&lt; 500 points):</strong> When a customer's wallet balance drops below approximately 500 points, SHAP value surges positive (+0.5 to +1.2), indicating rapid elevation in churn risk.</li>
                        <li><strong style="color: #00E5A0;">Safety Plateau (&gt; 700 points):</strong> Above 700 points, the attribution flattens into negative SHAP territory (-0.5), acting as a powerful retention stabilizer.</li>
                        <li><strong style="color: #00E5C3;">Business Implication:</strong> Retention incentive campaigns should specifically target accounts approaching the 500-point threshold to prevent customers from falling off this cliff.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        else: # Side-by-Side Global Analysis
            col_side1, col_side2 = st.columns(2, gap="medium")
            with col_side1:
                im_b = get_padded_image("visuals/shap_summary_bar.png", pad_left=35, pad_right=20)
                if im_b:
                    st.image(im_b, caption="Global Feature Importance (Mean |SHAP|)", use_container_width=True)
            with col_side2:
                im_w = get_padded_image("visuals/shap_summary_beeswarm.png", pad_left=35, pad_right=20)
                if im_w:
                    st.image(im_w, caption="SHAP Beeswarm Distribution (Value vs. Direction)", use_container_width=True)

    # ── Sub-tab 2: ROC Curves & Evaluation ──
    with diag_tab2:
        st.markdown("""
        <div style="margin-bottom: 16px;">
            <p style="font-size: 0.86rem; color: #CBD5E1; margin: 0 0 4px 0; font-weight: 600;">
                Holdout Validation & Multi-Model Comparative Benchmarking
            </p>
            <p style="font-size: 0.8rem; color: #8EA3BF; margin: 0; line-height: 1.55;">
                Model performance across 7,399 holdout records (20% stratified test split). XGBoost was selected for production deployment based on optimal F1-score and discrimination power.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Comparative Metrics Table
        st.markdown("""
        <div class="cg-card" style="padding: 16px 20px; margin-bottom: 20px;">
            <div class="cg-card-title" style="margin-bottom: 12px;">
                <span>Comparative Model Benchmark Summary (Holdout Test Set)</span>
                <span class="cg-card-title-badge">Verified Metrics</span>
            </div>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; text-align: left;">
                    <thead>
                        <tr style="border-bottom: 1px solid #1C2D4A; color: #64748B; text-transform: uppercase; font-size: 0.72rem;">
                            <th style="padding: 8px 12px;">Model Architecture</th>
                            <th style="padding: 8px 12px;">Accuracy</th>
                            <th style="padding: 8px 12px;">Precision</th>
                            <th style="padding: 8px 12px;">Recall</th>
                            <th style="padding: 8px 12px;">F1-Score</th>
                            <th style="padding: 8px 12px;">ROC-AUC</th>
                            <th style="padding: 8px 12px;">Production Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #131E33; color: #94A3B8;">
                            <td style="padding: 10px 12px; font-weight: 600;">Logistic Regression</td>
                            <td style="padding: 10px 12px;">77.56%</td>
                            <td style="padding: 10px 12px;">79.06%</td>
                            <td style="padding: 10px 12px;">79.62%</td>
                            <td style="padding: 10px 12px;">79.34%</td>
                            <td style="padding: 10px 12px;">83.29%</td>
                            <td style="padding: 10px 12px; color: #64748B;">Baseline Only</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #131E33; color: #CBD5E1;">
                            <td style="padding: 10px 12px; font-weight: 600;">Random Forest</td>
                            <td style="padding: 10px 12px;">93.16%</td>
                            <td style="padding: 10px 12px;">92.91%</td>
                            <td style="padding: 10px 12px;">94.58%</td>
                            <td style="padding: 10px 12px;">93.74%</td>
                            <td style="padding: 10px 12px;">97.55%</td>
                            <td style="padding: 10px 12px; color: #0099FF;">Benchmark Candidate</td>
                        </tr>
                        <tr style="background: rgba(0, 229, 195, 0.05); color: #F8FAFC;">
                            <td style="padding: 10px 12px; font-weight: 700; color: #00E5C3;">XGBoost (Deployed)</td>
                            <td style="padding: 10px 12px; font-weight: 700;">93.47%</td>
                            <td style="padding: 10px 12px; font-weight: 700;">93.29%</td>
                            <td style="padding: 10px 12px; font-weight: 700;">94.75%</td>
                            <td style="padding: 10px 12px; font-weight: 700;">94.01%</td>
                            <td style="padding: 10px 12px; font-weight: 700;">97.60%</td>
                            <td style="padding: 10px 12px; font-weight: 700; color: #00E5A0;">● Active Production</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visuals: ROC Curves and Model Comparison side-by-side
        col_eval1, col_eval2 = st.columns([1, 1.3], gap="large")
        with col_eval1:
            im_roc = get_padded_image("visuals/roc_curves.png", pad_left=25, pad_right=25)
            if im_roc:
                st.image(im_roc, caption="ROC Curves: XGBoost (AUC 97.60%) vs. RF (97.55%) vs. LR (83.29%)", use_container_width=True)
            else:
                st.info("Visual asset `roc_curves.png` not found.")
        with col_eval2:
            im_cmp = get_padded_image("visuals/model_comparison.png", pad_left=25, pad_right=25)
            if im_cmp:
                st.image(im_cmp, caption="Performance Comparison Across Accuracy, Precision, Recall, F1, and AUC", use_container_width=True)
            else:
                st.info("Visual asset `model_comparison.png` not found.")

        # Full-Width Confusion Matrices
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        im_cm = get_padded_image("visuals/confusion_matrices.png", pad_left=30, pad_right=30)
        if im_cm:
            st.image(im_cm, caption="Confusion Matrices Across All Models (Logistic Regression, Random Forest, XGBoost)", use_container_width=True)
            st.markdown("""
            <div style="background: #0E1626; border: 1px solid #1C2D4A; border-radius: 12px; padding: 14px 18px; margin-top: 8px;">
                <p style="font-size: 0.8rem; color: #8EA3BF; margin: 0; line-height: 1.55;">
                    <strong style="color: #00E5C3;">Confusion Matrix Diagnostics:</strong> In enterprise retention, <em>False Negatives (missed churners)</em> are significantly more costly than <em>False Positives (contacting a customer who was going to stay)</em>. XGBoost achieved a recall of 94.75%, capturing the overwhelming majority of at-risk customers while sustaining high precision (93.29%) to prevent budget wastage.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Sub-tab 3: Responsible AI & Governance ──
    with diag_tab3:
        st.markdown("""
        <div class="cg-card">
            <div class="cg-card-title">
                <span>Enterprise AI Governance & Causal Decision Guidelines</span>
                <span class="cg-card-title-badge">Operational Ethics</span>
            </div>
            <div style="font-size: 0.84rem; color: #8EA3BF; line-height: 1.7;">
                <p>
                    <strong style="color: #F1F5F9;">1. Association vs. Causation Guardrail:</strong><br>
                    SHAP attribution scores quantify predictive correlation in the training distribution—they do <em>not</em> prove causal mechanics. For example, while <code>points_in_wallet</code> strongly associates with customer departures, artificially depositing points into a dissatisfied customer's account will not necessarily prevent departure if unresolved support tickets or poor product quality are the true root causes.
                </p>
                <p>
                    <strong style="color: #F1F5F9;">2. Empirical Validation via Controlled Trials (A/B Testing):</strong><br>
                    Retention playbook initiatives recommended by ChurnGuard AI should be evaluated via randomized controlled trials (A/B testing) with dedicated holdout control groups before automated company-wide rollout. This measures net incremental retention and ensures customer lifetime value (LTV) gains exceed retention program expenses.
                </p>
                <p>
                    <strong style="color: #F1F5F9;">3. Production Drift & Model Recalibration:</strong><br>
                    Customer purchase dynamics, product features, and macro-economic factors drift over time. Production instances should log daily inference distributions and execute pipeline recalibration on a scheduled quarterly cadence to prevent concept drift.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Clean Enterprise Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="cg-footer">
    Built by <strong style="color: #00E5C3;">Sankeerth S</strong> ·
    ChurnGuard AI Enterprise Platform
    <br>
    <a href="https://github.com/sankeerth-26" target="_blank">GitHub Profile</a>
    &nbsp;·&nbsp;
    <a href="https://www.linkedin.com/in/sankeerth-s-1bbba1294" target="_blank">LinkedIn Profile</a>
    <br>
    <span style="font-size: 0.72rem; color: #475569;">Licensed under MIT · Production-Ready Customer Churn Analytics</span>
</div>
""", unsafe_allow_html=True)
