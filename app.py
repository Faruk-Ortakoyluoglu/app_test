import streamlit as st
import pandas as pd
import joblib

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="Professional Mushroom Classification System",
    page_icon="🍄",
    layout="wide",
)

# ---------------------- GLOBAL STYLE ---------------------- #
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #e3f2fd 0, #f8f9fa 40%, #f8f9fa 100%);
    }

    /* Başlık alanı */
    .hero-title {
        text-align: center;
        font-family: "Helvetica", sans-serif;
        font-weight: 800;
        font-size: 2.3rem;
        letter-spacing: .03em;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 2.2rem;
    }

    /* Kart görünümü */
    .card {
        background-color: #ffffff;
        border-radius: 18px;
        padding: 24px 28px 28px 28px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
    .card-header {
        font-weight: 700;
        font-size: 1.05rem;
        color: #111827;
        display: flex;
        align-items: center;
        gap: .4rem;
        margin-bottom: 1.2rem;
    }

    /* Selectbox label’ları */
    .stSelectbox label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }

    /* Buton */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: white;
        font-weight: 700;
        padding: 0.8rem 0;
        border-radius: 999px;
        border: none;
        letter-spacing: .05em;
        text-transform: uppercase;
        box-shadow: 0 10px 25px rgba(22, 163, 74, 0.35);
        transition: 0.18s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #22c55e, #15803d);
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 16px 35px rgba(22, 163, 74, 0.45);
    }

    /* Bilgi kutusu */
    .info-box {
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        color: #064e3b;
        padding: 18px 22px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.15);
        display: flex;
        align-items: center;
        margin-top: 28px;
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .info-icon {
        font-size: 22px;
        margin-right: 14px;
        background: rgba(255, 255, 255, 0.85);
        width: 44px;
        height: 44px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 999px;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.15);
    }

    /* Sonuç kartı */
    .result-card {
        margin-top: 1.2rem;
        padding: 1.3rem 1.5rem;
        border-radius: 16px;
        display: flex;
        gap: 1rem;
        align-items: center;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
    }
    .result-safe {
        background: linear-gradient(120deg, #dcfce7, #bbf7d0);
        border: 1px solid #4ade80;
        color: #065f46;
    }
    .result-toxic {
        background: linear-gradient(120deg, #fee2e2, #fecaca);
        border: 1px solid #f87171;
        color: #7f1d1d;
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: .2rem;
    }

    /* Uyarı kutusu */
    .disclaimer-box {
        background: #fff7e6;
        color: #92400e;
        padding: 14px 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.85rem;
        border: 1px solid #fed7aa;
        margin-top: 26px;
        box-shadow: 0 5px 15px rgba(15, 23, 42, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------- TITLE ---------------------- #
st.markdown(
    '<div class="hero-title">🍄 Professional Mushroom Analysis System</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Select the observable features of a mushroom and let the model estimate whether it is <strong>edible</strong> or <strong>poisonous</strong>.</div>',
    unsafe_allow_html=True,
)

# ---------------------- FEATURE MAPPINGS ---------------------- #
feature_mappings = {
    "cap-shape": {
        "bell": "b",
        "conical": "c",
        "convex": "x",
        "flat": "f",
        "knobbed": "k",
        "sunken": "s",
    },
    "cap-surface": {
        "fibrous": "f",
        "grooves": "g",
        "scaly": "y",
        "smooth": "s",
    },
    "cap-color": {
        "brown": "n",
        "buff": "b",
        "cinnamon": "c",
        "gray": "g",
        "green": "r",
        "pink": "p",
        "purple": "u",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
    "bruises": {"bruises": "t", "no": "f"},
    "odor": {
        "almond": "a",
        "anise": "l",
        "creosote": "c",
        "fishy": "y",
        "foul": "f",
        "musty": "m",
        "none": "n",
        "pungent": "p",
        "spicy": "s",
    },
    "gill-attachment": {
        "attached": "a",
        "descending": "d",
        "free": "f",
        "notched": "n",
    },
    "gill-spacing": {"close": "c", "crowded": "w", "distant": "d"},
    "gill-size": {"broad": "b", "narrow": "n"},
    "gill-color": {
        "black": "k",
        "brown": "n",
        "buff": "b",
        "chocolate": "h",
        "gray": "g",
        "green": "r",
        "orange": "o",
        "pink": "p",
        "purple": "u",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
}

# ---------------------- LOAD MODEL & DATA ---------------------- #
@st.cache_resource
def load_data():
    try:
        model = joblib.load("random_forest_model.pkl")
        train_data = pd.read_csv("mushrooms_mini.csv").iloc[:, 1:]
        return model, train_data
    except Exception as e:
        st.error(f"Error while loading model or dataset: {e}")
        return None, None


model, train_data = load_data()

# ---------------------- SIDEBAR ---------------------- #
with st.sidebar:
    st.markdown("### ℹ️ About this app")
    st.write(
        """
        This demo uses a **Random Forest classifier** trained on 
        the classic UCI Mushroom dataset.

        - Inputs: Categorical mushroom features  
        - Output: Edible vs. poisonous prediction  
        - Error rate: ~ **0.1%** on test data
        """
    )
    st.markdown("---")
    st.caption("Never rely solely on automated predictions for real-life consumption decisions.")

# ---------------------- MAIN FORM ---------------------- #
if (model is not None) and (train_data is not None):

    container = st.container()
    with container:
        st.markdown(
            '<div class="card"><div class="card-header">📝 Enter Mushroom Features</div>',
            unsafe_allow_html=True,
        )

        with st.form("analysis_form"):
            col1, col2, col3 = st.columns(3)
            cols = [col1, col2, col3]

            user_selections = {}

            for i, (feature_name, mapping) in enumerate(feature_mappings.items()):
                if feature_name in train_data.columns:
                    valid_codes = set(train_data[feature_name].unique())

                    display_options = {
                        f"{label} ({code})": code
                        for label, code in mapping.items()
                        if code in valid_codes
                    }

                    # fallback: eğitim setinde olup mapping’de olmayan kodlar
                    for code in valid_codes:
                        if code not in display_options.values():
                            display_options[f"Other ({code})"] = code

                    with cols[i % 3]:
                        selected_label = st.selectbox(
                            feature_name.replace("-", " ").title(),
                            options=list(display_options.keys()),
                        )
                        user_selections[feature_name] = display_options[selected_label]

            st.markdown("---")
            submitted = st.form_submit_button("🔍 Analyze")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- PREDICTION & RESULT ---------------------- #
    if submitted:
        try:
            new_row = pd.DataFrame([user_selections])
            final_set = pd.concat([train_data, new_row], ignore_index=True)
            final_set_encoded = pd.get_dummies(final_set, drop_first=True)
            prediction_input = final_set_encoded.iloc[[-1]]

            prediction = model.predict(prediction_input)
            result = prediction[-1]

            if str(result) == "0":
                # EDIBLE
                st.markdown(
                    """
                    <div class="result-card result-safe">
                        <div style="font-size: 2.2rem;">✅</div>
                        <div>
                            <div class="result-title">EDIBLE MUSHROOM</div>
                            <div>According to the model, this mushroom is <strong>likely safe to eat</strong>. 
                            Always double-check with reliable sources before making real-life decisions.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # POISONOUS
                st.markdown(
                    """
                    <div class="result-card result-toxic">
                        <div style="font-size: 2.2rem;">☠️</div>
                        <div>
                            <div class="result-title">POTENTIALLY POISONOUS</div>
                            <div>The model predicts this mushroom as <strong>poisonous or unsafe</strong>. 
                            Do <u>not</u> consume and always consult a professional mycologist.</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

else:
    st.error("Required files (model or dataset) were not found on the server.")

# ---------------------- INFO & DISCLAIMER ---------------------- #
st.markdown(
    """
    <div class="info-box">
        <div class="info-icon">📊</div>
        <div>
            <strong>Model Accuracy:</strong> Based on the UCI Mushroom dataset, the trained Random Forest model
            achieves an estimated error rate of <strong>around 0.1%</strong> on held-out test data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> This tool is built for educational and demonstrational purposes only.
        Automated predictions can be wrong. Never rely solely on this app to decide whether a mushroom is safe
        to eat.
    </div>
    """,
    unsafe_allow_html=True,
)