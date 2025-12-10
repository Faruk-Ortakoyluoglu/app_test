import streamlit as st
import pandas as pd
import joblib

# Page Settings
st.set_page_config(
    page_title="Mushroom Analysis System",
    page_icon="🍄",
    layout="wide"  # Wider layout
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E7D32;
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 12px;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        transform: scale(1.02);
    }
    h1 {
        text-align: center;
        color: #1B5E20;
        font-family: 'Helvetica', sans-serif;
    }
    .stSelectbox label {
        font-size: 16px;
        font-weight: 600;
        color: #424242;
    }
    .info-box {
        background: linear-gradient(135deg, #a8ff78 0%, #78ffd6 100%);
        color: #1b5e20;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        margin-bottom: 25px;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.4);
    }
    .info-icon {
        font-size: 24px;
        margin-right: 15px;
        background: rgba(255,255,255,0.8);
        width: 45px;
        height: 45px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 50%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .disclaimer-box {
        background: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 0.9rem;
        border: 1px solid #ffeeba;
        margin-top: 40px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown("""
    This demo uses a **Random Forest classifier** trained on the classic **UCI Mushroom dataset**.
    
    *   **Inputs:** Categorical mushroom features
    *   **Output:** Edible vs. poisonous prediction
    *   **Error rate:** ~ **0.1%** on test data
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="font-size: 12px; color: gray;">
        Never rely solely on automated predictions for real-life consumption decisions.
    </div>
    """, unsafe_allow_html=True)

# Title and Info
st.title("🍄 Professional Mushroom Analysis System")




# Mappings (User Friendly Labels)
feature_mappings = {
    'cap-shape': {
        'bell': 'b', 'conical': 'c', 'convex': 'x', 'flat': 'f', 'knobbed': 'k', 'sunken': 's'
    },
    'cap-surface': {
        'fibrous': 'f', 'grooves': 'g', 'scaly': 'y', 'smooth': 's'
    },
    'cap-color': {
        'brown': 'n', 'buff': 'b', 'cinnamon': 'c', 'gray': 'g', 'green': 'r', 
        'pink': 'p', 'purple': 'u', 'red': 'e', 'white': 'w', 'yellow': 'y'
    },
    'bruises': {
        'bruises': 't', 'no': 'f'
    },
    'odor': {
        'almond': 'a', 'anise': 'l', 'creosote': 'c', 'fishy': 'y', 'foul': 'f', 
        'musty': 'm', 'none': 'n', 'pungent': 'p', 'spicy': 's'
    },
    'gill-attachment': {
        'attached': 'a', 'descending': 'd', 'free': 'f', 'notched': 'n'
    },
    'gill-spacing': {
        'close': 'c', 'crowded': 'w', 'distant': 'd'
    },
    'gill-size': {
        'broad': 'b', 'narrow': 'n'
    },
    'gill-color': {
        'black': 'k', 'brown': 'n', 'buff': 'b', 'chocolate': 'h', 'gray': 'g', 
        'green': 'r', 'orange': 'o', 'pink': 'p', 'purple': 'u', 'red': 'e', 
        'white': 'w', 'yellow': 'y'
    }
}

# Model Loading
@st.cache_resource
def load_data():
    try:
        model = joblib.load('random_forest_model.pkl')
        train_data = pd.read_csv('mushrooms_mini.csv').iloc[:, 1:]
        return model, train_data
    except Exception as e:
        st.error(f"Error details: {e}")
        return None, None

model, train_data = load_data()

if model and train_data is not None:
    with st.form("analysis_form"):
        st.subheader("📝 Enter Mushroom Features")
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        user_selections = {}
        
        # Loop for each feature
        for i, (feature_name, mapping) in enumerate(feature_mappings.items()):
            # Find values present in training data (Values recognized by the model)
            if feature_name in train_data.columns:
                valid_codes = set(train_data[feature_name].unique())
                
                # Filter options only present in the training set
                # (We only show labels containing codes recognized by the model to prevent errors)
                display_options = {f"{label} ({code})": code for label, code in mapping.items() if code in valid_codes}
                
                # If there is a code in the training set that is not in the mapping, add it as raw code (Fallback)
                for code in valid_codes:
                    if code not in display_options.values():
                        display_options[f"Other ({code})"] = code
                
                # Create Dropdown
                with cols[i % 3]:
                    # Visible labels (Keys)
                    selected_label = st.selectbox(
                        f"{feature_name.replace('-', ' ').title()}", 
                        options=list(display_options.keys())
                    )
                    # Save the code corresponding to the selected label
                    user_selections[feature_name] = display_options[selected_label]

        st.markdown("---")
        submitted = st.form_submit_button("🔍 ANALYZE")

    if submitted:
        # Prediction Process
        try:
            # 1. Create DataFrame from user input
            new_row = pd.DataFrame([user_selections])
            
            # 2. Combine with training data (To preserve column structure)
            final_set = pd.concat([train_data, new_row], ignore_index=True)
            
            # 3. One-Hot Encoding
            final_set_encoded = pd.get_dummies(final_set, drop_first=True)
            
            # 4. Get the last row (data to be predicted)
            prediction_input = final_set_encoded.iloc[[-1]]
            
            # 5. Prediction
            prediction = model.predict(prediction_input)
            print(prediction[-1])
            result = prediction[-1]
            
            # Result Display
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if str(result) == '0':
                    st.image("https://cdn-icons-png.flaticon.com/512/1828/1828643.png", width=150) # Green Check
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/1828/1828843.png", width=150) # Red Warning

            with col_res2:
                st.markdown("### Analysis Result:")
                if str(result) == '0':
                    st.success("### ✅ EDIBLE")
                    st.write("According to the model analysis, this mushroom looks **safe**.")
                else:
                    st.error("### ☠️ POISONOUS")
                    st.write("Warning! According to the model analysis, this mushroom might be **poisonous**.")
                    
        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.error("Required files (model or dataset) not found.")


st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>Disclaimer:</strong> Errors may occur in automated predictions. Please be careful and verify before consumption.
    </div>
""", unsafe_allow_html=True)
