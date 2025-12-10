import streamlit as st
import pandas as pd
import joblib

# Sayfa Ayarları
st.set_page_config(
    page_title="Mantar Analiz Sistemi",
    page_icon="🍄",
    layout="wide"  # Daha geniş bir görünüm
)

# Özel CSS
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
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve Bilgi
st.title("🍄 Profesyonel Mantar Analiz Sistemi")


# Mappings (Kullanıcı Dostu Etiketler)
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

# Model Yükleme
@st.cache_resource
def load_data():
    try:
        model = joblib.load('random_forest_model.pkl')
        train_data = pd.read_csv('mushrooms_mini.csv').iloc[:, 1:]
        return model, train_data
    except Exception as e:
        st.error(f"Hata detayları: {e}")
        return None, None

model, train_data = load_data()

if model and train_data is not None:
    with st.form("analysis_form"):
        st.subheader("📝 Mantar Özelliklerini Giriniz")
        
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        
        user_selections = {}
        
        # Her özellik için döngü
        for i, (feature_name, mapping) in enumerate(feature_mappings.items()):
            # Eğitim verisinde var olan değerleri bul (Modelin tanıdığı değerler)
            if feature_name in train_data.columns:
                valid_codes = set(train_data[feature_name].unique())
                
                # Sadece eğitim setinde olan seçenekleri filtrele
                # (Modelin hata vermemesi için sadece tanıdığı kodları içeren etiketleri gösteriyoruz)
                display_options = {f"{label} ({code})": code for label, code in mapping.items() if code in valid_codes}
                
                # Eğer eğitim setinde olup mapping'de olmayan bir kod varsa, onu da ham koduyla ekle (Fallback)
                for code in valid_codes:
                    if code not in display_options.values():
                        display_options[f"Other ({code})"] = code
                
                # Dropdown oluştur
                with cols[i % 3]:
                    # Görünen etiketler (Keys)
                    selected_label = st.selectbox(
                        f"{feature_name.replace('-', ' ').title()}", 
                        options=list(display_options.keys())
                    )
                    # Seçilen etiketin kod karşılığını kaydet
                    user_selections[feature_name] = display_options[selected_label]

        st.markdown("---")
        submitted = st.form_submit_button("🔍 ANALİZ ET")

    if submitted:
        # Tahmin İşlemi
        try:
            # 1. Kullanıcı girdisinden DataFrame oluştur
            new_row = pd.DataFrame([user_selections])
            
            # 2. Eğitim verisiyle birleştir (Sütun yapısını korumak için)
            final_set = pd.concat([train_data, new_row], ignore_index=True)
            
            # 3. One-Hot Encoding
            final_set_encoded = pd.get_dummies(final_set, drop_first=True)
            
            # 4. Son satırı (tahmin edilecek veriyi) al
            prediction_input = final_set_encoded.iloc[[-1]]
            
            # 5. Tahmin
            prediction = model.predict(prediction_input)
            print(prediction[-1])
            result = prediction[-1]
            
            # Sonuç Gösterimi
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if str(result) == '0':
                    st.image("https://cdn-icons-png.flaticon.com/512/1828/1828643.png", width=150) # Green Check
                else:
                    st.image("https://cdn-icons-png.flaticon.com/512/1828/1828843.png", width=150) # Red Warning

            with col_res2:
                st.markdown("### Analiz Sonucu:")
                if str(result) == '0':
                    st.success("### ✅ YENEBİLİR (EDIBLE)")
                    st.write("Model analizine göre bu mantar **güvenli** görünüyor.")
                else:
                    st.error("### ☠️ ZEHİRLİ (POISONOUS)")
                    st.write("Dikkat! Model analizine göre bu mantar **zehirli** olabilir.")
                    
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

else:
    st.error("Gerekli dosyalar (model veya veri seti) bulunamadı.")
