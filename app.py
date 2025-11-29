import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2em;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    .prediction-box {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-top: 2rem;
        text-align: center;
    }
    .prediction-value {
        font-size: 3em;
        font-weight: 700;
        color: #333;
        margin: 0.5rem 0;
    }
    .prediction-label {
        font-size: 0.9em;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open('best_xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'best_xgb_model.pkl' not found!")
        st.info("Please save your model using: `pickle.dump(best_model, open('best_xgb_model.pkl', 'wb'))`")
        return None

model = load_model()

# Title and description
st.title("🏠 House Price Predictor")
st.markdown("### Predict property prices in lacs using XGBoost model")
st.markdown("---")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    posted_by = st.selectbox(
        "Posted By",
        options=["", "Dealer", "Owner", "Builder"],
        index=0
    )
    
    bhk = st.selectbox(
        "BHK",
        options=["", "1", "2", "3", "4", "5", "5+"],
        index=0
    )
    
    under_construction = st.checkbox("Under Construction")

with col2:
    city = st.selectbox(
        "City",
        options=["", "Bangalore", "Lalitpur", "Other", "Mumbai", "Pune", "Noida", 
                 "Kolkata", "Maharashtra", "Chennai", "Ghaziabad", "Jaipur", 
                 "Chandigarh", "Faridabad", "Mohali", "Vadodara", "Gurgaon", 
                 "Surat", "Nagpur", "Lucknow", "Indore", "Bhubaneswar", "Bhopal",
                 "Kochi", "Visakhapatnam", "Bhiwadi", "Coimbatore", "Goa", 
                 "Dehradun", "Ranchi", "Mangalore", "Sonipat", "Gandhinagar",
                 "Secunderabad", "Palghar", "Kanpur", "Guwahati", "Raipur",
                 "Jamshedpur", "Rajkot", "Siliguri", "Agra", "Patna", "Panchkula",
                 "Vijayawada", "Jamnagar", "Aurangabad", "Raigad", "Dharuhera",
                 "Thrissur", "Durgapur", "Gwalior", "Meerut"],
        index=0
    )
    
    square_ft = st.selectbox(
        "Square Feet Range",
        options=["", "0-100", "100-200", "200-300", "300-400", "400-500", 
                 "500-700", "700-1000", "1000-1500", "1500-2000", 
                 "2000-5000", "5000+"],
        index=0
    )
    
    rera = st.checkbox("RERA Approved")

st.markdown("---")

# Predict button
if st.button("🎯 Predict Price"):
    # Validation
    if not posted_by:
        st.error("❌ Please select 'Posted By'")
    elif not city:
        st.error("❌ Please select a city")
    elif not bhk:
        st.error("❌ Please select BHK")
    elif not square_ft:
        st.error("❌ Please select square feet range")
    elif model is None:
        st.error("❌ Model not loaded. Please check the model file.")
    else:
        try:
            # Create input DataFrame
            input_data = pd.DataFrame({
                'POSTED_BY': [posted_by],
                'UNDER_CONSTRUCTION': [1 if under_construction else 0],
                'RERA': [1 if rera else 0],
                'BHK_NO.': [bhk],
                'City': [city],
                'SQUARE_FT_BIN': [square_ft]
            })
            
            # Make prediction
            with st.spinner('Calculating...'):
                prediction = model.predict(input_data)[0]
            
            # Display result (multiply by 100 and convert to positive if negative)
            price_in_lacs = abs(prediction * 100)
            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">Predicted Price</div>
                    <div class="prediction-value">₹ {price_in_lacs:.2f} <span style="font-size: 0.5em; color: #666;">Lacs</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional info
            st.success("✅ Prediction completed successfully!")
            
            # Show input summary
            with st.expander("📋 Input Summary"):
                st.write(f"**Posted By:** {posted_by}")
                st.write(f"**City:** {city}")
                st.write(f"**BHK:** {bhk}")
                st.write(f"**Square Feet:** {square_ft}")
                st.write(f"**Under Construction:** {'Yes' if under_construction else 'No'}")
                st.write(f"**RERA Approved:** {'Yes' if rera else 'No'}")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em;">
        <p>💡 Tip: Make sure 'best_xgb_model.pkl' is in the same directory</p>
    </div>
""", unsafe_allow_html=True)