import streamlit as st
import time

# Configure page[cite: 1]
st.set_page_config(page_title="MediShield", page_icon="🛡️", layout="wide")

# Custom CSS for modern typography and lighting effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        font-weight: 300;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    .status-trusted {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid #10b981;
        color: #10b981;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .status-suspicious {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid #ef4444;
        color: #ef4444;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Headers[cite: 1, 2]
st.markdown('<p class="main-header">MEDISHIELD</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Medical AI Reliability Monitor</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 1. Input Image")
    # File uploader[cite: 1, 2]
    uploaded_file = st.file_uploader("Upload Skin Lesion Image (JPG/PNG)", type=["jpg", "png", "jpeg"]) 
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Lesion", use_container_width=True)

with col2:
    if uploaded_file is None:
        st.info("Awaiting image upload...")
    else:
        st.markdown("### 2. Analysis Engine")
        
        if 'stressed' not in st.session_state:
            st.session_state.stressed = False

        if not st.session_state.stressed:
            st.markdown("#### Clean Prediction")
            col2a, col2b, col2c = st.columns(3)
            # Default clean state metrics[cite: 1, 2]
            col2a.metric("Prediction", "Melanoma") 
            col2b.metric("Confidence", "94%") 
            col2c.metric("Reliability", "96%") 
            
            # Trusted status[cite: 1, 2]
            st.markdown('<div class="status-trusted">✓ STATUS: TRUSTED</div>', unsafe_allow_html=True) 
            
            st.write("---")
            # Attack button[cite: 1]
            if st.button("🚨 Run Adversarial Stress Test (FGSM)", use_container_width=True): 
                with st.spinner("Perturbing image..."):
                    time.sleep(1.5)
                st.session_state.stressed = True
                st.rerun()
                
        else:
            st.markdown("#### Perturbed Prediction (FGSM)")
            col2a, col2b, col2c = st.columns(3)
            # Perturbed state metrics[cite: 1, 2]
            col2a.metric("Prediction", "Nevus", "-12% (Changed)") 
            col2b.metric("Confidence", "82%", "-12%") 
            col2c.metric("Reliability", "18%", "-78%") 
            
            # Suspicious status[cite: 1, 2]
            st.markdown('<div class="status-suspicious">⚠ SUSPICIOUS / ABSTAIN — Human review recommended</div>', unsafe_allow_html=True) 
            
            st.write("---")
            # Show Grad-CAM[cite: 1, 2]
            if st.button("👁️ Show Explainability (Grad-CAM)", use_container_width=True): 
                st.markdown("#### Grad-CAM Comparison")
                # Clean vs Perturbed Grad-CAM comparison[cite: 1, 2]
                img_col1, img_col2 = st.columns(2) 
                with img_col1:
                    st.image("https://via.placeholder.com/224x224/1a202c/4facfe?text=Clean+Grad-CAM", caption="Stable features")
                with img_col2:
                    st.image("https://via.placeholder.com/224x224/1a202c/ef4444?text=Perturbed+Grad-CAM", caption="Shifted features")
            
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.stressed = False
                st.rerun()