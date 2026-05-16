import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
st.set_page_config(page_title="DR Screener", page_icon="👁️", layout="wide")

from database.db import SessionLocal
from database.crud import create_patient, save_prediction, get_all_patients, get_predictions_by_patient
from database.db import init_db

init_db()

import pandas as pd
import torch, numpy as np, cv2, sys, tempfile
from PIL import Image
from src.model import build_model
from src.dataset import get_transforms
from src.gradcam import (generate_gradcam, analyze_heatmap_regions, multi_method_comparison)
from src.lime_explain import (generate_lime_explanation, lime_heatmap, plot_lime_analysis)
from src.report import (save_combined_figure, generate_pdf_report, DR_CLASSES, RISK_LEVELS, DR_DESCRIPTIONS)

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH='models/best_model.pth'
STAGE_COLORS={0:'#27ae60', 1:'#f1c40f', 2:'#e67e22',
                3:'#e74c3c', 4:'#8e44ad'}

with st.sidebar:
    st.header("Settings")
    patient_id=st.text_input("Patient ID", "DR001")
    cam_method=st.selectbox("Grad-CAM method", ["gradcam++", "gradcam", "eigencam"])
    run_lime=st.checkbox("Run LIME", value=True)
    lime_samples=st.slider("LIME samples", 200, 1500, 600, step=100)
    show_compare=st.checkbox("Show multi-method XAI comparison", value=False)
    st.divider()
    st.caption(f"Running on: **{DEVICE.upper()}**")
    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found at models/best_model.pth - "
                   "train the model first.")
        
@st.cache_resource
def load_model():
    m=build_model(num_classes=5, device=DEVICE)
    ckpt=torch.load(MODEL_PATH, map_location=DEVICE)
    m.load_state_dict(ckpt.get('model_state_dict', ckpt))
    m.eval()
    return m
    
st.title("Diabetic Retinopathy AI Screener")
st.caption("Upload a retinal fundus image to get an AI prediction "
           "with Grad-CAM++ and LIME explanations.")

"""
Diabetic Retinopathy AI Screener

Streamlit app that:
- Predicts DR stage using CNN
- Visualizes using Grad-CAM and LIME
- Generates clinical PDF report
"""

uploaded=st.file_uploader("Choose a fundus image",
                          type=["png", "jpg", "jpeg"])

if uploaded and os.path.exists(MODEL_PATH):
    pil_img=Image.open(uploaded).convert('RGB')
    img_np=np.array(pil_img)
    transform=get_transforms('test', 300)
    tensor=transform(image=img_np)['image']
    model=load_model()
    with torch.no_grad():
        logits=model(tensor.unsqueeze(0).to(DEVICE))
        probs=torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class=int(probs.argmax())

    col1, col2=st.columns([1, 1.6])
    with col1:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)
    with col2:
        colour=STAGE_COLORS[pred_class]
        st.markdown(
            f"<div style='background:{colour};padding:14px 18px;"
            f"border-radius:10px;color:white;margin-bottom:12px'>"
            f"<div style='font-size:20px;font-weight:600'>"
            f"{DR_CLASSES[pred_class]}</div>"
            f"<div style='font-size:13px;opacity:0.9'>"
            f"Confidence {probs[pred_class]*100:.1f}% &nbsp;|&nbsp; "
            f"Risk: {RISK_LEVELS[pred_class]}</div>"
            f"</div>", unsafe_allow_html=True)
        
    for i, (cls, p) in enumerate(zip(DR_CLASSES, probs)):
        c= colour if i==pred_class else '#bdc3c7'
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;"
                f"margin:3px 0;font-size:13px'>"
                f"<span style='width:120px'>{cls}</span>"
                f"<div style='flex:1;background:#ecf0f1;border-radius:4px;height:16px'>"
                f"<div style='width:{p*100:.1f}%;background:{c};"
                f"height:100%;border-radius:4px'></div></div>"
                f"<span style='width:42px;text-align:right'>{p*100:.1f}%</span>"
                f"</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.info(DR_DESCRIPTIONS[pred_class])

if uploaded and os.path.exists(MODEL_PATH):
    st.markdown("### Explainability Analysis")
    img_resized=cv2.resize(img_np, (300, 300))
    with st.spinner(f"Generating {cam_method.upper()} heatmap..."):
        overlay, raw_cam, _, _=generate_gradcam(
            model, tensor, pred_class,
            method=cam_method, device=DEVICE)
        active_regions=analyze_heatmap_regions(raw_cam)
    lime_overlay=None
    lime_exp=None
    if run_lime:
        with st.spinner("Running LIME (this takes ~2-3 minutes)..."):
            lime_exp, lime_overlay, top_segs=generate_lime_explanation(
                model, img_resized, pred_class,
                device=DEVICE, num_samples=lime_samples)

    c1, c2, c3=st.columns(3)
    c1.image(img_resized, caption="Original", use_column_width=True)
    c2.image(overlay, caption=f"{cam_method.upper()} heatmap", use_column_width=True)
    if lime_overlay is not None:
        c3.image(lime_overlay, caption="LIME regions", use_column_width=True)
    else:
        c3.markdown("*(LIME disabled)*")
    if active_regions:
        st.markdown(f"**Model attention focused on:** {', '.join(active_regions)}")
    if show_compare:
        with st.spinner("Generating XAI method comparison..."):
            fig=multi_method_comparison(
                model, tensor, pred_class, device=DEVICE)
            st.pyplot(fig)

    if 'saved' not in st.session_state:
        st.session_state.saved = False
    if not st.session_state.saved:
        db = SessionLocal()
        try:
            create_patient(db, patient_id)
            save_prediction(
                db,
                patient_id=patient_id,
                image_name=uploaded.name,
                image_path=uploaded.name,
                predicted_class=pred_class,
                predicted_label=DR_CLASSES[pred_class],
                confidence=float(probs[pred_class]),
                risk_level=RISK_LEVELS[pred_class],
                probs=probs.tolist(),
                gradcam_path="gradcam.png",
                lime_path="lime.png" if lime_overlay is not None else "",
                active_regions=active_regions
            )
            st.session_state.saved = True
            st.success("Prediction automatically saved to database!")
        except Exception as e:
            st.error(f"DB Error: {e}")
        finally:
            db.close()
else:
    st.info("Please upload an image and ensure the model is available.")



st.markdown("---")
st.markdown("### Download Clinical Report")
if uploaded and os.path.exists(MODEL_PATH) and st.button("Generate PDF report"):
    with st.spinner("Building PDF..."):
        with tempfile.TemporaryDirectory() as tmp:
            fig_path=os.path.join(tmp, 'fig.png')
            save_combined_figure(
                img_resized, overlay, lime_overlay if lime_overlay is not None else overlay,
                pred_class, probs, active_regions, fig_path)
            pdf_path=os.path.join(tmp, 'report.pdf')
            generate_pdf_report(
                patient_id, uploaded.name, fig_path,
                pred_class, probs, active_regions, pdf_path)
            
            with open(pdf_path, 'rb') as f:
                st.download_button(
                    "Download PDF", 
                    data=f.read(),
                    file_name=f"DR_Report_{patient_id}.pdf",
                    mime="application/pdf")
elif uploaded and not os.path.exists(MODEL_PATH):
    st.error("Please train the model first (run `python src/main_train.py`), "
             "then reload this app.")

st.markdown("---")
st.markdown("## Prediction History")

db = SessionLocal()

try:
    patients = get_all_patients(db)
    total_patients = len(patients)
    all_preds = []
    for p in patients:
        preds = get_predictions_by_patient(db, p.patient_id)
        all_preds.extend(preds)
    total_scans = len(all_preds)
    if total_scans > 0:
        avg_conf = np.mean([p.confidence for p in all_preds]) * 100
    else:
        avg_conf = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Scans", total_scans)
    col2.metric("Total Patients", total_patients)
    col3.metric("Avg Confidence", f"{avg_conf:.1f}%")
    st.markdown("### Cases by DR stage")
    stage_counts = {}

    for p in all_preds:
        label = p.predicted_label
        stage_counts[label]=stage_counts.get(label, 0) + 1
    if stage_counts:
        stage_df = pd.DataFrame(
            list(stage_counts.items()),
            columns=["Stage", "Count"]
        )
        st.dataframe(stage_df, use_container_width=True)
    else:
        st.info("No prediction data available")
    st.markdown("### Recent predictions")

    if all_preds:
        recent_data = []
        all_preds_sorted = sorted(all_preds, key=lambda x: x.created_at, reverse=True)
        for i, p in enumerate(all_preds_sorted[:10]): 
            recent_data.append({
                "ID": i,
                "Patient": p.patient_id,
                "Image": p.image_name,
                "Prediction": p.predicted_label,
                "Confidence": f"{p.confidence*100:.1f}%",
                "Risk": p.risk_level,
                "Regions": p.active_regions if p.active_regions else "-",
                "Date": p.created_at.strftime("%Y-%m-%d %H:%M")
            })
        recent_df = pd.DataFrame(recent_data)
        st.dataframe(recent_df, use_container_width=True)

    else:
        st.info("No recent predictions found")
except Exception as e:
    st.error(f"Database Error: {e}")
finally:
    db.close()