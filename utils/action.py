import streamlit as st, numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
from lama_inpaint import inpaint_img_with_lama
from utils import dilate_mask

st.set_page_config(layout='wide')

def write(text): return st.markdown(f'<h1 style="text-align: center;">{text}</h1>', unsafe_allow_html=True) if isinstance(text, str) else st.markdown(f'<h1 style="font-size:100px; text-align: center; color: red; ">{text}</h1>', unsafe_allow_html=True)

def inpaint(im_path, ckpts_path, input_points, input_labels, device, lama_config, lama_ckpt, lang, dks = None):
        
        if   lang == "en": st.header("Building SAM model...!") 
        elif lang == "ko": st.header("SAM 모델을 구축하는 중입니다...")     

        # Initialize SAM model using the checkpoint
        sam = sam_model_registry["vit_h"](checkpoint=f"{ckpts_path}").to(device=device)
        predictor = SamPredictor(sam)
        if   lang == "en": st.header("The SAM segmentation model is successfully created!") 
        elif lang == "ko": st.header("SAM 세그멘테이션 모델이 성공적으로 생성되었습니다!") 

        image = np.array(Image.open(im_path).convert("RGB"))
        # image = load_img_to_array(im_path)

        # Preprocess image for SAM
        predictor.set_image(image)

        # Get segmentation masks
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
        if   lang == "en": st.header("The segmentation masks using SAM are obtained!\n") 
        elif lang == "ko": st.header("SAM을 사용하여 세그멘테이션 마스크가 생성되었습니다!") 

        masks = masks.astype(np.uint8) * 255

        # Dilate masks to avoid unmasked edge effect if the kernel size is specified
        if dks: masks = [dilate_mask(mask, dks) for mask in masks]

        # Save the segmentation masks
        if   lang == "en": st.header("Visualizing the segmentation masks...") 
        elif lang == "ko": st.header("세그멘테이션 마스크를 시각화하는 중입니다...") 
                 
        inpaintings = [inpaint_img_with_lama(image, mask, lama_config, lama_ckpt, device=device) for mask in masks]

        return masks, inpaintings