import streamlit as st, numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
from lama_inpaint import inpaint_img_with_lama
from utils import dilate_mask; from tqdm import tqdm
from stable_diffusion_inpaint import replace_img_with_sd

st.set_page_config(layout='wide')

class Action:
        
    def __init__(self, im_path, ckpts_path, input_points, input_labels, device, lama_config, lama_ckpt, lang, dks = None, text_prompt = None, final_header = None):
            
        self.im_path, self.ckpts_path, self.input_points, self.input_labels = im_path, ckpts_path, input_points, input_labels
        self.device, self.lama_config, self.lama_ckpt, self.lang, self.dks  = device, lama_config, lama_ckpt, lang, dks
        self.text_prompt, self.final_header = text_prompt, final_header

    def segment(self):
         
        if   self.lang == "en": st.header("Building SAM model...!") 
        elif self.lang == "ko": st.header("SAM 모델을 구축하는 중입니다...")     

        # Initialize SAM model using the checkpoint
        sam = sam_model_registry["vit_h"](checkpoint=f"{self.ckpts_path}").to(device=self.device)
        predictor = SamPredictor(sam)
        if   self.lang == "en": st.header("The SAM segmentation model is successfully created!") 
        elif self.lang == "ko": st.header("SAM 세그멘테이션 모델이 성공적으로 생성되었습니다!")

        # Preprocess image for SAM
        self.get_im(); predictor.set_image(self.image)

        # Get segmentation masks
        masks, _, _ = predictor.predict(point_coords=self.input_points, point_labels=self.input_labels, multimask_output=True)
        if   self.lang == "en": st.header("The segmentation masks using SAM are obtained!\n") 
        elif self.lang == "ko": st.header("SAM을 사용하여 세그멘테이션 마스크가 생성되었습니다!") 

        self.masks = masks.astype(np.uint8) * 255

        # Dilate masks to avoid unmasked edge effect if the kernel size is specified
        if self.dks: self.masks = [dilate_mask(mask, self.dks) for mask in self.masks]

        if   self.lang == "en": st.header("Visualizing the segmentation masks...") 
        elif self.lang == "ko": st.header("세그멘테이션 마스크를 시각화하는 중입니다...")
    
    def get_im(self): self.image = np.array(Image.open(self.im_path).convert("RGB"))    

    def summarize(self):

        to_show, writing = (self.replaces, "Replaced#") if self.text_prompt else (self.inpaintings, "Inpainting#")

        cols = st.columns(len(self.masks))

        for idx, col in enumerate(cols):
            with col: self.write(f"Mask#{idx+1}:"); st.image(self.masks[idx])

        cols = st.columns(len(self.masks))

        for idx, col in enumerate(cols):
            with col: self.write(f"{writing}{idx+1}:"); st.image(to_show[idx])

    def write(self, text): return st.markdown(f'<h1 style="text-align: center;">{text}</h1>', unsafe_allow_html=True) if isinstance(text, str) else st.markdown(f'<h1 style="font-size:100px; text-align: center; color: red; ">{text}</h1>', unsafe_allow_html=True)
    
    def inpaint(self): self.inpaintings = [inpaint_img_with_lama(self.image, mask, self.lama_config, self.lama_ckpt, device=self.device) for mask in self.masks]

    def replace(self): self.replaces = [(replace_img_with_sd(self.image, mask, self.text_prompt, device=self.device) / 255) for mask in tqdm(self.masks)]

    def inpainting(self): self.segment(); self.inpaint()

    def replacing(self):  self.segment(); self.replace()

    def run(self): self.replacing() if self.text_prompt else self.inpainting(); self.summarize(); st.header(self.final_header)