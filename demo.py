import cv2, argparse, numpy as np, streamlit as st
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
np.random.seed(seed = 2024)
from PIL import Image
from matplotlib import pyplot as plt
from lama_inpaint import inpaint_img_with_lama
from utils import get_ims_captions, choose, load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point, parse_coords
from streamlit_image_select import image_select

from segment_anything import sam_model_registry, SamPredictor

def run(args):
    
    options = ['vit_h', 'vit_l', 'vit_b', 'vit_t']
    coords  = ["click", "key_in"]

    option = choose(options, label = "Please choose SAM model!")
    coord  = choose(coords,  label = "Please choose coordinates type!")

    sam = sam_model_registry[option](checkpoint=args.ckpts_path).to(device=args.device)     

    st.header("이미지를 선택해주세요:")
    get_im = st.file_uploader('1', label_visibility='collapsed')
    if get_im:
        try: image = cv2.imdecode(np.frombuffer(get_im.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e: print(e); st.write('다른 이미지를 업로드해주세요')
        
    else:
        ims_lst, image_captions = get_ims_captions(path = args.root, n_ims = 7)
        im_path = image_select(label="", images = ims_lst, captions = image_captions)
        image = np.array(Image.open(im_path).convert("RGB"))
    print("Done")    
    # if option and size:
    
    #     resized_image = cv2.resize(image, (size, size))

    #     masks = mask_generator.generate(image); resized_masks = mask_generator.generate(resized_image)
    #     masks_to_plot = show_anns(masks); resized_masks_to_plot = show_anns(resized_masks)

    #     col1, col2, col3 = st.columns(3)

    #     with col1: write("원본 이미지:");                st.image(image)
    #     with col2: write("세포 수:");       write(len(masks)) 
    #     with col3: write("세그멘테이션:"); st.image(masks_to_plot)

    #     col1, col2, col3 = st.columns(3)

    #     with col1: write("resized 이미지:"); st.image(resized_image)
    #     with col2: write("세포 수:"); write(len(resized_masks))
    #     with col3: write("세그멘테이션 :"); st.image(resized_masks_to_plot)

    # else: st.write("이미지와 세그멘테이션 모델과 resize 크기를 선택해주세요")

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "Semantic Segmentation Demo")
    
    # Add arguments
    parser.add_argument("-r", "--root", type = str, default = "/mnt/data/cervical_screening/classification/mendeley", help = "Root folder for test images")
    parser.add_argument("-d", "--device", type = str, default = "cuda", help = "GPU or CPU")
    parser.add_argument("-cp", "--ckpts_path", type = str, default = "checkpoints", help = "Checkpoint path")
    parser.add_argument("-mc", "--config_path", type = str, default = "configs/sam2.1", help = "Model config path")

    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the code
    run(args) 