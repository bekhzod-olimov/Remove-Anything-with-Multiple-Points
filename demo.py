import cv2
import argparse
import numpy as np
import streamlit as st
import warnings, sys, os
warnings.filterwarnings('ignore')
np.random.seed(seed=2024)
from PIL import Image
from matplotlib import pyplot as plt
from utils import get_ims_captions, choose, inpaint, write, get_clicked_point, parse_coords, load_img_to_array
from streamlit_image_select import image_select

def run(args):
    # Streamlit UI for image selection

    assert args.lang in ["en", "ko"], "Please choose English (en) or Korean (ko) language."

    if   args.lang == "en": st.header("Please upload an image or choose from the list:") 
    elif args.lang == "ko": st.header("이미지를 업로드하거나 이미지를 선택해주세요:") 

    get_im = st.file_uploader('1', label_visibility='collapsed')
    ims_lst, image_captions = get_ims_captions(path=args.root, n_ims=7)   
    im_path, verbose = None, False

    # Use the selected or uploaded image to get points and labels
    if get_im:
        input_points, input_labels = get_clicked_point(get_im)          
        masks, inpaintings = inpaint(get_im, ckpts_path = args.ckpt_path, input_points = input_points, lama_config = args.lama_config, lama_ckpt = args.lama_ckpt,
                input_labels = input_labels, device = args.device, output_dir = args.output_dir, dks = args.dilate_kernel_size, lang = args.lang)

        cols = st.columns(len(masks))

        for idx, col in enumerate(cols):
            with col: write(f"Mask#{idx}:"); st.image(masks[idx])

        cols = st.columns(len(masks))

        for idx, col in enumerate(cols):
            with col: write(f"Inpainting#{idx}:"); st.image(inpaintings[idx])
        
    elif (get_im == None) and (im_path == None): 
        # print("get_im == None")
        select_label = "Images List" if args.lang == "en" else "이미지 목록"
        choice_label = "Available Images" if args.lang == "en" else "선택 가능한 이미지 목록"
        image_select(label=select_label, images=ims_lst, captions=image_captions)
        choice = choose(option = image_captions, label = choice_label)
        
        if   args.lang == "en": st.header("Please upload an image or choose from the list!") 
        elif args.lang == "ko": st.header("이미지를 선택하거나 업로드 해주세요!") 
        
        if choice:  
            # print("if choice")
            im_path = ims_lst[int(choice.split("#")[-1])-1]

            input_points, input_labels = get_clicked_point(im_path)          
            
            masks, inpaintings = inpaint(im_path, ckpts_path = args.ckpt_path, input_points = input_points, lama_config = args.lama_config, lama_ckpt = args.lama_ckpt,
                    input_labels = input_labels, device = args.device, output_dir = args.output_dir, dks = args.dilate_kernel_size, lang = args.lang)
            

            cols = st.columns(len(masks))

            for idx, col in enumerate(cols):
                with col: write(f"Mask#{idx+1}:"); st.image(masks[idx])

            cols = st.columns(len(masks))

            for idx, col in enumerate(cols):
                with col: write(f"Inpainting#{idx+1}:"); st.image(inpaintings[idx]) 
            if   args.lang == "en": st.header("Please rerun streamlit script from the terminal!") 
            elif args.lang == "ko": st.header("터미널에서 streamlit 스크립트를 다시 실행해 주세요!") 

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Image Inpainting Demo")

    # Add arguments
    parser.add_argument("-r", "--root", type=str, default="example/remove-anything", help="Root folder for test images")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="GPU or CPU")
    parser.add_argument("-cp", "--ckpt_path", type=str, default="../backup/pretrained_models/sam_vit_h_4b8939.pth", help="Checkpoint path")
    parser.add_argument("-pc", "--point_coords", type=parse_coords, nargs='*', help="The coordinate of the point prompt, [coord_W coord_H].")
    parser.add_argument("-pl", "--point_labels", type=parse_coords, nargs='*', help="The labels of the point prompt, 1 or 0.")
    parser.add_argument("-dk", "--dilate_kernel_size", type=int, default=None, help="Dilate kernel size. Default: None")
    parser.add_argument("-od", "--output_dir", type=str, default="results", help="Output path to the directory with results.")
    parser.add_argument("-lc", "--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="The path to the config file of lama model. Default: the config of big-lama")
    parser.add_argument("-ck", "--lama_ckpt", type=str, default="../backup/pretrained_models/big-lama", help="The path to the lama checkpoint.")
    parser.add_argument("-ln", "--lang", type=str, default="ko", help="Verbose language")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main code
    run(args)