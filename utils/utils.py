import cv2, ast
import numpy as np
from PIL import Image
from typing import List
from glob import glob
from streamlit_free_text_select import st_free_text_select
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
from lama_inpaint import inpaint_img_with_lama
from matplotlib import pyplot as plt
import streamlit as st

st.set_page_config(layout='wide')

def load_img_to_array(img_p):
    print(img_p)
    img = Image.open(img_p)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return np.array(img)

def save_array_to_img(img_arr, img_p):
    Image.fromarray(img_arr.astype(np.uint8)).save(img_p)

def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def erode_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.erode(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

def show_mask(ax, mask: np.ndarray, random_color=False):
    mask = mask.astype(np.uint8)
    if np.max(mask) == 255:
        mask = mask / 255
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_img = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_img)

def show_points(ax, coords: List[List[float]], labels: List[int], size=375):
    coords = np.array(coords)
    labels = np.array(labels)
    color_table = {0: 'red', 1: 'green'}
    for label_value, color in color_table.items():
        points = coords[labels == label_value]
        ax.scatter(points[:, 0], points[:, 1], color=color, marker='*',
                   s=size, edgecolor='white', linewidth=1.25)

def parse_coords(string):
    # Use ast.literal_eval to safely evaluate the string as a list of lists
    return ast.literal_eval(string)

def get_clicked_point(img_path):
    # Load the image
    img = cv2.cvtColor(load_img_to_array(img_path), cv2.COLOR_BGR2RGB)
    cv2.namedWindow("image"); cv2.imshow("image", img)

    # Lists to store points and labels
    pos_points, pos_lbls, neg_points, neg_lbls = [], [], [], []  

    # This will control the loop
    keep_looping = [True]

    # Define the callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal pos_points, neg_points, pos_lbls, neg_lbls, img, keep_looping
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: add point to pos_points
            pos_points.append([x, y])
            pos_lbls.append(1)  # Label for positive points
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red dot for positive point
            cv2.imshow("image", img)

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: add point to neg_points
            neg_points.append([x, y])
            neg_lbls.append(0)  # Label for negative points
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)  # Blue dot for negative point
            cv2.imshow("image", img)

        elif event == cv2.EVENT_MBUTTONDOWN:
            # Middle click: stop the program
            keep_looping[0] = False  # Use list to update the variable in the enclosing scope

    # Set mouse callback function
    cv2.setMouseCallback("image", mouse_callback)

    # Keep looping until the user clicks the middle mouse button
    while keep_looping[0]:
        cv2.waitKey(1)  # This allows OpenCV to process the events

    # Close the image window
    cv2.destroyAllWindows()

    # Return the points and labels
    return np.array(pos_points + neg_points), np.array(pos_lbls + neg_lbls)


def get_ims_captions(path, n_ims):

    im_paths = sorted(glob(f"{path}/*.jpg"))
    # np.random.shuffle(im_paths)
    ims = im_paths[:n_ims]
    captions = [f"Image #{i+1}" for i in range(len(ims))]

    return ims, captions

def choose(option, label):

    return st_free_text_select(
            label=label,
            options=option,
            index=None,
            placeholder="선택을 위해 클릭해주세요",
            disabled=False,
            delay=300,
            label_visibility="visible")

def inpaint(im_path, ckpts_path, input_points, input_labels, device, output_dir, lama_config, lama_ckpt, dks = None):
        st.header("Building SAM model...!") 
        
        print("Building SAM model...")

        # Initialize SAM model using the checkpoint
        sam = sam_model_registry["vit_h"](checkpoint=f"{ckpts_path}").to(device=device)
        predictor = SamPredictor(sam)
        print("The SAM segmentation model is successfully created!\n")
        st.header("The SAM segmentation model is successfully created!") 

        image = np.array(Image.open(im_path).convert("RGB"))
        # image = load_img_to_array(im_path)

        # Preprocess image for SAM
        predictor.set_image(image)

        # Get segmentation masks
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
        print("The segmentation masks using SAM are obtained!\n")
        st.header("The segmentation masks using SAM are obtained!\n")

        masks = masks.astype(np.uint8) * 255

        # Dilate masks to avoid unmasked edge effect if the kernel size is specified
        if dks: masks = [dilate_mask(mask, dks) for mask in masks]

        # Save the segmentation masks
        print("Visualizing the segmentation masks...")
        st.header("Visualizing the segmentation masks...")            
        inpaintings = [inpaint_img_with_lama(image, mask, lama_config, lama_ckpt, device=device) for mask in masks]

        return masks, inpaintings

def write(text): return st.markdown(f'<h1 style="text-align: center;">{text}</h1>', unsafe_allow_html=True) if isinstance(text, str) else st.markdown(f'<h1 style="font-size:100px; text-align: center; color: red; ">{text}</h1>', unsafe_allow_html=True)