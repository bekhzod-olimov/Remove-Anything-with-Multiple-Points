import cv2, ast
import numpy as np
from PIL import Image
from typing import List
from glob import glob
from streamlit_free_text_select import st_free_text_select

def load_img_to_array(img_p):
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
    img = cv2.imread(img_path)
    cv2.namedWindow("image")
    cv2.imshow("image", img)

    pos_points, neg_points = [], []  # Lists to store points
    pos_lbls, neg_lbls = [], []      # Lists to store labels

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

    im_paths = glob(f"{path}/*.jpg")
    np.random.shuffle(im_paths)
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