# import cv2, argparse, numpy as np, streamlit as st
# import warnings
# warnings.filterwarnings('ignore')
# import argparse
# import numpy as np
# from pathlib import Path
# np.random.seed(seed = 2024)
# from PIL import Image
# from matplotlib import pyplot as plt
# from lama_inpaint import inpaint_img_with_lama
# from utils import get_ims_captions, choose, load_img_to_array, save_array_to_img, dilate_mask, \
#     show_mask, show_points, get_clicked_point, parse_coords
# from streamlit_image_select import image_select

# from segment_anything import sam_model_registry, SamPredictor

# def run(args):
    
#     st.header("이미지를 선택해주세요:")
#     get_im = st.file_uploader('1', label_visibility='collapsed')
#     if get_im:
#         try: image = cv2.imdecode(np.frombuffer(get_im.read(), np.uint8), cv2.IMREAD_COLOR)
#         except Exception as e: print(e); st.write('다른 이미지를 업로드해주세요')
        
#     else:
#         ims_lst, image_captions = get_ims_captions(path = args.root, n_ims = 7)
#         im_path = image_select(label="", images = ims_lst, captions = image_captions)
#         image = load_img_to_array(im_path)
    
#     coords  = ["click", "key_in"]
#     coord  = choose(coords,  label = "Please choose coordinates type!")

#     try:
#         if coord == "click":
#             input_points, input_labels = get_clicked_point(im_path)
#         elif coord == "key_in":
#             input_points, input_labels = np.array(args.point_coords)[0], np.array(args.point_labels)[0]
#         else:
#             raise ValueError("Choose something valid")
#     except IndexError: print("Index error occurred with the point coordinates or labels.")
#     except ValueError as e: print(e)
    
#     if image is not None:

#         print("Building SAM model...")
    
#         sam = sam_model_registry["vit_h"](checkpoint=f"{args.ckpts_path}").to(device=args.device)
#         predictor = SamPredictor(sam)

#         print("The SamPredictor has been successfully built!\n")

#         predictor.set_image(image)

#         masks, _, _ = predictor.predict( point_coords=input_points, point_labels=input_labels, multimask_output=True)

#         print("The segmentations masks are obtained!\n")

#         masks = masks.astype(np.uint8) * 255

#         # dilate mask to avoid unmasked edge effect
#         if args.dilate_kernel_size: masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

#         # visualize the segmentation results
#         img_stem = Path(im_path).stem
#         out_dir  = Path(args.output_dir) / img_stem
#         out_dir.mkdir(parents=True, exist_ok=True)
        
#         # Save the segmentation masks
#         print("Saving the segmentation masks...")
#         for idx, mask in enumerate(masks):
#             # path to the results
#             mask_p = out_dir / f"mask_{idx}.png"
#             img_points_p = out_dir / f"with_points.png"
#             img_mask_p = out_dir / f"with_{Path(mask_p).name}"

#             # Save the mask
#             save_array_to_img(mask, mask_p)

#             # Save the pointed and masked image
#             dpi = plt.rcParams['figure.dpi']
#             height, width = image.shape[:2]
#             plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
#             plt.imshow(image)
#             plt.axis('off')

#             # Show the points
#             [show_points(plt.gca(), input_point, input_label, size=(width*0.04)**2) for input_point, input_label in zip(input_points, input_labels)]
            
#             plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
#             show_mask(plt.gca(), mask, random_color=False)
#             plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0); plt.close()        
#         print(f"The segmentation masks were successfully saved to {out_dir}.\n")

#         # Inpainting using the segmentation masks
#         print("Inpainting process is going to start...")
#         print(image.shape)
#         for idx, mask in enumerate(masks):
#             mask_p = out_dir / f"mask_{idx}.png"
#             img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
#             img_inpainted = inpaint_img_with_lama(image, mask, args.lama_config, args.lama_ckpt, device=args.device)
#             save_array_to_img(img_inpainted, img_inpainted_p)
#         print(f"The inpainting process is completed. The results can be found in {out_dir}.")
#     # if option and size:
    
#     #     resized_image = cv2.resize(image, (size, size))

#     #     masks = mask_generator.generate(image); resized_masks = mask_generator.generate(resized_image)
#     #     masks_to_plot = show_anns(masks); resized_masks_to_plot = show_anns(resized_masks)

#     #     col1, col2, col3 = st.columns(3)

#     #     with col1: write("원본 이미지:");                st.image(image)
#     #     with col2: write("세포 수:");       write(len(masks)) 
#     #     with col3: write("세그멘테이션:"); st.image(masks_to_plot)

#     #     col1, col2, col3 = st.columns(3)

#     #     with col1: write("resized 이미지:"); st.image(resized_image)
#     #     with col2: write("세포 수:"); write(len(resized_masks))
#     #     with col3: write("세그멘테이션 :"); st.image(resized_masks_to_plot)

#     # else: st.write("이미지와 세그멘테이션 모델과 resize 크기를 선택해주세요")

# if __name__ == "__main__":
    
#     # Initialize argument parser
#     parser = argparse.ArgumentParser(description = "Semantic Segmentation Demo")
    
#     # Add arguments
#     parser.add_argument("-r", "--root", type = str, default = "example/remove-anything", help = "Root folder for test images")
#     parser.add_argument("-d", "--device", type = str, default = "cpu", help = "GPU or CPU")
#     parser.add_argument("-cp", "--ckpts_path", type = str, default = "../backup/pretrained_models/sam_vit_h_4b8939.pth", help = "Checkpoint path")
#     parser.add_argument("-pc", "--point_coords", type=parse_coords, nargs='*', help="The coordinate of the point prompt, [coord_W coord_H].")
#     parser.add_argument("-pl", "--point_labels", type=parse_coords, nargs='*', help="The labels of the point prompt, 1 or 0.")
#     parser.add_argument("-dk", "--dilate_kernel_size", type=int, default=None, help="Dilate kernel size. Default: None")
#     parser.add_argument("-od", "--output_dir", type=str, default="results", help="Output path to the directory with results.")
#     parser.add_argument("-lc",  "--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="The path to the config file of lama model. Default: the config of big-lama")
#     parser.add_argument("-ck", "--lama_ckpt", type=str,  help="The path to the lama checkpoint.")
    
#     # Parse the arguments
#     args = parser.parse_args() 
    
#     # Run the code
#     run(args) 



import cv2
import argparse
import numpy as np
import streamlit as st
import warnings
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from lama_inpaint import inpaint_img_with_lama
from utils import get_ims_captions, choose, load_img_to_array, save_array_to_img, dilate_mask, show_mask, show_points, get_clicked_point, parse_coords
from streamlit_image_select import image_select
from segment_anything import sam_model_registry, SamPredictor

warnings.filterwarnings('ignore')
np.random.seed(seed=2024)


def run(args):
    # Streamlit UI for image selection
    st.header("이미지를 선택해주세요:")
    get_im = st.file_uploader('1', label_visibility='collapsed')

    # Handle image loading
    if get_im:
        try:
            image = cv2.imdecode(np.frombuffer(get_im.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(e)
            st.write('다른 이미지를 업로드해주세요')
    else:
        ims_lst, image_captions = get_ims_captions(path=args.root, n_ims=7)
        im_path = image_select(label="", images=ims_lst, captions=image_captions)
        image = load_img_to_array(im_path)

    # Choose coordinates type
    coords = ["click", "key_in"]
    coord = choose(coords, label="Please choose coordinates type!")

    try:
        if coord == "click":
            input_points, input_labels = get_clicked_point(im_path)
        elif coord == "key_in":
            input_points, input_labels = np.array(args.point_coords)[0], np.array(args.point_labels)[0]
        else:
            raise ValueError("Choose something valid")
    except IndexError:
        print("Index error occurred with the point coordinates or labels.")
    except ValueError as e:
        print(e)

    if image is not None:
        print("Building SAM model...")

        # Initialize SAM model using the checkpoint
        sam = sam_model_registry["vit_h"](checkpoint=f"{args.ckpts_path}").to(device=args.device)
        predictor = SamPredictor(sam)
        print("The SamPredictor has been successfully built!\n")

        # Preprocess image for SAM
        predictor.set_image(image)

        # Get segmentation masks
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=True)
        print("The segmentation masks are obtained!\n")

        masks = masks.astype(np.uint8) * 255

        # Dilate masks to avoid unmasked edge effect if the kernel size is specified
        if args.dilate_kernel_size:
            masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

        # Set output directory
        img_stem = Path(im_path).stem
        out_dir = Path(args.output_dir) / img_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save the segmentation masks
        print("Saving the segmentation masks...")
        for idx, mask in enumerate(masks):
            mask_p = out_dir / f"mask_{idx}.png"
            img_points_p = out_dir / f"with_points.png"
            img_mask_p = out_dir / f"with_{Path(mask_p).name}"

            # Save the mask
            save_array_to_img(mask, mask_p)

            # Save the pointed and masked image
            dpi = plt.rcParams['figure.dpi']
            height, width = image.shape[:2]
            plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
            plt.imshow(image)
            plt.axis('off')

            # Show the points on the image
            [show_points(plt.gca(), input_point, input_label, size=(width * 0.04) ** 2) for input_point, input_label in zip(input_points, input_labels)]

            plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
            show_mask(plt.gca(), mask, random_color=False)
            plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
            plt.close()
        print(f"The segmentation masks were successfully saved to {out_dir}.\n")

        # Inpainting using the segmentation masks
        print("Inpainting process is going to start...")
        for idx, mask in enumerate(masks):
            mask_p = out_dir / f"mask_{idx}.png"
            img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
            img_inpainted = inpaint_img_with_lama(image, mask, args.lama_config, args.lama_ckpt, device=args.device)
            save_array_to_img(img_inpainted, img_inpainted_p)
        print(f"The inpainting process is completed. The results can be found in {out_dir}.")

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Semantic Segmentation Demo")

    # Add arguments
    parser.add_argument("-r", "--root", type=str, default="example/remove-anything", help="Root folder for test images")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="GPU or CPU")
    parser.add_argument("-cp", "--ckpts_path", type=str, default="../backup/pretrained_models/sam_vit_h_4b8939.pth", help="Checkpoint path")
    parser.add_argument("-pc", "--point_coords", type=parse_coords, nargs='*', help="The coordinate of the point prompt, [coord_W coord_H].")
    parser.add_argument("-pl", "--point_labels", type=parse_coords, nargs='*', help="The labels of the point prompt, 1 or 0.")
    parser.add_argument("-dk", "--dilate_kernel_size", type=int, default=None, help="Dilate kernel size. Default: None")
    parser.add_argument("-od", "--output_dir", type=str, default="results", help="Output path to the directory with results.")
    parser.add_argument("-lc", "--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="The path to the config file of lama model. Default: the config of big-lama")
    parser.add_argument("-ck", "--lama_ckpt", type=str, default="../backup/pretrained_models/big-lama", help="The path to the lama checkpoint.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main code
    run(args)