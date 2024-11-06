import warnings
warnings.filterwarnings('ignore')
import sys, ast
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from segment_anything import sam_model_registry, SamPredictor


def parse_coords(string):
    # Use ast.literal_eval to safely evaluate the string as a list of lists
    return ast.literal_eval(string)

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=parse_coords, nargs='*', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=parse_coords, nargs='*', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )

    parser.add_argument(
        "--device", type=str, default="cuda",
        help="GPU / CPU",
    )


if __name__ == "__main__":
    
    """Example usage:
    python remove_anything.py \
    --input_img path/to/your/image \
    --coords_type key_in \
    --point_coords "[[580, 250], [580, 50], [700, 350]]" \
    --point_labels "[1, 1, 1]" \
    --dilate_kernel_size 15 \
    --output_dir ./results \
    --sam_model_type "vit_h" \
    --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama 
    """

    parser = argparse.ArgumentParser()
    setup_args(parser)    
    args = parser.parse_args(sys.argv[1:])
    assert args.coords_type in ["click", "key_in"]  

    input_points, input_labels = np.array(args.point_coords)[0], np.array(args.point_labels)[0] if args.coords_type == "key_in" else get_clicked_point(args.input_img)
    
    img = load_img_to_array(args.input_img)

    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_ckpt).to(device=args.device)

    predictor = SamPredictor(sam)
    print("The SamPredictor has been successfully built!\n")

    # Pre-processing
    predictor.set_image(img)

    # Get the predictions
    masks, _, _ = predictor.predict( point_coords=input_points, point_labels=input_labels, multimask_output=True)
    print("The segmentations masks are obtained!\n")

    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size: masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir  = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the segmentation masks
    print("Saving the segmentation masks...")
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # Save the mask
        save_array_to_img(mask, mask_p)

        # Save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')

        # Show the points
        [show_points(plt.gca(), input_point, input_label, size=(width*0.04)**2) for input_point, input_label in zip(input_points, input_labels)]
        
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0); plt.close()        
    print(f"The segmentation masks were successfully saved to {out_dir}.\n")

    # Inpainting using the segmentation masks
    print("Inpainting process is going to start...")
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=args.device)
        save_array_to_img(img_inpainted, img_inpainted_p)
    print(f"The inpainting process is completed. The results can be found in {out_dir}.")