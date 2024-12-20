import warnings, argparse, streamlit as st, numpy as np
warnings.filterwarnings('ignore')
np.random.seed(seed = 2024)
from utils import get_ims_captions, choose, get_clicked_point, parse_coords, get_coords, get_labels
from utils.action import Action
from streamlit_image_select import image_select

def run(args):
    
    assert args.lang in ["en", "ko"], "Please choose English (en) or Korean (ko) language."

    if   args.lang == "en": task1, task2, label, placeholder, final_header, text_prompt_label = "replace", "inpaint", "Please choose a task:", "Please click to choose", "Please rerun streamlit script from the terminal!", "Please type a text prompt to be replaced:"
    elif args.lang == "ko": task1, task2, label, placeholder, final_header, text_prompt_label = "대체", "인페인팅", "작업을 선택해 주세요:", "선택을 위해 클릭해주세요", "터미널에서 streamlit 스크립트를 다시 실행해 주세요!", "대체할 업젝트 프롬프트를 입력해 주세요:" 

    task_options = [task1, task2]
    
    task_name = choose(option = task_options, label = label, placeholder = placeholder)

    if task_name in [task1, task2]:

        if   args.lang == "en": type1, type2, label = "my_image", "existing_image", "Please choose a demo type:"
        elif args.lang == "ko": type1, type2, label = "본인 이미지", "리스트에 있는 이미지", "데모 종류를 선택해주세요:"

        text_prompt, input_points, demo_types = None, None, [type1, type2]
        
        type_name = choose(option = demo_types, label = label, placeholder = placeholder)
    
        if type_name == type1:
            

            if   args.lang == "en": im_path_lbl, input_points_lbl,  input_labels_lbl = "Please type image path:", "Please type input points:", "Please type input labels:"
            elif args.lang == "ko": im_path_lbl, input_points_lbl,  input_labels_lbl = "이미지 경로를 입력해 주세요:", "입력 포인트를 입력해 주세요:", "입력 레이블을 입력해 주세요:"

            im_path       = st.text_input(label = im_path_lbl,       value = None)
            input_points  = st.text_input(label = input_points_lbl,  value = None)
            input_labels  = st.text_input(label = input_labels_lbl,  value = None)

            if task_name == task1: text_prompt   = st.text_input(label = text_prompt_label, value = None)

            if (not input_labels is None) and (not input_points is None): input_points, input_labels = get_coords(input_points), get_labels(input_labels)      

        elif type_name == type2 and task_name == task2:

            if   args.lang == "en": st.header("Please upload an image or choose from the list:") 
            elif args.lang == "ko": st.header("이미지를 업로드하거나 이미지를 선택해주세요:") 

            input_points, input_labels = args.point_coords, args.point_labels    

            get_im  = st.file_uploader("1", label_visibility = "collapsed")
            im_path = None
            ims_lst, image_captions = get_ims_captions(path = args.root, n_ims = 7)   

            # Use the selected or uploaded image to get points and labels
            if get_im: input_points, input_labels = get_clicked_point(get_im)          
                
            elif (get_im == None) and (im_path == None): 
                
                select_label = "Images List" if args.lang == "en" else "이미지 목록"
                choice_label = "Available Images" if args.lang == "en" else "선택 가능한 이미지 목록"
                placeholder  = "Please click to choose" if args.lang == "en" else "선택을 위해 클릭해주세요"
                image_select(label=select_label, images=ims_lst, captions=image_captions)
                choice = choose(option = image_captions, label = choice_label, placeholder = placeholder)
                if   args.lang == "en": st.header("Please upload an image or choose from the list!") 
                elif args.lang == "ko": st.header("이미지를 선택하거나 업로드 해주세요!")             
                
                if choice:  input_points, input_labels = get_clicked_point(ims_lst[int(choice.split("#")[-1])-1])
        
        if text_prompt == None and task_name == task1: st.warning("Please enter a prompt to continue.")
                                  
        if ((not input_points is None) and (not input_labels is None) and (not text_prompt is None)) or ((task_name == task2) and (not input_points is None) and (not input_labels is None)):    
                     
            Action(im_path, ckpts_path = args.ckpt_path, input_points = input_points, lama_config = args.lama_config, lama_ckpt = args.lama_ckpt,
                input_labels = input_labels, device = args.device, dks = args.dilate_kernel_size, lang = args.lang, text_prompt = text_prompt, final_header = final_header).run()           
                                

if __name__ == "__main__":
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description = "SAM Semantic Segmentation & Inpainting Demo")

    # Add arguments
    parser.add_argument("-r", "--root", type=str, default="example/remove-anything", help="Root folder for test images")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="GPU or CPU")
    parser.add_argument("-cp", "--ckpt_path", type=str, default="../backup/pretrained_models/sam_vit_h_4b8939.pth", help="Checkpoint path")
    parser.add_argument("-pc", "--point_coords", type=parse_coords, nargs='*', default = None, help="The coordinate of the point prompt, [coord_W coord_H].") # [[300, 200] [350, 250]]
    parser.add_argument("-pl", "--point_labels", type=parse_coords, nargs='*', default = None, help="The labels of the point prompt, 1 or 0.")                # [1, 1]
    parser.add_argument("-dk", "--dilate_kernel_size", type=int, default=None, help="Dilate kernel size. Default: None")
    parser.add_argument("-od", "--output_dir", type=str, default="results", help="Output path to the directory with results.")
    parser.add_argument("-lc", "--lama_config", type=str, default="./lama/configs/prediction/default.yaml", help="The path to the config file of lama model. Default: the config of big-lama")
    parser.add_argument("-ck", "--lama_ckpt", type=str, default="../backup/pretrained_models/big-lama", help="The path to the lama checkpoint.")
    parser.add_argument("-ln", "--lang", type=str, default="en", help="Verbose language")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main code
    run(args)