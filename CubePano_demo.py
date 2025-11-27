import torch, os
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from CubemapAndERP import c2e
import numpy as np
import re
from openai import OpenAI
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

deepseek_api_key = "None"

with open('./script/deepseek_message.txt','r') as f:
    role_message = f.read()

whats_you_want = "a park"

role_message = role_message.replace('<>',whats_you_want)

def LLM_assistance():
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    # Round 1
    messages = [{"role": "user", "content": role_message}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages
    )

    reasoning_content = response.choices[0].message.reasoning_content #  This is the content of deep thinking
    content = response.choices[0].message.content # This is the actual content of the answer

    print('hi,the content is: \n', content) # The response format and order are specified in the prompt, so the large model will be described in the order of front, right, back, left, top, and bottom

    prompt_list = content.split('\n')

    prompt_front = re.search("(?<=Front:).*$", prompt_list[0])[0]
    prompt_right = re.search("(?<=Right:).*$", prompt_list[1])[0]
    prompt_back = re.search("(?<=Back:).*$", prompt_list[2])[0]
    prompt_left = re.search("(?<=Left:).*$", prompt_list[3])[0]
    prompt_top = re.search("(?<=Top:).*$", prompt_list[4])[0]
    prompt_bottom = re.search("(?<=Bottom:).*$", prompt_list[5])[0]

    prompt_dict ={'front':prompt_front,'right':prompt_right,'back':prompt_back,'left':prompt_left,'top':prompt_top,'bottom':prompt_bottom}

    print("hello, the prompt is : \n", prompt_dict)
    return prompt_dict

def get_the_baseIamge_Mask(image,flag=None): # flag is front2right or right2back or back2left or top or bottom
    # Next time, a canvas with image prompts and a mask should be returned
    if flag == "front2right":
        image_front = image
        image_front_crop = image_front.crop((512,0,576,512)) # Cut off the right 1/8 of the prompt image
        image_white = Image.new('RGB', (576, 512), (255, 255, 255)) # Creating Canvas
        image_white_mask = Image.new('RGB', (576, 512), (255, 255, 255)) # Create a mask with white representing the area to be generated
        image_black_mask = Image.new('RGB', (64, 512), (0, 0, 0)) # Make a mask, with black representing the retained parts and the indicated parts
        image_white.paste(image_front_crop,(0,0,64,512)) # Paste the picture prompt onto the canvas
        image_white_mask.paste(image_black_mask,(0,0,64,512)) # Stick on the black mask to get the mask
        return image_white, image_white_mask
    elif flag == "right2back":
        image_right = image
        image_right_crop = image_right.crop((512,0,576,512)) # Cut out 1/8 of the right side of the prompt image
        image_white = Image.new('RGB', (576, 512), (255, 255, 255)) # Make a canvas
        image_white_mask = Image.new('RGB', (576, 512), (255, 255, 255)) # When making a mask, white represents the area to be generated
        image_black_mask = Image.new('RGB', (64, 512), (0, 0, 0)) # Make a mask, with black representing the retained parts and the indicated parts
        image_white.paste(image_right_crop,(0,0,64,512)) # Paste the picture prompt onto the canvas
        image_white_mask.paste(image_black_mask,(0,0,64,512)) # Stick on the black mask to get the mask
        return image_white, image_white_mask
    elif flag == 'back2left':
        image_front = image[0] # "left" needs to refer to both the front view and the rear view to ensure the continuity of the panorama
        image_back = image[1]
        image_front_crop = image_front.crop((0,0,64,512))# Trim the left 1/8 of the prompt forward view image, which is continuous with the left view
        image_back_crop = image_back.crop((512,0,576,512)) # Cut out 1/8 of the right side of the prompt image
        image_white = Image.new('RGB', (576, 512), (255, 255, 255)) # Make a canvas
        image_white_mask = Image.new('RGB', (576, 512), (255, 255, 255)) # The white 255 of the mask represents the area to be generated
        image_black_mask = Image.new('RGB', (64, 512), (0, 0, 0)) # Make a mask. The black 0 represents the retained part and the hint part
        image_white.paste(image_back_crop,(0,0,64,512)) # 将Paste the picture prompt onto the canvas
        image_white.paste(image_front_crop,(512,0,576,512)) # Paste the front view image prompt to the left side of the canvas to ensure continuity

        image_white_mask.paste(image_black_mask,(0,0,64,512)) # Stick on the black mask to get the mask. Both the left and right sides need to be stuck on
        image_white_mask.paste(image_black_mask,(512,0,576,512))
        return image_white, image_white_mask
    
    if flag == "top":
        image_front = image[0]
        image_right = image[1]
        image_back = image[2]
        image_left = image[3]

        image_white = Image.new('RGB', (640, 640), (255, 255, 255))
        image_white_mask = Image.new('RGB', (640, 640), (255, 255, 255)) # The white 255 of the mask represents the area to be generated

        image_front_top = image_front.crop((64,0,576,64))

        image_right_top = image_right.crop((64,0,576,64))
        image_right_top = image_right_top.rotate(90,expand=True)

        image_back_top = image_back.crop((64,0,576,64))
        image_back_top = image_back_top.rotate(180,expand=True)
        # image_back_top = image_back_top.transpose(Image.FLIP_TOP_BOTTOM)

        image_left_top = image_left.crop((64,0,576,64))
        image_left_top = image_left_top.rotate(270,expand=True)

        image_white.paste(image_front_top,(64,576,576,640))
        image_white.paste(image_right_top,(576,64,640,576))
        image_white.paste(image_back_top,(64,0,576,64))
        image_white.paste(image_left_top,(0,64,64,576))


        image_black_mask1 = Image.new('RGB', (64, 512), (0, 0, 0))
        image_black_mask2 = Image.new('RGB', (512, 64), (0, 0, 0))

        image_white_mask.paste(image_black_mask2,(64,576,576,640))
        image_white_mask.paste(image_black_mask1,(576,64,640,576))
        image_white_mask.paste(image_black_mask2,(64,0,576,64))
        image_white_mask.paste(image_black_mask1,(0,64,64,576))

        return image_white,image_white_mask
    
    if flag == "bottom":
        image_front = image[0]
        image_right = image[1]
        image_back = image[2]
        image_left = image[3]

        image_white = Image.new('RGB', (640, 640), (255, 255, 255))
        image_white_mask = Image.new('RGB', (640, 640), (255, 255, 255)) # The white 255 of the mask represents the area to be generated

        image_front_top = image_front.crop((64,448,576,512))

        image_right_top = image_right.crop((64,448,576,512))
        image_right_top = image_right_top.rotate(270,expand=True)

        image_back_top = image_back.crop((64,448,576,512))
        image_back_top = image_back_top.rotate(180,expand=True)

        image_left_top = image_left.crop((64,448,576,512))
        image_left_top = image_left_top.rotate(90,expand=True)

        image_white.paste(image_back_top,(64,576,576,640))
        image_white.paste(image_right_top,(576,64,640,576))
        image_white.paste(image_front_top,(64,0,576,64))
        image_white.paste(image_left_top,(0,64,64,576))

        image_black_mask1 = Image.new('RGB', (64, 512), (0, 0, 0))
        image_black_mask2 = Image.new('RGB', (512, 64), (0, 0, 0))

        image_white_mask.paste(image_black_mask2,(64,576,576,640))
        image_white_mask.paste(image_black_mask1,(576,64,640,576))
        image_white_mask.paste(image_black_mask2,(64,0,576,64))
        image_white_mask.paste(image_black_mask1,(0,64,64,576))

        return image_white,image_white_mask



def create_Pano_image(prompt=None):

    timestamp = time.time()
    save_name = 'result'+str(timestamp)
    
    if not os.path.exists('./Pano_result/'+save_name):
        os.mkdir('./Pano_result/'+save_name)

    save_path = './Pano_result/'+save_name +'/'


    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        # torch_dtype=torch.float16,
    )

    pipe = pipe.to("cuda")

    image_base = Image.open('./custom_inpainting/b_img.png')
    mask_image_base = Image.open('./custom_inpainting/white_mask.png')

    print('create the front image...')
    image_front = pipe(prompt=prompt["front"], image=image_base, mask_image=mask_image_base, width=576, height=512).images[0] # generate view front


    print('create the right image...')
    image_base,mask_image_base = get_the_baseIamge_Mask(image_front,flag='front2right')
    image_right = pipe(prompt=prompt["right"], image=image_base, mask_image=mask_image_base, width=576, height=512).images[0] # generate view right


    print('create the back image...')
    image_base,mask_image_base = get_the_baseIamge_Mask(image_right,flag='right2back')
    image_back = pipe(prompt=prompt["back"], image=image_base, mask_image=mask_image_base, width=576, height=512).images[0] # generate view back

    print('create the left image...')
    image_base,mask_image_base = get_the_baseIamge_Mask([image_front, image_back],flag='back2left')
    image_left = pipe(prompt=prompt["left"], image=image_base, mask_image=mask_image_base, width=576, height=512).images[0] # generate view left


    print('create the top image...')
    image_base,mask_image_base = get_the_baseIamge_Mask([image_front, image_right, image_back, image_left],flag='top')
    image_top = pipe(prompt=prompt["top"], image=image_base, mask_image=mask_image_base, width=640, height=640).images[0] # generate view top


    print('create the bottom image...')
    image_base,mask_image_base = get_the_baseIamge_Mask([image_front, image_right, image_back, image_left],flag='bottom')
    image_bottom = pipe(prompt=prompt["bottom"], image=image_base, mask_image=mask_image_base, width=640, height=640).images[0] # generate view bottom



    image_front = image_front.crop((64,0,576,512))
    image_right = image_right.crop((64,0,576,512))
    image_back = image_back.crop((64,0,576,512))
    image_left = image_left.crop((64,0,576,512))
    image_top = image_top.crop((64,64,576,576))
    image_bottom = image_bottom.crop((64,64,576,576))

    image_box = Image.new('RGB', (2048, 1536), (0, 0, 0))

    image_box.paste(image_front,(512,512,1024,1024))
    image_box.paste(image_right,(1024,512,1536,1024))
    image_box.paste(image_back,(1536,512,2048,1024))
    image_box.paste(image_left,(0,512,512,1024))

    image_box.paste(image_top,(512,0,1024,512))
    image_box.paste(image_bottom,(512,1024,1024,1536))

    image_pano = np.array(image_box)
    image_pano = c2e(image_pano, 1024, 2048, cube_format='dice')
    image_pano = Image.fromarray(np.uint8(image_pano))
    image_pano.save(save_path+'Pano_image.png')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--deepseek_api_key', type=str, help='Put your Deepseek API key here')
    parser.add_argument('--key_prompt', default='a park',type=str, help='Put the panorama scene description you want to generate here')
    arg = parser.parse_args()

    deepseek_api_key = arg.deepseek_api_key
    whats_you_want = arg.key_prompt

    prompt_dict = LLM_assistance()

    create_Pano_image(prompt_dict)
