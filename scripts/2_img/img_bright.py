# -*- coding=utf-8 -*-
import cv2
from PIL import ImageEnhance,Image
import os
from tqdm import tqdm
import numpy as np
import sys


def core(input_dir):

    thres_bright_0 = 5.0
    thres_bright_1 = 20.0
    thres_bright_2 = 30.0

    def calculate_brightness(image):
        image = image.convert("L")  # Convert to grayscale
        pixel_values = np.array(image)
        return np.mean(pixel_values)

    def enhance_brightness(image, thres):
        enhancer = ImageEnhance.Brightness(image)
        brightened_img = enhancer.enhance(thres)
        return brightened_img

    uid = input_dir.split("/")[-1]
    counter = {-1:0, 0:0, 1:0, 2:0, 3:0}
    img_list = []
    for i in range(10):
        img_name = str(i).zfill(5)
        image_path = os.path.join(input_dir, f"{img_name}.png")
        img = Image.open(image_path)
        img_list.append(img)

        thres_enhance_1 = 2.0*2.0
        thres_enhance_2 = 4.0*2.0
        thres_enhance_3 = 6.0*2.0

        bright_enhance_2 = calculate_brightness(enhance_brightness(img, thres_enhance_2))
        bright_enhance_3 = calculate_brightness(enhance_brightness(img, thres_enhance_3))

        if bright_enhance_3 <= thres_bright_0:
            counter[-1] += 1
        elif bright_enhance_3 <= thres_bright_2:
            counter[3] += 1
        elif bright_enhance_2 <= thres_bright_2:
            counter[2] += 1
        else:
            counter[1] += 1

    flag = max(counter, key=counter.get)
    if flag == -1:
        pass
    elif flag == 0:
        for j in range(10):
            img_name = str(j).zfill(5)
            image_path = os.path.join(input_dir, f"{img_name}.png")
            image_path_out = image_path.replace(".png", f".bright.png")
            os.system(f"cp {image_path} {image_path_out}")
    else:
        thres = [thres_enhance_1, thres_enhance_2, thres_enhance_3][flag-1]
        for j in range(10):
            img_name = str(j).zfill(5)
            image_path = os.path.join(input_dir, f"{img_name}.png")
            img = img_list[j]
            img_enhance = enhance_brightness(img, thres)
            image_path_out = image_path.replace(".png", f".bright.png")
            img_enhance.save(image_path_out)
            
        

if __name__ == '__main__':
    input_dir = sys.argv[-1]
    core(input_dir)
