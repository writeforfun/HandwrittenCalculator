from PIL import Image
from itertools import groupby
from torchvision import transforms
from cnn_model import Net

import numpy as np
import torch
import warnings
import sys
warnings.filterwarnings("ignore")

def split_img(img_path=sys.argv[1]):
    # to graysacle
    image = Image.open(img_path).convert("L")

    # resize to 28 height pixels
    w = image.size[0]
    h = image.size[1]
    ratio = w / h 
    new_h = 28
    new_w = int(ratio * 28)
    new_image = image.resize((new_w, new_h))

    # converting to a numpy array
    new_image_arr = np.array(new_image)

    # invert image
    new_inv_image_arr = 255 - new_image_arr

    final_image_arr = new_inv_image_arr / 255.0
    
    area = final_image_arr.any(0)
    result = [final_image_arr[:,[*g]] for k, g in groupby(np.arange(len(area)), lambda x: area[x] != 0) if k]
    
    return result

def padding(splited_images):
    num_of_elements = len(splited_images)
    elements_list = []

    for x in range(0, num_of_elements):

        img = splited_images[x]
        
        # adding 0 value columns as fillers
        width = img.shape[1]
        filler = (img.shape[0] - width) / 2
        
        if filler.is_integer() == False:   
            filler_l = int(filler)
            filler_r = int(filler) + 1
        else:                              
            filler_l = int(filler)
            filler_r = int(filler)
        
        arr_l = np.zeros((img.shape[0], filler_l))
        arr_r = np.zeros((img.shape[0], filler_r))
        tmp = np.concatenate((arr_l, img), axis= 1)
        element_arr = np.concatenate((tmp, arr_r), axis= 1)
        
        element_arr.resize(1, 28, 28)
        element_arr_tensor = torch.tensor(element_arr)
        elements_list.append(element_arr_tensor.unsqueeze(0))
        tensor_list = torch.cat(elements_list, dim=0)

    return tensor_list

def digit_generator(tensor_list):
    
    arr = [x.item() for x in tensor_list]

    op = {
        10,   # /
        11,   # =
        12,   # -
        13,   # +
        14    # *
            }   
    
    m_exp = []
    temp = []
        
    for item in arr:
        if item not in op:
            temp.append(item)
        else:
            m_exp.append(temp)
            m_exp.append(item)
            temp = []
    if temp:
        m_exp.append(temp)
        
    # converting the elements to numbers and operators
    i = 0
    num = 0
    for item in m_exp:
        if type(item) == list:
            if not item:
                m_exp[i] = ""
                i = i + 1
            else:
                num_len = len(item)
                for digit in item:
                    num_len = num_len - 1
                    num = num + ((10 ** num_len) * digit)
                m_exp[i] = str(num)
                num = 0
                i = i + 1
        else:
            m_exp[i] = str(item)
            m_exp[i] = m_exp[i].replace("10","/")
            m_exp[i] = m_exp[i].replace("11","=")
            m_exp[i] = m_exp[i].replace("12","-")
            m_exp[i] = m_exp[i].replace("13","+")
            m_exp[i] = m_exp[i].replace("14","*")
            
            i = i + 1
    
    separator = ' '
    m_exp_str = separator.join(m_exp)
    
    return (m_exp_str)

def calculate(m_exp_str):
    while True:
        try:
            answer = eval(m_exp_str)
            answer = round(answer, 3)
            equation  = m_exp_str + " = " + str(answer)
            print(equation) 
            break

        except SyntaxError:
            print("Prediction Failed")
            print("Following is the predicted expression:")
            print(m_exp_str)
            break

if __name__ == '__main__':
    model = Net()
    model.load_cpu('model_v3.pt')
    imgs = split_img()
    resize_imgs = padding(imgs)
    

    result = model.predict(resize_imgs)
    digits = digit_generator(result)

    calculate(digits)
