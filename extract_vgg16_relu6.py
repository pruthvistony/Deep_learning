# write your codes here
import cv2
import numpy as np
import os
import scipy.io as sio
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn



def get_all_img_in_tensor(img, prep):
    img_list = []
    img1 = img[:224, :224]
    img_list.append(prep(img1))
    img2 = img[32:, :224]
    img_list.append(prep(img2))
    img3 = img[32:, 116:]
    img_list.append(prep(img3))
    img4 = img[:224, 116:]
    img_list.append(prep(img4))
    img5 = img[16:240, 56:280]
    img_list.append(prep(img5))
    #out = torch.stack([img1, img2])
    return img_list

def initialize_model():
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = models.vgg16_bn(pretrained=True)
    input_size = 224
    
    return model_ft, input_size


model, size = initialize_model()
model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
prep = transforms.Compose([ transforms.ToTensor(), normalize ])
#prep(img)

folder_list = sorted(os.listdir('UCF101_release/images_class1'))
try:
    folder_list. remove('.DS_Store')
except:
    pass
#print(folder_list)
j = 1
for fol_list in folder_list:
    path = 'UCF101_release/images_class1' + '/' + fol_list
    print("Feature extraction of the images in folder is in progress", fol_list)
    file_list = os.listdir(path)
    #print(file_list)
    i = 1
    out_list = np.empty(shape=[0,4096])
    for file_name in file_list:
        img = cv2.imread(path + '/' + file_name)
        #img = prep(img)
        img_list = get_all_img_in_tensor(img, prep)
        inp = torch.stack(img_list)
        #print(out.shape)
        outputs = model(inp)
        mean_output = outputs.mean(0)
        np_out = mean_output.detach().numpy()
        #print(np_out.shape)
        #out_list.append(mean_output)
        out_list = np.vstack((out_list, np_out))

        #print("image -", i, mean_output.shape)
        i = i + 1
        
    #out_tensor = torch.stack(out_list)
    #write the out_list to file
    print("saving the file ", fol_list+'.mat')
    sio.savemat('UCF101_release/vgg16_relu6' + '/' + fol_list +'.mat', {'Feature':out_list}, do_compression=True)
    
