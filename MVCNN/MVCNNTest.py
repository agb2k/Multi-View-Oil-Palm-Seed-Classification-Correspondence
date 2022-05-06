# M

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from glob2 import glob

from MVCNN.MVCNN import MVCNN

# Class labels to indices
class_id_map = {'Bad Seed': 0,
                'Good Seed': 1}

model = MVCNN()
model.load_state_dict(torch.load('mvcnn.pt'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

testSet_path = '../DatasetTest1/Test'

# FUNCTION TO GET PREDICTIONS FOR NEW CAR PLUGS
def mvcnn_pred(seed_name, data_dir, model, device):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    plug_fnames = glob.glob(data_dir + f'/{seed_name}_*.png')
    plug = torch.stack([transform(Image.open(fname).convert('RGB')) for fname in plug_fnames]).unsqueeze(0)
    plug = plug.to(device)
    pred = torch.nn.functional.softmax(model(plug)).argmax().item()
    return pred, {v: k for k, v in class_id_map.items()}[pred]


# GET NEW PREDICTIONS FROM RAW IMAGES USING A CAR PLUG NAME
plug_name = 'Set 10 - front Segmented Seed 1.png'
mvcnn_pred(plug_name, testSet_path + '/GoodSeed/front_s9.png', model, device)
