"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torchvision

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
from main import run_inference
from model import YoloNetV3
from roadmap_model import CNN_VAE


# import your model class
# import ...


def get_transform_task1():
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class ModelLoader():
    # Fill the information for your team
    team_name = 'Threat Score Underflow'
    team_number = 'unknown'
    round_number = 2
    team_member = ['Andrew Hopen', 'Eric Kosgey','Syed Rahman']
    contact_email = 'ah182@nyu.edu'

    def __init__(self, model_file= "/home/ek1905/weights/bounding_box.pt"):
        self.bounding_box_model = YoloNetV3(nms=True)
        self.bounding_box_model.load_state_dict(torch.load(model_file)['model_state_dict'])
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.bounding_box_model.to(self.device)
        print("Using device: " + str(self.device))
        self.bounding_box_model.eval()
        self.road_map_model = CNN_VAE().to(self.device)
        pretrained_dict = torch.load('/home/ek1905/weights/road_map.pt', map_location=self.device)
        model_dict = self.road_map_model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.road_map_model.load_state_dict(pretrained_dict)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        results = run_inference(self.bounding_box_model, samples, self.device, 0.5, 0.2)
        results = (((results * 1.92) - 400) / 10)
        return results

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800]
	
        return self.road_map_model.inference(samples.reshape(6,-1,3,256,256).to(self.device), self.device) > 0.5
