from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision
from torchview import draw_graph

class Yolo():
    def __init__(self):
        self.model = YOLO("yolo12n.pt")
        #self.model.overrides['classes'] = ['a', 'b']
        #self.model.overrides['nc'] = 2
        #torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        #self.model.dfl = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(16, config_dict['output_channels'])
        #self.model = self.model.load_state_dict(torch.load("yolo12n.pt"))

        
        count = 0
        params = 0
        for param in self.model.parameters():
            if param.requires_grad == True:
                count += 1

            params += 1
            
        print("Percent of grad req is: ", count/params)
        


    def get_model(self):
        #21
        #print(self.model['nc'])
        #self.model.model.model[21]
        return self.model