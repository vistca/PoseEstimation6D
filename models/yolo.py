from ultralytics import YOLO

class Yolo():
    def __init__(self):
        self.model = YOLO("yolo12n.pt")
        
        count = 0
        params = 0
        for param in self.model.parameters():
            if param.requires_grad == True:
                count += 1

            params += 1
            
        print("Percent of grad req is: ", count/params)
        


    def get_model(self):
        return self.model