from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

class ScheduleLoader(): 
        
    def __init__(self, optimizer, scheduler_name):
        if scheduler_name == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(optimizer)

        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(optimizer, patience=10)
        
        else:
            self.scheduler == None

    def get_scheduler(self):
        return self.scheduler
