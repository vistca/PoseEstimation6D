from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

class ScheduleLoader(): 
        
    def __init__(self, optimizer, scheduler_name, batch_size, sample_size):
        self.name = scheduler_name
        if scheduler_name == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3*sample_size//batch_size)

        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(optimizer, patience=10)
        else:
            self.scheduler = None

    def get_scheduler(self):
        return self.scheduler
