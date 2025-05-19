from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR

class OptimLoader(): 

    def __init__(self, optim_name, model_params, lr):
        self.optim = optim.SGD(params=model_params, lr=lr)
        if optim_name == "Adam":
            self.optim = optim.Adam(params=model_params, lr=lr)

        elif optim_name == "AdamW":
            self.optim = optim.AdamW(params=model_params, lr=lr)

    def get_optimizer(self):
        return self.optim
    
    def use_scheduler():
        scheduler1 = CosineAnnealingWarmRestarts()
        #scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        #scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        #scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], 

