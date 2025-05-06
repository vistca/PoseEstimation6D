from torch import optim

class OptimLoader(): 

    def __init__(self, optim_name, model_params, lr):
        self.optim = optim.SGD(params=model_params, lr=lr)
        if optim_name == "Adam":
            self.optim = optim.Adam(params=model_params, lr=lr)

        elif optim_name == "AdamW":
            self.optim = optim.AdamW(params=model_params, lr=lr)

    def get_optimizer(self):
        return self.optim