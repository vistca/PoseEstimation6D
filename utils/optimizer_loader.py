from torch import optim


class OptimLoader(): 

    def __init__(self, optim_name, model_params, lr):
              
        if optim_name == "Adam":
            self.optim = optim.Adam(params=model_params, lr=lr)

        elif optim_name == "AdamW":
            self.optim = optim.AdamW(params=model_params, lr=lr)

        elif optim_name == "SGD":
            self.optim = optim.SGD(params=model_params)

        elif optim_name == "RMSprop":
            self.optim = optim.RMSprop(params=model_params)

        else:
            self.optim = optim.Adagrad(params=model_params)

    def get_optimizer(self):
        return self.optim
