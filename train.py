from utils.wandb_setup import Wandb

wandb_instance = Wandb("round9")

wandb_instance.log_metric({"test_metric" : 3000})


