import yaml
import argparse

def add_runtime_args():

    parser = argparse.ArgumentParser()

    with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

    parser.add_argument('--lr', type=float,
                    help='The learning rate', default=config_dict['learning_rate'])
    
    parser.add_argument('--bs', type=int,
                    help='The bastch size', default=config_dict['batch_size'])
    
    parser.add_argument('--epochs', type=int,
                    help='The number of epochs', default=config_dict['epochs'])
    
    parser.add_argument('--dropout', type=float,
                    help='The percentage of dropout', default=config_dict['dropout'])
    
    parser.add_argument('--optimizer', type=str,
                    help='The optimizer to use', default=config_dict['optimizer'])
    
    parser.add_argument('--scheduler', type=str,
                        help='What kind of scheduler that should be applied', default=config_dict['scheduler'])
    
    parser.add_argument('--backbone', type=str,
                    help='The name of the backbone', default=config_dict['backbone'])

    parser.add_argument('--head', type=str,
                    help='The name of the head', default=config_dict['head'])
    
    parser.add_argument('--data', type=str,
                    help='The input folder', default=config_dict['data_dir'])
    
    parser.add_argument('--ld', type=str,
                    help='Google drive download path', default="")

    parser.add_argument('--wb', type=str,
                        help='If data is available locally or should be downloaded', default="")
    
    parser.add_argument('--log', type=bool, action=argparse.BooleanOptionalAction,
                        help='If run should be logged using wandb', default=True)
    
    parser.add_argument('--w', type=int,
                        help='The number of workers that should be used', default=2)
    
    parser.add_argument('--lm', type=str,
                        help='The name of the model that is to be loaded', default="")
    
    parser.add_argument('--sm', type=str,
                        help='The name of the model that is to be saved', default="")
    
    parser.add_argument('--tr', type=int,
                        help='Specifies the amount of trainable params, 0 min / 5 max', default=0)
    
    parser.add_argument('--fm', type=str,
                        help='The name of the mode that is to be used', default="transform")
    
    parser.add_argument('--test', type=bool,
                        help='If we should test the final model or solely train', default=False)
    
    return parser.parse_args()
    
