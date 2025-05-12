import os
import subprocess
import shutil
import argparse
import yaml
import json
from tqdm import tqdm

def download_data(google_folder, dataset_root):
    output_path = dataset_root + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        subprocess.run(["gdown", "--folder", str(google_folder), "-O", "tmp/"],
            check=True)
        subprocess.run(["unzip", "tmp/DenseFusion/Linemod_preprocessed.zip", "-d", output_path],
        check=True 
        )
        shutil.rmtree('tmp')

def transfer_data(base_path, ending):
    yml_path = base_path + "/" + ending + ".yml"
    json_path = base_path + "/" + ending + ".json"
    if not os.path.exists(json_path):
        with open(yml_path, 'r') as yaml_in, open(json_path, "w") as json_out:
            yaml_object = yaml.safe_load(yaml_in)
            json.dump(yaml_object, json_out)

def yaml_to_json(folder_path):
    print("Transforming yaml to json")
    dirs = os.listdir(folder_path)
    for dir in tqdm(dirs):
        if os.path.isdir(folder_path + dir):

            base_path = folder_path + dir
            transfer_data(base_path, "info")
            transfer_data(base_path, "gt")

def get_runtime_args():
    parser = argparse.ArgumentParser(description='Optional app description')

    with open('config/config.yaml') as f:
            config_dict = yaml.safe_load(f)

    parser.add_argument('--bf', type=str,
                    help='Shared google folder', default=config_dict['data_dir'])
    
    parser.add_argument('--f', type=str,
                    help='folder path for the data', default=config_dict['data_dir']+"Linemod_preprocessed/data/")
    
    parser.add_argument('--gf', type=str,
                    help='Shared google folder', default="")
    

    return parser.parse_args()
    
    

if __name__ == "__main__":
    args = get_runtime_args()
    download_data(args.gf, args.bf)
    yaml_to_json(args.f)
    print("Data prep complete!")
