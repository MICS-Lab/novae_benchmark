import argparse
import yaml

from novae_benchmark import AnnDataset
from novae_benchmark import get_model



parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='config file to use', type=str)

args = parser.parse_args()

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

config = load_yaml(args.config)

print(config)

if __name__ == "__main__":
    print("------------ Loading Dataset ----------------\n")

    print('Tissues considered : ')
    for tissue in config['dataset']['tissues']:
        print('----- ', tissue, '\n')

        dataset = AnnDataset(data_dir='../../data/spatial', metadata_filename='metadata_2024_06_21.csv')
        adataset = dataset.load_data(tissue_types=[tissue], mode=config['dataset']['mode'])

        print("------------ Dataset Loaded ! ----------------\n")

        results = {model_name: [] for model_name in config['params']['model_names']}

        for model_name in config['params']['model_names']:
            print("------------ Loading {} Model ----------------\n".format(model_name))
            model = get_model(model_name=model_name, hidden_dim=config['params']['hidden_dim'])
            print("------------ Model Loaded ! ----------------\n")
            if config['dataset']['mode'] == 'union':
                for adata in adataset:
                    model(adata=adata, n_clusters=config['params']['n_clusters'], batch_key=config['params']['batch_key'], 
                        device=config['params']['device'], fast_dev_run=config['params']['fast_dev_run'])
            
                    results[model_name].append(model.model_performances)
            else:
                    model(adata=adataset, n_clusters=config['params']['n_clusters'], batch_key=config['params']['batch_key'], 
                        device=config['params']['device'], fast_dev_run=config['params']['fast_dev_run'])
                    results[model_name].append(model.model_performances)

            print(results)
            
        



    
