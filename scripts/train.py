from novae_benchmark import get_model, AnnDataset


def train(tissue_types, model_names, data_dir='../../data/spatial', metadata_filename='metadata_2024_06_21.csv', hidden_dim=64, multi_slide=False):
    dataset = AnnDataset(data_dir=data_dir, metadata_filename=metadata_filename)
    adataset = dataset.load_data(tissue_types=tissue_types)

    for model_name in model_names:
        model = get_model(model_name=model_names, hidden_dim=hidden_dim)
        