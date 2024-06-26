import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata

class AnnDataset:
    def __init__(self, data_dir, metadata_filename):
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, metadata_filename)
        self.metadata = pd.read_csv(self.metadata_file)
        
    def load_data(self, tissue_types, use_common_genes=True, multi_slide=False):
        anndata_list = []
        anndata_list_original = []
        
        for tissue in tissue_types:
            files_to_load = self.metadata[self.metadata['tissue'] == tissue]['dataset_name']
            
            for dataset_name in files_to_load:
                file_path = os.path.join(self.data_dir, f"{dataset_name}.h5ad")
                adata = sc.read_h5ad(file_path)
                anndata_list_original.append(adata)

                
                # Convert gene names to lowercase to handle case insensitivity
                adata.var.index = adata.var.index.str.lower()
                
                # Add a column to indicate the dataset
                adata.obs['dataset'] = dataset_name
                
                anndata_list.append(adata)
        
        if anndata_list:
            if use_common_genes:
                # Find common genes across all datasets
                common_genes = set(anndata_list[0].var.index)
                for adata in anndata_list[1:]:
                    common_genes.intersection_update(adata.var.index)
                
                # Filter each AnnData to include only the common genes
                anndata_list = [adata[:, list(common_genes)] for adata in anndata_list]
            
            combined_adata = anndata.concat(
                anndata_list, 
                axis=0,
                join='inner', 
                label='slide_id', 
                keys=[adata.obs['slide_id'][0] for adata in anndata_list],
                pairwise=True
            )
        
        else:
            combined_adata = None
        
        if multi_slide:
            return combined_adata
        else:
            return anndata_list_original

