import os
import pandas as pd
import scanpy as sc
import anndata

class AnnDataset:
    def __init__(self, data_dir, metadata_filename):
        self.data_dir = data_dir
        self.metadata_file = os.path.join(data_dir, metadata_filename)
        self.metadata = pd.read_csv(self.metadata_file)
        
    def load_data(self, tissue_types, mode='union'):
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
            if mode=='inter':
                # Find common genes across all datasets
                common_genes = set(anndata_list[0].var.index)
                for adata in anndata_list[1:]:
                    common_genes.intersection_update(adata.var.index)
                
                # Filter each AnnData to include only the common genes
                anndata_list = [adata[:, list(common_genes)] for adata in anndata_list]
            
            if mode=='inter':
                combined_adata = anndata.concat(
                    anndata_list, 
                    axis=0,
                    join='inner', 
                    label='slide_id', 
                    keys=[adata.obs['slide_id'][0] for adata in anndata_list],
                    pairwise=True
                )
                return combined_adata
            else:
                # Group by gene panels
                gene_panels = {}
                for adata in anndata_list:
                    genes = tuple(sorted(adata.var.index))
                    if genes not in gene_panels:
                        gene_panels[genes] = []
                    gene_panels[genes].append(adata)
                
                # Concatenate within each group
                concatenated_adatas = []
                for genes, adatas in gene_panels.items():
                    
                    concatenated_adata = anndata.concat(
                        adatas,
                        axis=0,
                        join='inner', 
                        label='slide_id', 
                        keys=[adata.obs['slide_id'][0] for adata in adatas],
                        pairwise=True
                    )
                    concatenated_adatas.append(concatenated_adata)
                
                return concatenated_adatas
        else:
            return None
