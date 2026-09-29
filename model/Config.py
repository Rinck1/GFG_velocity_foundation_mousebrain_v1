import torch
import random
import numpy as np
class Config():
    def __init__(self, adata=None, seed=None):
        self.config = {
            "data" : {
                "num_top_gene": 2000, 
                "num_cell": None, 
                "num_gene": 2000, 
                "num_neighbor": 20, 
            }, 
            "model" : {
                "gene_dim": 8, 
                "codebook_size": 32,
                "use_vq": True,
                "hidden_state": (256, 512, 512, 256), 
                "attention_heads": 4,
                "attention_layers": 1,
                "attention_ff_mult": 2,
                "mlp_dropout": 0.2,
                "attention_dropout": 0.1,
            }, 
            "train" : {
                "seed": 0,
                "num_epochs": 10, 
                "batch_size": 128, 
                "lr": 1e-3, #1e-3,
                "early_stop": False, 
                "stop_width": 0.05, 
                
                "loss_weight": {
                    'loss_ode': 20, #10, #5, #2,
                    'loss_s': 2.5, #2.5,
                    'loss_u': 7.5, #7.5, 
                    'loss_m_vq': 1, 
                    'loss_v_vq': 1, 
                    'loss_smooth': 300, # 100,
                    'loss_align': 300, #300,  
                }
            },
        }
        self.history = None
        if adata is not None:
            self.update(adata)
            
        if seed is None:
            seed = self.config["train"]["seed"]
            
        print("训练种子为", seed)
        self.config["train"]["seed"] = seed
        self.set_seed(self.config["train"]["seed"]) 
        
    def update(self, adata):
        if adata is not None:
            self.config["data"]["num_gene"] = adata.n_vars
            self.config["data"]["num_cell"] = adata.n_obs 
            
    def clear_history(self):
        self.history = {k: [] for k in self.config["train"]["loss_weight"]}
        
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
