        # seed=42
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
                "hidden_state": (256, 512, 512, 256), 
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
        # 0.9714  0.9515  0.4968