# 切换到 Ours-2 目录并激活名为 velo_ours 的 conda 环境
'''
cd Ours-2
conda activate velo_ours
python train.py --draw
python traverse.py
'''

# 导入必要的模块和库
from model.model import VeloModel
from model.Config import Config
from model.dataset import Data
from model.graph import GraphBatchSampler, build_root_directed_adjacency
from tool.utils import * 
from tool.plot import *
from tool.evaluate import *
from preprocessing import * 
import numpy as np
import torch, os, argparse
from torch.utils.data import DataLoader 
from const.cluster_edges import all_edges
import random

import scvelo as scv

# 定义解析命令行参数的函数
def parse_args():
    parser = argparse.ArgumentParser()
    # 标志参数：是否在训练后绘制图形
    parser.add_argument("--draw", action="store_true", help="是否在训练后绘制图形")
    parser.add_argument("--load_model_path", type=str, default=None, help="加载已保存模型的路径")
    parser.add_argument("--data_path", type=str, default="/data/yuchang/dataset/MouseBrain.h5ad")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoint")
    parser.add_argument("--batch_sampling", choices=("graph", "random"), default="graph")
    parser.add_argument("--notpreprocess", action="store_false", help="是否在训练后绘制图形")
    args = parser.parse_args()
    return args

# 主训练函数
def train(args=parse_args()):
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 定义要处理的数据集列表（此处仅使用第一个数据集）
    datasets = [
        "DentateGyrus.h5ad",
        "Pancreas.h5ad",
        "MouseBrain.h5ad",
        "endocrinogenesis_day15.h5ad", 
        "Hindbrain_GABA_Glio.h5ad",
        "erythroid_lineage.h5ad", 
        "organoids.h5ad",
        "retina.h5ad",
        "reprogramming.h5ad",
        "hematopoiesis.h5ad",
        "human_limb.h5ad"
    ]
    data_index = 2
    cluster_edges = all_edges.get(datasets[data_index])
    print("data_index: ", data_index)
    if args.notpreprocess:
        adata = load_data(args.data_path)
        # adata = load_data("../Cell2LLM/adata/train_input.h5ad")  # (149844, 1501), (30001, 2000), (397193, 3000)
        seed = args.seed
        config = Config(adata, seed=seed)
        adata = preprocess_data_scv(adata, n_top_genes=None, n_neighbors=config.config["data"]["num_neighbor"], seed=seed)
        
        config.update(adata)
    else:
        load_data_path = 'data/preprocessed/' + datasets[data_index]
        adata = load_data(load_data_path)
        config = Config(adata, seed=args.seed)
    
    cluster_key, cluster_labels = find_cluster_key(adata)
    print(adata.obs[cluster_key].unique())
    adata.obs["cluster"] = adata.obs[cluster_key]
    
    batch_size = args.batch_size or config.config["train"]["batch_size"]

    dataset = Data(adata, spliced_key="Ms", unspliced_key="Mu")
    full_adj = adata.obsp["connectivities"]
    directed_adj, root_distance = build_root_directed_adjacency(
        full_adj, dataset.root_mask
    )
    adata.obs["root_distance"] = root_distance
    print(
        "图方向化完成: "
        f"roots={int(dataset.root_mask.sum())}, "
        f"undirected_nnz={full_adj.nnz}, "
        f"directed_nnz={directed_adj.nnz}, "
        f"unreachable={int(np.isnan(root_distance).sum())}"
    )

    if args.batch_sampling == "graph":
        batch_sampler = GraphBatchSampler(
            full_adj,
            batch_size=batch_size,
            seed=config.config["train"]["seed"],
        )
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        shuffle_generator = torch.Generator()
        shuffle_generator.manual_seed(config.config["train"]["seed"])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=shuffle_generator,
        )
    
    model = VeloModel(dataloader=dataloader, config=config, device=device, cluster_edge=None)
    # 如果指定了加载模型路径，则加载保存的模型
    if args.load_model_path:
        model = load_model(model, args.load_model_path, device=device)
    else:
        print("开始训练")
        model.fit(
            adjacency_matrix=full_adj,
            directed_adjacency_matrix=directed_adj,
            device=device,
            save_path=args.save_path,
        )

    # 预测
    print("开始预测")  # 开始进行预测
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 按顺序
    
    
    adata = compute_pred(adata, dataloader, model, device=device)
    adata = compute_latent(adata, dataloader, model, device=device)

    print("计算速度在 UMAP 空间投影")  # 计算速度在 UMAP 空间的投影
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")
    scv.tl.velocity_confidence(adata)
    
    velocity_consistency_score = adata.obs['velocity_confidence'].mean()
    print(f"速度可靠性得分: {velocity_consistency_score:.4f}")
    if cluster_key is not None:
        latent_z = adata.obsm["latent_presentation"]
        pca = PCA(n_components=10)
        latent_z = pca.fit_transform(latent_z)
        evaluate_ICCoh(adata, cluster_key=cluster_key, velocity_key='velocity')
        
        if cluster_edges is not None:
            _, cbdir = evaluate_CBDir2(adata, cluster_key=cluster_key, velocity_key="velocity", cluster_edges=cluster_edges)
            print(f"CBdir(without graph): {cbdir:.4f}")
            
            from tool.evaluate import compute_velocity_from_graph
            adata = compute_velocity_from_graph(adata, new_key="velocity_graph_umap")
            _, cbdir = evaluate_CBDir2(adata, cluster_key=cluster_key, velocity_key="velocity_graph", cluster_edges=cluster_edges)
            print(f"CBdir(with graph): {cbdir:.4f}")
            
    # adata.write("../Cell2LLM/code3/outputs/MouseBrain_GFG_3.h5ad")
     
    return adata
            
if __name__ == "__main__":
    # 运行主训练函数，传入命令行参数
    train()
    