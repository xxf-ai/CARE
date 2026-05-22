import torch
import numpy as np

def hr_ndcg_batch(scores, k_list=[5,10,20]):
    rank = (scores[:,1:] >= scores[:,0:1]).sum(dim=1) + 1
    results = {}
    for k in k_list:
        results[f"hr@{k}"]   = (rank<=k).float().mean().item()
        results[f"ndcg@{k}"] = (1.0/torch.log2(rank.float()+1.0)*(rank<=k).float()).mean().item()
    return results

def aggregate_metrics(metric_list):
    if not metric_list: return {}
    return {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}
