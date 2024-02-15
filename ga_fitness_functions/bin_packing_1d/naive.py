
def fitness_func(instance, x, out):
    bins = {}
    for idx, bin_idx in enumerate(x):
        bin_idx = int(bin_idx)
        bins[bin_idx] = bins.get(bin_idx, 0) + instance.weights[idx]
    
    num_bins = len(bins)
    max_bin_load = max(bins.values())
    
    # Objective: Minimize the number of bins used
    out["F"] = num_bins - max_bin_load / instance.bin_capacity
    out["placements"] = x.tolist()
    
    # Constraint: No bin should overflow
    out["G"] = max_bin_load - instance.bin_capacity

    return out
