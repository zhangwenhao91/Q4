"""Derive consistent HC K partitions without changing device source or storage layout."""
def split_config(base, tokens, flattened, splits):
    if splits not in (1, 2, 4, 8) or flattened % (256 * splits):
        raise ValueError('K splits must divide hidden blocks')
    # Preserve baseline active-core budget. Larger splits than the available budget
    # are rejected rather than silently oversubscribing the device.
    budget = base['num_aic_cores']
    if splits > budget:
        raise ValueError(f'Requested {splits} splits exceeds baseline active-core budget {budget}')
    cores_per_split = min(budget // splits, (tokens + 127) // 128)
    result = dict(base)
    result.update(num_k_splits=splits, cores_per_split=cores_per_split,
        num_aic_cores=splits * cores_per_split,
        num_hidden_blocks_per_split=flattened // 256 // splits,
        num_token_blocks_per_chunk=cores_per_split * base['num_l0c_stages'])
    return result
