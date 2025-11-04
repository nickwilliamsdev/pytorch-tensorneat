import torch
from functools import partial

# Infinite int, used to represent unavailable indices in int32 arrays
# (since we cannot use NaN in int32 arrays)
I_INF = torch.iinfo(torch.int32).max


def split_generator(base_gen, num_splits):
    """
    Split a base torch.Generator into multiple independent generators.
    """
    seeds = torch.randint(0, 2**32, (num_splits,), generator=base_gen)
    return [torch.Generator().manual_seed(seed.item()) for seed in seeds]

def attach_with_inf(arr, idx):
    """
    Attach values from `arr` using indices `idx`, replacing unavailable indices (I_INF) with NaN.
    """
    target_dim = arr.ndim + idx.ndim - 1
    expand_idx = idx.view(idx.shape + (1,) * (target_dim - idx.ndim))
    return torch.where(expand_idx == I_INF, float('nan'), arr[idx])


def fetch_first(mask, default=I_INF):
    """
    Fetch the first True index in a boolean mask.
    If no True value exists, return the default value.
    """
    idx = torch.argmax(mask).item()
    return idx if mask[idx] else default


def fetch_random(randkey, mask, default=I_INF):
    """
    Fetch a random True index from a boolean mask.
    If no True value exists, return the default value.
    """
    true_indices = torch.nonzero(mask, as_tuple=True)[0]
    if true_indices.numel() == 0:
        return default
    random_idx = torch.randint(0, true_indices.numel(), (1,), generator=randkey).item()
    return true_indices[random_idx].item()


def rank_elements(array, reverse=False):
    """
    Rank the elements in the array.
    If reverse is True, rank from smallest to largest. Default is largest to smallest.
    """
    if not reverse:
        array = -array
    return torch.argsort(torch.argsort(array))


def mutate_float(randkey, val, init_mean, init_std, mutate_power, mutate_rate, replace_rate):
    """
    Mutate a float value:
    - With probability `mutate_rate`, add noise.
    - With probability `replace_rate`, replace with a new random value.
    - Otherwise, keep the original value.
    """
    k1, k2, k3 = torch.Generator(), torch.Generator(), torch.Generator()
    k1.manual_seed(randkey.seed() + 1)
    k2.manual_seed(randkey.seed() + 2)
    k3.manual_seed(randkey.seed() + 3)

    noise = torch.normal(0, mutate_power, generator=k1)
    replace = torch.normal(init_mean, init_std, generator=k2)
    r = torch.rand((), generator=k3)

    if r < mutate_rate:
        return val + noise
    elif r < mutate_rate + replace_rate:
        return replace
    else:
        return val


def mutate_int(randkey, val, options, replace_rate):
    """
    Mutate an int value:
    - With probability `replace_rate`, replace with a new random value from `options`.
    - Otherwise, keep the original value.
    """
    k1, k2 = torch.Generator(), torch.Generator()
    k1.manual_seed(randkey.seed() + 1)
    k2.manual_seed(randkey.seed() + 2)

    r = torch.rand((), generator=k1)
    if r < replace_rate:
        return options[torch.randint(0, len(options), (1,), generator=k2).item()]
    else:
        return val


def argmin_with_mask(arr, mask):
    """
    Find the index of the minimum element in the array, considering only elements with True mask.
    """
    masked_arr = torch.where(mask, arr, float('inf'))
    return torch.argmin(masked_arr).item()


def hash_array(arr: torch.Tensor):
    """
    Hash an array of uint32 to a single uint.
    """
    arr = arr.to(dtype=torch.uint32)

    def update(hash_val, i):
        return hash_val ^ (
            arr[i].item() + 0x9E3779B9 + (hash_val << 6) + (hash_val >> 2)
        )

    hash_val = 0
    for i in range(arr.numel()):
        hash_val = update(hash_val, i)

    return hash_val