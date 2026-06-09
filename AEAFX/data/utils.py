import torch
import torch.utils.data

def get_tvt_loaders(*datasets:torch.utils.data.Dataset, batch_size:int, num_workers:int=0):
    loaders_list = []

    shuffle_list = [True, False, False]
    for i, ds in enumerate(datasets):
        loaders_list.append(
            torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_list[i],
                persistent_workers=True,
            )
        )
    return tuple(loaders_list)