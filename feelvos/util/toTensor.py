import torch


def list_to_tensor(t_list, x, y, device):
    for i in range(x):
        for j in range(y):
            t_list[i][j] = torch.from_numpy(t_list[i][j]).to(device=device)

    return t_list
