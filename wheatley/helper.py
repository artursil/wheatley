from torch import nn


def model_weights(model: nn.Module):
    for key in model.state_dict().keys():
        print(model.state_dict()[key].shape)
        # print(key)


def num_parameters(model: nn.Module):
    print(f'{sum(p.numel() for p in model.parameters()):,.0f} parameters')
    

