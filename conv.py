import torch
import torch.nn.functional as F


def pad_circular(x, pad):
    """
    :param x: shape [H, W]
    :param pad: int >= 0
    :return:
    """
    x = torch.cat([..., x, x[0:pad]], dim=-2)
    x = torch.cat([..., x, x[:, 0:pad]], dim=-1)
    x = torch.cat([..., x[-2 * pad:-pad], x], dim=-2)
    x = torch.cat([..., x[:, -2 * pad:-pad], x], dim=-1)
    return x


def conv2d(input, kernel, mode=None):
    # kernel = kernel.transpose(0, 1)
    if mode is not None:
        hks = kernel.shape[-1] // 2
        padding = (hks, hks, hks, hks)
        input = F.pad(input, padding, mode)
    input = input.transpose(0, 1)
    output = F.conv2d(input, kernel, groups=kernel.size(0))
    return output.transpose(0, 1)


def conv2d_color(input, kernel, mode=None):
    hks = kernel.shape[-1] // 2
    padding = (hks, hks, hks, hks)
    if mode is not None:
        in_pad = F.pad(input, padding, mode=mode)
    else:
        in_pad = input
    if input.shape[1] == 1:
        output = F.conv2d(in_pad, kernel)
    else:
        output = torch.zeros_like(input)
        if mode is not None:
            output = F.pad(output, (-hks, -hks, -hks, -hks))
        for c in range(3):
            tmp = in_pad[:, c].unsqueeze(1)
            output[:, c] = F.conv2d(tmp, kernel)
    return output


def conv_transpose2d(input, kernel, mode=None):
    hks = kernel.shape[-1] // 2
    padding = (-hks, -hks, -hks, -hks)
    if input.shape[1] == 1:
        output = F.conv_transpose2d(input, kernel)
        if mode != None:
            output = F.pad(output, padding)
    else:
        for c in range(3):
            tmp = input[:, c].unsqueeze(1)
            tmp = F.conv_transpose2d(tmp, kernel)
            output = torch.zeros_like(input)
            if mode != None:
                tmp = F.pad(tmp, padding)
            if c == 0:
                output = torch.zeros_like(tmp).expand(-1, 3, -1, 1)
            output[:, c] = tmp
    return output