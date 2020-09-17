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


def conv2d(x, k, pad_mode='constant', thresh=41):
    # derived from jodag's post
    # @ https://stackoverflow.com/questions/60561933/verify-convolution-theorem-using-pytorch
    if max(k.shape[-2:]) < thresh:
        # Regular convolution with F.conv2d
        pad_y = (k.shape[-2] - 1) // 2
        pad_x = (k.shape[-1] - 1) // 2
        x_padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), pad_mode)
        return F.conv2d(x_padded, k, bias=None)
    else:
        # FFT-based convolution
        assert len(x.size()) == 4
        assert len(k.size()) == 4
        # in general not necessary that inputs are odd shaped but makes life easier
#         assert x.size(2) % 2 == 1
#         assert x.size(3) % 2 == 1
        if x.size(2) % 2 == 0:
            py = 1
            x = F.pad(x, (0,0,0,py))
        else:
            py = 0
        if x.size(3) % 2 == 0:
            px = 1
            x = F.pad(x, (0,px,0,0))
        else:
            px = 0
        assert k.size(2) % 2 == 1
        assert k.size(3) % 2 == 1

        hks = k.shape[-1] //2
        x = F.pad(x, (hks,hks,hks,hks), 'reflect')

        size_y = x.size(2) + k.size(2) - 1
        size_x = x.size(3) + k.size(3) - 1

        x_new = torch.zeros((1, 1, size_y, size_x), device=x.device)
        k_new = torch.zeros((1, 1, size_y, size_x), device=k.device)

        # copy x to center
        x_pad_y = (x_new.size(2) - x.size(2)) // 2
        x_pad_x = (x_new.size(3) - x.size(3)) // 2
        x_new[..., x_pad_y:-x_pad_y, x_pad_x:-x_pad_x] = x

        # anchor of k is 0,0 (flip g and wrap circular)
        k_center_y = k.size(2) // 2
        k_center_x = k.size(3) // 2
        k_y, k_x = torch.meshgrid(torch.arange(k.size(2)), torch.arange(k.size(3)))
        k_new_y = (k_y.flip(0) - k_center_y) % k_new.size(2)
        k_new_x = (k_x.flip(1) - k_center_x) % k_new.size(3)
        k_new[..., k_new_y, k_new_x] = k[..., k_y, k_x]

        # take fft of both x and k
        F_x = torch.rfft(x_new, signal_ndim=2, onesided=False)
        F_k = torch.rfft(k_new, signal_ndim=2, onesided=False)

        # complex multiply
        FxY_real = F_x[..., :, :, 0] * F_k[..., :, :, 0] - F_x[..., :, :, 1] * F_k[..., :, :, 1]
        FxY_imag = F_x[..., :, :, 0] * F_k[..., :, :, 1] + F_x[..., :, :, 1] * F_k[..., :, :, 0]
        FxY = torch.stack([FxY_real, FxY_imag], dim=-1)

        # inverse fft
        fcy = torch.irfft(FxY, signal_ndim=2, onesided=False)

        # crop center before returning
        fcy = fcy[..., hks:-hks, hks:-hks]
        return fcy[..., x_pad_y:-x_pad_y-py, x_pad_x:-x_pad_x-px]


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
