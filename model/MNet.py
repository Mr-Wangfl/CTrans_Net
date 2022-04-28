import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class MNet(nn.Module):

    def __init__(self, filters, layers=4, kernel=3, inplanes=3, num_classes=2, active_func=None):
        super().__init__()
        self.filters = filters
        self.layers = layers
        self.kernel = kernel
        self.inplanes = inplanes
        self.active_func = active_func

        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.down_conv_list = nn.ModuleList()
        self.down_conv_2_list = nn.ModuleList()
        self.down_scale_conv_list = nn.ModuleList()
        for i in range(layers):
            self.down_conv_list.append(nn.Sequential(
                conv3x3(inplanes if i == 0 else filters * 2 ** i, filters * 2 ** i),
                nn.BatchNorm2d(filters * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                conv3x3(filters * 2 ** i, filters * 2 ** i),
                nn.BatchNorm2d(filters * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            self.down_conv_2_list.append(nn.Sequential(
                conv3x3(filters * 2 ** i, filters * 2 ** (i + 1), 2),
                nn.BatchNorm2d(filters * 2 ** (i + 1))
            ))
            if i != layers - 1:
                self.down_scale_conv_list.append(nn.Sequential(
                    conv3x3(inplanes, filters * 2 ** (i + 1)),
                    nn.BatchNorm2d(filters * 2 ** (i + 1))
                ))

        self.bottom = nn.Sequential(
            conv3x3(filters * 2 ** layers, filters * 2 ** layers),
            nn.BatchNorm2d(filters * 2 ** layers),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv3x3(filters * 2 ** layers, filters * 2 ** layers),
            nn.BatchNorm2d(filters * 2 ** layers),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.up_convtrans_list = nn.ModuleList()
        self.up_conv_list = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        for i in range(0, layers):
            self.up_convtrans_list.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=filters * 2 ** (layers - i), out_channels=filters * 2 ** max(0, layers - i - 1),
                                   kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1))
            ))
            self.up_conv_list.append(nn.Sequential(
                conv3x3(filters * 2 ** max(0, layers - i - 1), filters * 2 ** max(0, layers - i - 1)),
                nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1)),
                nn.ReLU(inplace=True),
                conv3x3(filters * 2 ** max(0, layers - i - 1), filters * 2 ** max(0, layers - i - 1)),
                nn.BatchNorm2d(filters * 2 ** max(0, layers - i - 1)),
                nn.ReLU(inplace=True),
            ))

            self.out_conv.append(conv3x3(filters * 2 ** (self.layers - i - 1), num_classes))

    def forward(self, x):
        x_down = down_out = x
        down_outs = []
        for i in range(0, self.layers):
            down_out = self.down_conv_list[i](down_out)
            down_outs.append(down_out)
            down_out = self.down_conv_2_list[i](down_out)
            if i != self.layers - 1:
                x_down = F.avg_pool2d(x_down, 2)
                down_out = self.leakyRelu(down_out + self.down_scale_conv_list[i](x_down))
            else:
                down_out = self.leakyRelu(down_out)

        # bottom branch
        up_out = self.bottom(down_out)

        outs = []
        out_final = None
        for j in range(self.layers):
            up_out = self.relu(down_outs[self.layers - j - 1] + self.up_convtrans_list[j](up_out))
            up_out = self.up_conv_list[j](up_out)

            out = F.interpolate(up_out, size=(up_out.shape[2] * 2 ** (self.layers - j - 1), up_out.shape[3] * 2 ** (self.layers - j - 1)))
            out = self.out_conv[j](out)
            if self.active_func is not None:
                out = self.active_func(out)
            outs.append(out)
            out_final = out if out_final is None else out_final + out

        out_final /= self.layers
        outs.append(out_final)
        return out_final



