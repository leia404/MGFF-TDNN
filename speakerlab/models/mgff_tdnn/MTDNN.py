from collections import OrderedDict

from speakerlab.models.mgff_tdnn.layers import *

class MTDNN(nn.Module):
    # A TDNN module with mutil-granularity feature fusion
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 se_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=2,
                 bias=False,
                 nonlinear_str='batchnorm-relu'):
        super(MTDNN, self).__init__()
        self.linear1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias)
        self.nonlinear1 = get_nonlinear(nonlinear_str, hidden_channels)
        self.tdnn = TDNNBlock(hidden_channels,
                              hidden_channels,
                              stride=stride,
                              kernel_size=kernel_size,
                              dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

        self.se_block = SEBlock(hidden_channels * 2, se_channels, hidden_channels * 2)
        self.linear2 = nn.Conv1d(hidden_channels * 2, out_channels, kernel_size=1, stride=1, bias=bias)
        self.nonlinear2 = get_nonlinear(nonlinear_str, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def phonemic_level_pooling(self, x, window=8):
        stride = math.ceil(window / 2)
        seg = F.avg_pool1d(x, kernel_size=window, stride=stride, ceil_mode=True)  # [1, 128, 74]
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, window).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)    # [1, 512, 298]

        x = self.nonlinear1(self.linear1(x))    # [1, 128, 298]
        global_context = self.tdnn(x)    # [1, 128, 298]
        local_context = self.phonemic_level_pooling(x, window=8)    # [1, 128, 298]
        fuse = torch.cat((global_context, local_context), dim=1)    # [1, 256, 298]
        x = self.se_block(fuse, lengths)   # [1, 256, 298]
        x = self.nonlinear2(self.linear2(x))   # [1, 512, 298]

        out = x + residual
        out = self.relu(out)

        return out


class MTDNN_WO_PLP(nn.Module):
    # A TDNN module with mutil-granularity feature fusion
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 se_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=2,
                 bias=False,
                 nonlinear_str='batchnorm-relu'):
        super(MTDNN_WO_PLP, self).__init__()
        self.linear1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1, bias=bias)
        self.nonlinear1 = get_nonlinear(nonlinear_str, hidden_channels)

        self.relu = nn.ReLU(inplace=True)

        # self.se_block = SEBlock(hidden_channels * 2, se_channels, hidden_channels * 2)
        self.se_block = SEBlock(hidden_channels, se_channels, hidden_channels)
        # self.linear2 = nn.Conv1d(hidden_channels * 2, out_channels, kernel_size=1, stride=1, bias=bias)
        self.linear2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        self.nonlinear2 = get_nonlinear(nonlinear_str, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def phonemic_level_pooling(self, x, window=8):
        stride = math.ceil(window / 2)
        seg = F.max_pool1d(x, kernel_size=window, stride=stride, ceil_mode=True)    # [1, 128, 74]
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, window).reshape(*shape[:-1], -1)
        seg = seg[..., :x.shape[-1]]
        return seg

    def forward(self, x, lengths=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)    # [1, 512, 298]

        x = self.nonlinear1(self.linear1(x))    # [1, 128, 298]
        # global_context = self.tdnn(x)    # [1, 128, 298]
        local_context = self.phonemic_level_pooling(x, window=8)    # [1, 128, 298]
        # dummy = global_context
        # fuse = torch.cat((global_context, dummy), dim=1)    # [1, 256, 298]
        # x = self.se_block(global_context, lengths)   # [1, 256, 298]
        x = self.se_block(local_context, lengths)   # [1, 256, 298]
        # x = self.se_block(global_context, lengths)   # [1, 256, 298]
        x = self.nonlinear2(self.linear2(x))   # [1, 512, 298]

        out = x + residual
        out = self.relu(out)

        return out


class MTDNNBlock(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 se_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 nonlinear_str='batchnorm-relu',
                 ):
        super(MTDNNBlock, self).__init__()
        self.in_channels = in_channels
        for i in range(num_layers):
            layer = MTDNN(in_channels=self.in_channels,
                          hidden_channels=hidden_channels,
                          se_channels=se_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          bias=bias,
                          nonlinear_str=nonlinear_str
                          )
            self.add_module("mtdnn%d" % (i + 1), layer)
            self.in_channels = out_channels

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class MTDNNBlock_WO_PLP(nn.ModuleList):
    def __init__(self,
                 num_layers,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 se_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=False,
                 nonlinear_str='batchnorm-relu',
                 ):
        super(MTDNNBlock_WO_PLP, self).__init__()
        self.in_channels = in_channels
        for i in range(num_layers):
            layer = MTDNN_WO_PLP(in_channels=self.in_channels,
                          hidden_channels=hidden_channels,
                          se_channels=se_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          dilation=dilation,
                          bias=bias,
                          nonlinear_str=nonlinear_str
                          )
            self.add_module("mtdnn%d" % (i + 1), layer)
            self.in_channels = out_channels

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x



class SeparableResNet(nn.Module):
    # the implement of depth-wise convolution and point-wise convolution
    def __init__(self, in_planes, planes, times=6, stride=1):
        super(SeparableResNet, self).__init__()
        self.pw1 = nn.Conv2d(in_planes, in_planes * times, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes * times)
        self.dw = nn.Conv2d(in_planes * times,
                            in_planes * times,
                            kernel_size=3,
                            stride=(stride, 1),
                            padding=1,
                            groups=in_planes * times,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes * times)
        self.pw2 = nn.Conv2d(in_planes * times, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if in_planes != planes or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = F.relu(self.bn1(self.pw1(x)))
        x = F.relu(self.bn2(self.dw(x)))
        x = self.bn3(self.pw2(x))
        x += residual
        x = F.relu(x)

        return x


class DSM(nn.Module):
    # depth-wise separable convolution module
    def __init__(self,
                 feat_dim=80,
                 init_channels=32):
        super(DSM, self).__init__()
        self.conv = nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(init_channels)
        self.layer1 = SeparableResNet(init_channels, init_channels, stride=2)
        self.layer2 = SeparableResNet(init_channels, init_channels, stride=2)
        self.layer3 = SeparableResNet(init_channels, init_channels, stride=2)
        self.out_channels = init_channels * (feat_dim // 8)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn(self.conv(x)))     # [1, 32, 80, 298]
        out = self.layer1(out)    # [1, 32, 40, 298]
        out = self.layer2(out)    # [1, 32, 20, 298]
        out = self.layer3(out)    # [1, 32, 10, 298]

        shape = out.shape
        out = out.reshape(shape[0], shape[1]*shape[2], shape[3])   # [1, 320, 298]

        return out


class MGFF_TDNN(nn.Module):
    '''
        multi-granularity feature fusion TNDD model with depth-wise separable module for speaker verification
    '''
    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 channels=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3],
                 se_channels=128,
                 nonlinear_str='batchnorm-relu'
                 ):
        super(MGFF_TDNN, self).__init__()
        self.dsm = DSM(feat_dim=feat_dim)
        out_channels = self.dsm.out_channels


        self.mtdnn = nn.Sequential(OrderedDict([]))
        self.in_channels = out_channels
        for i, (num_layer, stride,
                dilation) in enumerate(zip((3, 6, 4), (1, 1, 1), (1, 2, 2))):
            block = MTDNNBlock(num_layers=num_layer,
                               in_channels=self.in_channels,
                               hidden_channels=channels[i],
                               out_channels=channels[i+1],
                               se_channels=se_channels,
                               kernel_size=kernel_size[i],
                               stride=stride,
                               dilation=dilation,
                               nonlinear_str=nonlinear_str)
            self.in_channels = channels[i+1]
            self.mtdnn.add_module('block%d' % (i + 1), block)

        self.mtdnn.add_module('StatsPool', StatsPool())

        self.mtdnn.add_module('out_emb_layer', DenseLayer(channels[-1] * 2, embedding_size))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.dsm(x)
        x = self.mtdnn(x)

        return x



class MGFF_TDNN_without_DSM(nn.Module):
    '''
        Ablation experiments
    '''
    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 channels=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3],
                 se_channels=128,
                 nonlinear_str='batchnorm-relu'
                 ):
        super(MGFF_TDNN_without_DSM, self).__init__()
        # here because the output channel of original DSM module is 320
        out_channels = 320
        self.tdnn = TDNNBlock(feat_dim,
                           out_channels,
                           3,
                           stride=1,
                           dilation=1)

        self.mtdnn = nn.Sequential(OrderedDict([]))
        self.in_channels = out_channels
        for i, (num_layer, stride,
                dilation) in enumerate(zip((3, 6, 4), (1, 1, 1), (1, 2, 2))):
            block = MTDNNBlock(num_layers=num_layer,
                               in_channels=self.in_channels,
                               hidden_channels=channels[i],
                               out_channels=channels[i+1],
                               se_channels=se_channels,
                               kernel_size=kernel_size[i],
                               stride=stride,
                               dilation=dilation,
                               nonlinear_str=nonlinear_str)
            self.in_channels = channels[i+1]
            self.mtdnn.add_module('block%d' % (i + 1), block)

        self.mtdnn.add_module('StatsPool', StatsPool())

        self.mtdnn.add_module('out_emb_layer', DenseLayer(channels[-1] * 2, embedding_size))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.tdnn(x)
        x = self.mtdnn(x)

        return x


class MGFF_TDNN_without_PLP(nn.Module):
    '''
        Ablation experiments
    '''
    def __init__(self,
                 feat_dim=80,
                 embedding_size=192,
                 channels=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3],
                 se_channels=128,
                 nonlinear_str='batchnorm-relu'
                 ):
        super(MGFF_TDNN_without_PLP, self).__init__()

        self.dsm = DSM(feat_dim=feat_dim)
        out_channels = self.dsm.out_channels

        self.mtdnn = nn.Sequential(OrderedDict([]))
        self.in_channels = out_channels
        for i, (num_layer, stride,
                dilation) in enumerate(zip((3, 6, 4), (1, 1, 1), (1, 2, 2))):
            block = MTDNNBlock_WO_PLP(num_layers=num_layer,
                               in_channels=self.in_channels,
                               hidden_channels=channels[i],
                               out_channels=channels[i+1],
                               se_channels=se_channels,
                               kernel_size=kernel_size[i],
                               stride=stride,
                               dilation=dilation,
                               nonlinear_str=nonlinear_str)
            self.in_channels = channels[i+1]
            self.mtdnn.add_module('block%d' % (i + 1), block)

        self.mtdnn.add_module('StatsPool', StatsPool())

        self.mtdnn.add_module('out_emb_layer', DenseLayer(channels[-1] * 2, embedding_size))

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.dsm(x)
        x = self.mtdnn(x)

        return x

