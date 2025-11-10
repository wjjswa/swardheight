""" Operations """
import torch
import torch.nn as nn
import genotypes as gt

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: PoolBN('avg', C, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda C, stride, affine: PoolBN('max', C, 3, stride, 1, affine=affine),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda C, stride, affine: FacConv(C, C, 7, stride, 3, affine=affine)
}

tf_OPS = {
    'multiheadAttention_1_1':lambda C, stride, affine: TransformerEncoder(C, 1,1),
    'multiheadAttention_2_2':lambda C, stride, affine: TransformerEncoder(C, 2,2),
    # 'multiheadAttention_3_3':lambda C, stride, affine: TransformerEncoder(C, 2,2),
    'multiheadAttention_4_4':lambda C, stride, affine: TransformerEncoder(C, 4,4),
    # 'multiheadAttention_5_5':lambda C, stride, affine: TransformerEncoder(C, 4,4),
    # 'multiheadAttention_6_6':lambda C, stride, affine: TransformerEncoder(C, 4,4),
    'skip_connect': lambda C, stride, affine: \
        Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'none': lambda C, stride, affine: Zero(stride)

}

class TransformerEncoder(nn.Module):
    """Transformer encoder consisting of multiple transformer layers."""
    def __init__(self, hidden_size, num_layers, num_attention_heads, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            self.TransformerLayer(hidden_size, num_attention_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)

    class TransformerLayer(nn.Module):
        """A single transformer layer."""
        def __init__(self, hidden_size, num_attention_heads, dropout_rate):
            super().__init__()
            self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=dropout_rate)
            self.norm1 = nn.LayerNorm(hidden_size)
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            attn_output, _ = self.attention(x, x, x)
            x = x + self.dropout(attn_output)
            x = self.norm1(x)
            return x

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask; assuming x in cuda.
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)
        return x


class PoolBN(nn.Module):
    def __init__(self, pool_type, C, kernel_size, stride, padding, affine=True):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding,
                                     count_include_pad=False)  # count_include_pad=False表示不把填充的0计算进去
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(C, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(C_in, C_out, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size, stride, padding, dilation=dilation, groups=C_in,
                      bias=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(C_in, C_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(C_in, C_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.
        return x[:, :, ::self.stride, ::self.stride] * 0.


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))

class MixedOpTF(nn.Module):
    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in gt.tf_PRIMITIVES:
            op = tf_OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
