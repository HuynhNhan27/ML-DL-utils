import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



########################### Transformer ###################################

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: batch, seq, d
    # score = qkT -> batch, seq, seq. kT -> transpose hai chiều cuối
    d_k = k.size(0)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # mask

    if mask is not None:
        # masked_fill các vị trí = 0, fill -inf (-1e9)
        scores = scores.masked_fill(mask == 0, -1e9)

    # attention_weights: softmax theo hàng -> hàng = -1 -> dim = -1
    attention_weights = F.softmax(scores, dim=-1)

    # output: 
    output = torch.matmul(attention_weights, v)

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Cần: số head, d_model để khởi tạo trọng số
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Tạo linear cho qkv: d_model đầu vào giống nhau (do cùng từ X), d_model đầu ra có thể khác nhau, chỉ cần khớp shape
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1 ,self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x = scaled_dot_product_attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super.__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, mask):
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(self.dropout(attn_out) + x)

        ff_out = self.ff(x)
        x = self.norm2(self.dropout(ff_out) + x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.mask_self_attn = MultiHeadAttention(d_model, num_heads)
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encode_output, src_mask, tgt_mask):
        mask_attn_out = self.mask_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(mask_attn_out))

        attn_out = self.self_attn(encode_output, encode_output, x, src_mask)
        x = self.norm2(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x
    
###
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):        
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return x # Output: (batch_size, seq_len, d_model)


########################### ResNet ###################################

class BasicBlock(nn.Module):
    """Residual block cho ResNet-18 và ResNet-34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck block cho ResNet-50, ResNet-101, ResNet-152."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 conv giảm chiều
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 3x3 conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 1x1 conv tăng chiều (expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    ResNet tổng quát.
    Configs:
        ResNet-18:  block=BasicBlock,  num_blocks=[2, 2, 2, 2]
        ResNet-34:  block=BasicBlock,  num_blocks=[3, 4, 6, 3]
        ResNet-50:  block=Bottleneck,  num_blocks=[3, 4, 6, 3]
        ResNet-101: block=Bottleneck,  num_blocks=[3, 4, 23, 3]
        ResNet-152: block=Bottleneck,  num_blocks=[3, 8, 36, 3]
    """
    def __init__(self, block, num_blocks, num_classes=1000, in_channels=3):
        super().__init__()
        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


########################### MobileNet ###################################

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution: depthwise + pointwise."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNetV1(nn.Module):
    """
    MobileNetV1: sử dụng Depthwise Separable Convolution.
    Paper: MobileNets (Howard et al., 2017)
    """
    def __init__(self, num_classes=1000, alpha=1.0, in_channels=3):
        super().__init__()
        def c(channels):
            return max(1, int(channels * alpha))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(c(32),  c(64),  stride=1),
            DepthwiseSeparableConv(c(64),  c(128), stride=2),
            DepthwiseSeparableConv(c(128), c(128), stride=1),
            DepthwiseSeparableConv(c(128), c(256), stride=2),
            DepthwiseSeparableConv(c(256), c(256), stride=1),
            DepthwiseSeparableConv(c(256), c(512), stride=2),
            *[DepthwiseSeparableConv(c(512), c(512), stride=1) for _ in range(5)],
            DepthwiseSeparableConv(c(512),  c(1024), stride=2),
            DepthwiseSeparableConv(c(1024), c(1024), stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(c(1024), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class InvertedResidual(nn.Module):
    """Inverted Residual Block cho MobileNetV2."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True)
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=stride,
                      padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2: Inverted Residuals + Linear Bottlenecks.
    Paper: MobileNetV2 (Sandler et al., 2018)
    Config: (expand_ratio, out_channels, num_blocks, stride)
    """
    _cfg = [
        (1,  16, 1, 1),
        (6,  24, 2, 2),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes=1000, alpha=1.0, in_channels=3):
        super().__init__()
        def c(channels):
            return max(1, int(channels * alpha))

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c(32)),
            nn.ReLU6(inplace=True)
        )

        layers = []
        in_ch = c(32)
        for t, out_ch, n, s in self._cfg:
            out_ch = c(out_ch)
            for i in range(n):
                layers.append(InvertedResidual(in_ch, out_ch, stride=s if i == 0 else 1, expand_ratio=t))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv2d(in_ch, c(1280), kernel_size=1, bias=False),
            nn.BatchNorm2d(c(1280)),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(c(1280), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


########################### DenseNet ###################################

class DenseLayer(nn.Module):
    """Một layer trong DenseBlock: BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3."""
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(nn.Module):
    """Gồm nhiều DenseLayer nối tiếp nhau theo kiểu dense connection."""
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Giảm số channel và spatial size giữa các DenseBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    """
    DenseNet tổng quát.
    Paper: Densely Connected Convolutional Networks (Huang et al., 2017)
    Configs:
        DenseNet-121: num_blocks=[6, 12, 24, 16], growth_rate=32
        DenseNet-169: num_blocks=[6, 12, 32, 32], growth_rate=32
        DenseNet-201: num_blocks=[6, 12, 48, 32], growth_rate=32
        DenseNet-264: num_blocks=[6, 12, 64, 48], growth_rate=32
    """
    def __init__(self, num_blocks, growth_rate=32, num_init_features=64,
                 bn_size=4, theta=0.5, num_classes=1000, in_channels=3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        blocks = []
        in_ch = num_init_features
        for i, n_layers in enumerate(num_blocks):
            blocks.append(DenseBlock(n_layers, in_ch, growth_rate, bn_size))
            in_ch = in_ch + n_layers * growth_rate
            if i != len(num_blocks) - 1:
                out_ch = int(in_ch * theta)
                blocks.append(TransitionLayer(in_ch, out_ch))
                in_ch = out_ch

        blocks.append(nn.BatchNorm2d(in_ch))
        blocks.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def densenet121(num_classes=1000):
    return DenseNet([6, 12, 24, 16], growth_rate=32, num_classes=num_classes)

def densenet169(num_classes=1000):
    return DenseNet([6, 12, 32, 32], growth_rate=32, num_classes=num_classes)

def densenet201(num_classes=1000):
    return DenseNet([6, 12, 48, 32], growth_rate=32, num_classes=num_classes)

def densenet264(num_classes=1000):
    return DenseNet([6, 12, 64, 48], growth_rate=32, num_classes=num_classes)