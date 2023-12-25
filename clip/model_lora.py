from collections import OrderedDict
from typing import Tuple, Union, Optional
from torch import Tensor

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import loralib as lora
import math
from .model import Transformer, ModifiedResNet, VisionTransformer


class Lora_Linear(lora.Linear):
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0,
                 fan_in_fan_out: bool = False, merge_weights: bool = True, **kwargs):
        super().__init__(in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights,
                         **kwargs)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode == True:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True


class Lora_embeding(lora.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, r: int = 0, lora_alpha: int = 1,
                 merge_weights: bool = True, **kwargs):
        super().__init__(num_embeddings, embedding_dim, r, lora_alpha, merge_weights, **kwargs)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode == True:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).T * self.scaling
                self.merged = True


class Lora_conv2d(lora.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, r: int = 0, lora_alpha: int = 1,
                 lora_dropout: float = 0, merge_weights: bool = True, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, r, lora_alpha, lora_dropout, merge_weights, **kwargs)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if mode == True:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MultiheadAttention_Lora(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, r=1, a=0.):
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.r = r
        self.merged = False
        self.merge_weights = True
        self.lora_alpha = a

        self.lora_Q_A = nn.Parameter(self.in_proj_weight.new_zeros(r, embed_dim))
        self.lora_Q_B = nn.Parameter(self.in_proj_weight.new_zeros(embed_dim, r))
        self.lora_V_A = nn.Parameter(self.in_proj_weight.new_zeros(r, embed_dim))
        self.lora_V_B = nn.Parameter(self.in_proj_weight.new_zeros(embed_dim, r))
        self.lora_out_proj_A = nn.Parameter(self.out_proj.weight.new_zeros(r, embed_dim))
        self.lora_out_proj_B = nn.Parameter(self.out_proj.weight.new_zeros(embed_dim, r))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(self.lora_Q_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_Q_B)
        nn.init.kaiming_uniform_(self.lora_V_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_V_B)
        nn.init.kaiming_uniform_(self.lora_out_proj_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_out_proj_B)

    def train(self, mode: bool = True):
        nn.MultiheadAttention.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    deltQ = self.lora_Q_B @ self.lora_Q_A
                    deltK = torch.zeros_like(deltQ)
                    deltV = self.lora_V_B @ self.lora_V_A
                    self.in_proj_weight.data -= torch.cat((deltQ, deltK, deltV), dim=0) * self.scaling
                    self.out_proj.weight.data -= (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
                    del deltQ
                    del deltK
                    del deltV
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    deltQ = self.lora_Q_B @ self.lora_Q_A
                    deltK = torch.zeros_like(deltQ)
                    deltV = self.lora_V_B @ self.lora_V_A
                    self.in_proj_weight.data += torch.cat((deltQ, deltK, deltV), dim=0) * self.scaling
                    self.out_proj.weight.data += (self.lora_out_proj_B @ self.lora_out_proj_A) * self.scaling
                    del deltQ
                    del deltK
                    del deltV
                self.merged = True

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None):
        # Merge = False
        if self.r > 0 and not self.merged:
            result = nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask=key_padding_mask,
                                                   need_weights=need_weights, attn_mask=attn_mask)
            if self.r > 0:
                deltQ = self.lora_Q_B @ self.lora_Q_A
                deltK = self.in_proj_weight[self.embed_dim:self.embed_dim * 2]
                deltV = self.lora_V_B @ self.lora_V_A
                lora_in_proj_weight = torch.cat((deltQ, deltK, deltV), dim=0)
                lora_out_proj_weight = self.lora_out_proj_B @ self.lora_out_proj_A

                result = result[0] + F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    lora_in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, lora_out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)[0] * self.scaling, result[1]
            return result
        # Merge =True
        else:
            return nn.MultiheadAttention.forward(self, query, key, value, key_padding_mask=key_padding_mask,
                                                 need_weights=need_weights, attn_mask=attn_mask)


class ResidualAttentionBlock_Lora(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, r=1, a=0.):
        # def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, r=1, a=1):
        super().__init__()

        self.attn = MultiheadAttention_Lora(d_model, n_head, r=r, a=a)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", Lora_Linear(d_model, d_model * 4, r=r, lora_alpha=a)),
            ("gelu", QuickGELU()),
            ("c_proj", Lora_Linear(d_model * 4, d_model, r=r, lora_alpha=a))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_Lora_mlp(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, r=1, a=0.):
        # def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, r=1, a=1):
        super().__init__()

        # self.attn = MultiheadAttention_Lora(d_model, n_head, r=r, a=a)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", Lora_Linear(d_model, d_model * 4, r=r, lora_alpha=a)),
            ("gelu", QuickGELU()),
            ("c_proj", Lora_Linear(d_model * 4, d_model, r=r, lora_alpha=a))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer_Lora_mlp(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, r: int = 1, a: float = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_Lora_mlp(width, heads, attn_mask, r=r, a=a) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Transformer_Lora(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, r: int = 1, a: float = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_Lora(width, heads, attn_mask, r=r, a=a) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer_Lora_mlp(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 r: int = 1, a: float = 0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = Lora_conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size,
                                 bias=False, r=r, lora_alpha=a)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_Lora_mlp(width, layers, heads, r=r, a=a)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        cls = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            cls = cls @ self.proj

        return x, cls

class VisionTransformer_Lora(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 r: int = 1, a: float = 0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = Lora_conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size,
                                 bias=False, r=r, lora_alpha=a)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_Lora(width, layers, heads, r=r, a=a)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        cls = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj
            cls = cls @ self.proj

        return x, cls


class CLIP_Lora(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 r: int = 1,
                 # a: float = 0.
                 a=1,
                 lora_v=True,
                 lora_t=True,
                 lora_attn=False
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if lora_v:
                if lora_attn:
                    self.visual = VisionTransformer_Lora(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        r=r,
                        a=a,
                    )
                else:
                    self.visual = VisionTransformer_Lora_mlp(
                        input_resolution=image_resolution,
                        patch_size=vision_patch_size,
                        width=vision_width,
                        layers=vision_layers,
                        heads=vision_heads,
                        output_dim=embed_dim,
                        r=r,
                        a=a,
                    )
            else:
                self.visual = VisionTransformer(
                    input_resolution=image_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim
                )

        if lora_t:
            if lora_attn:
                self.transformer = Transformer_Lora(
                    width=transformer_width,
                    layers=transformer_layers,
                    heads=transformer_heads,
                    attn_mask=self.build_attention_mask(),
                    r=r,
                    a=a
                )
            else:
                self.transformer = Transformer_Lora_mlp(
                    width=transformer_width,
                    layers=transformer_layers,
                    heads=transformer_heads,
                    attn_mask=self.build_attention_mask(),
                    r=r,
                    a=a
                )
        else:
            self.transformer = Transformer(
                width=transformer_width,
                layers=transformer_layers,
                heads=transformer_heads,
                attn_mask=self.build_attention_mask()
            )

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.token_embedding = Lora_embeding(vocab_size, transformer_width, r=r, lora_alpha=a)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        # return self.visual.conv1.weight.dtype
        return self.visual.proj.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, r=1, a=0, lora_v=True, lora_t=True, lora_attn=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP_Lora(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, r, a,
        lora_v, lora_t, lora_attn
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    lora.mark_only_lora_as_trainable(model)
    # return model.eval()
    return model