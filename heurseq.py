import pywt
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from .model import Model
import torch.nn.functional as F
from pathlib import Path
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from models.WaveModel import Wave2D



class CLIPViewEmbedder:
    def __init__(self, model_path, device='cuda'):
        """
        model_path : 本地 BLIP 模型路径（已转成 safetensors）
        num_views  : 每条文本生成的 view 数
        """
        self.device = device


        # 加载 processor 和模型
        self.processor =CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(
            model_path,
            use_safetensors=True,
            local_files_only=True
        ).to(device)
        self.model.eval()




    def encode(self, texts_describe):
        """
        texts : List[str] 文本列表
        return: torch.Tensor [B, V, 768] view embeddings
        """
        # 文本处理
        texts = []
        for patent_views in texts_describe:
            # patent_views 是 tuple
            texts.extend([str(v) for v in patent_views])

        with torch.no_grad():
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            ).to(self.device)

            # 得到文本特征 [B, 768]

            text_feat = self.model.get_text_features(**inputs)  # [B, 768]


        return text_feat

class CrossModalBridge(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm_img = nn.LayerNorm(dim)
        self.norm_txt = nn.LayerNorm(dim)

        self.cross_attn_img = Attention(dim, num_heads=num_heads)
        self.cross_attn_txt = Attention(dim, num_heads=num_heads)

    def forward(self, img, txt):
        """
        img: [B, V, D]
        txt: [B, V, D]
        """

        # img ← txt
        img = img + self.cross_attn_img(self.norm_img(img + txt))

        # txt ← img
        txt = txt + self.cross_attn_txt(self.norm_txt(txt + img))

        return img, txt


def read_names_to_list(txt_path):
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"文件不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]  # 去除空行和换行符
    #
    # print(f"✅ 共读取 {len(names)} 个名称")
    return names

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def clip_loss(img_feat, txt_feat, temperature=0.07):
    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)

    logits = img_feat @ txt_feat.t() / temperature
    labels = torch.arange(len(logits)).to(logits.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)

    return (loss_i + loss_t) / 2

def cs_divergence(x, y, sigma=0.5):
    """
    x: [N, D]
    y: [M, D]
    """
    # 归一化（很重要，类似CLIP）
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    # pairwise distance
    xx = torch.cdist(x, x, p=2) ** 2   # [N, N]
    yy = torch.cdist(y, y, p=2) ** 2   # [M, M]
    xy = torch.cdist(x, y, p=2) ** 2   # [N, M]

    # Gaussian kernel
    k_xx = torch.exp(-xx / (2 * sigma ** 2))
    k_yy = torch.exp(-yy / (2 * sigma ** 2))
    k_xy = torch.exp(-xy / (2 * sigma ** 2))

    # 避免数值问题
    eps = 1e-8

    term_xx = torch.log(k_xx.mean() + eps)
    term_yy = torch.log(k_yy.mean() + eps)
    term_xy = torch.log(k_xy.mean() + eps)

    # CS divergence
    d_cs = term_xx + term_yy - 2 * term_xy

    return d_cs


class AlternatingAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads,mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm_intra = nn.LayerNorm(dim)
        self.attn_intra = Attention(dim, num_heads=num_heads,qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_inter = nn.LayerNorm(dim)
        self.attn_inter = Attention(dim, num_heads=num_heads,qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)



        self.norm_mlp = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        x: shape [B, N+1, D], where N=num_views
        """

        x = x + self.drop_path(self.attn_intra(self.norm_intra(x)))

        # Inter-attention across views
        cls_token = x[:, :1]  # [B, 1, D]
        views_intra = x[:, 1:]  # [B, N, D]
        x_cls = cls_token + self.drop_path(self.attn_inter(self.norm_inter(torch.cat([cls_token, views_intra], dim=1)))[:, :1])

        views_inter = []
        for i in range(views_intra.shape[1]):
            query = views_intra[:, i:i + 1]
            others = torch.cat([views_intra[:, :i], views_intra[:, i + 1:]], dim=1)
            inter = self.attn_inter(self.norm_inter(query + others.mean(dim=1, keepdim=True)))
            views_inter.append(inter)
        views_inter = self.drop_path(torch.cat(views_inter, dim=1)) + views_intra

        # views_inter=views_intra
        # 拼接 cls_token 和更新后的视角
        x = torch.cat([x_cls, views_inter], dim=1)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x




class TransposeLayer(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class HSL-Pat_S(nn.Module):
    """
    HSL-Pat_S（Swin Transformer 版本）
    输出:
        logits: [B*V, nclasses]
        fi:     [B*V, C, h, w]  —— Swin 最后 stage 的特征图
    """
    def __init__(self, name, nclasses=527, pretraining=True, cnn_name='swin_t'):
        super().__init__()

        self.nclasses = nclasses
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mean = Variable(torch.FloatTensor([0.0142, 0.0142, 0.0142]), requires_grad=False).to(device)
        self.std  = Variable(torch.FloatTensor([0.0818, 0.0818, 0.0818]), requires_grad=False).to(device)

        # ---------------------- Backbone ----------------------
        if cnn_name == "swin_t":
            self.net = torchvision.models.swin_t(
                weights="IMAGENET1K_V1" if pretraining else None
            )
            embed_dim = 768

        elif cnn_name == "swin_s":
            self.net = torchvision.models.swin_s(
                weights="IMAGENET1K_V1" if pretraining else None
            )
            embed_dim = 768

        elif cnn_name == "swin_b":
            self.net = torchvision.models.swin_b(
                weights="IMAGENET1K_V1" if pretraining else None
            )
            embed_dim = 1024

        else:
            raise ValueError("backbone must be one of: swin_t, swin_s, swin_b")

        # ---------------------- Remove original head ----------------------
        self.net.head = nn.Identity()
        self.fc = nn.Linear(embed_dim, nclasses)
        self.embed_dim = embed_dim



    # ---------------- Forward ----------------
    def forward(self, x):
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]

        feat = self.net(x)         # [B, embed_dim]
        logits = self.fc(feat)


        return logits, feat


class HSL-Pat_M(nn.Module):
    def __init__(self, name, model, pool_mode='PT', nclasses=527, cnn_name='densenet121', num_views=5):
        super().__init__()

        self.num_views = num_views
        self.nclasses = nclasses
        self.pool_mode = pool_mode

        # 保存 HSL-Pat_S
        self.net_1 = model

        # HSL-Pat_S 输出通道
        self.embed_dim = model.embed_dim  # ConvNeXt-Tiny 768

        # 投影 Conv（如果需要的话）
        self.proj = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1)

        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Multi-head 数量
        D = self.embed_dim
        if D >= 768:
            num_heads = 12
        elif D >= 512:
            num_heads = 8
        else:
            num_heads = 4

        norm_layer = nn.LayerNorm
        act_layer = nn.GELU
        
        self.clip_encoder=CLIPViewEmbedder(model_path="*/models/clip")
        

        self.stage1_img = nn.Sequential(
            Block(dim=D, num_heads=num_heads)
        )

        self.stage1_txt = nn.Sequential(
            Block(dim=D, num_heads=num_heads)
        )
        self.bridge = CrossModalBridge(dim=D, num_heads=num_heads)

        self.stage3 = nn.Sequential(
            AlternatingAttentionBlock(dim=D, num_heads=num_heads)
        )

        self.img_proj = nn.Linear(768, D)
        self.txt_proj = nn.Linear(768, D)

        self.pos_drop = nn.Dropout(0.)
        self.norm = nn.LayerNorm(D)
        self.pre_logits = nn.Identity()

        # 分类头
        self.head = nn.Linear(D, nclasses)


    def forward(self, x,view_describe):
        """
        x: [B*V, 3, H, W]
        """
        B = x.shape[0] // self.num_views
        # 1) HSL-Pat_S 提取 feature map
        _, img_fi = self.net_1(x)           # [B*V, D]



        clip_fi=self.clip_encoder.encode(view_describe)


        img_feat = self.img_proj(img_fi)
        clip_feat = self.txt_proj(clip_fi)

        img_feat = img_feat.view(B, self.num_views, self.embed_dim)
        clip_feat = clip_feat.view(B, self.num_views, self.embed_dim)


        # ====== ⭐ CS Distribution Alignment ======
        img_flat = img_feat.view(B * self.num_views, self.embed_dim)
        txt_flat = clip_feat.view(B * self.num_views, self.embed_dim)
        L_cs = cs_divergence(img_flat, txt_flat)

        # ====== Global CLIP alignment ======
        img_global = img_feat.mean(dim=1)  # [B, D]
        txt_global = clip_feat.mean(dim=1)  # [B, D]

        L_clip = clip_loss(img_global, txt_global)


        # feat=img_feat+clip_feat
        # ================= Stage 1 =================
        img_tokens = self.stage1_img(img_feat)
        txt_tokens = self.stage1_txt(clip_feat)

        # ================= Stage 2 (Bridge) =================
        img_tokens, txt_tokens = self.bridge(img_tokens, txt_tokens)

        
        feat = img_tokens + txt_tokens

        # ================= Stage 3 =================
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, feat], dim=1)

        x = self.stage3(x)

        
        x = self.norm(x)

        # 6) pooling
        if self.pool_mode == "PT":
            cls = x[:, 0]
            vp = x[:, 1:].max(1)[0]
            out = cls + vp
        else:
            out = x[:, 0]

        out=self.head(out)

        return out,L_cs,L_clip


