import torch
from torch import nn

from einops import rearrange,repeat
from einops.layers.torch import Rearrange

def pair(t):
    # 把单个输入成对输出
    return t if isinstance(t,tuple) else (t,t)

class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)

class FeedForward(nn.Module):
    # MLP
    def __init__(self,dim,hidden_dim,dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        # 当只有一个头且 这个头的输出维度与输入维度(token维度)一致时，project_out为False
        # 正常情况下project_out 为True
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        # 计算自注意力时的标准化参数
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 将一个输入转换为3个，而为了并行化，inner包含了head数量
        self.to_qkv = nn.Linear(dim,inner_dim*3,bias=False)

        # 只有一个头时，直接输出
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        # 包含三个值
        qkv = self.to_qkv(x).chunk(3,dim=-1)
        # qkv是可迭代对象，对于其中的每一个值，都做维度处理
        # 原始维度为(b,n,h*d),根据设置的h(head)维度，分离最后一维，并将head放到第2维
        q,k,v = map(lambda t:rearrange(t,'b n (h d) -> b h n d',h=self.heads),qkv)

        dots = torch.matmul(q,k.transpose(-1,-2)) * self.scale
        # 计算出注意力后，需要用softmax转换为权重，并进行dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,dim_head,mlp_dim,dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Attention(dim,heads,dim_head,dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))
    def forward(self,x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    # 命名关键词参数 在调用时必须输入参数名
    def __init__(self,*,image_size,patch_size,num_classes,dim,depth,heads,
                 mlp_dim,pool='cls',channels=3,dim_head=64,dropout=0.,
                 emb_dropout=0.):
        super(ViT, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_width % patch_width == 0 and image_height % patch_height == 0, 'Image dimensions must be divisible by the patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_width * patch_height
        assert pool in {'cls','mean'}, 'pool type must be either cls(cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=patch_height,p2=patch_width),
            nn.Linear(patch_dim,dim)
        )
        # 维度为1,num_patches+1,dim
        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim,depth,heads,dim_head,mlp_dim,dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_classes)
        )

    def forward(self,img):
        # 把图片转换为b (h w) (p1 p2 c)，然后 linear 变为b (h w) dim
        x = self.to_patch_embedding(img)
        b,n,_ = x.shape
        # 将单个cls_token复制b份，以应用于batch内的所有图像
        cls_token = repeat(self.cls_token,'1 n d -> b n d',b=b)
        # x维度变为b (h w)+1 dim
        x = torch.cat((cls_token,x),dim=1)
        # 利用广播机制加位置编码
        x += self.pos_embedding[:,:(n+1)]
        x = self.dropout(x)

        # transformer后的维度为 b (h w)+1 dim
        x = self.transformer(x)

        # 取cls_token 进行分类
        x = x.mean(dim=1) if self.pool == 'mean' else x[:,0]

        x = self.to_latent(x)

        return self.mlp_head(x)

if __name__ == '__main__':
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = v(img)  # (1, 1000)
    print(preds.shape)


