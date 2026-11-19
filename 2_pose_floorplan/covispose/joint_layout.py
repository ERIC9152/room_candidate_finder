"""LGT-Net-inspired local/global circular attention with two layout heads.

Original implementation of the architectural ideas, not a port of LGT-Net.
Reference: Jiang et al., CVPR 2022, https://arxiv.org/abs/2203.01824.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from .model import ColumnEncoder
from .floor_boundary import boundary_loss


class CircularAttention(nn.Module):
    def __init__(self, dim=128, heads=4, tokens=256, dropout=.1):
        super().__init__()
        if dim % heads:raise ValueError('dim must be divisible by heads')
        self.heads,self.tokens,self.scale=heads,tokens,(dim//heads)**-.5
        self.qkv=nn.Linear(dim,3*dim)
        self.relative_bias=nn.Parameter(torch.zeros(tokens,heads))
        self.projection=nn.Linear(dim,dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,positions):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        # Displacements use modular azimuth: 255 -> 0 is a one-token step.
        relative=(positions[:,None]-positions[None,:]) % self.tokens
        bias=self.relative_bias[relative].permute(2,0,1)
        scores=(q.float() @ k.float().transpose(-2,-1))*self.scale+bias.float()[None]
        weights=self.dropout(scores.softmax(-1)).to(v.dtype)
        result=(weights@v).transpose(1,2).reshape(b,n,d)
        return self.projection(result)


class LayoutAttentionBlock(nn.Module):
    def __init__(self,dim=128,heads=4,tokens=256,window=16,shift=0,global_attention=False,dropout=.1):
        super().__init__()
        if tokens % window:raise ValueError('window must divide token count')
        self.tokens,self.window,self.shift,self.global_attention=tokens,window,shift,global_attention
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)
        self.attention=CircularAttention(dim,heads,tokens,dropout)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*4),nn.GELU(),nn.Dropout(dropout),nn.Linear(dim*4,dim))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        b,n,d=x.shape
        if n!=self.tokens:raise ValueError('Unexpected token count')
        normalized=self.norm1(x)
        positions=torch.arange(n,device=x.device)
        if self.global_attention:
            attended=self.attention(normalized,positions)
        else:
            normalized=torch.roll(normalized,-self.shift,1)
            # Every window shares the same relative displacement pattern, even
            # the shifted window crossing the seam; no seam-blocking mask.
            window_positions=(positions[:self.window]+self.shift)%n
            attended=self.attention(normalized.reshape(b*n//self.window,self.window,d),window_positions)
            attended=torch.roll(attended.reshape(b,n,d),self.shift,1)
        x=x+self.dropout(attended)
        return x+self.dropout(self.mlp(self.norm2(x)))


def circular_upsample(x,width=1024):
    tokens=x.shape[-1]
    if width % tokens:raise ValueError('output width must be a multiple of tokens')
    scale=width//tokens
    return F.interpolate(torch.cat([x[...,-1:],x,x[...,:1]],-1),
                         size=width+2*scale,mode='linear',align_corners=False)[...,scale:-scale]


class JointLayoutNet(nn.Module):
    def __init__(self,dim=128,heads=4,tokens=256,window=16,depth=4,dropout=.1,attention='swg'):
        super().__init__()
        if attention not in ('swg','global'):raise ValueError('Unknown attention mode')
        self.architecture=dict(dim=dim,heads=heads,tokens=tokens,window=window,depth=depth,dropout=dropout,attention=attention)
        self.tokens=tokens
        self.column_encoder=ColumnEncoder('resnet34',False,img_h=512,img_w=1024,n_cols=tokens)
        self.project=nn.Conv1d(self.column_encoder.out_channels,dim,1)
        self.blocks=nn.ModuleList([LayoutAttentionBlock(dim,heads,tokens,window,
            shift=window//2 if i%4==2 else 0,global_attention=attention=='global' or i%2==1,
            dropout=dropout) for i in range(depth)])
        self.norm=nn.LayerNorm(dim)
        self.corner_head=nn.Conv1d(dim,1,1)
        self.boundary_head=nn.Conv1d(dim,1,1)
        nn.init.constant_(self.corner_head.bias,-3)
        nn.init.constant_(self.boundary_head.bias,math.log(1/2))

    def decode_features(self,features):
        x=F.relu(self.project(features)).transpose(1,2)
        for block in self.blocks:x=block(x)
        x=self.norm(x).transpose(1,2)
        logits=circular_upsample(self.corner_head(x))[:,0]
        bon=circular_upsample(self.boundary_head(x))[:,0].sigmoid()*(math.pi/2)
        return dict(corner_logits=logits,bon=bon)

    def forward(self,image):
        return self.decode_features(self.column_encoder(image,self.tokens))


def load_joint_records(split,root='outputs'):
    from pathlib import Path
    root=Path(root)
    corners=torch.load(root/'corners_rtx1070_v1'/f'{split}_records.pt')
    by_uid={r['uid']:r for r in corners}
    floors=torch.load(root/'floor_boundary_rtx1070_v1'/f'{split}_records.pt')
    records=[]
    for floor in floors:
        corner=by_uid[floor['uid']]
        if floor['image']!=corner['image']:raise ValueError('Image mismatch for '+floor['uid'])
        records.append(dict(uid=floor['uid'],image=floor['image'],bon=floor['target'],
            boundary_valid=floor['valid'],corner_target=corner['target'],corner_valid=corner['valid'],corners=corner['corners']))
    return records


class JointLayoutDataset(torch.utils.data.Dataset):
    def __init__(self,records,augment=False):self.records,self.augment=records,augment
    def __len__(self):return len(self.records)
    def __getitem__(self,i):
        r=self.records[i]
        with Image.open(r['image']) as im:
            im.draft('RGB',(1024,512));image=np.asarray(im.convert('RGB').resize((1024,512),Image.BILINEAR))
        labels={k:np.asarray(r[k],np.float32).copy() for k in ('bon','boundary_valid','corner_target','corner_valid')}
        if self.augment:
            if np.random.rand()<.5:
                image=image[:,::-1];labels={k:v[::-1] for k,v in labels.items()}
            roll=np.random.randint(1024);image=np.roll(image,roll,1);labels={k:np.roll(v,roll) for k,v in labels.items()}
            image=(np.power(image.astype(np.float32)/255,np.random.uniform(.8,1.2))*255).clip(0,255)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255,
                {k:torch.from_numpy(v.copy()) for k,v in labels.items()},i)


def planar_geometry_loss(pred,target,valid):
    """Supervise wall edge normals and their changes, not generic smoothing."""
    width=pred.shape[-1]
    theta=((torch.arange(width,device=pred.device,dtype=torch.float32)+.5)/width-.5)*2*math.pi
    dirs=torch.stack([-theta.sin(),theta.cos()],-1)
    def normals(bon):
        radius=(1/bon.float().clamp(math.radians(1),math.radians(89)).tan()).clamp(.1,20)
        points=radius[...,None]*dirs
        edge=torch.roll(points,-1,1)-points
        return F.normalize(torch.stack([-edge[...,1],edge[...,0]],-1),dim=-1,eps=1e-6)
    p,t=normals(pred),normals(target)
    pair=valid*torch.roll(valid,-1,1)
    normal=((1-(p*t).sum(-1))*pair).sum()/pair.sum().clamp_min(1)
    triple=pair*torch.roll(valid,-2,1)
    pg,tg=torch.roll(p,-1,1)-p,torch.roll(t,-1,1)-t
    gradient=((pg-tg).abs().mean(-1)*triple).sum()/triple.sum().clamp_min(1)
    return normal,gradient


def joint_loss(pred,labels,corner_weight=1.,boundary_weight=1.,geometry_weight=.05,positive_weight=15.):
    per=F.binary_cross_entropy_with_logits(pred['corner_logits'].float(),labels['corner_target'],
        reduction='none',pos_weight=pred['bon'].new_tensor(positive_weight))
    mask=labels['corner_valid']
    corner=(per*mask).sum()/mask.sum().clamp_min(1)
    # Divide the angular loss by 10 degrees to put initial task losses on
    # comparable scales. Log unscaled MAE separately for interpretability.
    boundary=boundary_loss(pred['bon'],labels['bon'],labels['boundary_valid'])/10
    if geometry_weight:
        normal,gradient=planar_geometry_loss(pred['bon'],labels['bon'],labels['boundary_valid'])
    else:normal=gradient=boundary.new_zeros(())
    total=corner_weight*corner+boundary_weight*boundary+geometry_weight*(normal+gradient)
    return total,dict(corner=corner,boundary=boundary,normal=normal,normal_gradient=gradient)
