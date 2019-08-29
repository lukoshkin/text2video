import torch
from torch import nn
from torch.nn.utils import spectral_norm

from blocks import DBlock
from convgru import ConvGRU
from functools import partial


class VideoEncoder(nn.Module):
    def __init__(
            self, in_colors=3, 
            base_width=32, bn=True, sn=False):
        super().__init__()
        block2d = partial(DBlock, '2d', bn=bn, sn=sn)
        block3d = partial(DBlock, '3d', bn=bn, sn=sn)
        SN = spectral_norm if sn else lambda x: x
        self.downsampler1 = nn.Sequential (
            SN(nn.Conv3d(in_colors, base_width, 1)),
            block3d(base_width, base_width*2, 2),
            block3d(base_width*2, base_width*4, (1,2,2))
        )
        self.cgru = ConvGRU(
                base_width*4, base_width*4, 3, spectral_norm=sn)
        self.downsampler2 = nn.Sequential (
            block2d(base_width*4, base_width*8, 2),
            block2d(base_width*8, base_width*16, 2),
            block2d(base_width*16, base_width*32, 2)
        )

    def forward(self, video):
        H = self.downsampler1(video)
        _, last = self.cgru(H)
        H = self.downsampler2(last)

        return H.view(H.size(0), -1)


class ProjectionVideoDiscriminator(VideoEncoder):
    def __init__(
            self, cond_size, 
            in_colors=3, base_width=32, logits=True):
        super().__init__(in_colors, base_width, bn=False, sn=True)
        self.proj = nn.Sequential (
            nn.Linear(cond_size, base_width*32),
            nn.LeakyReLU(.2, True)
        )
        self.pool = nn.Linear(base_width*32, 1)
        if logits: self.activation = nn.Sequential()
        else: self.activation = torch.sigmoid

    def forward(self, video, embedding):
        E = self.proj(embedding)
        H = super().forward(video)
        out = self.pool(H).squeeze()
        out += torch.einsum('ij,ij->i', E, H)
        
        return self.activation(out)


class ImageEncoder(nn.Module):
    def __init__(
            self, in_colors=3, k_frames=8, 
            base_width=32, bn=True, sn=False):
        super().__init__()
        self.k = k_frames
        block2d = partial(DBlock, '2d', bn=bn, sn=sn)
        SN = spectral_norm if sn else lambda x: x
        self.downsampler = nn.Sequential (
            SN(nn.Conv2d(in_colors, base_width, 1)),
            block2d(base_width, base_width*2, 2),
            block2d(base_width*2, base_width*4, 2),
            block2d(base_width*4, base_width*8, 2),
            block2d(base_width*8, base_width*16, 2),
            block2d(base_width*16, base_width*32, 2)
        )

    def forward(self, video):
        frame_ids = torch.multinomial(
                torch.ones(video.size(2)), self.k)
        images = torch.flatten(
                video[:, :, frame_ids, ...].permute(0,2,1,3,4), 0, 1)
        H = self.downsampler(images)

        # video.shape       (N, C, T, H, W)
        # images.shape      (N*k, C, H, W)
        # H.shape           (N*k, base_width*32, 1, 1)
        # output.shape      (N, k, base_width*32)

        return H.view(H.size(0)//self.k, self.k,  -1)


class ProjectionImageDiscriminator(ImageEncoder):
    def __init__(
            self, cond_size, 
            in_colors=3, k_frames=8, 
            base_width=32, logits=True):
        super().__init__(in_colors, k_frames, base_width, bn=False, sn=True)
        self.proj = nn.Sequential (
            nn.Linear(cond_size, base_width*32),
            nn.LeakyReLU(.2, True)
        )
        self.pool = nn.Linear(base_width*32, 1)
        if logits: self.activation = nn.Sequential()
        else: self.activation = torch.sigmoid

    def forward(self, video, embedding):
        E = self.proj(embedding)
        H = super().forward(video)
        out = self.pool(H).sum([1, 2])
        out += torch.einsum('ij,ikj->i', E, H)

        return self.activation(out)
