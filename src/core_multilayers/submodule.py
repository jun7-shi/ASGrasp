import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# https://github.com/FangjinhuaWang/IterMVS/

class DepthInitialization(nn.Module):
    def __init__(self, num_sample):
        super(DepthInitialization, self).__init__()
        self.num_sample = num_sample
    
    def forward(self, inverse_depth_min, inverse_depth_max, height, width, device):
        batch = inverse_depth_min.size()[0]      
        index = torch.arange(0, self.num_sample, 1, device=device).view(1, self.num_sample, 1, 1).float()
        normalized_sample = index.repeat(batch, 1, height, width) / (self.num_sample-1)
        depth_sample = inverse_depth_max + normalized_sample * (inverse_depth_min - inverse_depth_max)

        depth_sample = 1.0 / depth_sample

        return depth_sample

def differentiable_warping(src_fea, src_proj, ref_proj, depth_samples, return_mask=False):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, Ndepth, H, W] 
    # out: [B, C, Ndepth, H, W]
    batch, num_depth, height, width = depth_samples.size()
    height1, width1 = src_fea.size()[2:]

    with torch.no_grad():
        if batch==2:
            inv_ref_proj = []
            for i in range(batch):
                inv_ref_proj.append(torch.inverse(ref_proj[i]).unsqueeze(0))
            inv_ref_proj = torch.cat(inv_ref_proj, dim=0)
            assert (not torch.isnan(inv_ref_proj).any()), "nan in inverse(ref_proj)"
            proj = torch.matmul(src_proj, inv_ref_proj)
        else:
            proj = torch.matmul(src_proj, torch.inverse(ref_proj))
            assert (not torch.isnan(proj).any()), "nan in proj"

        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth_samples.device),
                            torch.arange(0, width, dtype=torch.float32, device=depth_samples.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        y = y*(height1/height)
        x = x*(width1/width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                            height * width)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        valid_mask = proj_xyz[:, 2:] > 1e-2
        proj_xyz[:, 0:1][~valid_mask] = width
        proj_xyz[:, 1:2][~valid_mask] = height
        proj_xyz[:, 2:3][~valid_mask] = 1
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        valid_mask = valid_mask & (proj_xy[:, 0:1] >=0) & (proj_xy[:, 0:1] < width) \
                    & (proj_xy[:, 1:2] >=0) & (proj_xy[:, 1:2] < height)
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width1 - 1) / 2) - 1 # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height1 - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy      

    dim = src_fea.size()[1]
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                padding_mode='zeros',align_corners=True)
    warped_src_fea = warped_src_fea.view(batch, dim, num_depth, height, width)
    if return_mask:
        valid_mask = valid_mask.view(batch,num_depth,height,width)
        return warped_src_fea, valid_mask
    else:
        return warped_src_fea

def depth_normalization(depth, inverse_depth_min, inverse_depth_max):
    '''convert depth map to the index in inverse range'''
    inverse_depth = 1.0 / (depth+1e-5)
    normalized_depth = (inverse_depth - inverse_depth_max) / (inverse_depth_min - inverse_depth_max)
    return normalized_depth

def depth_unnormalization(normalized_depth, inverse_depth_min, inverse_depth_max):
    '''convert the index in inverse range to depth map'''
    inverse_depth = inverse_depth_max + normalized_depth * (inverse_depth_min - inverse_depth_max) # [B,1,H,W]
    depth = 1.0 / inverse_depth
    return depth

