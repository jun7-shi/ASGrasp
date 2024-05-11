import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(MODULE_DIR)
from update import BasicMultiUpdateBlock, BasicMultiUpdateBlock_2layer
from extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from corr import  CorrBlock1D_Cost_Volume
from utils.utils import coords_grid, upflow8
from submodule import differentiable_warping, DepthInitialization
import numpy as np

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTMVS_2Layer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm,
                                      downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock_2layer(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

        self.ref_rgb = True
        self.depth_initialization = DepthInitialization(self.args.num_sample)
        self.G = 1
        self.sqrt_sample = np.sqrt(self.args.num_sample)

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze_bn(self):
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
        #         m.eval()
        for m in self.cnet.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                m.eval()
    def initialize_disp(self, img):
        B, _, H, W = img.shape
        disp = torch.zeros([B, 2, H, W]).to(img.device)
        return disp
    def upsample_disp(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        B, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3, 3], padding=1)
        up_flow = up_flow.view(B, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(B, D, factor * H, factor * W)

    def disp2depth(self, disp, inverse_depth_min, inverse_depth_max):
        '''convert the index in inverse range to depth map'''
        inverse_depth = inverse_depth_max + disp.clamp(min=1e-3) * (inverse_depth_min - inverse_depth_max) / (
                    self.args.num_sample - 1)  # [B,1,H,W]
        depth = 1.0 / inverse_depth
        return depth

    def depth_loss(self, depth_preds, depth_gt, valid, mask, loss_gamma=0.9, depth_cut=1e-3, only_bg = False):
        """ Loss function defined over sequence of flow predictions """

        n_predictions = len(depth_preds)
        assert n_predictions >= 1
        flow_loss = 0.0

        valid = ((valid >= 0.5))
        # assert valid.shape == depth_gt.shape, [valid.shape, depth_gt.shape]
        assert not torch.isinf(depth_gt[valid.bool()]).any()

        for i in range(n_predictions):
            assert not torch.isnan(depth_preds[i]).any() and not torch.isinf(depth_preds[i]).any()
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations

            adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
            i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
            i_loss = 25 * (1 / depth_preds[i].clamp(min=depth_cut) - 1 / depth_gt.clamp(min=depth_cut)).abs()
            assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, depth_gt.shape, depth_preds[i].shape]
            # flow_loss += i_weight * i_loss[valid.bool()].mean()
            ts_object_mask = torch.logical_and(mask >= 2, valid)
            other = torch.logical_and(mask < 2, valid)
            flow_loss += i_weight * (i_loss[ts_object_mask].mean() + i_loss[other].mean() * 0.4)

        epe = 25 * (1 / depth_preds[-1].clamp(min=depth_cut) - 1 / depth_gt.clamp(min=depth_cut)).abs()  # only disp
        epe = epe.view(-1)[valid.view(-1)]

        mde = (depth_preds[-1] - depth_gt).abs()
        mde = mde.view(-1)[valid.view(-1)]

        metrics = {
            'loss': flow_loss.item(),
            'epe': epe.mean().item(),
            'mde': mde.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
        }

        return flow_loss, metrics

    def get_error(self, batch_data_label):
        depth_loss, depth_metric = self.depth_loss(self.depth_predictions, batch_data_label['depth_gt'], batch_data_label['mvs_valid'], batch_data_label['mvs_mask'])
        return depth_loss, depth_metric

    def forward(self, rgb, ir1, ir2, proj_matrices, depth_min, depth_max, batch_data_label=None, iters=12, flow_init=None, test_mode=False):
        end_points = {}
        """ Estimate optical flow between pair of frames """
        depth_min = depth_min.float()
        depth_max = depth_max.float()

        rgb = (2 * (rgb / 255.0) - 1.0).contiguous()
        ir1 = (2 * (ir1 / 255.0) - 1.0).contiguous()
        ir2 = (2 * (ir2 / 255.0) - 1.0).contiguous()

        rgb_proj = proj_matrices[:, 0, :, :].clone()
        proj_matrices[:, :, :2, :] *= 0.25  # as we proj feature map at 1/4

        with autocast(enabled=self.args.mixed_precision):
            images = torch.cat((ir1, ir2), dim=0)
            fmap1, fmap2 = self.fnet(images).split(dim=0, split_size=images.shape[0] // 2)

            if self.ref_rgb:
                ref_proj = proj_matrices[:, 0]
                src_projs = [proj_matrices[:, 1], proj_matrices[:, 2]]
                ref_feature = None
                src_features = [fmap1, fmap2]
                cnet_list = self.cnet(rgb, num_layers=self.args.n_gru_layers)
            else:
                ref_proj = proj_matrices[:, 1]
                src_projs = [proj_matrices[:, 2]]
                ref_feature = fmap1
                src_features = [fmap2]
                cnet_list = self.cnet(ir1, num_layers=self.args.n_gru_layers)

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

            batch, dim, height, width = fmap1.size()
            inverse_depth_min = (1.0 / depth_min).view(batch, 1, 1, 1)
            inverse_depth_max = (1.0 / depth_max).view(batch, 1, 1, 1)

        if self.args.corr_implementation == "reg":  # Default
            corr_block = CorrBlock1D_Cost_Volume

        device = fmap1.device
        # ref_feature = ref_feature.float()
        warped_features = []
        corr_frames = []
        depth_samples = self.depth_initialization(inverse_depth_min, inverse_depth_max, height, width, device)
        for src_feature, src_proj in zip(src_features, src_projs):
            src_feature = src_feature.float()
            warped_feature = differentiable_warping(src_feature, src_proj, ref_proj, depth_samples)
            warped_feature = warped_feature.view(batch, self.G, dim // self.G, self.args.num_sample, height, width)
            # correlation = torch.mean(warped_feature * ref_feature.view(batch, self.G, dim // self.G, 1, height, width),  dim=2, keepdim=False)
            # corr_frames.append(correlation)
            warped_features.append(warped_feature)
            del warped_feature, src_feature, src_proj

        correlation = torch.mean(warped_features[0] * warped_features[1], dim=2, keepdim=False) / self.sqrt_sample

        corr_fn = corr_block(correlation, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        disp = self.initialize_disp(net_list[0])

        if flow_init is not None:
            disp = disp + flow_init

        self.depth_predictions = []

        for itr in range(iters):
            disp = disp.detach()
            corr0 = corr_fn(disp[:,0:1,:,:])  # index correlation volume
            if self.args.train_2layer:
                corr1 = corr_fn(disp[:,1:2,:,:])  # index correlation volume
            else:
                corr1 = corr0.clone()
            corr = torch.cat([corr0, corr1], axis=1)
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru:  # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False,
                                                 update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:  # Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers == 3, iter16=True,
                                                 iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, disp,
                                                                  iter32=self.args.n_gru_layers == 3,
                                                                  iter16=self.args.n_gru_layers >= 2)

            # F(t+1) = F(t) + \Delta(t)
            disp = disp + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters - 1:
                continue

            # upsample predictions
            if up_mask is None:
                disp_up = upflow8(disp)
            else:
                disp_up = self.upsample_disp(disp, up_mask)

            # disp2depth
            depth_prediction = self.disp2depth(disp_up, inverse_depth_min, inverse_depth_max)
            if not self.args.train_2layer:
                depth_prediction = depth_prediction[:,0:1,:,:]

            self.depth_predictions.append(depth_prediction)
            end_points['depth_pred'] = depth_prediction
        if test_mode:
            return depth_prediction
        if self.args.dp_mode:
            return self.depth_predictions
        loss, metrics = self.get_error(batch_data_label)
        return loss, metrics, end_points

