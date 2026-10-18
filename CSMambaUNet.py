import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p):
        super(UpBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CrossScanMambaBlock2D(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand_factor=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random",
                 dt_scale=1.0, dt_init_floor=1e-4,
                 conv_bias=True, bias=False, use_fast_path=True,
                 layer_norm_epsilon=1e-5):
        super().__init__()
        self.dim = dim


        self.norm = nn.LayerNorm(dim, eps=layer_norm_epsilon)

        try:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=bias,
                use_fast_path=use_fast_path
            )
        except TypeError:
            try:
                self.mamba = Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                )
            except TypeError:
                self.mamba = Mamba(d_model=dim)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def _scan_and_mamba(self, x):

        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)
        # (B, H*W, C) -> (B, C, H, W)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):

        identity = x


        out1 = self._scan_and_mamba(x)


        out2 = self._scan_and_mamba(x.flip(-1)).flip(-1)


        out3 = self._scan_and_mamba(x.transpose(-2, -1)).transpose(-2, -1)


        out4 = self._scan_and_mamba(
            x.transpose(-2, -1).flip(-1)
        ).flip(-1).transpose(-2, -1)


        out = (out1 + out2 + out3 + out4) / 4.0


        attn = self.channel_attn(identity)
        out = out * attn + identity

        return out


class MultiScaleMamba(nn.Module):

    def __init__(self, chn_list, d_state=16, bottleneck_blocks=2):
        super().__init__()

        self.mamba_l2 = CrossScanMambaBlock2D(chn_list[2], d_state=d_state)

        self.mamba_l3 = CrossScanMambaBlock2D(chn_list[3], d_state=d_state)


    def forward(self, features):

        processed = list(features)


        processed[2] = self.mamba_l2(processed[2])


        processed[3] = self.mamba_l3(processed[3])

        return processed


class Encoder(nn.Module):

    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        self.dropout = self.params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):

    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p=0.0)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p=0.0)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p=0.0)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p=0.0)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x_last = self.up4(x, x0)
        output = self.out_conv(x_last)
        return output, x_last


class UNet_Mamba(nn.Module):

    def __init__(self, in_chns, class_num, feat_dim=32,
                 num_classes_contrastive=4, d_state=16,
                 bottleneck_blocks=2):
        super(UNet_Mamba, self).__init__()

        params = {
            'in_chns': in_chns,
            'feature_chns': [16, 32, 64, 128, 256],
            'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num': class_num,
            'acti_func': 'relu'
        }

        self.encoder = Encoder(params)

        self.multi_scale_mamba = MultiScaleMamba(
            chn_list=params['feature_chns'],
            d_state=d_state,
            bottleneck_blocks=bottleneck_blocks
        )

        self.decoder = Decoder(params)

        dim_in = 16
        self.feat_dim = feat_dim

        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )

        self.num_classes_contrastive = num_classes_contrastive
        for class_c in range(num_classes_contrastive):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__(f'contrastive_class_selector_{class_c}', selector)

        for class_c in range(num_classes_contrastive):
            selector = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(feat_dim, 1)
            )
            self.__setattr__(f'contrastive_class_selector_memory_{class_c}', selector)

    def forward_projection_head(self, features):
        return self.projection_head(features)

    def forward_prediction_head(self, features):
        return self.prediction_head(features)

    def get_contrastive_weights(self, features, memory=False):
        weights = []
        prefix = 'contrastive_class_selector_memory_' if memory else 'contrastive_class_selector_'
        for class_c in range(self.num_classes_contrastive):
            selector = self.__getattr__(f'{prefix}{class_c}')
            w = selector(features)
            weights.append(w)
        return torch.cat(weights, dim=1)

    def forward(self, x, return_features=False):

        feature = self.encoder(x)


        feature = self.multi_scale_mamba(feature)

        output, features = self.decoder(feature)

        if return_features:
            return output, features
        return output


def dice_loss_with_logits(pred, target, smooth=1.0):
    pred_sigmoid = torch.sigmoid(pred)

    pred_flat = pred_sigmoid.reshape(pred.shape[0], pred.shape[1], -1)
    target_flat = target.reshape(target.shape[0], target.shape[1], -1)

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class SemiSupervisedContrastiveLoss(nn.Module):


    def __init__(self, temperature=0.1, lambda_contrastive=0.5,
                 dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.seg_loss = self._hybrid_seg_loss

    def _hybrid_seg_loss(self, pred, target):

        dice_l = dice_loss_with_logits(pred, target)
        bce_l = F.binary_cross_entropy_with_logits(pred, target)
        return self.dice_weight * dice_l + self.bce_weight * bce_l

    def contrastive_loss(self, z1, z2, labels=None):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim_matrix = z1 @ z2.T

        if labels is not None:

            labels_flat = labels.view(labels.size(0), -1) if labels.dim() > 1 else labels

            mask = labels_flat.unsqueeze(0) == labels_flat.unsqueeze(1)
            pos_mask = mask.float()
            neg_mask = (~mask).float()


            sim_exp = torch.exp(sim_matrix / self.temperature)

            pos_sim = (sim_exp.unsqueeze(-1) * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)
            neg_sim = (sim_exp.unsqueeze(-1) * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)

            loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        else:

            sim_exp = torch.exp(sim_matrix / self.temperature)
            loss = -torch.log(sim_exp.diag() / (sim_exp.sum(dim=1) + 1e-8)).mean()

        return loss

    def forward(self, pred, target, z1, z2, labels=None):


        dice_l = dice_loss_with_logits(pred, target)
        bce_l = F.binary_cross_entropy_with_logits(pred, target)
        seg_loss = self.dice_weight * dice_l + self.bce_weight * bce_l


        contrastive_l = self.contrastive_loss(z1, z2, labels)


        total_loss = seg_loss + self.lambda_contrastive * contrastive_l

        return total_loss, seg_loss, contrastive_l



