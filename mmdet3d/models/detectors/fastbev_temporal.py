# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.models import DETECTORS
from .fastbev import FastBEV4D
from ..fusion_layers.temporal_fusion import BEVTemporalFusionConcat


@DETECTORS.register_module()
class FastBEVTemporalConcat(FastBEV4D):
    """FastBEV4D variant using a dedicated, identity-initialized fusion
    module instead of raw channel-concat + shared re-encoding.

    Each frame is encoded independently through the full bev_encoder,
    then aligned frames are fused via BEVTemporalFusionConcat. Because
    that module is identity-initialized, this model is mathematically
    equivalent to the single-frame FastBEV baseline at init.

    Streaming/sequential inference (pred_prev) is not yet supported.
    """

    def __init__(self, feat_channels=256, fusion_dropout=0.3, **kwargs):
        super(FastBEVTemporalConcat, self).__init__(**kwargs)
        self.temporal_fusion = BEVTemporalFusionConcat(
            feat_channels=feat_channels,
            num_prev_frames=self.num_frame - 1,
            dropout=fusion_dropout,
        )

    def extract_img_feat(self, img, img_metas, pred_prev=False,
                          sequential=False, **kwargs):
        assert not sequential, \
            'sequential (streaming) inference not supported yet for ' \
            'FastBEVTemporalConcat'
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
            bda, _ = self.prepare_inputs(img)

        bev_feat_list, depth_list = [], []
        key_frame = True
        for img_t, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
                imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            mlp_input = self.img_view_transformer.get_mlp_input(
                sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
            inputs_curr = (img_t, sensor2keyego, ego2global, intrin,
                           post_rot, post_tran, bda, mlp_input)
            if key_frame:
                bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            else:
                with torch.no_grad():
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
            bev_feat = self.bev_encoder(bev_feat)  # full encode, per frame
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False

        # align every non-key frame's *encoded* feature into the key frame
        for adj_id in range(1, self.num_frame):
            bev_feat_list[adj_id] = self.shift_feature(
                bev_feat_list[adj_id],
                [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)

        bev_feat_fused = self.temporal_fusion(
            bev_feat_list[0], bev_feat_list[1:], already_warped=True)
        return [bev_feat_fused], depth_list[0]