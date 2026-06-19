# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .bevdet import BEVDepth4D, BEVDet, BEVDet4D, BEVDetTRT, BEVStereo4D
from .fastbev import FastBEV, FastBEV4D, FastBEVTRT
from .fastbev_temporal import FastBEVTemporalConcat

try:
    from .bevdet_occ import BEVStereo4DOCC
except ImportError:
    BEVStereo4DOCC = None
try:
    from .centerpoint import CenterPoint
except ImportError:
    CenterPoint = None
try:
    from .dal import DAL
except ImportError:
    DAL = None
try:
    from .dynamic_voxelnet import DynamicVoxelNet
except ImportError:
    DynamicVoxelNet = None
try:
    from .fcos_mono3d import FCOSMono3D
except ImportError:
    FCOSMono3D = None
try:
    from .groupfree3dnet import GroupFree3DNet
except ImportError:
    GroupFree3DNet = None
try:
    from .h3dnet import H3DNet
except ImportError:
    H3DNet = None
try:
    from .imvotenet import ImVoteNet
except ImportError:
    ImVoteNet = None
try:
    from .imvoxelnet import ImVoxelNet
except ImportError:
    ImVoxelNet = None
try:
    from .mink_single_stage import MinkSingleStage3DDetector
except ImportError:
    MinkSingleStage3DDetector = None
try:
    from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
except ImportError:
    DynamicMVXFasterRCNN = None
    MVXFasterRCNN = None
try:
    from .mvx_two_stage import MVXTwoStageDetector
except ImportError:
    MVXTwoStageDetector = None
try:
    from .parta2 import PartA2
except ImportError:
    PartA2 = None
try:
    from .point_rcnn import PointRCNN
except ImportError:
    PointRCNN = None
try:
    from .sassd import SASSD
except ImportError:
    SASSD = None
try:
    from .single_stage_mono3d import SingleStageMono3DDetector
except ImportError:
    SingleStageMono3DDetector = None
try:
    from .smoke_mono3d import SMOKEMono3D
except ImportError:
    SMOKEMono3D = None
try:
    from .ssd3dnet import SSD3DNet
except ImportError:
    SSD3DNet = None
try:
    from .votenet import VoteNet
except ImportError:
    VoteNet = None
try:
    from .voxelnet import VoxelNet
except ImportError:
    VoxelNet = None

__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'PointRCNN', 'SMOKEMono3D',
    'MinkSingleStage3DDetector', 'SASSD', 'BEVDet', 'BEVDet4D', 'BEVDepth4D',
    'BEVDetTRT', 'BEVStereo4D', 'BEVStereo4DOCC',
    'FastBEV', 'FastBEV4D', 'FastBEVTRT','FastBEVTemporalConcat'
]