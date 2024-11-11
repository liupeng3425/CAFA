from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.dacs_depth_init_pred import DACSDepthInitPred
from mmseg.models.uda.dacs_depth import DACSDepth
from mmseg.models.uda.dacs_aux import DACSAux
from mmseg.models.uda.dacs_aux_cross import DACSAuxCross
from mmseg.models.uda.dacs_dual import DACSDual
from mmseg.models.uda.dacs_depth_bb import DACSDepthBB
from mmseg.models.uda.dacs_dual_attnavg import DACSDualAttnAvg
from mmseg.models.uda.dacs_dual_cor_loss import DACSDualCorLoss
from mmseg.models.uda.dacs_dual_cor_weight import DACSDualCorWeight
from mmseg.models.uda.dacs_dual_adv import DACSDualAdv
from mmseg.models.uda.dacs_dual_adv_nomixup import DACSDualAdvNomixup
from mmseg.models.uda.dacs_adv import DACSAdv
from mmseg.models.uda.dacs_adv_feature import DACSAdvFeature
from mmseg.models.uda.dacs_adv_feature_nomixup import DACSAdvFeatureNomixup
from mmseg.models.uda.dacs_dual_adv_feature import DACSDualAdvFeature
from mmseg.models.uda.dacs_dual_adv_feature_nomixup import DACSDualAdvFeatureNomixup
from mmseg.models.uda.dacs_adv_feature_cat import DACSAdvFeatureCat
from mmseg.models.uda.dacs_dual_adv_feature_cat import DACSDualAdvFeatureCat
from mmseg.models.uda.dacs_dual_syn import DACSDualSyn
from mmseg.models.uda.dada import DACSDualAdvFeatureDADA


__all__ = [
    "DACS",
    "DACSDepth",
    "DACSDepthInitPred",
    "DACSAux",
    "DACSAuxCross",
    "DACSDual",
    'DACSDepthBB',
    'DACSDualAttnAvg',
    'DACSDualCorLoss',
    'DACSDualCorWeight',
    'DACSDualAdv',
    'DACSDualAdvNomixup',
    'DACSAdv',
    'DACSAdvFeature',
    'DACSAdvFeatureNomixup',
    'DACSDualAdvFeature',
    'DACSDualAdvFeatureNomixup',
    'DACSAdvFeatureCat',
    'DACSDualAdvFeatureCat',
    'DACSDualSyn',
    'DACSDualAdvFeatureDADA',
]
