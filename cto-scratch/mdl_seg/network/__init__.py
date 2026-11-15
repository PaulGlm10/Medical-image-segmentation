from .cto_net import CTO
from .unet import UNet


def build_model(opt):
    """Construit le réseau à partir des options (voir cto/options.py)."""
    if opt.model == "cto":
        return CTO(
            num_class=opt.num_class,
            backbone=opt.backbone,
            pretrained=opt.pretrained,
            trans_dim=opt.trans_dim,
            trans_depth=opt.trans_depth,
            fusion_head=opt.fusion_head,
            bim_attention=opt.bim_attention,
        )
    if opt.model == "unet":
        return UNet(num_class=opt.num_class)
    raise ValueError(f"Modèle inconnu : {opt.model}")
