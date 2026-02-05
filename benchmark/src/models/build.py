from __future__ import annotations

from typing import Dict, List

from torch import nn

from src.models.backbones.lstm import BiLSTMEncoder
from src.models.backbones.tcn import TCNEncoder
from src.models.backbones.transformer import TransformerEncoder
from src.models.fusion import FusionModel
from src.models.heads import JointRegressor


def _select_modalities(data_cfg: Dict, modality_dims: Dict[str, int]) -> List[str]:
    order = ["imu"]
    if data_cfg.get("include_pressure", False):
        order.append("pressure")
    if data_cfg.get("include_fpv", False):
        order.append("fpv")
    for name in order:
        if name not in modality_dims:
            raise ValueError(f"missing_modality_dim: {name}")
    return order


def _build_backbone(backbone_cfg: Dict, in_dim: int) -> nn.Module:
    name = backbone_cfg.get("name", "tcn").lower()
    if name == "tcn":
        return TCNEncoder(
            in_dim=in_dim,
            hidden_dim=backbone_cfg.get("hidden_dim", 256),
            num_layers=backbone_cfg.get("num_layers", 3),
            dropout=backbone_cfg.get("dropout", 0.1),
        )
    if name == "lstm":
        return BiLSTMEncoder(
            in_dim=in_dim,
            hidden_dim=backbone_cfg.get("hidden_dim", 256),
            num_layers=backbone_cfg.get("num_layers", 2),
            dropout=backbone_cfg.get("dropout", 0.1),
            bidirectional=backbone_cfg.get("bidirectional", True),
        )
    if name == "transformer":
        return TransformerEncoder(
            in_dim=in_dim,
            model_dim=backbone_cfg.get("model_dim", 256),
            num_layers=backbone_cfg.get("num_layers", 4),
            num_heads=backbone_cfg.get("num_heads", 4),
            dropout=backbone_cfg.get("dropout", 0.1),
            max_len=backbone_cfg.get("max_len", 2000),
        )
    raise ValueError(f"unknown_backbone: {name}")


def build_model(cfg: Dict) -> nn.Module:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    modality_dims = model_cfg["modality_dims"]
    modality_order = _select_modalities(data_cfg, modality_dims)

    fusion = {
        name: modality_dims[name] for name in modality_order
    }
    latent_dim = model_cfg.get("latent_dim", 128)
    backbone_in_dim = latent_dim * len(modality_order)
    backbone = _build_backbone(model_cfg["backbone"], in_dim=backbone_in_dim)
    head = JointRegressor(
        in_dim=getattr(backbone, "out_dim", latent_dim),
        out_dim=model_cfg.get("out_dim", 63),
        dropout=model_cfg.get("head_dropout", 0.1),
    )
    return FusionModel(
        modality_dims=fusion,
        latent_dim=latent_dim,
        backbone=backbone,
        head=head,
        modality_order=modality_order,
        dropout=model_cfg.get("fusion_dropout", 0.0),
        modality_dropout=model_cfg.get("modality_dropout", 0.0),
    )
