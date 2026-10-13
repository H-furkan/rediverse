"""Image metrics of the paper: the main comparison table (images), the SD3 ablation table and the
design-choice tables (supplementary), with the definitions of App. F.

Stages, run as modules from the repository root:
  extract_feats     A  CPU fp32 feature caches (DINOv2-base, CLIP ViT-B/32, DreamSim)
  run_metrics       H  GPU metrics (LPIPS, HPSv2.1, ImageReward)
  tpips_embed       T  TPIPS-Embed embeddings per factor
  feature_metrics   B  DINO, Vendi, DreamSim, CLIPScore and TPIPS from the caches
evaluate_set.sh runs them for one set; score_ablation.sh for one SD3 ablation arm.
"""
