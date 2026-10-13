"""Video evaluation: the main comparison table (videos) and the video half of the per-aspect TPIPS
table.

Feature extractors write per-prompt caches under <set>/feats; video_metrics.py reduces them to
<set>/metrics_video.json; evaluation/tables/video_table.py prints the paper rows. See
evaluation/README.md (Video metrics) for the environments and commands.
"""
