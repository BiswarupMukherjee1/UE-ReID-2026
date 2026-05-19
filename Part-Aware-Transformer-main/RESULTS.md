# Urban ReID 2026 — Experiment Results

## Best Result: 0.13267
**File:** submissions/exp13_ep40_classfilter_mergedbins.csv
**Pipeline:** 
1. update.py with exp13_vitlarge_uam_unified/part_attention_vit_40.pth
2. class_filter_rerank.py --merge_bins

## All Submitted Results (chronological)

| File | Score | Notes |
|------|-------|-------|
| exp01_submission_submission.csv | ? | Baseline |
| exp02_submission_submission.csv | 0.12052 | ViT-Base |
| exp02_repro_submission.csv | ? | Repro |
| exp03_submission_submission.csv | ? | |
| exp04_submission_submission.csv | ? | |
| exp04_epoch30_submission_submission.csv | ? | |
| exp05_submission_submission.csv | ? | |
| exp11_submission_submission.csv | ? | |
| exp12_submission_submission.csv | ? | |
| exp06_vit_large_submission_submission.csv | ? | |
| exp06_rerank_k20k4_submission.csv | ? | |
| exp13_ep35_submission_submission.csv | 0.12434 | ViT-Large ep35 |
| exp13_ep20_submission_submission.csv | 0.12128 | ViT-Large ep20 |
| exp13_ep40_submission_submission.csv | 0.12998 | ViT-Large ep40 |
| exp13cont_ep20_submission_submission.csv | ? | Continued training |
| exp13cont_ep30_submission_submission.csv | ? | Continued training |
| exp13cont_ep40_submission_submission.csv | 0.12127 | Continued training hurt |
| exp13_ep40_cls_concat_submission.csv | 0.00365 | BROKEN - wrong hooks |
| exp13_ep40_clsconcat_norerank.csv | 0.11167 | CLS concat no rerank |
| exp13_ep40_clsconcat_rerank.csv | 0.00361 | BROKEN - wrong distance |
| exp13_ep40_clsconcat_rerank_v3.csv | 0.12836 | CLS concat fixed |
| exp13_ep40_classfilter.csv | 0.13133 | Class filtering |
| exp13_ep40_classfilter_mergedbins.csv | 0.13267 | Class filter + merged bins BEST |

## Key Scripts
- update.py — standard inference, saves qf.npy gf.npy
- class_filter_rerank.py — post-processing with class filtering
- eval_cls_concat.py — experimental CLS averaging inference

## Best Checkpoint
models/exp13_vitlarge_uam_unified/part_attention_vit_40.pth
