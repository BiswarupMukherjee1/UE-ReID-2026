# Experiment log

All scores are mAP on the Kaggle public leaderboard (49% of the test set). "Standard re-ranking" is PAT's default k-reciprocal re-ranking; "CA-Jaccard" is the camera-aware re-ranking with a same-class gallery.

## Main line

| Run | What changed | Training images | Scoring | mAP |
|---|---|---|---|---|
| exp01 | PAT baseline, ViT-B/16 | 11.2K | standard re-ranking | 0.102 |
| exp02 | + external data | | standard re-ranking | 0.121 |
| exp13 | ViT-L/16 + external training split | 17.6K | standard re-ranking | 0.130 |
| exp15 | + domain-adversarial branch (weight 0.2), colour jitter, random patch | 17.6K | standard re-ranking | 0.143 |
| exp20 | + all external splits, adversarial weight 0.3 | 19.9K | standard re-ranking | 0.144 |
| exp24 | camera-balanced sampler, adversarial branch off | 19.9K | standard re-ranking | 0.152 |
| exp17 | camera-balanced sampler + adversarial branch | 19.9K | standard re-ranking | 0.157 |

## Retrieval on exp17

| Step | mAP |
|---|---|
| CA-Jaccard, k1 = 15, rubbish bins and containers merged into one class | 0.166 |
| CA-Jaccard, k1 = 10 | 0.171 |
| Checkpoint ensemble, epochs 50/55/60 | 0.173 |
| Query-side horizontal flip, epochs 45/50/55/60 | 0.175 |

## Sampling study

| Run | Batch composition | Camera classifier accuracy | mAP (CA-Jaccard, k1 = 15) |
|---|---|---|---|
| exp17 | one backward image per group for backward-capable identities | 0.69 | 0.166 |
| exp18 | four backward-capable identities forced into every batch | 0.60 | 0.153 |

exp18 made the features less camera-specific but moved each batch towards the external campus domain (about 37% competition identities per batch), which lowered retrieval accuracy on the city test set.

## Hardware

Single NVIDIA RTX 2080 Ti (11 GB). ViT-L/16 at 256×128, batch 32, mixed precision: about 60 images/s, 5.5 hours for 60 epochs.
