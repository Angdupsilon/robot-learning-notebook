## shapes

| variant | seeds | params (M) | layers | GFLOPs/sample | iid acc | iid binding-pair acc | OOD acc | tag gap | FG-ARI frames | FG-ARI content | dial sync |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 5 | 4.79 | 6 | 0.687 | 60.7 ± 6.7 | 80.5 ± 10.0 | 5.4 ± 6.8 | – | – | 52.9 ± 8.4 | – |
| A_prime | 5 | 5.58 | 7 | 0.800 | 54.9 ± 2.6 | 70.7 ± 5.2 | 0.2 ± 0.2 | – | – | 51.6 ± 4.1 | – |
| B | 5 | 4.86 | 6 | 0.702 | 95.4 ± 0.9 | 97.9 ± 0.3 | 60.9 ± 2.3 | 0.443 ± 0.029 | 60.0 ± 2.6 | 54.1 ± 5.8 | 0.577 ± 0.018 |
| C | 5 | 4.91 | 6 | 0.716 | 95.0 ± 2.8 | 98.1 ± 0.3 | 61.8 ± 5.6 | – | 56.5 ± 2.8 | 53.8 ± 4.8 | – |
| D | 5 | 4.97 | 6 | 0.731 | 96.5 ± 0.4 | 98.2 ± 0.2 | 66.5 ± 6.2 | 0.415 ± 0.049 | 55.3 ± 4.8 | 51.6 ± 2.9 | 0.609 ± 0.019 |
| K | 3 | 4.98 | 6 | 0.808 | 60.8 ± 4.0 | 79.7 ± 7.7 | 1.9 ± 2.7 | 0.145 ± 0.017 | 46.6 ± 1.6 | 42.7 ± 15.3 | 0.922 ± 0.009 |
| D_nosync | 3 | 4.97 | 6 | 0.717 | 96.3 ± 0.7 | 97.9 ± 0.5 | 59.9 ± 4.3 | 0.396 ± 0.030 | 47.5 ± 1.0 | 48.5 ± 1.6 | 0.567 ± 0.016 |
| D_nomoves | 3 | 4.83 | 6 | 0.705 | 95.1 ± 1.5 | 98.0 ± 0.6 | 59.3 ± 7.5 | 0.428 ± 0.024 | 55.8 ± 4.6 | 50.4 ± 2.0 | 0.604 ± 0.018 |
| D_frozen | 3 | 4.83 | 6 | 0.691 | 95.2 ± 1.3 | 97.5 ± 0.4 | 51.4 ± 12.8 | 0.393 ± 0.009 | 46.5 ± 2.1 | 45.7 ± 1.8 | 0.545 ± 0.016 |

Pass/fail (OOD accuracy):
- D beats A (RoPE baseline) by more than 2x the seed spread: **PASS** (difference +61.1 points, pooled std 6.5, seeds 5/5)
- D beats A_prime (compute-matched) by more than 2x the seed spread: **PASS** (difference +66.3 points, pooled std 4.4, seeds 5/5)
- D beats K (published KoPE) by more than 2x the seed spread: **PASS** (difference +64.7 points, pooled std 4.8, seeds 5/3)
- B vs D: different (+5.6 points)
- C vs D: +4.7 points (what the dials add on top of gyros)
- D beats D_nosync (sync off) by more than 2x the seed spread: **no** (difference +6.7 points, pooled std 5.4, seeds 5/3)
- D beats D_nomoves (moves off) by more than 2x the seed spread: **no** (difference +7.2 points, pooled std 6.9, seeds 5/3)
- D beats D_frozen (all frame updates off) by more than 2x the seed spread: **no** (difference +15.1 points, pooled std 10.0, seeds 5/3)

