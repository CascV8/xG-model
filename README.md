# Expected Goals model

Expected goals (xG) for NHL unblocked shots, built on [puckscrape](../puckscrape).
A gradient-boosted model trained on 2022-23 to 2024-25 play-by-play, excluding empty-net shots.
On the held-out 2025-26 season it beats MoneyPuck on log loss (0.2156 vs 0.2179) and AUC (0.774 vs 0.768);
its season total runs 4.6% high, as 2025-26 shots were logged closer to the net without a matching rise in goals.

```
uv run xg-model collect 2022 2023 2024 2025   # download play-by-play
uv run xg-model train                         # train 2022-24, test 2025-26, write reports/
uv run xg-model game 2025020001               # xG for one game
```

```python
import puckscrape as ps
from xg_model import predict

shots = predict.score(ps.nhl.play_by_play(2025020001))   # one row per shot, with an xg column
```
