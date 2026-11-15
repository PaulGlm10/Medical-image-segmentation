#!/bin/sh
# Évalue tous les runs du fold 0 sur le jeu de test, puis les compare.
DATA=${DATA:-data/isic2018_256}
RUNS=${RUNS:-runs}
for run in "$RUNS"/*/fold_0; do
    [ -f "$run/best.pth" ] || continue
    echo "== $run"
    python test.py --run-dir "$run" --data-dir "$DATA" --save-pred
done
python compare.py --runs-dir "$RUNS" --data-dir "$DATA"
