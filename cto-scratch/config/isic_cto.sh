#!/bin/sh
# CTO : Transformer sur x4 + BIM à attention sur l'incertitude + loss de contours Dice.
DATA=${DATA:-data/isic2018_256}
RUNS=${RUNS:-runs}
python train.py --name cto --data-dir "$DATA" --save-dir "$RUNS" --fold 0
