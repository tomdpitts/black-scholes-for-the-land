#!/bin/zsh
# Lane 1: GBM baseline, then the three driver-ablation fits, sequentially.
cd /Users/tompitts/dphil/ProjectB
R=runs/20261006_1802_drivers
.venv/bin/python -u revision/baselines.py gbm > runs/20261006_1759_baselines/gbm.log 2>&1 || echo "GBM FAILED" >> $R/queue.log
for v in rain_temp temp rain_all; do
  .venv/bin/python -u revision/drivers.py fit $v $R > $R/stdout_$v.log 2>&1 || echo "$v FAILED" >> $R/queue.log
done
echo "lane1 done" >> $R/queue.log
