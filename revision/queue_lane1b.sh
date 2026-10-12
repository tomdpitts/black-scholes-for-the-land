#!/bin/zsh
# Sequential (one fit in memory at a time): after rain_temp finishes, run temp, then rain_all.
cd /Users/tompitts/dphil/ProjectB
R=runs/20261006_1802_drivers
until ! pgrep -f "drivers.py fit rain_temp" >/dev/null; do sleep 30; done
for v in temp rain_all; do
  .venv/bin/python -u revision/drivers.py fit $v $R > $R/stdout_$v.log 2>&1 || echo "$v FAILED" >> $R/queue.log
done
echo "lane1b done" >> $R/queue.log
