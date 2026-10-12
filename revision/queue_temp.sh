#!/bin/zsh
# Start the temperature-only fit once the rain+temp fit has written its metrics (memory: two fits at a time).
cd /Users/tompitts/dphil/ProjectB
R=runs/20261006_1802_drivers
until [ -f $R/metrics_rain_temp.json ] || ! pgrep -f "drivers.py fit rain_temp" >/dev/null; do sleep 30; done
.venv/bin/python -u revision/drivers.py fit temp $R > $R/stdout_temp.log 2>&1 || echo "temp FAILED" >> $R/queue.log
