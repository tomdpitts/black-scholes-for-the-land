#!/bin/sh
# Addendum 4: topo_pet seeds 0 and 1, after the running seed-2 fits. Usage: sh revision/queue_m3_pet.sh <run dir> <pid> <pid>
R=$1
while kill -0 $2 2>/dev/null || kill -0 $3 2>/dev/null; do sleep 30; done
fit() { .venv/bin/python -u revision/m3.py fit $1 $2 $R > $R/stdout_$1_s$2.log 2>&1 || echo "$1 s$2 FAILED" >> $R/queue.log; }
fit topo_pet 0 & fit topo_pet 1 & wait
echo "pet done" >> $R/queue.log
