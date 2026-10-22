#!/bin/sh
# M3 runs after addendum 3. Usage: sh revision/queue_m3.sh <run dir>
R=$1
fit() { .venv/bin/python -u revision/m3.py fit $1 $2 $R > $R/stdout_$1_s$2.log 2>&1 || echo "$1 s$2 FAILED" >> $R/queue.log; }
( fit base 2; .venv/bin/python -u revision/baselines.py gbm $R B_test > $R/stdout_gbm_B_test.log 2>&1 || echo "gbm FAILED" >> $R/queue.log; echo "lane1 done" >> $R/queue.log ) &
( fit topo 2; echo "lane2 done" >> $R/queue.log ) &
wait
echo "all done" >> $R/queue.log
