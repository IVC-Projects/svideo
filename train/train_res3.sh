#!/usr/bin/env sh
LOG=./log/res3_ldp_lr0.003_qp37_fore.log
CAFFE=/home/tjc/caffe/build/tools/caffe # your caffe path

$CAFFE train --solver=./res3_solver.prototxt -gpu 0 2>&1 | tee $LOG

## resume training
#$CAFFE train --solver=./res3_solver.prototxt \
#--snapshot=../../model/res3_qp42_rb0.003_fore_Filter_iter_45000.solverstate -gpu 0 2>&1 | tee $LOG

