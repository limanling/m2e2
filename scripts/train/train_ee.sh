#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/ee/'$1
mkdir -p $logdir

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"

python ../../src/engine/EErunner.py \
       --train_ee $datadir"/ace/JMEE_train_filter_no_timevalue.json" \
       --test_ee $datadir"/ace/JMEE_test_filter_no_timevalue.json" \
       --dev_ee $datadir"/ace/JMEE_dev_filter_no_timevalue.json" \
       --earlystop 10 --optimizer "adam" --lr 1e-4 \
       --webd $glovedir"/glove.840B.300d.txt" \
       --batch 32 --epochs 100 --device "cuda" --out $logdir \
       --shuffle \
       --hps "{
       'wemb_dim': 300,
       'wemb_ft': True,
       'wemb_dp': 0.5,
       'pemb_dim': 50,
       'pemb_dp': 0.5,
       'eemb_dim': 50,
       'eemb_dp': 0.5,
       'psemb_dim': 50,
       'psemb_dp': 0.5,
       'lstm_dim': 220,
       'lstm_layers': 1,
       'lstm_dp': 0,
       'gcn_et': 3,
       'gcn_use_bn': True,
       'gcn_layers': 2,
       'gcn_dp': 0.5,
       'sa_dim': 300,
       'use_highway': True,
       'loss_alpha': 5
       }" \
       >& $logdir/stdout.log &
