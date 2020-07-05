#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/grounding/'$1
mkdir -p $logdir

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"

python ../../src/engine/Groundingrunner.py \
       --train $datadir"/grounding/grounding_train.json" \
       --test $datadir"/grounding/grounding_test.json" \
       --dev $datadir"/grounding/grounding_valid.json" \
       --earlystop 10 --restart 999999 --optimizer "adam" --lr 0.0001 \
       --wnebd $datadir"/vocab/embedding_situation_noun.npy" \
       --wvebd $datadir"/vocab/embedding_situation_verb.npy" \
       --wrebd $datadir"/vocab/embedding_situation_role.npy" \
       --vocab $datadir"/vocab/" \
       --webd $glovedir"/glove.840B.300d.txt" \
       --img_dir $datadir"/voa/rawdata/VOA_image_en/" \
       --shuffle \
       --batch 32 --epochs 100 --device "cuda" --out $logdir \
       --ee_hps "{
       'wemb_dim': 300,
       'wemb_ft': True,
       'wemb_dp': 0.5,
       'pemb_dim': 50,
       'pemb_dp': 0.5,
       'eemb_dim': 50,
       'eemb_dp': 0.5,
       'psemb_dim': 50,
       'psemb_dp': 0.5,
       'lstm_dim': 150,
       'lstm_layers': 1,
       'lstm_dp': 0,
       'gcn_et': 3,
       'gcn_use_bn': True,
       'gcn_layers': 3,
       'gcn_dp': 0.5,
       'sa_dim': 300,
       'use_highway': True,
       'loss_alpha': 5
       }" \
       --sr_hps "{
       'wemb_dim': 300,
       'wemb_ft': True,
       'wemb_dp': 0.0,
       'iemb_backbone': 'vgg16',
       'iemb_dim':4096,
       'iemb_ft': False,
       'iemb_dp': 0.0,
       'posemb_dim': 512,
       'fmap_dim': 512,
       'fmap_size': 7,
       'att_dim': 1024,
       'loss_weight_verb': 1.0,
       'loss_weight_noun': 0.1,
       'loss_weight_role': 0.0
       }" \
       >& $logdir/stdout.log &
