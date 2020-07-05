#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/joint/'$1
mkdir -p $logdir

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"

python ../../src/engine/JOINTRunner.py \
       --train_sr $datadir"/imSitu/train.json" \
       --test_sr $datadir"/imSitu/test.json" \
       --dev_sr $datadir"/imSitu/dev.json" \
       --wnebd $datadir"/vocab/embedding_situation_noun.npy" \
       --wvebd $datadir"/vocab/embedding_situation_verb.npy" \
       --wrebd $datadir"/vocab/embedding_situation_role.npy" \
       --vocab $datadir"/vocab/" \
       --image_dir $datadir"/imSitu/of500_images_resized" \
       --imsitu_ontology_file $datadir"/imSitu/imsitu_space.json" \
       --verb_mapping_file $datadir"/ace/ace_sr_mapping.txt" \
       --object_class_map_file $datadir"/object/class-descriptions-boxable.csv" \
       --object_detection_pkl_file $datadir"/imSitu/object_detection/det_results_imsitu_oi_1.pkl" \
       --train_ee $datadir"/ace/JMEE_train_filter_no_timevalue.json" \
       --test_ee $datadir"/ace/JMEE_test_filter_no_timevalue.json" \
       --dev_ee $datadir"/ace/JMEE_dev_filter_no_timevalue.json" \
       --webd $glovedir"/glove.840B.300d.txt" \
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
       --train_grounding $datadir"/grounding/grounding_train_20000.json" \
       --test_grounding $datadir"/grounding/grounding_test_20000.json" \
       --dev_grounding $datadir"/grounding/grounding_valid_20000.json" \
       --img_dir_grounding $datadir"/voa/rawdata/VOA_image_en/" \
       --object_detection_pkl_file_g $datadir"/voa/object_detection/det_results_voa_oi_1.pkl" \
       --earlystop 10 --restart 999999 --optimizer "adadelta" --lr 1 \
       --finetune_sr '/scratch/manling2/mm-event-graph/log/sr/retest/model.pt' \
       --sr_hps_path '/scratch/manling2/mm-event-graph/log/sr/retest/sr_hyps.json' \
       --object_detection_threshold 0.0 \
       --shuffle \
       --filter_place \
       --add_object \
       --batch 4 --epochs 100 --device "cuda" --out $logdir \
       >& $logdir/stdout.log &
