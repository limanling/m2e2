#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/sr/'$1
mkdir -p $logdir

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"

python ../../src/engine/SRrunner.py \
       --train_sr $datadir"/imSitu/train.json" \
       --test_sr $datadir"/imSitu/test.json" \
       --dev_sr $datadir"/imSitu/dev.json" \
       --train_ee $datadir"/ace/JMEE_train_filter_no_timevalue.json" \
       --webd $glovedir"/glove.840B.300d.txt" \
       --earlystop 10 --restart 999999 --optimizer "adam" --lr 0.0001 \
       --wnebd $datadir"/vocab/embedding_situation_noun.npy" \
       --wvebd $datadir"/vocab/embedding_situation_verb.npy" \
       --wrebd $datadir"/vocab/embedding_situation_role.npy" \
       --vocab $datadir"/vocab/" \
       --image_dir $datadir"/imSitu/of500_images_resized" \
       --imsitu_ontology_file $datadir"/imSitu/imsitu_space.json" \
       --verb_mapping_file $datadir"/ace/ace_sr_mapping.txt" \
       --object_class_map_file $datadir"/object/class-descriptions-boxable.csv" \
       --object_detection_pkl_file $datadir"/imSitu/object_detection/det_results_imsitu_oi_1.pkl" \
       --object_detection_threshold 0.2 \
       --shuffle \
       --add_object \
       --filter_place \
       --batch 12 --epochs 100 --device "cuda" --out $logdir \
       --hps "{
       'wemb_dim': 300, 
       'wemb_ft': False, 
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
       'loss_weight_role': 0.0,
       'gcn_layers': 1,
       'gcn_dp': False,
       'gcn_use_bn': False,
       'use_highway': False,
       }" \
       >& $logdir/stdout.log &

#--filter_irrelevant_verbs
#       --train_ace \