#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/ee_test/'$1
mkdir -p $logdir

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"
checkpoint="/scratch/manling2/mm-event-graph/log/joint/obj_gcn2/model/model_ee_17.pt"
checkpoint_params="/scratch/manling2/mm-event-graph/log/joint/obj_gcn2/ee_hyps.json"

python ../../src/engine/TestRunnerEE_m2e2.py \
       --train_grounding $datadir"/grounding/grounding_train_20000.json" \
       --test_grounding $datadir"/grounding/grounding_test_20000.json" \
       --dev_grounding $datadir"/grounding/grounding_valid_20000.json" \
       --img_dir_grounding $datadir"/voa/rawdata/VOA_image_en/" \
       --object_detection_pkl_file_g $datadir"/voa/object_detection/det_results_voa_oi_1.pkl" \
       --object_class_map_file $datadir"/object/class-descriptions-boxable.csv" \
       --object_detection_threshold 0.2 \
       --train_ee $datadir"/ace/JMEE_train_filter_no_timevalue.json" \
       --test_ee $datadir"/ace/JMEE_test_filter_no_timevalue.json" \
       --dev_ee $datadir"/ace/JMEE_dev_filter_no_timevalue.json" \
       --webd $glovedir"/glove.840B.300d.txt" \
       --batch 32 --device "cuda" --out $logdir \
       --finetune ${checkpoint} \
       --hps_path ${checkpoint_params} \
       --gt_voa_text $datadir"/voa_anno_m2e2/article_event.json" \
       --keep_events 1 \
       --load_grounding \
       >& $logdir/testout.log &

