#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$2

logdir='../../log/sr_test/'$1
mkdir -p $logdir
mkdir -p $logdir/'image_result'

datadir="/scratch/manling2/data/mm-event-graph"
glovedir="/scratch/manling2/data/glove"
checkpoint="/scratch/manling2/mm-event-graph/log/joint/att_gcn3/model/model_sr_4.pt"
checkpoint_params="/scratch/manling2/mm-event-graph/log/joint/att_gcn3/sr_hyps.json"

python ../../src/engine/TestRunnerSR_m2e2.py \
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
       --filter_irrelevant_verbs --filter_place \
       --train_ee $datadir"/ace/JMEE_train_filter_no_timevalue.json" \
       --test_ee $datadir"/ace/JMEE_test_filter_no_timevalue.json" \
       --dev_ee $datadir"/ace/JMEE_dev_filter_no_timevalue.json" \
       --webd $glovedir"/glove.840B.300d.txt" \
       --train_grounding $datadir"/grounding/grounding_train_20000.json" \
       --test_grounding $datadir"/grounding/grounding_test_20000.json" \
       --dev_grounding $datadir"/grounding/grounding_valid_20000.json" \
       --batch 1 --device "cuda" --out $logdir \
       --test_voa_image $datadir"/voa/rawdata/VOA_image_en/" \
       --gt_voa_image $datadir"/voa_anno_m2e2/image_event.json" \
       --gt_voa_text $datadir"/voa_anno_m2e2/article_event.json" \
       --finetune_sr ${checkpoint} \
       --sr_hps_path ${checkpoint_params} \
       --object_detection_pkl_file $datadir"/voa/object_detection/det_results_voa_oi_1.pkl" \
       --ignore_place_sr_test \
       --ignore_time_test \
       --object_detection_threshold 0.1 \
       --keep_events 1 \
       --keep_events_sr 0 \
       --visual_voa_sr_path $logdir/'image_result' \
       >& $logdir/testout.log &



