import os
from math import ceil

import torch
from torchtext.data import BucketIterator
from torch.nn import functional as F
from PIL import Image
import math
from random import shuffle
from collections import defaultdict

import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.dataflow.numpy.data_loader_situation import image_collate_fn
from src.engine.EEtraining import run_over_batch_ee
from src.engine.SRtraining import run_over_batch_sr
from src.engine.Groundingtraining import run_over_batch_grounding

def run_over_data_joint(model_ee, model_sr, model_g, optimizer, data_iters, dataset_lens, BATCH_SIZE, need_backward,
                            ee_tester, sr_tester, g_tester, j_tester,
                            ee_hyps, sr_hyps, device,
                            maxnorm,
                            img_dir_grounding, transform,
                            word_i2s, label_i2s, role_i2s, weight, arg_weight, verb_roles, save_output, ee_role_masks,
                            load_object=False, object_results_g=None, object_label_g=None,
                            object_detection_threshold_g=.2, vocab_objlabel=None, ee_visualizer=None
                        ):
    # model_ee = model_ee.module
    # model_sr = model_sr.module

    if need_backward:
        model_ee.train()
        model_sr.train()
        model_g.train()
    else:
        model_ee.eval()
        model_sr.eval()
        model_g.eval()

    sr_running_loss = 0.0
    ee_running_loss = 0.0
    g_running_loss = 0.0

    print()

    # ee
    all_tokens = []
    all_y = []
    all_y_ = []
    all_events = []
    all_events_ = []
    all_sentids = []
    text_result = defaultdict(lambda : defaultdict())
    # sr
    all_results = dict()
    all_verbs = []
    all_verbs_ = []
    all_role = []
    all_role_ = []
    all_noun = []
    all_noun_ = []
    all_triple = []
    all_triple_ = []
    all_triple_group = []
    all_triple_group_ = []
    # grounding
    all_images = []
    all_images_ = []
    all_captions = []
    all_captions_ = []

    if need_backward:
        # Calculate mixing rates
        batch_num = ceil(dataset_lens['ee'] / BATCH_SIZE)  # the main task
        r_sr = math.sqrt(dataset_lens['sr'])
        r_ee = math.sqrt(dataset_lens['ee']) #1.0 * .1 * math.sqrt(len(data_iters['ee']))
        r_grounding = math.sqrt(dataset_lens['g']) #.1 * 1.0 * math.sqrt(len(data_iters['g']))
        num_ee = int(r_ee / r_sr * batch_num)
        num_grounding = int(r_grounding / r_sr * batch_num)
        # print('Training batch_num, num_ee, num_grounding {}, {}, {}'.format(batch_num, num_ee, num_grounding))
        # print(len(data_iters['sr']), len(data_iters['ee']), len(data_iters['g']))
        # Training batch_num, num_ee, num_grounding 1786, 862, 806
        # Training batch_num, num_ee, num_grounding 3, 77, 3    3 1667 3
        # Training batch_num, num_ee, num_grounding 3, 100, 3
        #  VOA_EN_NW_2017.05.14.3850859_0.jpg
        MAX_STEP_sr = batch_num
        MAX_STEP_ee = batch_num  # num_ee
        MAX_STEP_g = batch_num  # num_grounding

        # tasks = ['sr'] * batch_num + ['ee'] * num_ee + ['g'] * num_grounding
        tasks = ['sr'] * batch_num + ['ee'] * batch_num + ['g'] * batch_num
        # tasks = ['sr'] * 5 + ['ee'] * 5 + ['g'] * 5
        shuffle(tasks)
    else:
        MAX_STEP_sr = ceil(dataset_lens['sr'] / BATCH_SIZE)
        MAX_STEP_ee = ceil(dataset_lens['ee'] / BATCH_SIZE)
        MAX_STEP_g = ceil(dataset_lens['g'] / BATCH_SIZE)
        tasks = ['sr'] * MAX_STEP_sr + ['ee'] * MAX_STEP_ee + ['g'] * MAX_STEP_g
        print('Dev/Test MAX_STEP_sr, MAX_STEP_ee, MAX_STEP_g {}, {}, {}'.format(MAX_STEP_sr, MAX_STEP_ee, MAX_STEP_g))
        # print(len(data_iters['sr']), len(data_iters['ee']), len(data_iters['g']))
        # Testing MAX_STEP_sr, MAX_STEP_ee, MAX_STEP_g 1, 44, 1
        # Testing MAX_STEP_sr, MAX_STEP_ee, MAX_STEP_g 1, 37, 1

    cnt_ee = 0
    cnt_sr = 0
    cnt_g = 0
    # MAX_STEP = len(tasks)

    sr_iter = iter(data_iters['sr'])
    ee_iter = iter(data_iters['ee'])
    g_iter = iter(data_iters['g'])
    for task in tasks:
        # # batch = next(data_iters[task])
        if task == 'sr':
            # batch = next(sr_iter)
            try:
                batch = next(sr_iter)
            except StopIteration:
                sr_iter = iter(data_iters['sr'])
                batch = next(sr_iter)
            # print(task, batch)
            sr_running_loss, cnt_sr, all_verbs, all_verbs_, all_role, all_role_, all_noun, all_noun_, \
                all_triple, all_triple_, all_triple_group, all_triple_group_, img_id_batch = \
                run_over_batch_sr(batch, sr_running_loss, cnt_sr, all_verbs, all_verbs_, all_role, all_role_,
                                  all_noun, all_noun_, all_triple, all_triple_, all_triple_group, all_triple_group_, verb_roles,
                                  model_sr, optimizer, MAX_STEP_sr, need_backward, sr_tester, sr_hyps, device, maxnorm,
                                  label_i2s, role_i2s, word_i2s,
                                  train_ace=True, load_object=load_object, visualize_path=None)

        elif task == 'ee':
            try:
                batch = next(ee_iter)
            except StopIteration:
                ee_iter = iter(data_iters['ee'])
                batch = next(ee_iter)
            # print(task, batch)
            ee_running_loss, cnt_ee, all_events, all_events_, all_y, all_y_, all_tokens = \
                run_over_batch_ee(batch, ee_running_loss, cnt_ee, all_events, all_events_, all_y, all_y_, all_tokens, all_sentids, text_result,
                                  model_ee, optimizer, MAX_STEP_ee, need_backward, ee_tester, ee_hyps, device, word_i2s, label_i2s,
                                  role_i2s, maxnorm, weight, arg_weight, ee_role_masks, ee_visualizer)
        elif task == 'g':
            try:
                batch = next(g_iter)
            except StopIteration:
                g_iter = iter(data_iters['g'])
                batch = next(g_iter)
            # print(task, batch)
            g_running_loss, cnt_g, all_captions, all_captions_, all_images, all_images_ = \
                run_over_batch_grounding(batch, g_running_loss, cnt_g, all_captions, all_captions_, all_images, all_images_,
                                         model_g, optimizer, MAX_STEP_g, need_backward, g_tester, ee_hyps, device, maxnorm,
                                         img_dir_grounding, transform, add_object=load_object,
                                         object_results=object_results_g,
                                         object_label=object_label_g,
                                          object_detection_threshold=object_detection_threshold_g,
                                          vocab_objlabel=vocab_objlabel)

    if save_output:
        with open(save_output, "w", encoding="utf-8") as f:
            for tokens in all_tokens:
                for token in tokens:
                    # to match conll2000 format
                    f.write("%s %s %s\n" % (token.word, token.triggerLabel, token.predictedLabel))
                f.write("\n")

    sr_running_loss = sr_running_loss / cnt_sr
    vp, vr, vf = sr_tester.calculate_lists(all_verbs, all_verbs_)
    rp, rr, rf = sr_tester.calculate_sets_no_order(all_role, all_role_)
    np, nr, nf = sr_tester.calculate_sets_no_order(all_noun, all_noun_)
    tp, tr, tf = sr_tester.calculate_sets_no_order(all_triple, all_triple_)
    np_group, nr_group, nf_group = sr_tester.calculate_sets_noun(all_triple_group, all_triple_group_)
    tp_group, tr_group, tf_group = sr_tester.calculate_sets_triple(all_triple_group, all_triple_group_)

    ee_running_loss = ee_running_loss / cnt_ee
    ep, er, ef = ee_tester.calculate_report(all_y, all_y_, transform=False)
    ap, ar, af = ee_tester.calculate_sets(all_events, all_events_)

    g_running_loss = g_running_loss / cnt_g
    p_caption, r_caption, f_caption = g_tester.calculate_lists(all_captions, all_captions_)
    p_image, r_image, f_image = g_tester.calculate_lists(all_images, all_images_)

    # print()

    return sr_running_loss, vp, vr, vf, rp, rr, rf, np, nr, nf, tp, tr, tf, np_group, nr_group, nf_group, tp_group, tr_group, tf_group, \
           ee_running_loss, ep, er, ef, ap, ar, af, \
            g_running_loss, p_caption, r_caption, f_caption, p_image, r_image, f_image


def joint_train(model_ee, model_sr, model_g,
                train_set_g, dev_set_g, test_set_g,
                train_set_ee, dev_set_ee, test_set_ee,
                train_set_sr, dev_set_sr, test_set_sr,
                optimizer_constructor, epochs,
                ee_tester, sr_tester, g_tester, j_tester,
                parser, other_testsets, transform,
                vocab_objlabel
                ):
    data_iters = {}
    dataset_lens = defaultdict(lambda : defaultdict())
    # build batch on cpu
    train_ee_iter = BucketIterator(train_set_ee, batch_size=parser.batch,
                                train=parser.shuffle, shuffle=parser.shuffle, device=-1,
                                sort_key=lambda x: len(x.POSTAGS))
    dev_ee_iter = BucketIterator(dev_set_ee, batch_size=parser.batch, train=False,
                              shuffle=False, device=-1,
                              sort_key=lambda x: len(x.POSTAGS))
    test_ee_iter = BucketIterator(test_set_ee, batch_size=parser.batch, train=False,
                               shuffle=False, device=-1,
                               sort_key=lambda x: len(x.POSTAGS))

    # train_sr_iter = train_set_sr
    # dev_sr_iter = dev_set_sr
    # test_sr_iter = test_set_sr
    train_sr_iter = torch.utils.data.DataLoader(dataset=train_set_sr,
                                             batch_size=parser.batch,
                                             shuffle=parser.shuffle,
                                             num_workers=2,
                                             collate_fn=image_collate_fn)
    dev_sr_iter = torch.utils.data.DataLoader(dataset=dev_set_sr,
                                           batch_size=parser.batch,
                                           shuffle=False,
                                           num_workers=2,
                                           collate_fn=image_collate_fn)
    test_sr_iter = torch.utils.data.DataLoader(dataset=test_set_sr,
                                            batch_size=parser.batch,
                                            shuffle=False,
                                            num_workers=2,
                                            collate_fn=image_collate_fn)

    train_grounding_iter = BucketIterator(train_set_g, batch_size=parser.batch,
                                train=True, shuffle=parser.shuffle, device=-1,
                                sort_key=lambda x: len(x.POSTAGS))
    dev_grounding_iter = BucketIterator(dev_set_g, batch_size=parser.batch, train=False,
                              shuffle=False, device=-1,
                              sort_key=lambda x: len(x.POSTAGS))
    test_grounding_iter = BucketIterator(test_set_g, batch_size=parser.batch, train=False,
                               shuffle=False, device=-1,
                               sort_key=lambda x: len(x.POSTAGS))

    data_iters['train'] = {}
    data_iters['train']['ee'] = train_ee_iter
    data_iters['train']['sr'] = train_sr_iter
    data_iters['train']['g'] = train_grounding_iter
    data_iters['dev'] = {}
    data_iters['dev']['ee'] = dev_ee_iter
    data_iters['dev']['sr'] = dev_sr_iter
    data_iters['dev']['g'] = dev_grounding_iter
    data_iters['test'] = {}
    data_iters['test']['ee'] = test_ee_iter
    data_iters['test']['sr'] = test_sr_iter
    data_iters['test']['g'] = test_grounding_iter
    # train_MAX_STEPs[0] = ceil(len(train_set_ee) / parser.batch)
    # train_MAX_STEPs[1] = len(train_sr_iter)
    # train_MAX_STEPs[2] = ceil(len(train_set_g) / parser.batch)
    # dev_MAX_STEPs[0] = ceil(len(dev_set_ee) / parser.batch)
    # dev_MAX_STEPs[1] = len(dev_sr_iter)
    # dev_MAX_STEPs[2] = ceil(len(dev_set_g) / parser.batch)
    # test_MAX_STEPs[0] = ceil(len(test_set_ee) / parser.batch)
    # test_MAX_STEPs[1] = len(test_sr_iter)
    # test_MAX_STEPs[2] = ceil(len(test_set_g) / parser.batch)
    # for progress in data_iters:
    #     for task in data_iters[progress]:
    #         dataset_lens[progress][task] = len()
    dataset_lens['train']['ee'] = len(train_set_ee)
    dataset_lens['train']['sr'] = len(train_set_sr)
    dataset_lens['train']['g'] = len(train_set_g)
    dataset_lens['dev']['ee'] = len(dev_set_ee)
    dataset_lens['dev']['sr'] = len(dev_set_sr)
    dataset_lens['dev']['g'] = len(dev_set_g)
    dataset_lens['test']['ee'] = len(test_set_ee)
    dataset_lens['test']['sr'] = len(test_set_sr)
    dataset_lens['test']['g'] = len(test_set_g)


    scores = 0.0
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)
    os.makedirs(os.path.join(parser.out, 'model'), exist_ok=True)

    object_results, object_label, object_detection_threshold = train_set_g.get_object_results()

    for i in range(epochs):
        print("Epoch", i + 1)
        # Training Phrase
        sr_training_loss, training_verb_p, training_verb_r, training_verb_f1, \
        training_role_p, training_role_r, training_role_f1, \
        training_noun_p, training_noun_r, training_noun_f1, \
        training_triple_p, training_triple_r, training_triple_f1, \
        training_noun_p_relaxed, training_noun_r_relaxed, training_noun_f1_relaxed, \
        training_triple_p_relaxed, training_triple_r_relaxed, training_triple_f1_relaxed, \
        ee_training_loss, training_ed_p, training_ed_r, training_ed_f1, \
        training_ae_p, training_ae_r, training_ae_f1, \
        g_training_loss, training_caption_p, training_caption_r, training_caption_f1, \
        training_image_p, training_image_r, training_image_f1 = run_over_data_joint(
            model_ee=model_ee,
            model_sr=model_sr,
            model_g=model_g,
            optimizer=optimizer,
            data_iters=data_iters['train'],
            dataset_lens=dataset_lens['train'],
            BATCH_SIZE=parser.batch,
            need_backward=True,
            ee_tester=ee_tester,
            sr_tester=sr_tester,
            g_tester=g_tester,
            j_tester=j_tester,
            ee_hyps=parser.ee_hps,
            sr_hyps=parser.sr_hps,
            device=parser.device,
            maxnorm=parser.maxnorm,
            img_dir_grounding=parser.img_dir_grounding,
            transform=transform,
            word_i2s=parser.ee_word_i2s,
            label_i2s=parser.ee_label_i2s,
            role_i2s=parser.ee_role_i2s,
            weight=parser.ee_label_weight,
            arg_weight=parser.ee_arg_weight,
            verb_roles=train_set_sr.get_verb_role_mapping(),
            save_output=os.path.join(parser.out,
                                     "training_epoch_%d.txt" % (
                                             i + 1)),
            ee_role_masks=parser.role_mask,
            load_object=parser.add_object,
            object_results_g=object_results,
            object_label_g=object_label,
            object_detection_threshold_g=object_detection_threshold,
            vocab_objlabel=vocab_objlabel
            )
        print("\nEpoch", i + 1,
              " sr training loss: ", sr_training_loss,
              "\nsr training verb p: ", training_verb_p,
              " sr training verb r: ", training_verb_r,
              " sr training verb f1: ", training_verb_f1,
              "\nsr training role p: ", training_role_p,
              " sr training role r: ", training_role_r,
              " sr training role f1: ", training_role_f1,
              "\nsr training noun p: ", training_noun_p,
              " sr training noun r: ", training_noun_r,
              " sr training noun f1: ", training_noun_f1,
              "\nsr training triple p: ", training_triple_p,
              " sr training triple r: ", training_triple_r,
              " sr training triple f1: ", training_triple_f1,
              "\nsr training noun p relaxed: ", training_noun_p_relaxed,
              " sr training noun r relaxed: ", training_noun_r_relaxed,
              " sr training noun f1 relaxed: ", training_noun_f1_relaxed,
              "\nsr training triple p relaxed: ", training_triple_p_relaxed,
              " sr training triple r relaxed: ", training_triple_r_relaxed,
              " sr training triple f1 relaxed: ", training_triple_f1_relaxed,
              "\nee training loss: ", ee_training_loss,
              "\nee training ed p: ", training_ed_p,
              " ee training ed r: ", training_ed_r,
              " ee training ed f1: ", training_ed_f1,
              "\nee training ae p: ", training_ae_p,
              " ee training ae r: ", training_ae_r,
              " ee training ae f1: ", training_ae_f1,
              "\ngrounding training loss: ", g_training_loss,
              "\ngrounding training caption p: ", training_caption_p,
              " grounding training caption r: ", training_caption_r,
              " grounding training caption f1: ", training_caption_f1,
              "\ngrounding training image p: ", training_image_p,
              " grounding training image r: ", training_image_r,
              " grounding training image f1: ", training_image_f1)
        parser.writer.add_scalar('train_sr/loss', sr_training_loss, i)
        parser.writer.add_scalar('train_sr/verb/p', training_verb_p, i)
        parser.writer.add_scalar('train_sr/verb/r', training_verb_r, i)
        parser.writer.add_scalar('train_sr/verb/f1', training_verb_f1, i)
        parser.writer.add_scalar('train_sr/role/p', training_role_p, i)
        parser.writer.add_scalar('train_sr/role/r', training_role_r, i)
        parser.writer.add_scalar('train_sr/role/f1', training_role_f1, i)
        parser.writer.add_scalar('train_sr/noun/p', training_noun_p, i)
        parser.writer.add_scalar('train_sr/noun/r', training_noun_r, i)
        parser.writer.add_scalar('train_sr/noun/f1', training_noun_f1, i)
        parser.writer.add_scalar('train_sr/triple/p', training_triple_p, i)
        parser.writer.add_scalar('train_sr/triple/r', training_triple_r, i)
        parser.writer.add_scalar('train_sr/triple/f1', training_triple_f1, i)
        parser.writer.add_scalar('train_sr/noun/p_relaxed', training_noun_p_relaxed, i)
        parser.writer.add_scalar('train_sr/noun/r_relaxed', training_noun_r_relaxed, i)
        parser.writer.add_scalar('train_sr/noun/f1_relaxed', training_noun_f1_relaxed, i)
        parser.writer.add_scalar('train_sr/triple/p_relaxed', training_triple_p_relaxed, i)
        parser.writer.add_scalar('train_sr/triple/r_relaxed', training_triple_r_relaxed, i)
        parser.writer.add_scalar('train_sr/triple/f1_relaxed', training_triple_f1_relaxed, i)
        parser.writer.add_scalar('train_ee/loss', ee_training_loss, i)
        parser.writer.add_scalar('train_ee/ed/p', training_ed_p, i)
        parser.writer.add_scalar('train_ee/ed/r', training_ed_r, i)
        parser.writer.add_scalar('train_ee/ed/f1', training_ed_f1, i)
        parser.writer.add_scalar('train_ee/ae/p', training_ae_p, i)
        parser.writer.add_scalar('train_ee/ae/r', training_ae_r, i)
        parser.writer.add_scalar('train_ee/ae/f1', training_ae_f1, i)
        parser.writer.add_scalar('train_grounding/loss', g_training_loss, i)
        parser.writer.add_scalar('train_grounding/caption/p', training_caption_p, i)
        parser.writer.add_scalar('train_grounding/caption/r', training_caption_r, i)
        parser.writer.add_scalar('train_grounding/caption/f1', training_caption_f1, i)
        parser.writer.add_scalar('train_grounding/image/p', training_image_p, i)
        parser.writer.add_scalar('train_grounding/image/r', training_image_r, i)
        parser.writer.add_scalar('train_grounding/image/f1', training_image_f1, i)

        # Validation Phrase
        with torch.no_grad():
            sr_dev_loss, dev_verb_p, dev_verb_r, dev_verb_f1, \
            dev_role_p, dev_role_r, dev_role_f1, \
            dev_noun_p, dev_noun_r, dev_noun_f1, \
            dev_triple_p, dev_triple_r, dev_triple_f1, \
            dev_noun_p_relaxed, dev_noun_r_relaxed, dev_noun_f1_relaxed, \
            dev_triple_p_relaxed, dev_triple_r_relaxed, dev_triple_f1_relaxed, \
            ee_dev_loss, dev_ed_p, dev_ed_r, dev_ed_f1, \
            dev_ae_p, dev_ae_r, dev_ae_f1, \
            g_dev_loss, dev_caption_p, dev_caption_r, dev_caption_f1, \
            dev_image_p, dev_image_r, dev_image_f1 = run_over_data_joint(
                model_ee=model_ee,
                model_sr=model_sr,
                model_g=model_g,
                optimizer=optimizer,
                data_iters=data_iters['dev'],
                dataset_lens=dataset_lens['dev'],
                BATCH_SIZE=parser.batch,
                need_backward=False,
                ee_tester=ee_tester,
                sr_tester=sr_tester,
                g_tester=g_tester,
                j_tester=j_tester,
                ee_hyps=parser.ee_hps,
                sr_hyps=parser.sr_hps,
                device=parser.device,
                maxnorm=parser.maxnorm,
                img_dir_grounding=parser.img_dir_grounding,
                transform=transform,
                word_i2s=parser.ee_word_i2s,
                label_i2s=parser.ee_label_i2s,
                role_i2s=parser.ee_role_i2s,
                weight=parser.ee_label_weight,
                arg_weight=parser.ee_arg_weight,
                verb_roles=train_set_sr.get_verb_role_mapping(),
                save_output=os.path.join(parser.out,
                                         "dev_epoch_%d.txt" % (
                                                 i + 1)),
                ee_role_masks=parser.role_mask,
                load_object=parser.add_object,
                object_results_g=object_results,
                object_label_g=object_label,
                object_detection_threshold_g=object_detection_threshold,
                vocab_objlabel=vocab_objlabel
            )
        print("\nEpoch", i + 1,
              " sr dev loss: ", sr_dev_loss,
              "\nsr dev verb p: ", dev_verb_p,
              " sr dev verb r: ", dev_verb_r,
              " sr dev verb f1: ", dev_verb_f1,
              "\nsr dev role p: ", dev_role_p,
              " sr dev role r: ", dev_role_r,
              " sr dev role f1: ", dev_role_f1,
              "\nsr dev noun p: ", dev_noun_p,
              " sr dev noun r: ", dev_noun_r,
              " sr dev noun f1: ", dev_noun_f1,
              "\nsr dev triple p: ", dev_triple_p,
              " sr dev triple r: ", dev_triple_r,
              " sr dev triple f1: ", dev_triple_f1,
              "\nsr dev noun p relaxed: ", dev_noun_p_relaxed,
              " sr dev noun r relaxed: ", dev_noun_r_relaxed,
              " sr dev noun f1 relaxed: ", dev_noun_f1_relaxed,
              "\nsr dev triple p relaxed: ", dev_triple_p_relaxed,
              " sr dev triple r relaxed: ", dev_triple_r_relaxed,
              " sr dev triple f1 relaxed: ", dev_triple_f1_relaxed,
              "\nee dev loss: ", ee_dev_loss,
              "\nee dev ed p: ", dev_ed_p,
              " ee dev ed r: ", dev_ed_r,
              " ee dev ed f1: ", dev_ed_f1,
              "\nee dev ae p: ", dev_ae_p,
              " ee dev ae r: ", dev_ae_r,
              " ee dev ae f1: ", dev_ae_f1,
              "\ngrounding dev loss: ", g_dev_loss,
              "\ngrounding dev caption p: ", dev_caption_p,
              " grounding dev caption r: ", dev_caption_r,
              " grounding dev caption f1: ", dev_caption_f1,
              "\ngrounding dev image p: ", dev_image_p,
              " grounding dev image r: ", dev_image_r,
              " grounding dev image f1: ", dev_image_f1)
        parser.writer.add_scalar('dev_sr/loss', sr_dev_loss, i)
        parser.writer.add_scalar('dev_sr/verb/p', dev_verb_p, i)
        parser.writer.add_scalar('dev_sr/verb/r', dev_verb_r, i)
        parser.writer.add_scalar('dev_sr/verb/f1', dev_verb_f1, i)
        parser.writer.add_scalar('dev_sr/role/p', dev_role_p, i)
        parser.writer.add_scalar('dev_sr/role/r', dev_role_r, i)
        parser.writer.add_scalar('dev_sr/role/f1', dev_role_f1, i)
        parser.writer.add_scalar('dev_sr/noun/p', dev_noun_p, i)
        parser.writer.add_scalar('dev_sr/noun/r', dev_noun_r, i)
        parser.writer.add_scalar('dev_sr/noun/f1', dev_noun_f1, i)
        parser.writer.add_scalar('dev_sr/triple/p', dev_triple_p, i)
        parser.writer.add_scalar('dev_sr/triple/r', dev_triple_r, i)
        parser.writer.add_scalar('dev_sr/triple/f1', dev_triple_f1, i)
        parser.writer.add_scalar('dev_sr/noun/p_relaxed', dev_noun_p_relaxed, i)
        parser.writer.add_scalar('dev_sr/noun/r_relaxed', dev_noun_r_relaxed, i)
        parser.writer.add_scalar('dev_sr/noun/f1_relaxed', dev_noun_f1_relaxed, i)
        parser.writer.add_scalar('dev_sr/triple/p_relaxed', dev_triple_p_relaxed, i)
        parser.writer.add_scalar('dev_sr/triple/r_relaxed', dev_triple_r_relaxed, i)
        parser.writer.add_scalar('dev_sr/triple/f1_relaxed', dev_triple_f1_relaxed, i)
        parser.writer.add_scalar('dev_ee/loss', ee_dev_loss, i)
        parser.writer.add_scalar('dev_ee/ed/p', dev_ed_p, i)
        parser.writer.add_scalar('dev_ee/ed/r', dev_ed_r, i)
        parser.writer.add_scalar('dev_ee/ed/f1', dev_ed_f1, i)
        parser.writer.add_scalar('dev_ee/ae/p', dev_ae_p, i)
        parser.writer.add_scalar('dev_ee/ae/r', dev_ae_r, i)
        parser.writer.add_scalar('dev_ee/ae/f1', dev_ae_f1, i)
        parser.writer.add_scalar('dev_grounding/loss', g_dev_loss, i)
        parser.writer.add_scalar('dev_grounding/caption/p', dev_caption_p, i)
        parser.writer.add_scalar('dev_grounding/caption/r', dev_caption_r, i)
        parser.writer.add_scalar('dev_grounding/caption/f1', dev_caption_f1, i)
        parser.writer.add_scalar('dev_grounding/image/p', dev_image_p, i)
        parser.writer.add_scalar('dev_grounding/image/r', dev_image_r, i)
        parser.writer.add_scalar('dev_grounding/image/f1', dev_image_f1, i)

        # Testing Phrase
        with torch.no_grad():
            sr_test_loss, test_verb_p, test_verb_r, test_verb_f1, \
            test_role_p, test_role_r, test_role_f1, \
            test_noun_p, test_noun_r, test_noun_f1, \
            test_triple_p, test_triple_r, test_triple_f1, \
            test_noun_p_relaxed, test_noun_r_relaxed, test_noun_f1_relaxed, \
            test_triple_p_relaxed, test_triple_r_relaxed, test_triple_f1_relaxed, \
            ee_test_loss, test_ed_p, test_ed_r, test_ed_f1, \
            test_ae_p, test_ae_r, test_ae_f1, \
            g_test_loss, test_caption_p, test_caption_r, test_caption_f1, \
            test_image_p, test_image_r, test_image_f1 = run_over_data_joint(
                model_ee=model_ee,
                model_sr=model_sr,
                model_g=model_g,
                optimizer=optimizer,
                data_iters=data_iters['test'],
                dataset_lens=dataset_lens['test'],
                BATCH_SIZE=parser.batch,
                need_backward=False,
                ee_tester=ee_tester,
                sr_tester=sr_tester,
                g_tester=g_tester,
                j_tester=j_tester,
                ee_hyps=parser.ee_hps,
                sr_hyps=parser.sr_hps,
                device=parser.device,
                maxnorm=parser.maxnorm,
                img_dir_grounding=parser.img_dir_grounding,
                transform=transform,
                word_i2s=parser.ee_word_i2s,
                label_i2s=parser.ee_label_i2s,
                role_i2s=parser.ee_role_i2s,
                weight=parser.ee_label_weight,
                arg_weight=parser.ee_arg_weight,
                verb_roles=train_set_sr.get_verb_role_mapping(),
                save_output=os.path.join(parser.out,
                                         "test_epoch_%d.txt" % (
                                                 i + 1)),
                ee_role_masks=parser.role_mask,
                load_object=parser.add_object,
                object_results_g=object_results,
                object_label_g=object_label,
                object_detection_threshold_g=object_detection_threshold,
                vocab_objlabel=vocab_objlabel
            )
        print("\nEpoch", i + 1,
              " sr test loss: ", sr_test_loss,
              "\nsr test verb p: ", test_verb_p,
              " sr test verb r: ", test_verb_r,
              " sr test verb f1: ", test_verb_f1,
              "\nsr test role p: ", test_role_p,
              " sr test role r: ", test_role_r,
              " sr test role f1: ", test_role_f1,
              "\nsr test noun p: ", test_noun_p,
              " sr test noun r: ", test_noun_r,
              " sr test noun f1: ", test_noun_f1,
              "\nsr test triple p: ", test_triple_p,
              " sr test triple r: ", test_triple_r,
              " sr test triple f1: ", test_triple_f1,
              "\nsr test noun p relaxed: ", test_noun_p_relaxed,
              " sr test noun r relaxed: ", test_noun_r_relaxed,
              " sr test noun f1 relaxed: ", test_noun_f1_relaxed,
              "\nsr test triple p relaxed: ", test_triple_p_relaxed,
              " sr test triple r relaxed: ", test_triple_r_relaxed,
              " sr test triple f1 relaxed: ", test_triple_f1_relaxed,
              "\nee test loss: ", ee_test_loss,
              "\nee test ed p: ", test_ed_p,
              " ee test ed r: ", test_ed_r,
              " ee test ed f1: ", test_ed_f1,
              "\nee test ae p: ", test_ae_p,
              " ee test ae r: ", test_ae_r,
              " ee test ae f1: ", test_ae_f1,
              "\ngrounding test loss: ", g_test_loss,
              "\ngrounding test caption p: ", test_caption_p,
              " grounding test caption r: ", test_caption_r,
              " grounding test caption f1: ", test_caption_f1,
              "\ngrounding test image p: ", test_image_p,
              " grounding test image r: ", test_image_r,
              " grounding test image f1: ", test_image_f1)
        parser.writer.add_scalar('test_sr/loss', sr_test_loss, i)
        parser.writer.add_scalar('test_sr/verb/p', test_verb_p, i)
        parser.writer.add_scalar('test_sr/verb/r', test_verb_r, i)
        parser.writer.add_scalar('test_sr/verb/f1', test_verb_f1, i)
        parser.writer.add_scalar('test_sr/role/p', test_role_p, i)
        parser.writer.add_scalar('test_sr/role/r', test_role_r, i)
        parser.writer.add_scalar('test_sr/role/f1', test_role_f1, i)
        parser.writer.add_scalar('test_sr/noun/p', test_noun_p, i)
        parser.writer.add_scalar('test_sr/noun/r', test_noun_r, i)
        parser.writer.add_scalar('test_sr/noun/f1', test_noun_f1, i)
        parser.writer.add_scalar('test_sr/triple/p', test_triple_p, i)
        parser.writer.add_scalar('test_sr/triple/r', test_triple_r, i)
        parser.writer.add_scalar('test_sr/triple/f1', test_triple_f1, i)
        parser.writer.add_scalar('test_sr/noun/p_relaxed', test_noun_p_relaxed, i)
        parser.writer.add_scalar('test_sr/noun/r_relaxed', test_noun_r_relaxed, i)
        parser.writer.add_scalar('test_sr/noun/f1_relaxed', test_noun_f1_relaxed, i)
        parser.writer.add_scalar('test_sr/triple/p_relaxed', test_triple_p_relaxed, i)
        parser.writer.add_scalar('test_sr/triple/r_relaxed', test_triple_r_relaxed, i)
        parser.writer.add_scalar('test_sr/triple/f1_relaxed', test_triple_f1_relaxed, i)
        parser.writer.add_scalar('test_ee/loss', ee_test_loss, i)
        parser.writer.add_scalar('test_ee/ed/p', test_ed_p, i)
        parser.writer.add_scalar('test_ee/ed/r', test_ed_r, i)
        parser.writer.add_scalar('test_ee/ed/f1', test_ed_f1, i)
        parser.writer.add_scalar('test_ee/ae/p', test_ae_p, i)
        parser.writer.add_scalar('test_ee/ae/r', test_ae_r, i)
        parser.writer.add_scalar('test_ee/ae/f1', test_ae_f1, i)
        parser.writer.add_scalar('test_grounding/loss', g_test_loss, i)
        parser.writer.add_scalar('test_grounding/caption/p', test_caption_p, i)
        parser.writer.add_scalar('test_grounding/caption/r', test_caption_r, i)
        parser.writer.add_scalar('test_grounding/caption/f1', test_caption_f1, i)
        parser.writer.add_scalar('test_grounding/image/p', test_image_p, i)
        parser.writer.add_scalar('test_grounding/image/r', test_image_r, i)
        parser.writer.add_scalar('test_grounding/image/f1', test_image_f1, i)

        # Early Stop
        # if scores <= dev_caption_f1 + dev_triple_f1 + dev_ed_f1 + dev_ae_f1:
        #     scores = dev_caption_f1 + dev_triple_f1 + dev_ed_f1 + dev_ae_f1
        #     # Move model parameters to CPU
        model_sr.save_model(os.path.join(parser.out, 'model', "model_sr_%d.pt" % (i+1)))
        model_ee.save_model(os.path.join(parser.out, 'model', "model_ee_%d.pt" % (i+1)))
        #     print("Save CPU model at Epoch", i + 1)
        #     now_bad = 0
        # else:
        #     now_bad += 1
        #     if now_bad >= parser.earlystop:
        #         if restart_used >= parser.restart:
        #             print("Restart opportunity are run out")
        #             break
        #         restart_used += 1
        #         print("lr decays and best model is reloaded")
        #         lr = lr * 0.1
        #         model_sr.load_model(os.path.join(parser.out, "model_sr.pt"))
        #         model_ee.load_model(os.path.join(parser.out, "model_ee.pt"))
        #         optimizer = optimizer_constructor(lr=lr)
        #         print("Restart in Epoch %d" % (i + 2))
        #         now_bad = 0

    # Testing Phrase
    sr_test_loss, test_verb_p, test_verb_r, test_verb_f1, \
    test_role_p, test_role_r, test_role_f1, \
    test_noun_p, test_noun_r, test_noun_f1, \
    test_triple_p, test_triple_r, test_triple_f1, \
    test_noun_p_relaxed, test_noun_r_relaxed, test_noun_f1_relaxed, \
    test_triple_p_relaxed, test_triple_r_relaxed, test_triple_f1_relaxed, \
    ee_test_loss, test_ed_p, test_ed_r, test_ed_f1, \
    test_ae_p, test_ae_r, test_ae_f1, \
    g_test_loss, test_caption_p, test_caption_r, test_caption_f1, \
    test_image_p, test_image_r, test_image_f1 = run_over_data_joint(
        model_ee=model_ee,
        model_sr=model_sr,
        model_g=model_g,
        optimizer=optimizer,
        data_iters=data_iters['test'],
        dataset_lens=dataset_lens['test'],
        BATCH_SIZE=parser.batch,
        need_backward=False,
        ee_tester=ee_tester,
        sr_tester=sr_tester,
        g_tester=g_tester,
        j_tester=j_tester,
        ee_hyps=parser.ee_hps,
        sr_hyps=parser.sr_hps,
        device=parser.device,
        maxnorm=parser.maxnorm,
        img_dir_grounding=parser.img_dir_grounding,
        transform=transform,
        word_i2s=parser.ee_word_i2s,
        label_i2s=parser.ee_label_i2s,
        role_i2s=parser.ee_role_i2s,
        weight=parser.ee_label_weight,
        verb_roles=train_set_sr.get_verb_role_mapping(),
        save_output=os.path.join(parser.out,
                                 "test_final_%d.txt" % (
                                         i + 1)),
        ee_role_masks=parser.role_mask,
        arg_weight=parser.ee_arg_weight,
        load_object=parser.add_object,
        object_results_g=object_results,
        object_label_g=object_label,
        object_detection_threshold_g=object_detection_threshold,
        vocab_objlabel=vocab_objlabel
    )
    print("Finally sr test loss: ", sr_test_loss,
          "\nsr test verb p: ", test_verb_p,
          " sr test verb r: ", test_verb_r,
          " sr test verb f1: ", test_verb_f1,
          "\nsr test role p: ", test_role_p,
          " sr test role r: ", test_role_r,
          " sr test role f1: ", test_role_f1,
          "\nsr test noun p: ", test_noun_p,
          " sr test noun r: ", test_noun_r,
          " sr test noun f1: ", test_noun_f1,
          "\nsr test triple p: ", test_triple_p,
          " sr test triple r: ", test_triple_r,
          " sr test triple f1: ", test_triple_f1,
          "\nsr test noun p relaxed: ", test_noun_p_relaxed,
          " sr test noun r relaxed: ", test_noun_r_relaxed,
          " sr test noun f1 relaxed: ", test_noun_f1_relaxed,
          "\nsr test triple p relaxed: ", test_triple_p_relaxed,
          " sr test triple r relaxed: ", test_triple_r_relaxed,
          " sr test triple f1 relaxed: ", test_triple_f1_relaxed,
          "\nFinally ee test loss: ", ee_test_loss,
          "\nee test ed p: ", test_ed_p,
          " ee test ed r: ", test_ed_r,
          " ee test ed f1: ", test_ed_f1,
          "\nee test ae p: ", test_ae_p,
          " ee test ae r: ", test_ae_r,
          " ee test ae f1: ", test_ae_f1,
          "\nFinally grounding test loss: ", g_test_loss,
          "\ngrounding test caption p: ", test_caption_p,
          " grounding test caption r: ", test_caption_r,
          " grounding test caption f1: ", test_caption_f1,
          "\ngrounding test image p: ", test_image_p,
          " grounding test image r: ", test_image_r,
          " grounding test image f1: ", test_image_f1)

    print("Training Done!")
