import os
from math import ceil
import torch
import ujson as json
from collections import defaultdict

# import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util.util_model import progressbar
from src.dataflow.numpy.data_loader_situation import unpack, image_collate_fn

def run_over_batch_sr(batch, running_loss, cnt,
                      all_verbs, all_verbs_, all_role, all_role_, all_noun, all_noun_,
                      all_triple, all_triple_, all_triple_group, all_triple_group_,
                      verb_roles,
                      model, optimizer, MAX_STEP, need_backward, tester, hyps, device, maxnorm,
                      label_i2s, role_i2s, word_i2s, train_ace=False, load_object=False, visualize_path=None):
    # for _ in range(100):
    # print('in run_over_batch_sr')
    # print('get batch', device)
    # print('model type', type(model))
    img_id_batch, image_batch, verb_gt_batch, event_gt_batch, roles_gt_batch, ee_roles_gt_batch, args_gt_batch, \
        bbox_entities_id, bbox_entities_region, bbox_entities_label, \
        arg_num_batch, bbox_num_batch = unpack(batch, device, load_object)
    # print('get batch', arg_num_batch, bbox_num_batch, load_object)

    if need_backward:
        optimizer.zero_grad()

    y_role = roles_gt_batch
    y_arg = args_gt_batch
    y_verb = verb_gt_batch

    img_id_batch, verb_emb_common, noun_emb_common, verb_emb, noun_emb, role_logits, verb_logits, noun_logits, \
        event_logits, event_ae_logits, num_obj_proj \
        = model(
        img_id_batch, image_batch,
        bbox_entities_id, bbox_entities_region, bbox_entities_label, bbox_num_batch
    )  # ! note that the logits must be after mask and log_softmax
    loss_sr_verb, loss_sr_terms, verb_logits, noun_logits, role_logits = model.calculate_loss_all(
        verb_emb, verb_gt_batch,
        noun_emb, args_gt_batch,
        role_logits, verb_logits, noun_logits, roles_gt_batch,
        num_obj_proj, arg_num_batch
    )
    if train_ace:
        loss_sr_ace, loss_ace_terms, event_logits, event_ae_logits = model.calculate_loss_ace(
            event_logits, event_ae_logits, event_gt_batch, ee_roles_gt_batch, num_obj_proj, arg_num_batch
        )
        loss_sr = loss_sr_verb + loss_sr_ace
    else:
        loss_sr = loss_sr_verb

    # evaluate verbs
    # emb_verb_all = model.get_all_verbs_emb()
    # predicted_verbs = model.get_predicted_event(emb_verb_proj, emb_verb_all)
    # y_verb_ = predicted_verbs.tolist()
    y_verb_ = torch.max(verb_logits, dim=1)[1].tolist()  # [batch,]
    y_verb = y_verb.tolist()
    all_verbs_.extend(y_verb_)
    all_verbs.extend(y_verb)
    bp_v, br_v, bf_v = tester.calculate_lists(y_verb, y_verb_)
    # print('verb', bp_v, br_v, bf_v)

    # evaluate roles
    # role_logits_masked = role_logits.masked_fill(mask_proj.unsqueeze(2).expand_as(role_logits), 0.)
    y_role_ = torch.max(role_logits, dim=2)[1].tolist()  # batch * obj_num
    y_role = y_role.tolist()  # batch * arg_gt_num
    # unpad
    for i, ll in enumerate(num_obj_proj):
        y_role_[i] = y_role_[i][:ll]  # list is truncated
    for i, ll in enumerate(arg_num_batch):
        y_role[i] = y_role[i][:ll]
    all_role.extend(y_role)
    all_role_.extend(y_role_)
    bp_r, br_r, bf_r = tester.calculate_sets_no_order(y_role, y_role_)

    # evaluate nouns
    # get the most similar noun from logits
    # noun_logits_masked = noun_logits.masked_fill(mask_proj.unsqueeze(2).expand_as(noun_logits), 0.)
    y_noun_ = torch.max(noun_logits, dim=2)[1].tolist()  # batch * obj_num
    y_arg = y_arg.tolist()
    # unpad
    for i, ll in enumerate(num_obj_proj):
        y_noun_[i] = y_noun_[i][:ll]
    for i, ll in enumerate(arg_num_batch):
        y_arg[i] = y_arg[i][:ll]
    all_noun.extend(y_arg)

    # evaluate triples
    y_triple = []
    y_triple_ = []
    y_triple_set = []  # defaultdict(lambda : defaultdict(set))
    y_triple_set_ = []  # defaultdict(lambda : defaultdict(set))
    for i, ll in enumerate(arg_num_batch):
        y_triple.append([])
        y_triple_set.append(defaultdict(set))
        for j in range(ll):
            y_triple[i].append('(%d,%d,%d)' % (y_verb[i], y_role[i][j], y_arg[i][j]))
            y_triple_set[i][y_role[i][j]].add(y_arg[i][j])
    # get nouns that only contained the ones of each role:
    y_noun__ = []
    y_triple_gt_ = []
    y_triple_set_gt_ = []
    y_noun__gt_ = []
    for i, ll in enumerate(num_obj_proj):
        y_triple_.append([])
        y_triple_set_.append(defaultdict(set))
        y_noun__.append([])
        y_triple_gt_.append([])
        y_triple_set_gt_.append(defaultdict(set))
        y_noun__gt_.append([])
        for j in range(ll):
            if y_role_[i][j] in verb_roles[y_verb_[i]]:
                y_triple_[i].append('(%d,%d,%d)' % (y_verb_[i], y_role_[i][j], y_noun_[i][j]))
                y_triple_set_[i][y_role_[i][j]].add(y_noun_[i][j])
                y_noun__[i].append(y_noun_[i][j])
            if y_role_[i][j] in verb_roles[y_verb[i]]:
                y_triple_gt_[i].append('(%d,%d,%d)' % (y_verb[i], y_role_[i][j], y_noun_[i][j]))
                y_triple_set_gt_[i][y_role_[i][j]].add(y_noun_[i][j])
                y_noun__gt_[i].append(y_noun_[i][j])
    all_triple.extend(y_triple)
    all_triple_.extend(y_triple_)
    all_triple_group.extend(y_triple_set)
    all_triple_group_.extend(y_triple_set_)
    bp_t, br_t, bf_t = tester.calculate_sets_no_order(y_triple, y_triple_)
    bp_n_group, br_n_group, bf_n_group = tester.calculate_sets_noun(y_triple_set, y_triple_set_)
    bp_t_group, br_t_group, bf_t_group = tester.calculate_sets_triple(y_triple_set, y_triple_set_)
    bp_t_gt, br_t_gt, bf_t_gt = tester.calculate_sets_no_order(y_triple, y_triple_gt_)
    bp_n_group_gt, br_n_group_gt, bf_n_group_gt = tester.calculate_sets_noun(y_triple_set, y_triple_set_gt_)
    bp_t_group_gt, br_t_group_gt, bf_t_group_gt = tester.calculate_sets_triple(y_triple_set, y_triple_set_gt_)
    all_noun_.extend(y_noun__)
    bp_n, br_n, bf_n = tester.calculate_sets_no_order(y_arg, y_noun__)
    bp_n_gt, br_n_gt, bf_n_gt = tester.calculate_sets_no_order(y_arg, y_noun__gt_)

    if visualize_path:
        tester.visualize_sets_triple(img_id_batch, y_verb, y_verb_, y_triple_set, y_triple_set_,
                                 label_i2s, role_i2s, word_i2s, visualize_path, None)

    #### evaluate with ground truth verb
    # all_triple_gt_.extend(y_triple_gt_)
    # all_triple_group_gt_.extend(y_triple_set_gt_)
    # all_noun__gt_.extend(y_noun__gt_)

    # add_results(all_results, img_id_batch, y_verb, y_verb_, y_role, y_arg, y_role_, bbox_entities_label, label_i2s,
    #            word_i2s, role_i2s)

    loss = loss_sr
    cnt += 1
    other_information = ""
    if need_backward:
        loss.backward()
        if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)

        optimizer.step()
        other_information = 'SR Iter[{}] loss: {:.6f} loss_terms: {} \nverbP:{:.4f}% verbR:{:.4f}% verbF1:{:.4f}% ; \n' \
                            'roleP:{:.4f}% roleR: {:.4f}% roleF1: {:.4f}% ; \n' \
                            'nounP:{:.4f}% nounR: {:.4f}% nounF1: {:.4f}% ; \n' \
                            'tripleP:{:.4f}% tripleR: {:.4f}% tripleF1: {:.4f}% ; \n' \
                            'nounP_relaxed:{:.4f}% nounR_relaxed: {:.4f}% nounF1_relaxed: {:.4f}% ; \n' \
                            'tripleP_relaxed:{:.4f}% tripleR_relaxed: {:.4f}% tripleF1_relaxed: {:.4f}% ; ' \
                            '(with ground truth verb) nounP:{:.4f}% nounR: {:.4f}% nounF1: {:.4f}% ; \n' \
                            '(with ground truth verb) tripleP:{:.4f}% tripleR: {:.4f}% tripleF1: {:.4f}% ; \n' \
                            '(with ground truth verb) nounP_relaxed:{:.4f}% nounR_relaxed: {:.4f}% nounF1_relaxed: {:.4f}% ; \n' \
                            '(with ground truth verb) tripleP_relaxed:{:.4f}% tripleR_relaxed: {:.4f}% tripleF1_relaxed: {:.4f}% ; '.format(
            cnt,
            loss.item(),
            loss_sr_terms,
            bp_v * 100.0,
            br_v * 100.0,
            bf_v * 100.0,
            bp_r * 100.0,
            br_r * 100.0,
            bf_r * 100.0,
            bp_n * 100.0,
            br_n * 100.0,
            bf_n * 100.0,
            bp_t * 100.0,
            br_t * 100.0,
            bf_t * 100.0,
            bp_n_group * 100.0,
            br_n_group * 100.0,
            bf_n_group * 100.0,
            bp_t_group * 100.0,
            br_t_group * 100.0,
            bf_t_group * 100.0,
            bp_n_gt * 100.0,
            br_n_gt * 100.0,
            bf_n_gt * 100.0,
            bp_t_gt * 100.0,
            br_t_gt * 100.0,
            bf_t_gt * 100.0,
            bp_n_group_gt * 100.0,
            br_n_group_gt * 100.0,
            bf_n_group_gt * 100.0,
            bp_t_group_gt * 100.0,
            br_t_group_gt * 100.0,
            bf_t_group_gt * 100.0
                                                                                                                   )

    progressbar(cnt, MAX_STEP, other_information)
    running_loss += loss.item()
    # raise

    return running_loss, cnt, all_verbs, all_verbs_, all_role, all_role_, all_noun, all_noun_, all_triple, all_triple_, all_triple_group, all_triple_group_, img_id_batch


def add_results(all_results, img_id_batch, y_verb, y_verb_, y_role, y_arg, y_role_, bbox_entities_label, verb_id2word, noun_id2word, role_id2word):
    for idx, img_id in enumerate(img_id_batch):
        if img_id not in all_results:
            all_results[img_id] = dict()
        all_results[img_id]['verb_gt'] = verb_id2word[y_verb[idx]]
        all_results[img_id]['verb_pred'] = verb_id2word[y_verb_[idx]]
        all_results[img_id]['role_pred'] = dict()
        all_results[img_id]['role_gt'] = dict()
        for obj_idx, obj_role in enumerate(y_role_[idx]):
            if role_id2word[obj_role] not in all_results[img_id]['role_pred']:
                all_results[img_id]['role_pred'][role_id2word[obj_role]] = []
            # print('bbox_entities_label[idx][obj_idx]', int(bbox_entities_label[idx][obj_idx].item()))
            all_results[img_id]['role_pred'][role_id2word[obj_role]].append(noun_id2word[int(bbox_entities_label[idx][obj_idx].item())])
        for arg_idx, arg_role in enumerate(y_role[idx]):
            if role_id2word[arg_role] not in all_results[img_id]['role_gt']:
                all_results[img_id]['role_gt'][role_id2word[arg_role]] = []
            all_results[img_id]['role_gt'][role_id2word[arg_role]].append(noun_id2word[y_arg[idx][arg_idx]])


def run_over_data_sr(model, optimizer, data_iter, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                     role_i2s, verb_roles, maxnorm, load_object, train_ace, save_output, visualize_path=None):
    if need_backward:
        # model.test_mode = False
        model.train()
    else:
        # model.test_mode = True
        model.eval()

    running_loss = 0.0

    print()

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

    cnt = 0
    # MAX_STEP = len(data_iter)
    # print('MAX_STEP', MAX_STEP)
    # print('data_iter', data_iter)

    for batch in data_iter:
        # print('batch', batch[0])
        # print('start to run_over_batch_sr')
        running_loss, cnt, all_verbs, all_verbs_, all_role, all_role_, all_noun, all_noun_, all_triple, all_triple_, all_triple_group, all_triple_group_, image_id_batch = \
            run_over_batch_sr(batch, running_loss, cnt, all_verbs, all_verbs_, all_role, all_role_,
                            all_noun, all_noun_, all_triple, all_triple_, all_triple_group, all_triple_group_, verb_roles,
                            model, optimizer, MAX_STEP, need_backward, tester, hyps, device, maxnorm, label_i2s, role_i2s, word_i2s,
                            train_ace=train_ace, load_object=load_object, visualize_path=visualize_path)


    if save_output:
        json.dump(all_results, open(save_output, "w", encoding="utf-8"), indent=4)

    running_loss = running_loss / cnt
    vp, vr, vf = tester.calculate_lists(all_verbs, all_verbs_)
    rp, rr, rf = tester.calculate_sets_no_order(all_role, all_role_)
    np, nr, nf = tester.calculate_sets_no_order(all_noun, all_noun_)
    tp, tr, tf = tester.calculate_sets_no_order(all_triple, all_triple_)
    np_group, nr_group, nf_group = tester.calculate_sets_noun(all_triple_group, all_triple_group_)
    tp_group, tr_group, tf_group = tester.calculate_sets_triple(all_triple_group, all_triple_group_)

    print()
    return running_loss, vp, vr, vf, rp, rr, rf, np, nr, nf, tp, tr, tf, np_group, nr_group, nf_group, tp_group, tr_group, tf_group



def sr_train(model, train_set, dev_set, test_set, optimizer_constructor, epochs, tester, parser, other_testsets):
    # build batch on cpu
    # train_iter = train_set
    # dev_iter = dev_set
    # test_iter = test_set
    train_iter = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=parser.batch,
                                              shuffle=parser.shuffle,
                                              num_workers=2,
                                              collate_fn=image_collate_fn)
    dev_iter = torch.utils.data.DataLoader(dataset=dev_set,
                                             batch_size=parser.batch,
                                             shuffle=parser.shuffle,
                                             num_workers=2,
                                             collate_fn=image_collate_fn)
    test_iter = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=parser.batch,
                                             shuffle=parser.shuffle,
                                             num_workers=2,
                                             collate_fn=image_collate_fn)

    verb_roles = train_set.get_verb_role_mapping()

    scores = 0.0
    testscores = 0.0
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)

    if 'visualize_path' not in parser:
        visualize_path = None
    else:
        visualize_path = parser.visualize_path

    for i in range(epochs):
        # Training Phrase
        print("Epoch", i + 1)
        print('train len', len(train_iter))
        training_loss, training_verb_p, training_verb_r, training_verb_f1, \
        training_role_p, training_role_r, training_role_f1, \
        training_noun_p, training_noun_r, training_noun_f1, \
        training_triple_p, training_triple_r, training_triple_f1, \
        training_noun_p_relaxed, training_noun_r_relaxed, training_noun_f1_relaxed, \
        training_triple_p_relaxed, training_triple_r_relaxed, training_triple_f1_relaxed = run_over_data_sr(
            data_iter=train_iter,
            optimizer=optimizer,
            model=model,
            need_backward=True,
            MAX_STEP=ceil(len(train_set) / parser.batch),
            tester=tester,
            hyps=model.hyperparams,
            device=model.device,
            maxnorm=parser.maxnorm,
            word_i2s=parser.word_i2s,
            label_i2s=parser.label_i2s,
            role_i2s=parser.role_i2s,
            verb_roles=verb_roles,
            load_object=parser.add_object,
            train_ace=parser.train_ace,
            visualize_path=visualize_path,
            save_output=os.path.join(parser.out,
                                  "training_epoch_%d.txt" % (
                                      i + 1))
        )
        print("\nEpoch", i + 1,
              " training loss: ", training_loss,
              "\ntraining verb p: ", training_verb_p,
              " training verb r: ", training_verb_r,
              " training verb f1: ", training_verb_f1,
              "\ntraining role p: ", training_role_p,
              " training role r: ", training_role_r,
              " training role f1: ", training_role_f1,
              "\ntraining noun p: ", training_noun_p,
              " training noun r: ", training_noun_r,
              " training noun f1: ", training_noun_f1,
              "\ntraining triple p: ", training_triple_p,
              " training triple r: ", training_triple_r,
              " training triple f1: ", training_triple_f1,
              "\ntraining noun p relaxed: ", training_noun_p_relaxed,
              " training noun r relaxed: ", training_noun_r_relaxed,
              " training noun f1 relaxed: ", training_noun_f1_relaxed,
              "\ntraining triple p relaxed: ", training_triple_p_relaxed,
              " training triple r relaxed: ", training_triple_r_relaxed,
              " training triple f1 relaxed: ", training_triple_f1_relaxed
              )
        parser.writer.add_scalar('train/loss', training_loss, i)
        parser.writer.add_scalar('train/verb/p', training_verb_p, i)
        parser.writer.add_scalar('train/verb/r', training_verb_r, i)
        parser.writer.add_scalar('train/verb/f1', training_verb_f1, i)
        parser.writer.add_scalar('train/role/p', training_role_p, i)
        parser.writer.add_scalar('train/role/r', training_role_r, i)
        parser.writer.add_scalar('train/role/f1', training_role_f1, i)
        # parser.writer.add_scalar('train/noun/p', training_noun_p, i)
        # parser.writer.add_scalar('train/noun/r', training_noun_r, i)
        # parser.writer.add_scalar('train/noun/f1', training_noun_f1, i)
        parser.writer.add_scalar('train/triple/p', training_triple_p, i)
        parser.writer.add_scalar('train/triple/r', training_triple_r, i)
        parser.writer.add_scalar('train/triple/f1', training_triple_f1, i)
        # parser.writer.add_scalar('train/noun/p_relaxed', training_noun_p_relaxed, i)
        # parser.writer.add_scalar('train/noun/r_relaxed', training_noun_r_relaxed, i)
        # parser.writer.add_scalar('train/noun/f1_relaxed', training_noun_f1_relaxed, i)
        parser.writer.add_scalar('train/triple/p_relaxed', training_triple_p_relaxed, i)
        parser.writer.add_scalar('train/triple/r_relaxed', training_triple_r_relaxed, i)
        parser.writer.add_scalar('train/triple/f1_relaxed', training_triple_f1_relaxed, i)

        # Validation Phrase
        with torch.no_grad():
            dev_loss, dev_verb_p, dev_verb_r, dev_verb_f1, \
            dev_role_p, dev_role_r, dev_role_f1, \
            dev_noun_p, dev_noun_r, dev_noun_f1, \
            dev_triple_p, dev_triple_r, dev_triple_f1, \
            dev_noun_p_relaxed, dev_noun_r_relaxed, dev_noun_f1_relaxed, \
            dev_triple_p_relaxed, dev_triple_r_relaxed, dev_triple_f1_relaxed = run_over_data_sr(
                data_iter=dev_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(dev_set) / parser.batch),
                tester=tester,
                hyps=model.hyperparams,
                device=model.device,
                maxnorm=parser.maxnorm,
                word_i2s=parser.word_i2s,
                label_i2s=parser.label_i2s,
                role_i2s=parser.role_i2s,
                verb_roles=verb_roles,
                load_object=parser.add_object,
                train_ace=parser.train_ace,
                visualize_path=visualize_path,
                save_output=os.path.join(parser.out,
                                       "dev_epoch_%d.txt" % (
                                           i + 1))
            )
        print("\nEpoch", i + 1, " dev loss: ", dev_loss,
              "\ndev verb p: ", dev_verb_p,
              " dev verb r: ", dev_verb_r,
              " dev verb f1: ", dev_verb_f1,
              "\ndev role p: ", dev_role_p,
              " dev role r: ", dev_role_r,
              " dev role f1: ", dev_role_f1,
              "\ndev noun p: ", dev_noun_p,
              " dev noun r: ", dev_noun_r,
              " dev noun f1: ", dev_noun_f1,
              "\ndev triple p: ", dev_triple_p,
              " dev triple r: ", dev_triple_r,
              " dev triple f1: ", dev_triple_f1,
              "\ndev noun p relaxed: ", dev_noun_p_relaxed,
              " dev noun r relaxed: ", dev_noun_r_relaxed,
              " dev noun f1 relaxed: ", dev_noun_f1_relaxed,
              "\ndev triple p relaxed: ", dev_triple_p_relaxed,
              " dev triple r relaxed: ", dev_triple_r_relaxed,
              " dev triple f1 relaxed: ", dev_triple_f1_relaxed
              )
        parser.writer.add_scalar('dev/loss', dev_loss, i)
        parser.writer.add_scalar('dev/verb/p', dev_verb_p, i)
        parser.writer.add_scalar('dev/verb/r', dev_verb_r, i)
        parser.writer.add_scalar('dev/verb/f1', dev_verb_f1, i)
        parser.writer.add_scalar('dev/role/p', dev_role_p, i)
        parser.writer.add_scalar('dev/role/r', dev_role_r, i)
        parser.writer.add_scalar('dev/role/f1', dev_role_f1, i)
        # parser.writer.add_scalar('dev/noun/p', dev_noun_p, i)
        # parser.writer.add_scalar('dev/noun/r', dev_noun_r, i)
        # parser.writer.add_scalar('dev/noun/f1', dev_noun_f1, i)
        parser.writer.add_scalar('dev/triple/p', dev_triple_p, i)
        parser.writer.add_scalar('dev/triple/r', dev_triple_r, i)
        parser.writer.add_scalar('dev/triple/f1', dev_triple_f1, i)
        # parser.writer.add_scalar('dev/noun/p_relaxed', dev_noun_p_relaxed, i)
        # parser.writer.add_scalar('dev/noun/r_relaxed', dev_noun_r_relaxed, i)
        # parser.writer.add_scalar('dev/noun/f1_relaxed', dev_noun_f1_relaxed, i)
        parser.writer.add_scalar('dev/triple/p_relaxed', dev_triple_p_relaxed, i)
        parser.writer.add_scalar('dev/triple/r_relaxed', dev_triple_r_relaxed, i)
        parser.writer.add_scalar('dev/triple/f1_relaxed', dev_triple_f1_relaxed, i)


        # Testing Phrase
        with torch.no_grad():
            test_loss, test_verb_p, test_verb_r, test_verb_f1, \
            test_role_p, test_role_r, test_role_f1, \
            test_noun_p, test_noun_r, test_noun_f1, \
            test_triple_p, test_triple_r, test_triple_f1, \
            test_noun_p_relaxed, test_noun_r_relaxed, test_noun_f1_relaxed, \
            test_triple_p_relaxed, test_triple_r_relaxed, test_triple_f1_relaxed = run_over_data_sr(
                data_iter=test_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(test_set) / parser.batch),
                tester=tester,
                hyps=model.hyperparams,
                device=model.device,
                maxnorm=parser.maxnorm,
                word_i2s=parser.word_i2s,
                label_i2s=parser.label_i2s,
                role_i2s=parser.role_i2s,
                verb_roles=verb_roles,
                load_object=parser.add_object,
                train_ace=parser.train_ace,
                visualize_path=visualize_path,
                save_output=os.path.join(parser.out,
                                      "test_epoch_%d.txt" % (
                                          i + 1))
            )
        print("\nEpoch", i + 1, " test loss: ", test_loss,
              "\ntest verb p: ", test_verb_p,
              " test verb r: ", test_verb_r,
              " test verb f1: ", test_verb_f1,
              "\ntest role p: ", test_role_p,
              " test role r: ", test_role_r,
              " test role f1: ", test_role_f1,
              "\ntest noun p: ", test_noun_p,
              " test noun r: ", test_noun_r,
              " test noun f1: ", test_noun_f1,
              "\ntest triple p: ", test_triple_p,
              " test triple r: ", test_triple_r,
              " test triple f1: ", test_triple_f1,
              "\ntest noun p relaxed: ", test_noun_p_relaxed,
              " test noun r relaxed: ", test_noun_r_relaxed,
              " test noun f1 relaxed: ", test_noun_f1_relaxed,
              "\ntest triple p relaxed: ", test_triple_p_relaxed,
              " test triple r relaxed: ", test_triple_r_relaxed,
              " test triple f1 relaxed: ", test_triple_f1_relaxed
              )
        parser.writer.add_scalar('test/loss', test_loss, i)
        parser.writer.add_scalar('test/verb/p', test_verb_p, i)
        parser.writer.add_scalar('test/verb/r', test_verb_r, i)
        parser.writer.add_scalar('test/verb/f1', test_verb_f1, i)
        parser.writer.add_scalar('test/role/p', test_role_p, i)
        parser.writer.add_scalar('test/role/r', test_role_r, i)
        parser.writer.add_scalar('test/role/f1', test_role_f1, i)
        # parser.writer.add_scalar('test/noun/p', test_noun_p, i)
        # parser.writer.add_scalar('test/noun/r', test_noun_r, i)
        # parser.writer.add_scalar('test/noun/f1', test_noun_f1, i)
        parser.writer.add_scalar('test/triple/p', test_triple_p, i)
        parser.writer.add_scalar('test/triple/r', test_triple_r, i)
        parser.writer.add_scalar('test/triple/f1', test_triple_f1, i)
        # parser.writer.add_scalar('test/noun/p_relaxed', test_noun_p_relaxed, i)
        # parser.writer.add_scalar('test/noun/r_relaxed', test_noun_r_relaxed, i)
        # parser.writer.add_scalar('test/noun/f1_relaxed', test_noun_f1_relaxed, i)
        parser.writer.add_scalar('test/triple/p_relaxed', test_triple_p_relaxed, i)
        parser.writer.add_scalar('test/triple/r_relaxed', test_triple_r_relaxed, i)
        parser.writer.add_scalar('test/triple/f1_relaxed', test_triple_f1_relaxed, i)

        # Early Stop
        if scores <= dev_verb_f1 + dev_triple_f1:
            scores = dev_verb_f1 + dev_triple_f1
            # Move model parameters to CPU
            model.save_model(os.path.join(parser.out, "model.pt"))
            print("Save CPU model at Epoch", i + 1)
            now_bad = 0
        else:
            now_bad += 1
            if now_bad >= parser.earlystop:
                if restart_used >= parser.restart:
                    print("Restart opportunity are run out")
                    break
                restart_used += 1
                print("lr decays and best model is reloaded")
                lr = lr * 0.1
                model.load_model(os.path.join(parser.out, "model.pt"))
                optimizer = optimizer_constructor(lr=lr)
                print("Restart in Epoch %d" % (i + 2))
                now_bad = 0

        # save best test model
        if testscores <= test_verb_f1 + test_triple_f1:
            testscores = test_verb_f1 + test_triple_f1
            # Move model parameters to CPU
            model.save_model(os.path.join(parser.out, "model_test.pt"))
            print("Save test best CPU model at Epoch", i + 1)

    # Testing Phrase
    with torch.no_grad():
        test_loss, test_verb_p, test_verb_r, test_verb_f1, \
        test_role_p, test_role_r, test_role_f1, \
        test_noun_p, test_noun_r, test_noun_f1, \
        test_triple_p, test_triple_r, test_triple_f1, \
        test_noun_p_relaxed, test_noun_r_relaxed, test_noun_f1_relaxed, \
        test_triple_p_relaxed, test_triple_r_relaxed, test_triple_f1_relaxed = run_over_data_sr(
            data_iter=test_iter,
            optimizer=optimizer,
            model=model,
            need_backward=False,
            MAX_STEP=ceil(len(test_set) / parser.batch),
            tester=tester,
            hyps=model.hyperparams,
            device=model.device,
            maxnorm=parser.maxnorm,
            word_i2s=parser.word_i2s,
            label_i2s=parser.label_i2s,
            role_i2s=parser.role_i2s,
            verb_roles=verb_roles,
            load_object=parser.add_object,
            train_ace=parser.train_ace,
            visualize_path=visualize_path,
            save_output=os.path.join(parser.out, "test_final.txt")
        )
    print("\nFinally test loss: ", test_loss,
          "\ntest verb p: ", test_verb_p,
          " test verb r: ", test_verb_r,
          " test verb f1: ", test_verb_f1,
          "\ntest role p: ", test_role_p,
          " test role r: ", test_role_r,
          " test role f1: ", test_role_f1,
          "\ntest noun p: ", test_noun_p,
          " test noun r: ", test_noun_r,
          " test noun f1: ", test_noun_f1,
          "\ntest triple p: ", test_triple_p,
          " test triple r: ", test_triple_r,
          " test triple f1: ", test_triple_f1,
          "\ntest noun p relaxed: ", test_noun_p_relaxed,
          " test noun r relaxed: ", test_noun_r_relaxed,
          " test noun f1 relaxed: ", test_noun_f1_relaxed,
          "\ntest triple p relaxed: ", test_triple_p_relaxed,
          " test triple r relaxed: ", test_triple_r_relaxed,
          " test triple f1 relaxed: ", test_triple_f1_relaxed
          )

    print("Training Done!")
