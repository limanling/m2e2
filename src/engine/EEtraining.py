import os
from math import ceil

import torch
from torchtext.data import BucketIterator

# import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util.util_model import progressbar
from src.dataflow.torch.Sentence import Token


def batch_process_ee(batch, all_events, all_events_, all_y, all_y_, all_tokens,
                     hyps, word_i2s, label_i2s, role_i2s, weight, arg_weight,
                     model, tester, device, need_backward, role_mask):
    words, x_len = batch.WORDS
    postags = batch.POSTAGS
    entitylabels = batch.ENTITYLABELS  # entity type
    adjm = batch.ADJM
    y = batch.LABEL  # event type
    entities = batch.ENTITIES
    events = batch.EVENT
    all_events.extend(events)

    SEQ_LEN = words.size()[1]

    adjm = torch.stack([torch.sparse.FloatTensor(torch.LongTensor(adjmm[0]),
                                                 torch.FloatTensor(adjmm[1]),
                                                 torch.Size([hyps["gcn_et"], SEQ_LEN, SEQ_LEN])).to_dense() for
                        adjmm in adjm])
    words = words.to(device)
    # lemmas = lemmas.to(device)
    x_len = x_len.detach().numpy()
    # x_len = x_len.to(device)
    postags = postags.to(device)
    adjm = adjm.to(device)
    y = y.to(device)

    # print('x_len', x_len)

    if need_backward:
        y_, mask, ae_logits, ae_logits_key = model(words, x_len, postags, entitylabels, adjm, entities, y,
                                                       label_i2s)
    else:
        y_, mask, ae_logits, ae_logits_key = model.predict(words, x_len, postags, entitylabels, adjm, entities, y,
                                                   label_i2s)
    loss_ed = model.calculate_loss_ed(y_, mask, y, weight)
    if len(ae_logits_key) > 0:
        loss_ae, predicted_events = model.calculate_loss_ae(ae_logits, ae_logits_key, events, len(x_len), arg_weight, role_mask)
        loss = loss_ed + hyps["loss_alpha"] * loss_ae
    else:
        loss = loss_ed
        predicted_events = [{} for _ in range(len(x_len))]
    all_events_.extend(predicted_events)

    y__ = torch.max(y_, 2)[1].view(y.size()).tolist()
    y = y.tolist()

    add_tokens(words, y, y__, x_len, all_tokens, word_i2s, label_i2s)

    # unpad
    for i, ll in enumerate(x_len):
        y[i] = y[i][:ll]
        y__[i] = y__[i][:ll]
    bp, br, bf = tester.calculate_report(y, y__, transform=True)
    all_y.extend(y)
    all_y_.extend(y__)

    return loss, bp, br, bf

def run_over_batch_ee(batch, running_loss, cnt, all_events, all_events_, all_y, all_y_, all_tokens,
                   model, optimizer, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                  role_i2s, maxnorm, weight, arg_weight, role_mask):
    if need_backward:
        optimizer.zero_grad()

    loss, bp, br, bf = batch_process_ee(batch, all_events, all_events_, all_y, all_y_, all_tokens,
                     hyps, word_i2s, label_i2s, role_i2s, weight, arg_weight,
                     model, tester, device, need_backward, role_mask)

    cnt += 1
    other_information = ""

    if need_backward:
        loss.backward()
        if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)

        optimizer.step()
        other_information = 'EE Iter[{}] loss: {:.6f} edP: {:.4f}% edR: {:.4f}% edF1: {:.4f}%'.format(cnt, loss.item(),
                                                                                                   bp * 100.0,
                                                                                                   br * 100.0,
                                                                                                   bf * 100.0)
    progressbar(cnt, MAX_STEP, other_information)
    running_loss += loss.item()

    return running_loss, cnt, all_events, all_events_, all_y, all_y_, all_tokens

def add_tokens(words, y, y_, x_len, all_tokens, word_i2s, label_i2s):
    words = words.tolist()
    for s, ys, ys_, sl in zip(words, y, y_, x_len):
        s = s[:sl]
        ys = ys[:sl]
        ys_ = ys_[:sl]
        tokens = []
        for w, yw, yw_ in zip(s, ys, ys_):
            atoken = Token(word=word_i2s[w], posLabel="", entityLabel="", triggerLabel=label_i2s[yw])
            atoken.addPredictedLabel(label_i2s[yw_])
            tokens.append(atoken)
        all_tokens.append(tokens)


def run_over_data(model, optimizer, data_iter, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                  role_i2s, maxnorm, weight, arg_weight, role_mask, save_output):
    if need_backward:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    print()

    all_tokens = []
    all_y = []
    all_y_ = []
    all_events = []
    all_events_ = []

    cnt = 0
    # print(data_iter)
    for batch in data_iter:
        running_loss, cnt, all_events, all_events_, all_y, all_y_, all_tokens = \
            run_over_batch_ee(batch, running_loss, cnt, all_events, all_events_, all_y, all_y_, all_tokens,
                model, optimizer, MAX_STEP, need_backward, tester, hyps, device, word_i2s, label_i2s,
                role_i2s, maxnorm, weight, arg_weight, role_mask)

    if save_output:
        with open(save_output, "w", encoding="utf-8") as f:
            for tokens in all_tokens:
                for token in tokens:
                    # to match conll2000 format
                    f.write("%s %s %s\n" % (token.word, token.triggerLabel, token.predictedLabel))
                f.write("\n")

    running_loss = running_loss / cnt
    ep, er, ef = tester.calculate_report(all_y, all_y_, transform=False)
    ap, ar, af = tester.calculate_sets(all_events, all_events_)
    print()
    return running_loss, ep, er, ef, ap, ar, af




def ee_train(model, train_set, dev_set, test_set, optimizer_constructor, epochs,
          tester, parser, other_testsets, role_mask):


    # build batch on cpu
    train_iter = BucketIterator(train_set, batch_size=parser.batch,
                                train=True, shuffle=parser.shuffle, device=-1,
                                sort_key=lambda x: len(x.POSTAGS))
    dev_iter = BucketIterator(dev_set, batch_size=parser.batch, train=False,
                              shuffle=parser.shuffle, device=-1,
                              sort_key=lambda x: len(x.POSTAGS))
    test_iter = BucketIterator(test_set, batch_size=parser.batch, train=False,
                               shuffle=parser.shuffle, device=-1,
                               sort_key=lambda x: len(x.POSTAGS))

    scores = 0.0
    testscores = 0.0
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)

    for i in range(epochs):
        # Training Phrase
        print("Epoch", i + 1)
        training_loss, training_ed_p, training_ed_r, training_ed_f1, \
        training_ae_p, training_ae_r, training_ae_f1 = run_over_data(
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
            weight=parser.label_weight,
            arg_weight=parser.arg_weight,
            role_mask=role_mask,
            save_output=os.path.join(parser.out,
                                     "training_epoch_%d.txt" % (
                                             i + 1)))
        print("\nEpoch", i + 1, " training loss: ", training_loss,
              "\ntraining ed p: ", training_ed_p,
              " training ed r: ", training_ed_r,
              " training ed f1: ", training_ed_f1,
              "\ntraining ae p: ", training_ae_p,
              " training ae r: ", training_ae_r,
              " training ae f1: ", training_ae_f1)
        parser.writer.add_scalar('train/loss', training_loss, i)
        parser.writer.add_scalar('train/ed/p', training_ed_p, i)
        parser.writer.add_scalar('train/ed/r', training_ed_r, i)
        parser.writer.add_scalar('train/ed/f1', training_ed_f1, i)
        parser.writer.add_scalar('train/ae/p', training_ae_p, i)
        parser.writer.add_scalar('train/ae/r', training_ae_r, i)
        parser.writer.add_scalar('train/ae/f1', training_ae_f1, i)

        # Validation Phrase
        with torch.no_grad():
            dev_loss, dev_ed_p, dev_ed_r, dev_ed_f1, \
            dev_ae_p, dev_ae_r, dev_ae_f1 = run_over_data(data_iter=dev_iter,
                                                          optimizer=optimizer,
                                                          model=model,
                                                          need_backward=False,
                                                          MAX_STEP=ceil(len(
                                                              dev_set) /
                                                                        parser.batch),
                                                          tester=tester,
                                                          hyps=model.hyperparams,
                                                          device=model.device,
                                                          maxnorm=parser.maxnorm,
                                                          word_i2s=parser.word_i2s,
                                                          label_i2s=parser.label_i2s,
                                                          role_i2s=parser.role_i2s,
                                                          weight=parser.label_weight,
                                                          arg_weight=parser.arg_weight,
                                                          role_mask=role_mask,
                                                          save_output=os.path.join(
                                                              parser.out,
                                                              "dev_epoch_%d.txt" % (
                                                                      i + 1)))
        print("\nEpoch", i + 1, " dev loss: ", dev_loss,
              "\ndev ed p: ", dev_ed_p,
              " dev ed r: ", dev_ed_r,
              " dev ed f1: ", dev_ed_f1,
              "\ndev ae p: ", dev_ae_p,
              " dev ae r: ", dev_ae_r,
              " dev ae f1: ", dev_ae_f1)
        parser.writer.add_scalar('dev/loss', dev_loss, i)
        parser.writer.add_scalar('dev/ed/p', dev_ed_p, i)
        parser.writer.add_scalar('dev/ed/r', dev_ed_r, i)
        parser.writer.add_scalar('dev/ed/f1', dev_ed_f1, i)
        parser.writer.add_scalar('dev/ae/p', dev_ae_p, i)
        parser.writer.add_scalar('dev/ae/r', dev_ae_r, i)
        parser.writer.add_scalar('dev/ae/f1', dev_ae_f1, i)

        # Testing Phrase
        with torch.no_grad():
            test_loss, test_ed_p, test_ed_r, test_ed_f1, \
            test_ae_p, test_ae_r, test_ae_f1 = run_over_data(
                data_iter=test_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(
                    test_set) /
                              parser.batch),
                tester=tester,
                hyps=model.hyperparams,
                device=model.device,
                maxnorm=parser.maxnorm,
                word_i2s=parser.word_i2s,
                label_i2s=parser.label_i2s,
                role_i2s=parser.role_i2s,
                weight=parser.label_weight,
                arg_weight=parser.arg_weight,
                role_mask=role_mask,
                save_output=os.path.join(
                    parser.out,
                    "test_epoch_%d.txt" % (
                            i + 1)))
        print("\nEpoch", i + 1, " test loss: ", test_loss,
              "\ntest ed p: ", test_ed_p,
              " test ed r: ", test_ed_r,
              " test ed f1: ", test_ed_f1,
              "\ntest ae p: ", test_ae_p,
              " test ae r: ", test_ae_r,
              " test ae f1: ", test_ae_f1)
        parser.writer.add_scalar('test/loss', test_loss, i)
        parser.writer.add_scalar('test/ed/p', test_ed_p, i)
        parser.writer.add_scalar('test/ed/r', test_ed_r, i)
        parser.writer.add_scalar('test/ed/f1', test_ed_f1, i)
        parser.writer.add_scalar('test/ae/p', test_ae_p, i)
        parser.writer.add_scalar('test/ae/r', test_ae_r, i)
        parser.writer.add_scalar('test/ae/f1', test_ae_f1, i)

        # Early Stop
        if scores <= dev_ed_f1 + dev_ae_f1:
            scores = dev_ed_f1 + dev_ae_f1
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
        if testscores <= test_ed_f1 + test_ae_f1:
            testscores = test_ed_f1 + test_ae_f1
            # Move model parameters to CPU
            model.save_model(os.path.join(parser.out, "model_test.pt"))
            print("Save test best CPU model at Epoch", i + 1)

    # Testing Phrase
    test_loss, test_ed_p, test_ed_r, test_ed_f1, \
    test_ae_p, test_ae_r, test_ae_f1 = run_over_data(data_iter=test_iter,
                                                     optimizer=optimizer,
                                                     model=model,
                                                     need_backward=False,
                                                     MAX_STEP=ceil(len(
                                                         test_set) /
                                                                   parser.batch),
                                                     tester=tester,
                                                     hyps=model.hyperparams,
                                                     device=model.device,
                                                     maxnorm=parser.maxnorm,
                                                     word_i2s=parser.word_i2s,
                                                     label_i2s=parser.label_i2s,
                                                     role_i2s=parser.role_i2s,
                                                     weight=parser.label_weight,
                                                     role_mask=role_mask,
                                                     save_output=os.path.join(
                                                         parser.out,
                                                         "test_final.txt"))
    print("\nFinally test loss: ", test_loss,
          "\ntest ed p: ", test_ed_p,
          " test ed r: ", test_ed_r,
          " test ed f1: ", test_ed_f1,
          "\ntest ae p: ", test_ae_p,
          " test ae r: ", test_ae_r,
          " test ae f1: ", test_ae_f1)

    for name, additional_test_set in other_testsets.items():
        additional_test_iter = BucketIterator(additional_test_set,
                                              batch_size=parser.batch,
                                              train=False, shuffle=True,
                                              device=-1,
                                              sort_key=lambda x: len(
                                                  x.POSTAGS))

        additional_test_loss, additional_test_ed_p, additional_test_ed_r, \
        additional_test_ed_f1, \
        additional_test_ae_p, additional_test_ae_r, additional_test_ae_f1 = \
            run_over_data(
                data_iter=additional_test_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(additional_test_set) / parser.batch),
                tester=tester,
                hyps=model.hyperparams,
                device=model.device,
                maxnorm=parser.maxnorm,
                word_i2s=parser.word_i2s,
                label_i2s=parser.label_i2s,
                role_i2s=parser.role_i2s,
                weight=parser.label_weight,
                role_mask=role_mask,
                save_output=os.path.join(parser.out, "%s.txt") % (name))
        print("\nFor ", name, ", additional test loss: ", additional_test_loss,
              " additional ed test p: ", additional_test_ed_p,
              " additional ed test r: ", additional_test_ed_r,
              " additional ed test f1: ", additional_test_ed_f1,
              " additional ae test p: ", additional_test_ae_p,
              " additional ae test r: ", additional_test_ae_r,
              " additional ae test f1: ", additional_test_ae_f1)

    print("Training Done!")
