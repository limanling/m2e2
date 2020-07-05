import os
from math import ceil

import torch
from torchtext.data import BucketIterator
from torch.nn import functional as F
from PIL import Image

import sys
#sys.path.append('/dvmm-filer2/users/manling/mm-event-graph2')
from src.util.util_model import progressbar
from src.dataflow.numpy.data_loader_grounding import unpack_grounding


def batch_process_grounding(batch_unpacked, all_captions, all_captions_, all_images, all_images_,
                            model, tester, device, add_object=False):
    words, x_len, postags, entitylabels, adjm, \
        image_id, image, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num_batch, \
        sent_id, entities = batch_unpacked
    emb_image, emb_sentence, word2noun_att_output, word_common, noun2word_att_output, noun_emb = \
        model(words, x_len, postags, entitylabels, adjm, image_id, image,
              add_object=add_object, bbox_entities_id=bbox_entities_id,
              bbox_entities_region=bbox_entities_region,
              bbox_entities_label=bbox_entities_label,
              object_num_batch=object_num_batch
              )

    loss_grounding, loss_terms, caption_logits, image_logits = model.calculate_loss_grounding(emb_sentence, emb_image, word2noun_att_output, word_common, noun2word_att_output, noun_emb)
    # image_logits = F.log_softmax(caption_logits.t())

    # negative sample: all other caption in the sentecne, compare each image with all sentence in the batch
    # caption_logits = torch.mm(emb_image, emb_sentence.t())
    # caption_logits = F.log_softmax(caption_logits)
    BATCH_SIZE = words.size()[0]
    y_caption_ = torch.max(caption_logits, dim=1)[1].tolist()  # [batch,]
    y_caption = list(range(BATCH_SIZE))
    all_captions_.extend(y_caption_)
    all_captions.extend(y_caption)
    bp_caption, br_caption, bf_caption = tester.calculate_lists(y_caption, y_caption_)

    y_image_ = torch.max(image_logits, dim=1)[1].tolist()  # [batch,]
    y_image = list(range(BATCH_SIZE))
    all_images_.extend(y_image_)
    all_images.extend(y_image)
    bp_image, br_image, bf_image = tester.calculate_lists(y_image, y_image_)

    return loss_grounding, loss_terms, bp_caption, br_caption, bf_caption, bp_image, br_image, bf_image


def run_over_batch_grounding(batch, running_loss, cnt, all_captions, all_captions_, all_images, all_images_,
                             model, optimizer, MAX_STEP, need_backward, tester, ee_hyps, device, maxnorm,
                             img_dir, transform, add_object=False,
                             object_results=None, object_label=None,
                             object_detection_threshold=.2, vocab_objlabel=None):

    try:
        # words, x_len, postags, entitylabels, adjm, image_id, image = unpack_grounding(batch, device, transform,
        #                                                                           img_dir, ee_hyps)
        batch_unpacked = unpack_grounding(batch, device, transform, img_dir, ee_hyps,
                                          load_object=add_object, object_results=object_results,
                                          object_label=object_label,
                                          object_detection_threshold=object_detection_threshold,
                                          vocab_objlabel=vocab_objlabel)
    except:
        # if the batch is a bad batch, Nothing changed, return directly
        return running_loss, cnt, all_captions, all_captions_, all_images, all_images_

    if batch_unpacked is None:
        # if the batch is a bad batch, Nothing changed, return directly
        return running_loss, cnt, all_captions, all_captions_, all_images, all_images_

    if need_backward:
        optimizer.zero_grad()

    loss_grounding, loss_items, bp_caption, br_caption, bf_caption, bp_image, br_image, bf_image \
        = batch_process_grounding(batch_unpacked, all_captions, all_captions_, all_images, all_images_,
                                model, tester, device,
                                add_object=add_object
                                  )

    cnt += 1
    other_information = ""

    if need_backward:
        loss_grounding.backward()
        if 1e-6 < maxnorm and model.parameters_requires_grad_clipping() is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters_requires_grad_clipping(), maxnorm)

        optimizer.step()
        other_information = 'Grounding Iter[{}] loss: {:.6f} loss_terms: {} \n' \
                            'capationP: {:.4f}% captionR: {:.4f}% captionF1: {:.4f}% \n' \
                            'imageP: {:.4f}% imageR: {:.4f}% imageF1: {:.4f}%'.format(cnt, loss_grounding.item(),
                                                                                      loss_items,
                                                                                      bp_caption * 100.0,
                                                                                      br_caption * 100.0,
                                                                                      bf_caption * 100.0,
                                                                                      bp_image * 100.0,
                                                                                      br_image * 100.0,
                                                                                      bf_image * 100.0)
    progressbar(cnt, MAX_STEP, other_information)
    running_loss += loss_grounding.item()

    return running_loss, cnt, all_captions, all_captions_, all_images, all_images_


def run_over_data_grounding(model, optimizer, data_iter, MAX_STEP, need_backward, tester, ee_hyps, device, maxnorm,
                            img_dir, transform, add_object=False,
                             object_results=None, object_label=None,
                             object_detection_threshold=.2, vocab_objlabel=None):
    if need_backward:
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    print()

    all_images = []
    all_images_ = []
    all_captions = []
    all_captions_ = []

    cnt = 0
    # print(data_iter)
    for batch in data_iter:
        running_loss, cnt, all_captions, all_captions_, all_images, all_images_ = \
            run_over_batch_grounding(batch, running_loss, cnt, all_captions, all_captions_, all_images, all_images_,
                                    model, optimizer, MAX_STEP, need_backward, tester, ee_hyps, device, maxnorm,
                                    img_dir, transform,
                                     add_object=add_object,
                                     object_results=object_results,
                                     object_label=object_label,
                                     object_detection_threshold=object_detection_threshold,
                                     vocab_objlabel=vocab_objlabel
                                     )

    running_loss = running_loss / cnt
    p_caption, r_caption, f_caption = tester.calculate_lists(all_captions, all_captions_)
    p_image, r_image, f_image = tester.calculate_lists(all_images, all_images_)
    print()
    return running_loss, p_caption, r_caption, f_caption, p_image, r_image, f_image


def grounding_train(model, train_set, dev_set, test_set, optimizer_constructor, epochs,
            tester, parser, other_testsets, transform, vocab_objlabel=None):
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
    now_bad = 0
    restart_used = 0
    print("\nStarting training...\n")
    lr = parser.lr
    optimizer = optimizer_constructor(lr=lr)

    object_results, object_label, object_detection_threshold = train_set.get_object_results()

    for i in range(epochs):
        # Training Phrase
        print("Epoch", i + 1)
        training_loss, training_caption_p, training_caption_r, training_caption_f1, \
        training_image_p, training_image_r, training_image_f1 = run_over_data_grounding(
            data_iter=train_iter,
            optimizer=optimizer,
            model=model,
            need_backward=True,
            MAX_STEP=ceil(len(train_set) / parser.batch),
            tester=tester,
            ee_hyps=parser.ee_hps,
            device=model.device,
            maxnorm=parser.maxnorm,
            img_dir=parser.img_dir,
            transform=transform,
            add_object=parser.add_object,
            object_results=object_results,
            object_label=object_label,
            object_detection_threshold=object_detection_threshold,
            vocab_objlabel=vocab_objlabel
            )
        # if training_loss is not None:
        print("\nEpoch", i + 1, " training loss: ", training_loss,
              "\ntraining caption p: ", training_caption_p,
              " training caption r: ", training_caption_r,
              " training caption f1: ", training_caption_f1,
              "\ntraining image p: ", training_image_p,
              " training image r: ", training_image_r,
              " training image f1: ", training_image_f1)
        parser.writer.add_scalar('train/loss', training_loss, i)
        parser.writer.add_scalar('train/caption/p', training_caption_p, i)
        parser.writer.add_scalar('train/caption/r', training_caption_r, i)
        parser.writer.add_scalar('train/caption/f1', training_caption_f1, i)
        parser.writer.add_scalar('train/image/p', training_image_p, i)
        parser.writer.add_scalar('train/image/r', training_image_r, i)
        parser.writer.add_scalar('train/image/f1', training_image_f1, i)

        # Validation Phrase
        with torch.no_grad():
            dev_loss, dev_caption_p, dev_caption_r, dev_caption_f1, \
            dev_image_p, dev_image_r, dev_image_f1 = run_over_data_grounding(
                data_iter=dev_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(dev_set) / parser.batch),
                tester=tester,
                ee_hyps=parser.ee_hps,
                device=model.device,
                maxnorm=parser.maxnorm,
                img_dir=parser.img_dir,
                transform=transform,
                add_object=parser.add_object,
                object_results=object_results,
                object_label=object_label,
                object_detection_threshold=object_detection_threshold,
                vocab_objlabel=vocab_objlabel
            )
        # if dev_loss is not None:
        print("\nEpoch", i + 1, " dev loss: ", dev_loss,
              "\ndev caption p: ", dev_caption_p,
              " dev caption r: ", dev_caption_r,
              " dev caption f1: ", dev_caption_f1,
              "\ndev image p: ", dev_image_p,
              " dev image r: ", dev_image_r,
              " dev image f1: ", dev_image_f1)
        parser.writer.add_scalar('dev/loss', dev_loss, i)
        parser.writer.add_scalar('dev/caption/p', dev_caption_p, i)
        parser.writer.add_scalar('dev/caption/r', dev_caption_r, i)
        parser.writer.add_scalar('dev/caption/f1', dev_caption_f1, i)
        parser.writer.add_scalar('dev/image/p', dev_image_p, i)
        parser.writer.add_scalar('dev/image/r', dev_image_r, i)
        parser.writer.add_scalar('dev/image/f1', dev_image_f1, i)

        # Testing Phrase
        with torch.no_grad():
            test_loss, test_caption_p, test_caption_r, test_caption_f1, \
            test_image_p, test_image_r, test_image_f1 = run_over_data_grounding(
                data_iter=test_iter,
                optimizer=optimizer,
                model=model,
                need_backward=False,
                MAX_STEP=ceil(len(test_set) / parser.batch),
                tester=tester,
                ee_hyps=parser.ee_hps,
                device=model.device,
                maxnorm=parser.maxnorm,
                img_dir=parser.img_dir,
                transform=transform,
                add_object=parser.add_object,
                object_results=object_results,
                object_label=object_label,
                object_detection_threshold=object_detection_threshold,
                vocab_objlabel=vocab_objlabel
                )
        # if test_loss is not None:
        print("\nEpoch", i + 1, " test loss: ", test_loss,
              "\ntest caption p: ", test_caption_p,
              " test caption r: ", test_caption_r,
              " test caption f1: ", test_caption_f1,
              "\ntest image p: ", test_image_p,
              " test image r: ", test_image_r,
              " test image f1: ", test_image_f1)
        parser.writer.add_scalar('test/loss', test_loss, i)
        parser.writer.add_scalar('test/caption/p', test_caption_p, i)
        parser.writer.add_scalar('test/caption/r', test_caption_r, i)
        parser.writer.add_scalar('test/caption/f1', test_caption_f1, i)
        parser.writer.add_scalar('test/image/p', test_image_p, i)
        parser.writer.add_scalar('test/image/r', test_image_r, i)
        parser.writer.add_scalar('test/image/f1', test_image_f1, i)

        # Early Stop
        if scores <= dev_caption_f1 + dev_image_f1:
            scores = dev_caption_f1 + dev_image_f1
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

    # Testing Phrase
    test_loss, test_caption_p, test_caption_r, test_caption_f1, \
    test_image_p, test_image_r, test_image_f1 = run_over_data_grounding(
        data_iter=test_iter,
        optimizer=optimizer,
        model=model,
        need_backward=False,
        MAX_STEP=ceil(len(test_set) / parser.batch),
        tester=tester,
        ee_hyps=parser.ee_hps,
        device=model.device,
        maxnorm=parser.maxnorm,
        img_dir=parser.img_dir,
        transform=transform,
        add_object=parser.add_object,
        object_results=object_results,
        object_label=object_label,
        object_detection_threshold=object_detection_threshold,
        vocab_objlabel=vocab_objlabel
    )
    # if test_loss is not None:
    print("\nFinally test loss: ", test_loss,
          "\ntest caption p: ", test_caption_p,
          " test caption r: ", test_caption_r,
          " test caption f1: ", test_caption_f1,
          "\ntest image p: ", test_image_p,
          " test image r: ", test_image_r,
          " test image f1: ", test_image_f1)

    print("Training Done!")
