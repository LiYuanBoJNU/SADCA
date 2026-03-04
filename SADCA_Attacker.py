import sys

import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import random
import time
import scipy.stats as st
import os
from collections import defaultdict

class Attacker():
    def __init__(self, model, img_attacker, txt_attacker, tokenizer, lam, dynamic_num):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker
        self.lam = lam
        self.dynamic_num = dynamic_num
        self.tokenizer = tokenizer

    # def save_img(self, img_name, norm_img):
    #     pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
    #     pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
    #     img_path = "./flickr30k-adv/"
    #     pil_img.save(img_path + img_name)

    def attack(self, imgs, txts, txt2img, neg_img_embeddings, neg_txt_embeddings, device='cpu', max_length=30, masks=None, **kwargs):
        start_time = time.time()
        with torch.no_grad():

            txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

            # origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            # img_supervisions = origin_img_output['image_feat'][txt2img]


        ali_imgs = self.img_attacker.img_pos(self.model, imgs, txt2img, device, txt_embeds=txt_supervisions)
        with torch.no_grad():
            ali_img_output = self.model.inference_image(self.img_attacker.normalization(ali_imgs))
            ali_img_supervisions = ali_img_output['image_feat'][txt2img]

        adv_imgs = imgs.clone()
        momentum = 0
        for i in range(self.dynamic_num):
            print(i)
            with torch.no_grad():
                adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
                adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=ali_img_supervisions,
                                                            adv_img_embeds=adv_img_supervisions,
                                                            txt_embeds=txt_supervisions,
                                                            neg_img_embeds=neg_img_embeddings)
            print(adv_txts)

            with torch.no_grad():
                adv_txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                             max_length=max_length, return_tensors="pt").to(device)
                adv_txts_output = self.model.inference_text(adv_txts_input)
                adv_txt_supervisions = adv_txts_output['text_feat']
            adv_imgs, momentum = self.img_attacker.txt_guided_attack(self.model, adv_imgs, txt2img, device, txt_embeds=txt_supervisions,
                                                           adv_txt_embeds=adv_txt_supervisions,
                                                           img_embeds=ali_img_supervisions,
                                                           neg_txt_embeds=neg_txt_embeddings,
                                                           clean_imgs=imgs,
                                                           ali_imgs=ali_imgs, momentum=momentum,
                                                           txts=txts, adv_txts=adv_txts, tokenizer=self.tokenizer)


        end_time = time.time()
        execuate_time = end_time - start_time
        print(f"The function execution time: {execuate_time} seconds")

        # print(adv_imgs.max(), adv_imgs.min())
        # print(txts)
        # print(adv_txts)
        # self.save_img('0.png',adv_imgs[0])
        # sys.exit()
        return adv_imgs, adv_txts, execuate_time




class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5, lam=0.2, input_trans_num=10,  max_length=30):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers
        self.lam = lam
        self.max_length = max_length
        self.input_trans_num = input_trans_num

        # transformation
        self.num_block = 3
        self.kernel = self.gkern()
        self.op = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180,
                   self.scale, self.add_noise, self.BSR]

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low=0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low=0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16 / 255, 16 / 255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        # return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)
        return torch.from_numpy(stack_kernel.astype(np.float32)).cuda()

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)


    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand / rand.sum() * length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)
    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips
    def BSR(self, x):
        dims = [2, 3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat(
            [torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips],
            dim=dims[0])

    def local_semantic_augmentation(self, x):
        B, C, H, W = x.shape
        local_img_transform = transforms.RandomResizedCrop(H, scale=(0.4, 0.8))

        x = local_img_transform(x)
        chosen = np.random.randint(0, high=len(self.op), dtype=np.int32)
        # x = self.op[chosen](x)
        augmentation_img = self.op[chosen](x)

        return augmentation_img

    def mix_texts_in_groups(self, texts, txt2img, method='concat', alpha=0.7):
        mixed_texts = []
        # Group text by image ID
        img2texts = defaultdict(list)
        for txt, img_id in zip(texts, txt2img):
            img2texts[img_id].append(txt)

        # The text inside each set of images is blended
        for group in img2texts.values():
            shuffled_group = group.copy()
            random.shuffle(shuffled_group)
            for original, shuffled in zip(group, shuffled_group):
                if method == 'concat':
                    mixed = original + ' ' + shuffled
                elif method == 'interpolate':
                    l = int(len(original) * alpha)
                    r = int(len(shuffled) * (1 - alpha))
                    mixed = original[:l] + shuffled[-r:]
                else:
                    raise ValueError("Unsupported method")
                mixed_texts.append(mixed)
        return mixed_texts

    def loss_func_contrast(self, adv_imgs_embeds, txts_embeds, txt2img, neg_txts_embeds):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        neg_sim_matrix = adv_imgs_embeds @ neg_txts_embeds.T
        it_labels = torch.zeros(neg_sim_matrix.shape).to(device)
        k = int(len(neg_txts_embeds) / len(adv_imgs_embeds))
        txt2img_neg = torch.arange(len(adv_imgs_embeds)).repeat_interleave(k)
        for i in range(len(txt2img_neg)):
            it_labels[txt2img_neg[i], i] = 1
        loss_IaTcneg = (neg_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos + self.lam * loss_IaTcneg

        return loss

    def loss_func_self(self, adv_imgs_embeds, imgs_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ imgs_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()

        loss = 0.2 * loss_IaTcpos
        # print(loss)
        return loss

    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img):
        device = adv_imgs_embeds.device

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        # print(it_labels)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        # print(it_labels)
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        # print(loss_IaTcpos)
        # sys.exit()
        return loss

    def txt_guided_attack(self, model, imgs, txt2img, device, txt_embeds=None, adv_txt_embeds=None, img_embeds=None, neg_txt_embeds=None,
                          clean_imgs=None, ali_imgs=None, momentum=None,
                          txts=None, adv_txts=None, tokenizer=None):

        model.eval()

        b, _, h, w = imgs.shape

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        input_trans_num = self.input_trans_num
        for step in range(self.steps):  # self.steps=10
            adv_imgs.requires_grad_()
            scaled_imgs = torch.cat([adv_imgs, ] + [self.local_semantic_augmentation(adv_imgs) for _ in range(input_trans_num - 1)], 0)

            with torch.no_grad():
                mix_txts = self.mix_texts_in_groups(txts, txt2img)
                mix_adv_txts = self.mix_texts_in_groups(adv_txts, txt2img)
                mix_txts_input = tokenizer(mix_txts, padding='max_length', truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(device)
                mix_txts_output = model.inference_text(mix_txts_input)
                mix_txt_supervisions = mix_txts_output['text_feat']
                txt_embeds = mix_txt_supervisions

                mix_adv_txts_input = tokenizer(mix_adv_txts, padding='max_length', truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(device)
                mix_adv_txts_output = model.inference_text(mix_adv_txts_input)
                mix_adv_txt_supervisions = mix_adv_txts_output['text_feat']
                adv_txt_embeds = mix_adv_txt_supervisions

            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))

            else:
                adv_imgs_output = model.inference_image(scaled_imgs)

            adv_imgs_embeds = adv_imgs_output['image_feat']

            model.zero_grad()
            with torch.enable_grad():
                loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                for i in range(int(len(scaled_imgs) / b)):
                    loss_item = self.loss_func_contrast(adv_imgs_embeds[i * b:i * b + b], adv_txt_embeds, txt2img, neg_txt_embeds) + \
                                self.loss_func_contrast(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img, neg_txt_embeds)
                    loss += loss_item
            loss.backward()
            print(step, "loss:", loss)
            grad = adv_imgs.grad

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = momentum * 1.0 + grad
            perturbation = self.step_size * momentum.sign()
            adv_imgs = adv_imgs.detach() + perturbation
            adv_imgs = torch.min(torch.max(adv_imgs, clean_imgs - self.eps), clean_imgs + self.eps)
            adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
        print("adv loss", loss)
        return adv_imgs.detach(), momentum

    def img_pos(self, model, imgs, txt2img, device, txt_embeds=None):
        model.eval()
        b, _, _, _ = imgs.shape

        col_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(device)
        col_imgs = torch.clamp(col_imgs, 0.0, 1.0)
        momentum = 0
        # for step in range(self.steps):  # self.steps=10
        for step in range(5):  # self.steps=10

            col_imgs.requires_grad_()

            if self.normalization is not None:
                col_imgs_output = model.inference_image(self.normalization(col_imgs))
            else:
                col_imgs_output = model.inference_image(col_imgs)

            col_imgs_embeds = col_imgs_output['image_feat']
            model.zero_grad()
            with torch.enable_grad():
                loss = - self.loss_func_old(col_imgs_embeds, txt_embeds, txt2img)
            loss.backward()
            # print("loss", loss)
            grad = col_imgs.grad

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            momentum = momentum * 1.0 + grad
            perturbation = self.step_size * momentum.sign()

            # perturbation = self.step_size * grad.sign()
            col_imgs = col_imgs.detach() + perturbation
            col_imgs = torch.min(torch.max(col_imgs, imgs - self.eps), imgs + self.eps)
            col_imgs = torch.clamp(col_imgs, 0.0, 1.0)
        print("col loss", loss)

        # print(f"The function execution time: {elapsed_time} seconds")
        return col_imgs.detach()

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2], lam=0.2):
        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.text_ratios = text_ratios
        self.lam = lam

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, txt_embeds=None, neg_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_feat'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_feat'].flatten(1)

                loss = self.loss_func(replace_embeds, adv_img_embeds, i, neg_img_embeds) + self.loss_func(replace_embeds, img_embeds, i, neg_img_embeds)

                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))
        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label, neg_img_embeds):

        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss_TaIcneg = 0
        for k in range(neg_img_embeds.shape[1]):
            loss_neg = txt_embeds.mul(neg_img_embeds[label][k].repeat(len(txt_embeds), 1)).sum(-1)
            loss_TaIcneg += loss_neg

        loss = loss_TaIcpos + self.lam * (loss_TaIcneg / neg_img_embeds.shape[1])

        return loss


    def loss_txt_func(self, txt_embeds, txt_embeds2):
        loss_TaIcpos = -txt_embeds.mul(txt_embeds2).sum(-1)
        loss = loss_TaIcpos
        return loss

    def attack(self, net, texts):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # substitutes
        mlm_logits = self.ref_net(text_inputs.input_ids, attention_mask=text_inputs.attention_mask).logits
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)  # seq-len k

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_embed'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='none')
        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            important_scores = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)

            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)

            words, sub_words, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)
            change = 0

            for top_index in list_of_index:
                if change >= self.num_perturbation:
                    break

                tgt_word = words[top_index[0]]
                if tgt_word in filter_words:
                    continue
                if keys[top_index[0]][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
                word_pred_scores = word_pred_scores_all[i, keys[top_index[0]][0]:keys[top_index[0]][1]]

                substitutes = get_substitues(substitutes, self.tokenizer, self.ref_net, 1, word_pred_scores,
                                             self.threshold_pred_score)

                replace_texts = [' '.join(final_words)]
                available_substitutes = [tgt_word]
                for substitute_ in substitutes:
                    substitute = substitute_

                    if substitute == tgt_word:
                        continue  # filter out original word
                    if '##' in substitute:
                        continue  # filter out sub-word

                    if substitute in filter_words:
                        continue
                    '''
                    # filter out atonyms
                    if substitute in w2i and tgt_word in w2i:
                        if cos_mat[w2i[substitute]][w2i[tgt_word]] < 0.4:
                            continue
                    '''
                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index[0]] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(' '.join(temp_replace))
                replace_text_input = self.tokenizer(replace_texts, padding='max_length', truncation=True,
                                                    max_length=self.max_length, return_tensors='pt').to(device)
                replace_output = net.inference_text(replace_text_input)
                if self.cls:
                    replace_embeds = replace_output['text_embed'][:, 0, :]
                else:
                    replace_embeds = replace_output['text_embed'].flatten(1)

                loss = criterion(replace_embeds.log_softmax(dim=-1),
                                 origin_embeds[i].softmax(dim=-1).repeat(len(replace_embeds), 1))

                loss = loss.sum(dim=-1)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words