import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import random
import time
import scipy.stats as st
from collections import defaultdict

class Attacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, all_txt_supervisions,device='cpu', max_length=30, scales=None, masks=None, **kwargs):
        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions)

        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(adv_txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']

            # all_texts_input = self.txt_attacker.tokenizer(all_texts, padding='max_length', truncation=True,
            #                                          max_length=max_length, return_tensors="pt").to(device)
            # all_texts_output = self.model.inference_text(all_texts_input)
            # all_txt_supervisions = all_texts_output['text_feat']


        start_time = time.time()
        adv_imgs, last_adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img,all_txt_supervisions, device,
                                                                      scales=scales, txt_embeds=txt_supervisions)
        end_time = time.time()
        execuate_time = end_time - start_time

        with torch.no_grad():
            adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_imgs_outputs['image_feat'][txt2img]
            last_adv_imgs_outputs = self.model.inference_image(self.img_attacker.normalization(last_adv_imgs))
            last_adv_img_supervisions = last_adv_imgs_outputs['image_feat'][txt2img]
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
                                                       adv_img_embeds=adv_img_supervisions,
                                                       last_adv_img_embeds=last_adv_img_supervisions)
        return adv_imgs, adv_txts, execuate_time


class ImageAttacker():
    def __init__(self, normalization, eps=2 / 255, steps=10, step_size=0.5 / 255, sample_numbers=5):
        self.normalization = normalization
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.sample_numbers = sample_numbers

        # transformation
        self.num_block = 3
        self.kernel = self.gkern()
        self.op = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180,
                   self.scale, self.add_noise]

        self.op_SA = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180,
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

    def SIA(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0, ] + np.random.choice(list(range(1, h)), self.num_block - 1, replace=False).tolist() + [h, ]
        x_axis = [0, ] + np.random.choice(list(range(1, w)), self.num_block - 1, replace=False).tolist() + [w, ]
        y_axis.sort()
        x_axis.sort()

        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](
                    x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

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

        img2texts = defaultdict(list)
        for txt, img_id in zip(texts, txt2img):
            img2texts[img_id].append(txt)

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


    def loss_func(self, adv_imgs_embeds, txts_embeds, txt2img,all_txt_supervisions):
        device = adv_imgs_embeds.device

        U, S, V = torch.svd(all_txt_supervisions.T.to(torch.float32))
        # projection_matrix = U[:, :30] @ U[:, :30].t()
        projection_matrix = U[:, 1:len(U)] @ U[:, 1:len(U)].t()
        # projection_matrix = U @ U.t()
        # print("txts_embeds.shape",txts_embeds.shape)
        # print("len(txt2img)",len(txt2img))
        # print("projection_matrix.shape",projection_matrix.shape)
        # print(a)

        adv_imgs_embeds = adv_imgs_embeds @ projection_matrix
        txts_embeds = txts_embeds @ projection_matrix

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)

        for i in range(len(txt2img)):
            it_labels[txt2img[i], i] = 1
        # print(it_labels)
        # print(a)
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos

        return loss
    
    def loss_func_old(self, adv_imgs_embeds, txts_embeds, txt2img):  
        device = adv_imgs_embeds.device    

        it_sim_matrix = adv_imgs_embeds @ txts_embeds.T
        it_labels = torch.zeros(it_sim_matrix.shape).to(device)
        
        for i in range(len(txt2img)):
            it_labels[txt2img[i], i]=1
        
        loss_IaTcpos = -(it_sim_matrix * it_labels).sum(-1).mean()
        loss = loss_IaTcpos
        
        return loss

    def rand3Num(self): ### num1 -> adv num2-> clean num3->last
        while True:
            num1 = random.randint(1, 100)
            if 100 - num1 > 1:
                num2 = random.randint(1, 100 - num1)
            else:
                num1 = 98
                num2 = 1
            num3 = 100 - num1 - num2
            
            if 1 <= num3 <= 100 and num1 < num3 and num3 < num2:
                break

        return (num1, num2, num3)

    def txt_guided_attack(self, model, imgs, txt2img, all_txt_supervisions,device, scales=None, txt_embeds=None):

        model.eval()

        b, _, _, _ = imgs.shape

        if scales is None:
            scales_num = 1
        else:
            scales_num = len(scales) + 1

        adv_imgs = imgs.detach() + torch.from_numpy(np.random.uniform(-self.eps, self.eps, imgs.shape)).float().to(
            device)
        adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        last_adv_imgs = None

        start_time = time.time()
        ratio_list = []
        momentum = 0
        for step in range(self.steps):  # self.steps=10
            if last_adv_imgs != None:
                samples = []
                clone_adv_imgs = adv_imgs.clone()
                loss_list = []
                for k in range(self.sample_numbers):
                    samples.append(self.rand3Num())
                for sample in samples:
                    adv_imgs = (sample[0] / 100) * clone_adv_imgs + (sample[1] / 100) * imgs + (
                                sample[2] / 100) * last_adv_imgs
                    adv_imgs.requires_grad_()

                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)

                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img,all_txt_supervisions)
                    adv_imgs.retain_grad()
                    loss.backward()
                    grad = adv_imgs.grad
                    grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                    perturbation = self.step_size * grad.sign()

                    adv_imgs = clone_adv_imgs.detach() + perturbation
                    adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                    adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)



                    if self.normalization is not None:
                        adv_imgs_output = model.inference_image(self.normalization(adv_imgs))
                    else:
                        adv_imgs_output = model.inference_image(adv_imgs)
                    adv_imgs_embeds = adv_imgs_output['image_feat']
                    model.zero_grad()
                    with torch.enable_grad():
                        loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                        loss = self.loss_func(adv_imgs_embeds, txt_embeds, txt2img,all_txt_supervisions)
                    loss.backward()
                    loss_list.append(loss.item())
                #candidate_index = loss_list.index(max(loss_list))

                candidate_index = loss_list.index(max(loss_list))
                ratio_list.append(samples[candidate_index])

                adv_imgs = (samples[candidate_index][0] / 100) * clone_adv_imgs + (
                            samples[candidate_index][1] / 100) * imgs + (
                                       samples[candidate_index][2] / 100) * last_adv_imgs
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)
                # scaled_imgs = torch.cat([self.SIA(adv_imgs) for _ in range(10)])  # SIA
                # scaled_imgs = torch.cat([adv_imgs, ] + [self.local_semantic_augmentation(adv_imgs) for _ in range(10 - 1)], 0) # SA

                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img,all_txt_supervisions)
                        loss += loss_item
                adv_imgs.retain_grad()
                print("loss", loss)
                loss.backward()

                grad = adv_imgs.grad
                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()

                adv_imgs = clone_adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)
                last_adv_imgs = clone_adv_imgs.clone()
            else:
                last_adv_imgs = adv_imgs.clone()
                adv_imgs.requires_grad_()
                scaled_imgs = self.get_scaled_imgs(adv_imgs, [0.5, 0.75, 1.25, 1.5], device)
                # scaled_imgs = torch.cat([self.SIA(adv_imgs) for _ in range(10)])  # SIA
                # scaled_imgs = torch.cat([adv_imgs, ] + [self.local_semantic_augmentation(adv_imgs) for _ in range(10 - 1)], 0)  # SA

                if self.normalization is not None:
                    adv_imgs_output = model.inference_image(self.normalization(scaled_imgs))
                else:
                    adv_imgs_output = model.inference_image(scaled_imgs)

                adv_imgs_embeds = adv_imgs_output['image_feat']
                model.zero_grad()
                with torch.enable_grad():
                    loss = torch.tensor(0.0, dtype=torch.float32).to(device)
                    for i in range(5):
                        loss_item = self.loss_func(adv_imgs_embeds[i * b:i * b + b], txt_embeds, txt2img,all_txt_supervisions)
                        loss += loss_item
                loss.backward()
                print("loss",loss)
                grad = adv_imgs.grad

                grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
                perturbation = self.step_size * grad.sign()

                adv_imgs = adv_imgs.detach() + perturbation
                adv_imgs = torch.min(torch.max(adv_imgs, imgs - self.eps), imgs + self.eps)
                adv_imgs = torch.clamp(adv_imgs, 0.0, 1.0)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"The function execution time: {elapsed_time} seconds")

        return adv_imgs, last_adv_imgs

    def save_img(self, img_name, norm_img):
        pil_array = (norm_img * 255).to(torch.uint8).cpu().numpy()
        pil_img = Image.fromarray(np.transpose(pil_array, (1, 2, 0)))
        img_path = "./mscoco_imgs/"
        pil_img.save(img_path + img_name)


    def get_scaled_imgs(self, imgs, scales=None, device='cuda'):
        if scales is None:
            return imgs

        ori_shape = (imgs.shape[-2], imgs.shape[-1])

        reverse_transform = transforms.Resize(ori_shape,
                                              interpolation=transforms.InterpolationMode.BICUBIC)
        result = []
        for ratio in scales:
            scale_shape = (int(ratio * ori_shape[0]),
                           int(ratio * ori_shape[1]))
            scale_transform = transforms.Resize(scale_shape,
                                                interpolation=transforms.InterpolationMode.BICUBIC)
            scaled_imgs = imgs + torch.from_numpy(np.random.normal(0.0, 0.05, imgs.shape)).float().to(device)
            scaled_imgs = scale_transform(scaled_imgs)
            scaled_imgs = torch.clamp(scaled_imgs, 0.0, 1.0)

            reversed_imgs = reverse_transform(scaled_imgs)

            result.append(reversed_imgs)

        return torch.cat([imgs, ] + result, 0)


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
                 threshold_pred_score=0.3, batch_size=32, text_ratios=[0.6, 0.2, 0.2]):
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

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None, last_adv_img_embeds=None):
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

                # print(replace_embeds.size(), img_embeds.size(), img_embeds[i].size(), i)
                if adv_img_embeds == None:
                    loss = self.loss_func(replace_embeds, img_embeds, i)
                else:
                    loss = self.text_ratios[0] * self.loss_func(replace_embeds, img_embeds, i) + self.text_ratios[
                        1] * self.loss_func(replace_embeds, adv_img_embeds, i) + self.text_ratios[2] * self.loss_func(
                        replace_embeds, last_adv_img_embeds, i)
                candidate_idx = loss.argmax()

                final_words[top_index[0]] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != tgt_word:
                    change += 1

            final_adverse.append(' '.join(final_words))

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
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