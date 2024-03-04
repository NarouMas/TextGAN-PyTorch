import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.SeqGAN_D import SeqGAN_D
from models.SeqGAN_G import SeqGAN_G
from utils import rollout
from utils.data_loader import GenDataIter, DisDataIter
from utils.text_process import load_dict, write_tokens, tensor_to_tokens


# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : seqgan_instructor.py
# @Time         : Created at 2019-06-05
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.optim as optim

import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.SeqGAN_D import SeqGAN_D
from models.SeqGAN_G import SeqGAN_G
from utils import rollout
from utils.data_loader import GenDataIter, DisDataIter


class SeqGANInstructor2(BasicInstructor):
    def __init__(self, opt):
        super(SeqGANInstructor2, self).__init__(opt)

        # generator, discriminator
        self.gen = SeqGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, cfg.vocab_size, cfg.max_seq_len,
                            cfg.padding_idx, gpu=cfg.CUDA)
        self.dis = SeqGAN_D(cfg.dis_embed_dim, cfg.vocab_size, cfg.padding_idx, gpu=cfg.CUDA)
        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)
        self.opt = opt

    def _run(self):
        print("hello")
        # load model from file save\20240302\emnlp_news\seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl51_temp1_lfd0.0_T0302_2222_12\models\gen_MLE_00009.pt
        self.gen.load_state_dict(torch.load('save/20240302/emnlp_news/seqgan_vanilla_dt-Ra_lt-rsgan_mt-ra_et-Ra_sl51_temp1_lfd0.0_T0302_2222_12/models/gen_MLE_00009.pt'))
        neg_sample = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
        print("cft.samples_num", cfg.samples_num)
        print("cfg.batch_size", cfg.batch_size)
        print("neg_sample", neg_sample)
        inp, target = GenDataIter.prepare(neg_sample, gpu=cfg.CUDA)
        print("inp", inp)
        print("target", target)
        b_ins = BasicInstructor(self.opt)
        tokens = tensor_to_tokens(neg_sample, b_ins.idx2word_dict)
        print("tokens", tokens)

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

                # ===Test===
                if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
                    self.log.info(
                        '[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, self.cal_metrics(fmt_str=True)))
                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        rollout_func = rollout.ROLLOUT(self.gen, cfg.CUDA)
        total_g_loss = 0
        for step in range(g_step):
            inp, target = GenDataIter.prepare(self.gen.sample(cfg.batch_size, cfg.batch_size), gpu=cfg.CUDA)

            # ===Train===
            rewards = rollout_func.get_reward(target, cfg.rollout_num, self.dis)
            adv_loss = self.gen.batchPGLoss(inp, target, rewards)
            self.optimize(self.gen_adv_opt, adv_loss)
            total_g_loss += adv_loss.item()

        # ===Test===
        self.log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, self.cal_metrics(fmt_str=True)))

    def train_discriminator(self, d_step, d_epoch, phase='MLE'):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        # prepare loader for validate
        global d_loss, train_acc
        for step in range(d_step):
            # prepare loader for training
            pos_samples = self.train_data.target
            neg_samples = self.gen.sample(cfg.samples_num, 4 * cfg.batch_size)
            dis_data = DisDataIter(pos_samples, neg_samples)
            print('pos_samples', pos_samples)
            print('neg_samples', neg_samples)
            print('dis_data', dis_data)

            for epoch in range(d_epoch):
                # ===Train===
                d_loss, train_acc = self.train_dis_epoch(self.dis, dis_data.loader, self.dis_criterion,
                                                         self.dis_opt)

            # ===Test===
            self.log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (
                phase, step, d_loss, train_acc))

            if cfg.if_save and not cfg.if_test:
                torch.save(self.dis.state_dict(), cfg.pretrained_dis_path)


def main():
    inst = SeqGANInstructor2(cfg.run_cfg)

if __name__ == "__main__":
    main()
