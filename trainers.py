import time

import numpy as np
from pathlib import Path

from logger import Logger

import torch
import torch.optim as optim
from torch import nn

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch


def images_to_numpy(tensor):
    generated = (torch.clamp(tensor, -1, 1) + 1) / 2 * 255
    generated = generated.data.cpu().numpy().transpose(0, 2, 3, 1)
    return generated.astype('uint8')


def videos_to_numpy(tensor):
    generated = (torch.clamp(tensor, -1, 1) + 1) / 2 * 255
    generated = generated.data.cpu().numpy()
    return generated.astype('uint8')

class Trainer(object):
    def __init__(self, image_sampler, video_sampler, 
                 log_interval, train_batches, log_folder, 
                 use_cuda=False, use_infogan=True, use_categories=True):

        self.use_categories = use_categories

        self.gan_criterion = nn.BCEWithLogitsLoss()
        self.category_criterion = nn.CrossEntropyLoss()

        self.image_sampler = image_sampler
        self.video_sampler = video_sampler

        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = log_interval
        self.train_batches = train_batches

        self.log_folder = Path(log_folder)

        self.use_cuda = use_cuda
        self.use_infogan = use_infogan

        self.image_enumerator = None
        self.video_enumerator = None

    @staticmethod
    def create_by_shape(tensor, val):
        return T.FloatTensor(tensor.size()).fill_(val) 

    def sample_real_image_batch(self):
        if self.image_enumerator is None:
            self.image_enumerator = enumerate(self.image_sampler)

        batch_idx, batch = next(self.image_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.items():
                b[k] = v.cuda()

        if batch_idx == len(self.image_sampler) - 1:
            self.image_enumerator = enumerate(self.image_sampler)

        return b

    def sample_real_video_batch(self):
        if self.video_enumerator is None:
            self.video_enumerator = enumerate(self.video_sampler)

        batch_idx, batch = next(self.video_enumerator)
        b = batch
        if self.use_cuda:
            for k, v in batch.items():
                b[k] = v.cuda()

        if batch_idx == len(self.video_sampler) - 1:
            self.video_enumerator = enumerate(self.video_sampler)

        return b

    def train_discriminator(self, discriminator, 
                            sample_true, sample_fake, 
                            opt, batch_size, use_categories):
        opt.zero_grad()

        real_batch = sample_true()
        batch = real_batch['values']

        fake_batch, generated_categories = sample_fake(batch_size)

        real_labels, real_categorical = discriminator(batch)
        fake_labels, fake_categorical = discriminator(fake_batch.detach())

        ones = self.create_by_shape(real_labels, 1)
        zeros = self.create_by_shape(fake_labels, 0)

        l_discriminator = self.gan_criterion(real_labels, ones) + \
                          self.gan_criterion(fake_labels, zeros)

        if use_categories:
            # Ask the video discriminator to learn categories from training videos
            categories_gt = torch.squeeze(real_batch['categories'].long())
            l_discriminator += self.category_criterion(real_categorical.squeeze(), categories_gt)

        l_discriminator.backward()
        opt.step()

        return l_discriminator

    def train_generator(self,
                        image_discriminator, video_discriminator,
                        sample_fake_images, sample_fake_videos,
                        opt):

        opt.zero_grad()

        # train on images

        fake_batch, generated_categories = sample_fake_images(self.image_batch_size)
        fake_labels, fake_categorical = image_discriminator(fake_batch)
        all_ones = self.create_by_shape(fake_labels, 1)

        l_generator = self.gan_criterion(fake_labels, all_ones)

        # train on videos

        fake_batch, generated_categories = sample_fake_videos(self.video_batch_size)
        fake_labels, fake_categorical = video_discriminator(fake_batch)
        all_ones = self.create_by_shape(fake_labels, 1)

        l_generator += self.gan_criterion(fake_labels, all_ones)

        if self.use_infogan:
            # Ask the generator to generate categories recognizable by the discriminator
            l_generator += self.category_criterion(fake_categorical.squeeze(), generated_categories)

        l_generator.backward()
        opt.step()

        return l_generator

    def train(self, generator, image_discriminator, video_discriminator):
        if self.use_cuda:
            generator.cuda()
            image_discriminator.cuda()
            video_discriminator.cuda()

        logger = Logger(self.log_folder)

        # create optimizers
        opt_generator = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)
        opt_image_discriminator = optim.Adam(image_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)
        opt_video_discriminator = optim.Adam(video_discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),
                                             weight_decay=0.00001)

        # training loop

        def sample_fake_image_batch(batch_size):
            return generator.sample_images(batch_size)

        def sample_fake_video_batch(batch_size):
            return generator.sample_videos(batch_size)

        def init_logs():
            return {'l_gen': 0, 'l_image_dis': 0, 'l_video_dis': 0}

        batch_num = 0

        logs = init_logs()

        start_time = time.time()

        while True:
            generator.train()
            image_discriminator.train()
            video_discriminator.train()

            opt_generator.zero_grad()

            opt_video_discriminator.zero_grad()

            # train image discriminator
            l_image_dis = self.train_discriminator(image_discriminator, self.sample_real_image_batch,
                                                   sample_fake_image_batch, opt_image_discriminator,
                                                   self.image_batch_size, use_categories=False)

            # train video discriminator
            l_video_dis = self.train_discriminator(video_discriminator, self.sample_real_video_batch,
                                                   sample_fake_video_batch, opt_video_discriminator,
                                                   self.video_batch_size, use_categories=self.use_categories)

            # train generator
            l_gen = self.train_generator(image_discriminator, video_discriminator,
                                         sample_fake_image_batch, sample_fake_video_batch,
                                         opt_generator)

            logs['l_gen'] += l_gen.item()

            logs['l_image_dis'] += l_image_dis.item()
            logs['l_video_dis'] += l_video_dis.item()

            batch_num += 1

            if batch_num % self.log_interval == 0:

                log_string = "Batch %d" % batch_num
                for k, v in logs.items():
                    log_string += " [%s] %5.3f" % (k, v / self.log_interval)

                log_string += ". Took %5.2f" % (time.time() - start_time)

                print(log_string)

                for tag, value in logs.items():
                    logger.scalar_summary(tag, value / self.log_interval, batch_num)

                logs = init_logs()
                start_time = time.time()

                generator.eval()

                images, _ = sample_fake_image_batch(self.image_batch_size)
                logger.image_summary("Images", images_to_numpy(images), batch_num)

                videos, _ = sample_fake_video_batch(self.video_batch_size)
                logger.video_summary("Videos", videos_to_numpy(videos), batch_num)

                torch.save(generator, self.log_folder / 'generator_%05d.pytorch' % batch_num)

            if batch_num >= self.train_batches:
                torch.save(generator, self.log_folder / 'generator_%05d.pytorch' % batch_num)
                break
