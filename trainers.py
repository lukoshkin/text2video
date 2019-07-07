import time

import torch
import torch.optim as optim

from torch import nn
from pathlib import Path
from tensorboardX import SummaryWriter

def images_to_numpy(tensor):
    generated = (torch.clamp(tensor, -1, 1) + 1) / 2 * 255
    generated = generated.data
                         .cpu()
                         .numpy()
                         .transpose(0, 2, 3, 1)

    return generated.astype('uint8')

def videos_to_numpy(tensor):
    generated = (torch.clamp(tensor, -1, 1) + 1) / 2 * 255
    generated = generated.data
                         .cpu()
                         .numpy()
                         .transpose(0, 2, 1, 3, 4)

    return generated.astype('uint8')


class MultiOptimizer:
    def __init__(self, *args):
        self.opts = []
        for arg in args:
            self.opts.append(arg)

    def step(self):
        for opt in self.opts:
            opt.step()

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()

class Trainer:
    def __init__(self, image_loader, video_loader, 
                 log_interval, num_batches, log_folder) 

        self.image_loader = image_loader
        self.video_loader = video_loader

        self.video_batch_size = self.video_sampler.batch_size
        self.image_batch_size = self.image_sampler.batch_size

        self.log_interval = log_interval
        self.num_batches = num_batches

        self.log_folder = Path(log_folder)

    def train(self, generator, discriminator, encoder):
        writer = SummaryWriter()
        opt1 = optim.Adam(
                    generator.parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        opt2 = optim.Adam(
                    discriminator['image'].parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        opt3 = optim.Adam(
                    discriminator['video'].parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        opt4 = optim.Adam(
                    encoder.parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        optimizers = MultiOptimizer(opt1, opt2, opt3)

        def sampler(n_samples, type, vlen=None):
            generator.eval()
            if type == 'image':
                return generator.sample_images(
                            n_samples, conditions
                        )
            if type == 'video':
                return generator.sample_videos(
                            n_samples, conditions, vlen
                        )
            else:
                raise TypeError('Wrong discriminator type')

        batch_No = 0
        logs = dict.fromkeys(logs, 0)
        time_per_epoch =- time.time()

        while True:
            optimizers.zero_grad()
            for type in ['image', 'video']:
                gen_pairs = makePairsFromGenerated()
                neg_pairs = sampleWrongAssociations()
                pos_pairs = sampleProperAssociations()

                pos_scores = discriminator[type](pos_pairs)
                neg_scores = discriminator[type](neg_pairs)
                gen_scores = discriminator[type](gen_pairs)

                l1 = torch.log(pos_scores).mean()
                l2 = torch.log1p(-neg_scores).mean()
                l3 = torch.log1p(-gen_scores).mean()
                
                generator.eval()
                discriminator[type].train()

                dis_loss = l1 + .5 * (l2 + l3)
                logs[f"{type} dis"] += dis_loss.item()
                (-dis_loss).backward()

                # calculate_grad_penalty

                generator.train()
                discriminator[type].eval()

                gen_loss = .5 * (l3 - l2)
                logs['generator'] += gen_loss.item()
                gen_loss.backward()

                logs['encoder'] += (l1 + l2).item()

            optimizers.step()
            batch_No += 1

            if batch_No % self.log_interval == 0:

                log_string = f"Batch {batch_No}" 
                for k, v in logs.items():
                    log_string += " [%s] %5.3f" % (k, v / self.log_interval)

                time_per_epoch += time.time()
                log_string += ". Took %5.2f" % time_per_epoch 

                print(log_string)

                for tag, value in logs.items():
                    writer.add_scalar(tag, value / self.log_interval, batch_No)

                logs = dict.fromkeys(logs, 0)
                time_per_epoch =- time.time()

                generator.eval()

                images = generator.sample_images(self.image_batch_size)
                writer.add_images("Images", images, batch_No)

                videos = generator.sample_videos(self.video_batch_size)
                writer.add_video("Videos", videos_to_numpy(videos), batch_No)

                torch.save(
                        generator.state_dict(), 
                        self.log_folder / 'gen_%05d.pytorch' % batch_No
                    )

            if batch_No >= self.num_batches:
                torch.save(
                        generator.state_dict(), 
                        self.log_folder / 'gen_%05d.pytorch' % batch_No
                    )
                break
