import time

import numpy as np

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



class Trainer:
    def __init__(self, video_loader, batch_size,
                 log_interval, num_batches, log_folder) 

        self.num_batches  = num_batches
        self.video_loader = video_loader
        self.log_interval = log_interval
        self.batch_size = self.batch_size
        self.log_folder = Path(log_folder)

    def train(self, generator, discriminator, encoder):
        optimizer1 = optim.Adam (
                    generator.parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        optimizer2 = optim.Adam (
                    discriminator['image'].parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        optimizer3 = optim.Adam (
                    discriminator['video'].parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )
        optimizer4 = optim.Adam (
                    encoder.parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5
                )

        batch_No = 0
        writer = SummaryWriter()
        time_per_epoch =- time.time()
        pos_pairs, neg_pairs = {}, {}
        logs = {'image dis': 0, 
                'video dis': 0, 
                'encoder'  : 0, 
                'generator': 0}

        while True:
            pos_pairs['video'] = next(self.video_loader)

            # if this does not work, try np.rec.array right here

            labels = pos_pairs['video'].f0
            videos = pos_pairs['video'].f1

            images = videos [
                np.arange(self.batch_size), 
                np.random.choice(self.vlen, self.batch_size), ...
            ]
            im_sh = images.shape[1:]

            neg_pairs['video'] = np.rec.fromarrays (
                (np.roll(labels, 1), videos),
                dtype=pos_pairs.dtype
            )
            pos_pairs['image'] = np.rec.fromarrays (
                (labels, images),
                dtype=[('', '<U2'), ('', 'uint8', im_sh)]
            )
            neg_pairs['image'] = np.rec.fromarrays (
                (np.roll(labels, -1), images),
                dtype=pos_pairs['image'].dtype
            )

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()
            for type in ['image', 'video']:
                pos_scores = discriminator[type](pos_pairs[type])
                neg_scores = discriminator[type](neg_pairs[type])
                gen_scores = discriminator[type](gen_pairs[type])

                l1 = torch.log(pos_scores).mean()
                l2 = torch.log1p(-neg_scores).mean()
                l3 = torch.log1p(-gen_scores).mean()
                
                generator.eval()
                discriminator[type].train()

                dis_loss = l1 + .5 * (l2 + l3)
                logs[f"{type} dis"] += dis_loss.item()
                (-dis_loss).backward()

                # write grad_penalty

                generator.train()
                discriminator[type].eval()

                gen_loss = .5 * (l3 - l2)
                logs['generator'] += gen_loss.item()
                gen_loss.backward()

                logs['encoder'] += (l1 + l2).item()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            batch_No += 1

            if batch_No % self.log_interval == 0:

                print(f"Batch {batch_No}")
                for k, v in logs.items():
                    print("\t%s\t%5.3f" % (k, v / self.log_interval)

                time_per_epoch += time.time()
                print("Completed in %.f" % time_per_epoch)

                for tag, value in logs.items():
                    writer.add_scalar (
                        tag, value / self.log_interval, batch_No
                    )

                logs = dict.fromkeys(logs, 0)
                time_per_epoch =- time.time()

                generator.eval()

                images = generator.sample_images(self.image_batch_size)
                writer.add_images("Images", images, batch_No)

                videos = generator.sample_videos(self.video_batch_size)
                writer.add_video("Videos", videos_to_numpy(videos), batch_No)

                torch.save (
                        generator.state_dict(), 
                        self.log_folder / 'gen_%05d.pytorch' % batch_No
                    )

            if batch_No >= self.num_batches:
                torch.save (
                        generator.state_dict(), 
                        self.log_folder / 'gen_%05d.pytorch' % batch_No
                    )
                break
