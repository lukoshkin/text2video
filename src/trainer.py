import time
import torch

from pathlib import Path

from torch import autograd, optim
from torch.utils.tensorboard import SummaryWriter

def to_video(tensor):
    generated = (torch.clamp(tensor,-1,1) + 1) / 2 * 255
    generated = generated.data \
                         .cpu() \
                         .numpy() \
                         .transpose(0, 2, 1, 3, 4)

    return generated.astype('uint8')



class Trainer:
    def __init__(
        self, video_loader, log_folder, 
        log_interval, training_time, val_example):

        self.vloader = video_loader
        self.log_interval = log_interval
        self.num_batches = training_time 
        self.log_folder = Path(log_folder)
        self.batch_size = self.vloader.batch_size
        self.val_example = val_example
        self.hyppar = 1.

    def zeroCentredGradPenalty(self, output, inputs):
        jacobian = autograd.grad (
                    outputs=output,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    allow_unused=True
                )[0]
        jacobian = jacobian.view(jacobian.size(0), -1)

        return self.hyppar * ((jacobian.norm(dim=1) - 1) ** 2).mean()

    def composeBatchOfImages(self, videos):
        images = videos [
            torch.arange(self.batch_size),
            :,
            torch.multinomial (
                torch.ones(videos.size(2)), 
                self.batch_size
            ),
            ... ]

        return images

    def train(self, generator, dis_dict, text_encoder):
        coded_example,_ = text_encoder(self.val_example.unsqueeze(0))

        optimizer1 = optim.Adam(
                    generator.parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5)
        optimizer2 = optim.Adam(
                    dis_dict['image'].parameters(), 
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5)
        optimizer3 = optim.Adam(
                    dis_dict['video'].parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5)
        optimizer4 = optim.Adam(
                    text_encoder.parameters(),
                    lr=2e-4, betas=(.5, .999), weight_decay=1e-5)

        batch_No = 0
        writer = SummaryWriter()
        time_per_epoch =- time.time()
        pos_pairs, neg_pairs, gen_pairs = {}, {}, {}
        logs = {'image dis': 0, 
                'video dis': 0, 
                'encoder'  : 0, 
                'generator': 0}

        autograd.set_detect_anomaly(True)
        while True:
            # >>> form training pairs >>>
            labels, videos = next(iter(self.vloader)).values()

            videos.requires_grad_(True)
            images = self.composeBatchOfImages(videos)

            (at_video, at_image), A = text_encoder(labels)

            pos_pairs['video'] = (videos, at_video)
            neg_pairs['video'] = (videos, torch.roll(at_video, 1, 0)) 
            neg_pairs['video'][0].register_hook(lambda grad: grad * 2)

            pos_pairs['image'] = (images, at_image)
            neg_pairs['image'] = (images, torch.roll(at_image, -1, 0))
            neg_pairs['image'][0].register_hook(lambda grad: grad * 2)

            conditions  = (at_image.detach(), at_video.detach())
            fake_videos = generator(torch.cat(conditions, 1))
            fake_images = self.composeBatchOfImages(fake_videos)

            fake_videos.register_hook(lambda grad: -grad)
            fake_images.register_hook(lambda grad: -grad)
            
            gen_pairs['image'] = (fake_images, conditions[0])
            gen_pairs['video'] = (fake_videos, conditions[1])
            samples = {'image': images, 'video': videos}
            # <<< form training pairs <<<

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            optimizer4.zero_grad()

            for kind in ['image', 'video']:
                #pos_pairs[kind][0].requires_grad_(True)
                pos_scores = dis_dict[kind](*pos_pairs[kind])
                neg_scores = dis_dict[kind](*neg_pairs[kind])
                gen_scores = dis_dict[kind](*gen_pairs[kind])

                # base loss temrs (enc., gen., disc.)
                L1 = torch.log(pos_scores).mean()
                L2 = .5 * torch.log1p(-neg_scores).mean()
                L3 = .5 * torch.log1p(-gen_scores).mean()
                # grad. penalty loss (disc.)
                gp_loss = self.zeroCentredGradPenalty(
                                pos_scores, samples[kind])

                autograd.backward(
                        [-L1, -L2, -L3, gp_loss], retain_graph=True)

                logs[f'{kind} dis'] += (L1 + L2 + L3 - gp_loss).item()
                logs['encoder'] += (L1 + 2 * L2).item()
                logs['generator'] += L3.item()

            # sameness penalty (enc.)
            AAt = torch.einsum('ikp,ikq->ipq', A, A)
            mask = torch.eye(AAt.size(1)).byte()
            sp_loss = torch.mean(
                        torch.norm(
                            AAt.masked_fill(mask, 0),
                            dim=(1, 2)) ** 2)
            sp_loss.backward()
            logs['encoder'] -= sp_loss.item()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            batch_No += 1

            # write log info with tensorboardX; 
            # save generator every several epochs (perhaps unnecessary)
            if batch_No % self.log_interval == 0:
                print(f"Batch {batch_No}")
                for k, v in logs.items():
                    print("\t%s:\t%5.3f" % (k, v / self.log_interval))

                time_per_epoch += time.time()
                print("Completed in %.f" % time_per_epoch)

                logs = {k: v / self.log_interval for k, v in logs.items()}
                writer.add_scalars('Losses', logs, batch_No)

                logs = dict.fromkeys(logs, 0)
                time_per_epoch =- time.time()

                generator.eval()

                videos = generator(torch.cat(coded_example, 1))
                writer.add_video("Videos", to_video(videos), batch_No)

            if batch_No >= self.num_batches:
                torch.save(
                        generator.state_dict(), 
                        self.log_folder / 'gen_%05d.pytorch' % batch_No)
                break
