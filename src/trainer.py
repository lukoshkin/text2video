import time
import torch

from pathlib import Path

from torch import autograd
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pack_padded_sequence


def to_video(tensor):
    """
    Takes generator output and converts it to
    the format which can later be written by
    SummaryWriter's instance

    input: tensor of shape (N, C, D, H, W)
           obtained passed through nn.Tanh
    """
    generated = (tensor + 1) / 2 * 255
    generated = generated.cpu() \
                         .numpy() \
                         .transpose(0, 2, 1, 3, 4)

    return generated.astype('uint8')


class Trainer:
    """
    Args:
        dis_dict        dict object containing image and video
                        discriminators under the keys 'image'
                        and 'video' respectively
        opt_list        list with the optimizers of all model 
                        components
        log_folder      folder where to save the generator's 
                        weights after training is over 
        val_samples     np.rec.array of pairs: video and 
                        corresponding to it description. 
        
    """
    def __init__(
        self, encoder, dis_dict, generator,
        opt_list, video_loader, val_samples,
        log_folder, num_epochs=100000):

        self.encoder = encoder
        self.dis_dict = dis_dict
        self.generator = generator
        self.opt_list = opt_list

        self.vloader = video_loader
        self.num_epochs = num_epochs
        self.log_folder = log_folder
        self.val_samples = val_samples

        self.pairs = {}
        self.hyppar = 1e-3
        self.logs = {'image dis': 0,
                     'video dis': 0,
                     'generator': 0,
                     'encoder': 0}

    def composeBatchOfImages(self, videos):
        bs = videos.size(0)
        nf = videos.size(2)
        images = videos[
                torch.arange(bs), :,
                torch.multinomial(
                    torch.ones(nf), bs, replacement=True), ... ]

        return images

    def zeroCentredGradPenalty(self, output, inputs):
        jacobians = autograd.grad(
                outputs=output, inputs=inputs, 
                grad_outputs=torch.ones_like(output), 
                create_graph=True)
        jacobian = torch.flatten(jacobians[0], 1)
        res = ((jacobian.norm(dim=1) - 1) ** 2).mean()
        jacobian = torch.flatten(jacobians[1], 1)
        res += self.hyppar * ((jacobian.norm(dim=1) - 1) ** 2).mean()

        return res 

    def formTrainingPairs(self, videos, condition):
        images = self.composeBatchOfImages(videos)

        self.pairs['pos','video'] = (videos, condition)
        self.pairs['neg','video'] = (videos, torch.roll(condition, 1, 0))
        self.pairs['neg','video'][1].register_hook(lambda grad: grad * 2)

        self.pairs['pos','image'] = (images, condition)
        self.pairs['neg','image'] = (images, torch.roll(condition, -1, 0))
        self.pairs['neg','image'][1].register_hook(lambda grad: grad * 2)

        fake_videos = self.generator(condition.detach())
        fake_images = self.composeBatchOfImages(fake_videos)

        fake_videos.register_hook(lambda grad: -grad)
        fake_images.register_hook(lambda grad: -grad)
        
        self.pairs['gen','image'] = (fake_images, condition.detach())
        self.pairs['gen','video'] = (fake_videos, condition.detach())
    
    def calculateBaseLossTerms(self, kind):
        pos_scores = self.dis_dict[kind](*self.pairs['pos',kind])
        neg_scores = self.dis_dict[kind](*self.pairs['neg',kind])
        gen_scores = self.dis_dict[kind](*self.pairs['gen',kind])

        L1 = torch.log(pos_scores).mean()
        L2 = .5 * torch.log1p(-neg_scores).mean()
        L3 = .5 * torch.log1p(-gen_scores).mean()
        gp_loss = self.zeroCentredGradPenalty(
                pos_scores, self.pairs['pos',kind])

        self.logs[f'{kind} dis'] += (L1 + L2 + L3 - gp_loss).item()
        self.logs['encoder'] += (L1 + 2 * L2).item()
        self.logs['generator'] += L3.item()

        return -(L1 + L2 + L3 - gp_loss)

    def passBatchThroughNetwork(self, labels, videos, senlen):
        videos.requires_grad_(True)
        condition = self.encoder(labels, senlen)
        self.formTrainingPairs(videos, condition)

        loss = videos.new(1).fill_(0) 
        for opt in self.opt_list:
            opt.zero_grad()
        for kind in ['image', 'video']:
            loss += self.calculateBaseLossTerms(kind)
        loss.backward()
        for opt in self.opt_list:
            opt.step()

    def train(self):
        writer = SummaryWriter()
        device = next(self.generator.parameters()).device

        lens = torch.tensor(self.val_samples.f0, device=device)
        texts = torch.tensor(self.val_samples.f1, device=device)
        movies = torch.tensor(self.val_samples.f2, device=device)
        writer.add_video("Real Clips", to_video(movies))

        time_per_epoch =- time.time()
        for epoch in range(self.num_epochs):
            for No, batch in enumerate(self.vloader):
                labels = batch['label'].to(device, non_blocking=True)
                videos = batch['video'].to(device, non_blocking=True)
                senlen = batch['sen_len'].to(device, non_blocking=True)
                self.passBatchThroughNetwork(labels, videos, senlen)

            time_per_epoch += time.time()
            # --------------------------
            print(f'Epoch {epoch}/{self.num_epochs}')
            for k, v in self.logs.items():
                print("\t%s:\t%5.4f" % (k, v/(No+1)))
                self.logs[k] = v / (No+1)
            print('Completed in %.f s' % time_per_epoch)
            writer.add_scalars('Loss', self.logs, epoch)
            self.logs = dict.fromkeys(self.logs, 0)

            self.generator.eval()
            with torch.no_grad():
                condition = self.encoder(texts, lens)
                movies = self.generator(condition)
            writer.add_video('Fakes', to_video(movies), epoch)
            self.generator.train()
            # --------------------------
            time_per_epoch =- time.time()

        torch.save(
            self.generator.state_dict(),
            self.log_folder / ('gen_%05d.pytorch' % epoch))
        print('Training has been completed successfully!')
