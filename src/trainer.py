import time
import torch

from pathlib import Path

from torch import autograd
from torch.utils.tensorboard import SummaryWriter


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
        log_folder, log_period=10, num_epochs=100000):

        self.encoder = encoder
        self.dis_dict = dis_dict
        self.generator = generator
        self.opt_list = opt_list

        self.vloader = video_loader
        self.num_epochs = num_epochs
        self.log_folder = Path(log_folder)
        self.val_samples = val_samples
        self.log_period = log_period

        self.pairs = {}
        self.hyppar = 1e-3
        self.logs = {'image dis': 0,
                     'video dis': 0,
                     'generator': 0,
                     'encoder': 0}

    def getImageBatchIndices(self, batch_size, vlen):
        sample_ids = torch.arange(batch_size)
        frame_ids = torch.multinomial(
                torch.ones(vlen), batch_size, replacement=True)

        return sample_ids, frame_ids

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

    def formTrainingPairs(self, videos, at_video):
        sample_ids, frame_ids = self.getImageBatchIndices(
                videos.size(0), videos.size(2))
        images = videos[sample_ids, :, frame_ids, ...]
        at_image = at_video[sample_ids, frame_ids, :]

        self.pairs['pos','video'] = (videos, at_video)
        self.pairs['neg','video'] = (videos, torch.roll(at_video, 1, 0))
        self.pairs['neg','video'][1].register_hook(lambda grad: grad * 2)

        self.pairs['pos','image'] = (images, at_image)
        self.pairs['neg','image'] = (images, torch.roll(at_image, -1, 0))
        self.pairs['neg','image'][1].register_hook(lambda grad: grad * 2)

        fake_videos = self.generator(at_video.detach())
        fake_images = fake_videos[sample_ids, :, frame_ids, ...]

        fake_videos.register_hook(lambda grad: -grad)
        fake_images.register_hook(lambda grad: -grad)
        
        self.pairs['gen','image'] = (fake_images, at_image.detach())
        self.pairs['gen','video'] = (fake_videos, at_video.detach())

    def samenessPenaltyBackward(self, A):
        AAt = torch.einsum('ikp,ikq->ipq', A, A)
        mask = torch.eye(
                AAt.size(1), dtype=torch.uint8, device=AAt.device)
        sp_loss = torch.mean(
                torch.norm(AAt.masked_fill(mask, 0), dim=(1, 2)) ** 2)
        sp_loss.backward()
        self.logs['encoder'] -= sp_loss.item()
    
    def baseLossTermsBackward(self, kind):
        pos_scores = self.dis_dict[kind](*self.pairs['pos',kind])
        neg_scores = self.dis_dict[kind](*self.pairs['neg',kind])
        gen_scores = self.dis_dict[kind](*self.pairs['gen',kind])

        L1 = torch.log(pos_scores).mean()
        L2 = .5 * torch.log1p(-neg_scores).mean()
        L3 = .5 * torch.log1p(-gen_scores).mean()
        gp_loss = self.zeroCentredGradPenalty(
                pos_scores, self.pairs['pos',kind])

        autograd.backward([-L1, -L2, -L3, gp_loss], retain_graph=True)

        self.logs[f'{kind} dis'] += (L1 + L2 + L3 - gp_loss).item()
        self.logs['encoder'] += (L1 + 2 * L2).item()
        self.logs['generator'] += L3.item()

    def passBatchThroughNetwork(self, labels, videos):
        videos.requires_grad_(True)
        at_video, A = self.encoder(labels)
        self.formTrainingPairs(videos, at_video)

        for opt in self.opt_list:
            opt.zero_grad()
        for kind in ['image', 'video']:
            self.baseLossTermsBackward(kind)
        self.samenessPenaltyBackward(A)
        for opt in self.opt_list:
            opt.step()

    def train(self):
        writer = SummaryWriter()
        device = next(self.generator.parameters()).device

        texts = torch.tensor(self.val_samples.f0, device=device)
        movies = torch.tensor(self.val_samples.f1, device=device)
        writer.add_video("Real Clips", to_video(movies))

        time_per_epoch =- time.time()
        for epoch in range(self.num_epochs):
            batch_No = 0
            for batch in self.vloader:
                labels = batch['label'].to(device, non_blocking=True)
                videos = batch['video'].to(device, non_blocking=True)
                self.passBatchThroughNetwork(labels, videos)
                batch_No += 1

            time_per_epoch += time.time()
            # --------------------------
            print(f'Epoch {epoch}/{self.num_epochs}')
            for k, v in self.logs.items():
                print("\t%s:\t%5.4f" % (k, v/batch_No))
                self.logs[k] = v / batch_No
            print('Completed in %.f s' % time_per_epoch)
            writer.add_scalars('Loss', self.logs, epoch)
            self.logs = dict.fromkeys(self.logs, 0)

            if not epoch % self.log_period:
                self.generator.eval()
                with torch.no_grad():
                    at_video,_ = self.encoder(texts)
                    movies = self.generator(at_video)
                writer.add_video('Fakes', to_video(movies), epoch)
                self.generator.train()
            # --------------------------
            time_per_epoch =- time.time()

        torch.save(
            self.generator.state_dict(),
            self.log_folder / ('gen_%05d.pytorch' % epoch))
        print('Training has been completed successfully!')
