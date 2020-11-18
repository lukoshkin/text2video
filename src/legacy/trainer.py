import time
import torch

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
        train_enc       whether to train encoder or not
    """
    def __init__(
        self, encoder, dis_dict, generator, opt_list,
        video_loader, log_folder, train_enc, num_epochs=100000):

        self.encoder = encoder
        self.dis_dict = dis_dict
        self.generator = generator
        self.opt_list = opt_list
        self.train_enc = train_enc

        self.vloader = video_loader
        self.num_epochs = num_epochs
        self.log_folder = log_folder

        self.pairs = {}
        self.logs = {'image dis': 0,
                     'video dis': 0,
                     'generator': 0}

    def formTrainingPairs(self, videos, condition):
        sp = len(condition) // 2
        self.pairs['pos'] = (videos[:sp], condition[:sp])
        roll_condition = torch.roll(condition[sp:], 1, 0)
        self.pairs['neg'] = (videos[sp:], roll_condition)

        art_condition = condition.detach()[:sp]
        fake_videos = self.generator(art_condition)
        fake_videos.register_hook(lambda grad: -grad)
        self.pairs['gen'] = (fake_videos, art_condition)

    def calculateBaseLossTerms(self, kind):
        pos_scores = self.dis_dict[kind](*self.pairs['pos'])
        neg_scores = self.dis_dict[kind](*self.pairs['neg'])
        gen_scores = self.dis_dict[kind](*self.pairs['gen'])

        L1 = torch.log(pos_scores).mean()
        L2 = torch.log1p(-neg_scores).mean()
        L3 = torch.log1p(-gen_scores).mean()

        self.logs[f'{kind} dis'] += (L1 + L2 + L3).item()
        self.logs['generator'] += L3.item()

        return -(L1 + L2 + L3)

    def passBatchThroughNetwork(self, labels, videos, senlen):
        with torch.set_grad_enabled(self.train_enc):
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

    def train(self, lens, texts, movies, pp=0, lp=20):
        """
        lp: log period
        pp: print period (0 - no print to stdout)
        """
        writer = SummaryWriter()
        writer.add_video("Real Clips", to_video(movies))
        device = next(self.generator.parameters()).device

        time_per_epoch =- time.time()
        for epoch in range(self.num_epochs):
            for No, batch in enumerate(self.vloader):
                labels = batch['label'].to(device, non_blocking=True)
                videos = batch['video'].to(device, non_blocking=True)
                senlen = batch['slens'].to(device, non_blocking=True)
                self.passBatchThroughNetwork(labels, videos, senlen)

            if pp and epoch % pp == 0:
                time_per_epoch += time.time()
                print(f'Epoch {epoch}/{self.num_epochs}')
                for k, v in self.logs.items():
                    print("\t%s:\t%5.4f" % (k, v/(No+1)))
                    self.logs[k] = v / (No+1)
                print('Completed in %.f s' % time_per_epoch)
                time_per_epoch =- time.time()
            
            if epoch % lp == 0:
                self.generator.eval()
                with torch.no_grad():
                    condition = self.encoder(texts, lens)
                    movies = self.generator(condition)
                writer.add_scalars('Loss', self.logs, epoch)
                writer.add_video('Fakes', to_video(movies), epoch)
                self.generator.train()
            self.logs = dict.fromkeys(self.logs, 0)

        torch.save(
            self.generator.state_dict(),
            self.log_folder / ('gen_%05d.pytorch' % epoch))
        print('Training has been completed successfully!')
