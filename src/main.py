"""
Usage:
    main.py [options] <metadata> <log_folder>

Options:
    --device=<count>            if specified, stands for gpu node number
    --batch-size=<count>        number of samples in all types of batches [default: 8]

    --noise                     when specified, Gaussian noise is added to both discriminators
    --sigma=<float>             if there is a noise, controls its standard deviation [default: .1]

    --video-length=<len>        length of videos in the video batch [default: 32]
    --print-period=<count>      print log info every few iterations [default: 10]
    --training-time=<count>     number of training batches [default: 100000]
"""
import docopt
import sys
from math import ceil

import torch
from torch.utils.data import DataLoader

import models
from trainer import Trainer
from data_prep import LabeledVideoDataset
from text_processing import getGloveEmbeddings, selectTemplates 



def to_tensor(video):
    return video.transpose(3, 0, 1, 2) \
                .astype('float32')

if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    print(args)

    sys.exit()

    templates = ['Pushing [something] from left to right']
    new_path = selectTemplates(
            args['<metadata>'], templates, '1-template.pkl')

    vlen = int(args['--video-length'])
    video_dataset = LabeledVideoDataset(
            new_path, '../data_logs', 
            (vlen, 64, 64, 3), check_spell=True, 
            transform=to_tensor)
    video_loader = DataLoader(
            video_dataset, batch_size=int(args['--batch-size']), 
            shuffle=True, drop_last=True, num_workers=2)

    emb_weights = getGloveEmbeddings('../embeddings', '../data_logs')
    text_encoder = models.TextEncoder(n_spots, emb_weights)

    r_v = 8  # n_spots 
    r_i = ceil(ceil(video_loader.max_sen_len/2) / 2)
    generator = models.VideoGenerator(dim_Z, (r_i+r_v, 128))

    image_discriminator = models.ImageDiscriminator(
            cond_shape=(r_i, 128), noise=args['--noise'], 
            sigma=float(args['--sigma']))
    video_discriminator = models.VideoDiscriminator(
            cond_shape=(r_v, 128), noise=args['--noise'],
            sigma=float(args['--sigma']))

    if args['--device']:
        device = torch.device(f"cuda:{args['--device']}")
        generator.to(device)
        text_encoder.to(device)
        image_discriminator.to(device)
        video_discriminator.to(device)

    dis_dict = {'image': image_discriminator,
                'video': video_discriminator}

    trainer = Trainer (
            video_loader,
            args['<log_folder>'],
            int(args['--print-period']),
            int(args['--training-time'])
    )
    trainer.train(generator, dis_dict, text_encoder)