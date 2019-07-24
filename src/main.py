"""
Usage:
    main.py [options] <metadata> <log_folder>

Options:
    --device=<count>            if specified, stands for gpu node number
    --batch-size=<count>        number of samples in all types of batches [default: 3]

    --noise                     when specified, Gaussian noise is added to both discriminators
    --sigma=<float>             if there is a noise, controls its standard deviation [default: .1]

    --video-length=<len>        length of videos in the video batch [default: 32]
    --print-period=<count>      print log info every few iterations [default: 10]
    --training-time=<count>     number of training batches [default: 100000]
"""
import sys
import docopt
import pickle
from math import ceil

import torch
from torch.utils.data import DataLoader

import models
from trainer import Trainer
from data_prep import LabeledVideoDataset
from text_processing import getGloveEmbeddings, selectTemplates, sen2vec



if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    print(args)

    cache = '../data_logs'
    device = torch.device(
        f"cuda:{args['--device']}" if args['--device'] else 'cpu')

    templates = ['Pushing [something] from left to right']
    text_example = 'pushing book from left to right'.split()

    new_path = selectTemplates(
            args['<metadata>'], templates, '1-template.pkl')

    vlen = int(args['--video-length'])
    video_dataset = LabeledVideoDataset(
            new_path, cache, (vlen, 64, 64, 3),
            check_spell=True, device=device)
    video_loader = DataLoader(
            video_dataset, batch_size=int(args['--batch-size']), 
            shuffle=True, drop_last=True)

    with open(cache + '/vocab.pkl', 'rb') as fp:
        t2i = pickle.load(fp)
        max_sen_len = int(fp.readline())

    coded_example = sen2vec(text_example, t2i, max_sen_len)
    coded_example = torch.tensor(
            coded_example, dtype=torch.long, device=device)

    r_v = 8  # n_spots 
    r_i = ceil(.5 * ceil(.5*max_sen_len))

    emb_weights = getGloveEmbeddings('../embeddings', cache, t2i) 
    emb_weights = torch.tensor(emb_weights, device=device)
    text_encoder = models.TextEncoder(r_v, emb_weights)

    generator = models.VideoGenerator(50, (r_i+r_v, 128))

    image_discriminator = models.ImageDiscriminator(
            cond_shape=(r_i, 128), noise=args['--noise'], 
            sigma=float(args['--sigma']))
    video_discriminator = models.VideoDiscriminator(
            cond_shape=(r_v, 128), noise=args['--noise'],
            sigma=float(args['--sigma']))

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
            int(args['--training-time']),
            coded_example
    )
    trainer.train(generator, dis_dict, text_encoder)
