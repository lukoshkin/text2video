"""
Usage:
    main.py [options] <path> [<cache>]

path      path to a json or pkl file with data
cache     folder where you store the processed data
          and the generator weights

Options:
    --from-scratch              remove cached files before training begins
    --device=<count>            stands for gpu node number
    --num_workers=<count>       number of cpu processes to load batches on gpu [default: 8]
    --batch-size=<count>        batch size (currently, all types of batches are of the same size) [default: 3]

    --noise                     when specified, Gaussian noise is added to both discriminators
    --sigma=<float>             if there is a noise, controls its standard deviation [default: .1]

    --video-length=<len>        original length of videos in the video batch [default: 32]
    --training-time=<count>     number of training epochs [default: 100000]

    --encoder=<str>             type of encoder: simple, mere, joint [default: mere]
    --pp=<count>                print period [default: 0]
    --lp=<count>                log period [default: 20]
"""
import docopt
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import optim

import models
from trainer import Trainer
from data_prep import LabeledVideoDataset
from text_processing import getGloveEmbeddings, selectTemplates


if __name__ == "__main__":
    args = docopt.docopt(__doc__)
    print(args)

    cache = Path(args['<cache>'])
    cache = cache if cache else Path('../logdir')
    if args['--from-scratch']:
        if cache.exists(): 
            for f in cache.glob('*'):
                f.unlink()

    device = args['--device']
    device = torch.device(f'cuda:{device}' if device else 'cpu')

    templates = ['Pushing [something] from left to right']
    new_path = selectTemplates(
            args['<path>'], templates, 
            f'{len(templates)}-template.pkl')

    vlen = int(args['--video-length'])
    video_dataset = LabeledVideoDataset(
            new_path, cache, (vlen, 64, 64, 3), check_spell=True)

    video_loader = DataLoader(
            video_dataset, int(args['--batch-size']), 
            shuffle=True, num_workers=int(args['--num_workers']), 
            pin_memory=True, drop_last=True)

    val_samples = [168029, 157604, 71563, 82109]
    val_samples = video_dataset.getById(val_samples)
    lens = torch.tensor(val_samples.f0, device=device)
    texts = torch.tensor(val_samples.f1, device=device)
    movies = torch.tensor(val_samples.f2, device=device)

    with open(cache / 'vocab.pkl', 'rb') as fp:
        t2i = pickle.load(fp)
        max_sen_len = int(fp.readline())

    device = torch.device(device)
    emb_weights = getGloveEmbeddings('../embeddings', cache, t2i) 
    emb_weights = torch.tensor(emb_weights, device=device)

    if args['--encoder'] == 'simple':
        emb_size = 50
        text_encoder = models.SimpleTextEncoder(emb_weights)
    elif args['--encoder'] == 'mere':
        emb_size = 64
        text_encoder = models.TextEncoder(emb_weights, proj=True)
    elif args['--encoder'] == 'joint':
        pass
    else:
        raise TypeError('Invalid encoder type')

    dim_Z = 50
    generator = models.VideoGenerator(dim_Z, emb_size)

    image_discriminator = models.ImageDiscriminator(
            cond_size=emb_size, noise=args['--noise'],
            sigma=float(args['--sigma']))
    video_discriminator = models.VideoDiscriminator(
            cond_size=emb_size, noise=args['--noise'],
            sigma=float(args['--sigma']))

    generator.to(device)
    text_encoder.to(device)
    image_discriminator.to(device)
    video_discriminator.to(device)

    dis_dict = {'image': image_discriminator,
                'video': video_discriminator}

    opt_list = [
        optim.Adam(
            generator.parameters(), lr=2e-4, 
            betas=(.3, .999), weight_decay=1e-5),
        optim.Adam(
            dis_dict['image'].parameters(), lr=2e-4, 
            betas=(.3, .999), weight_decay=1e-5),
        optim.Adam(
            dis_dict['video'].parameters(), lr=2e-4, 
            betas=(.3, .999), weight_decay=1e-5),
    ]

    train_enc = (args['--encoder'] == 'mere')
    if train_enc:
        opt_list += [optim.Adam(
            text_encoder.parameters(), lr=2e-4,
            betas=(.3, .999), weight_decay=1e-5)]

    trainer = Trainer (
            text_encoder, dis_dict, generator,
            opt_list, video_loader, cache,
            train_enc, int(args['--training-time']),
    )
    trainer.train(lens, texts, movies, int(args['--pp']), int(args['--lp']))
