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
"""
import docopt
import pickle
from math import ceil
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

    batch_size = int(args['--batch-size'])
    num_workers = int(args['--num_workers'])
    video_loader = DataLoader(
            video_dataset, batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True)

    val_samples = [168029]  # pushing book from left to right
    val_samples = video_dataset.getById(val_samples)

    with open(cache / 'vocab.pkl', 'rb') as fp:
        t2i = pickle.load(fp)
        max_sen_len = int(fp.readline())

    r_v = 8  # n_spots 
    r_i = ceil(.5 * ceil(.5*max_sen_len))

    device = torch.device(device)
    emb_weights = getGloveEmbeddings('../embeddings', cache, t2i) 
    emb_weights = torch.tensor(emb_weights, device=device)
    text_encoder = models.TextEncoder(r_v, emb_weights)

    dim_Z = 50
    emb_size = 128
    generator = models.VideoGenerator(dim_Z, (r_i+r_v, emb_size))

    image_discriminator = models.ImageDiscriminator(
            cond_shape=(r_i, emb_size), noise=args['--noise'], 
            sigma=float(args['--sigma']))
    video_discriminator = models.VideoDiscriminator(
            cond_shape=(r_v, emb_size), noise=args['--noise'],
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
            betas=(.5, .999), weight_decay=1e-5),
        optim.Adam(
            dis_dict['image'].parameters(), lr=2e-4, 
            betas=(.5, .999), weight_decay=1e-5),
        optim.Adam(
            dis_dict['video'].parameters(), lr=2e-4, 
            betas=(.5, .999), weight_decay=1e-5),
        optim.Adam(
            text_encoder.parameters(), lr=2e-4, 
            betas=(.5, .999), weight_decay=1e-5)
    ]

    trainer = Trainer (
            text_encoder, dis_dict, generator,
            opt_list, video_loader, val_samples,
            cache, int(args['--training-time'])
    )
    trainer.train()
