import pickle

import copy
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from smart_open import open
from collections import Counter
from nltk import wordpunct_tokenize
from spellchecker import SpellChecker


def selectTemplates(path, templates, new_name):
    """
    Remove from the database (json file specified by 'path' argument)
    label categories which are not in 'templates' list. The new file
    is created under the same folder as the old one.
    """
    path = Path(path)
    df = pd.read_json(path)

    mask = df.template.isin(templates)
    new_df = df[mask]

    new_df.index = np.arange(mask.sum())
    new_path = path.parent / new_name
    new_df.to_pickle(new_path)

    return new_path


def doTextPart(path, cache, min_freq=2, check_spell=False):
    path = Path(path)
    file_name = path.stem 
    ext = path.suffix[1:]

    cache = Path(cache)
    if ((cache / 'vocab.pkl').exists() and 
            (cache / f'{file_name}.pkl').exists()):
        df = pd.read_pickle(cache / f'{file_name}.pkl')
        with open(cache / 'vocab.pkl', 'rb') as fp:
            t2i = pickle.load(fp)
            max_len = int(fp.readline())

        return max_len, t2i, df

    cache.mkdir(parents=True, exist_ok=True)

    if ext == 'json':
        df = pd.read_json(path)
    elif ext == 'pkl':
        df = pd.read_pickle(path)
    else:
        raise TypeError("'json' and 'pkl' are only supported")

    sentences = list(df.label.apply(wordpunct_tokenize).values)
    token_counts = Counter()
    token_counts.update(np.concatenate(sentences))

    vague_words = [
        x for x in token_counts.keys() 
        if token_counts[x] <= min_freq
    ]

    if check_spell:
        spell = SpellChecker()
        checked_words = []
        for sen in tqdm(sentences, 'Spell-check'):
            for i, w in enumerate(sen):
                if w in vague_words:
                    sen[i] = spell.correction(w)
                    checked_words.append(sen[i])
            
        token_counts.update(checked_words)

        vague_words = [
            x for x in token_counts.keys()
            if token_counts[x] <= min_freq
        ]

    mended = copy.deepcopy(sentences)
    pbar = tqdm(sentences, "Removing 'bad samples'")
    k = 0
    for i, sen in enumerate(pbar):
        flag = True
        for w in sen:
            if w in vague_words:
                df.drop(i, inplace=True)
                del mended[k]
                flag = False
                break 
        if flag:
            k += 1

    df.label = mended
    df.to_pickle(cache / f'{file_name}.pkl')
                
    for w in vague_words:
        token_counts.pop(w)

    tokens = ['PAD'] + list(token_counts.keys())
    t2i = {t: i for i, t in enumerate(tokens)}
    max_len = max(map(len, mended))
    with open(cache / 'vocab.pkl', 'wb') as fp:
        pickle.dump(t2i, fp)
        fp.write(b'%d' % max_len)

    return (max_len, t2i, df)


def getGloveEmbeddings(folder, cache, t2i, emb_size=50):
    cache = Path(cache)
    file_path = cache / 'emb_matrix.npy'
    if file_path.exists():
        return np.load(file_path)

    folder = Path(folder)
    with open(folder / f'glove.6B.{emb_size}d.txt') as fp:
        raw_data = fp.readlines()

    glove = {}
    pbar = tqdm(raw_data, 'Reading glove embeddings')
    for line in pbar:
        t, *v = line.split()
        glove[t] = np.array(v, 'float32')
            
    emb_matrix = np.empty((len(t2i), emb_size), 'float32')
    for t, i in t2i.items():
        try:
            emb_matrix[i] = glove[t]
        except KeyError:
            emb_matrix[i] = .6 * np.random.randn(emb_size)
    emb_matrix[0] = 0 

    np.save(cache / 'emb_matrix', emb_matrix)

    return emb_matrix 


def sen2vec(sen, t2i, max_len):
    """
    Converts a sentence to a sequence of positive
    integers of length 'max_len' (according to 't2i'
    dictionary), padded with zeros where necessary

    Output type: int64 - necessary for nn.Embedding
    """
    numerated = np.zeros(max_len, 'int')
    filling = [t2i[w] for w in sen]
    numerated[:len(filling)] = filling

    return numerated
