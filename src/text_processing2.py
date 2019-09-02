import re
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


def getGloveEmbeddings(folder, cache, t2i, emb_size=50):
    """
    Prepares word embedding matrix of width `emb_size`
    with `t2i` vocab. from the corresponding file in the `folder`.
    Caches the result at `cache`
    """
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


class TextProcessor:
    def __init__(self, path, cache, mode='toy', min_freq=2):
        self.mode = mode
        self.path = Path(path)
        self.cache = Path(cache)

        self._min_freq = min_freq
        self._save_df_as = f'{self.path.stem}.pkl'
        self._save_t2i_as = f'{mode}_vocab.pkl'

        if mode not in ['toy', 'simple', 'casual']:
            raise TypeError("mode must be 'toy', 'simple' or 'casual'")
        if mode == 'casual':
            raise NotImplemented('This feature is not added')

    def doTextPart(self):
        """
        Does all the work related to text part:
        filtration by word frequency, spell correction,
        data frame columns update. Also, caches the results
        """
        if ((self.cache / self._save_t2i_as).exists() and 
                (self.cache / self._save_df_as).exists()):
            self.df = pd.read_pickle(self.cache / self._save_df_as)
            with open(self.cache / self._save_t2i_as, 'rb') as fp:
                self.t2i = pickle.load(fp)
                self._max_len, self._act_max_len = pickle.load(fp)
            return 

        ext = self.path.suffix
        if ext == '.json':
            self.df = pd.read_json(self.path)
        elif ext == '.pkl':
            self.df = pd.read_pickle(self.path)
        else:
            raise TypeError("'json' and 'pkl' are only supported")

        tokens = ['PAD']
        self._extractTokens('placeholders', self.mode, tokens)
        self.df.label = self.df.apply(
                lambda x: re.sub('\[.*?\]', 
                    ' '.join(x.placeholders), x.template), axis=1)
        self.df.label = self.df.label.map(wordpunct_tokenize)
        self.df.template = self.df.template.map(
                #lambda x: re.sub('\[|\]', '', x))
                lambda x: re.sub('\[.*?\]', 'something', x))
                # there are strings where there are more than just
                # word 'something'. In this case, the words characterize
                # the object better, however, it may not help to
                # train the network
        self._extractTokens('template', 'action', tokens)
        self._act_max_len = max(map(len, self.df.template))
        self._max_len = max(map(len, self.df.label))

        self.t2i = {t: i for i, t in enumerate(tokens)}
        self.cache.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(self.cache / self._save_df_as)
        with open(self.cache / self._save_t2i_as, 'wb') as fp:
            pickle.dump(self.t2i, fp)
            pickle.dump((self._max_len, self._act_max_len), fp)

    def _extractTokens(self, column, mode, tokens):
        """
        Updates the specified `column` in `self.df`
        according to processing `mode` policy. Collects
        tokens encountered in the `column` into `tokens`
        """
        sentences = self.df[column]
        if mode in ['simple', 'toy']:
            mask = (sentences.map(len) == 1)
            sentences = sentences[mask]
            self.df = self.df[mask]
            if mode == 'toy':
                mask = (sentences.map(
                    lambda x: len(x[0].split()) == 1))
                sentences = sentences[mask]
                self.df = self.df[mask]

            self.df.index = np.arange(mask.sum())
            sentences = sentences.map(lambda x: x[0])

        check_spell = False if mode == 'action' else True
        sentences = list(sentences.apply(wordpunct_tokenize).values)
        # << converting to list to be able to delete elements
        token_counts = self._findTypos(sentences, check_spell)
        self._updateSeries(column, sentences)

        for w in self._vague_words:
            token_counts.pop(w)
        tokens += list(token_counts.keys())

    def _findTypos(self, sentences, check_spell):
        """
        Colects suspicious (rare) words in `self._vague_words`.
        `self._min_freq` defines the rarity extent. 
        If check_spell == True, then the words found will 
        be corrected with pyspellchecker

        Returns counts of the words in the sentences
        """
        def freqFilter(C):
            return [] if self._min_freq < 1 else [
                    x for x in C.keys() if C[x] <= self._min_freq]

        token_counts = Counter(np.concatenate(sentences))
        self._vague_words = freqFilter(token_counts)

        if not (check_spell and self._vague_words): 
            return token_counts

        spell = SpellChecker()
        checked_words = []
        for sen in tqdm(sentences, 'Spell-check'):
            for i, w in enumerate(sen):
                if w in self._vague_words:
                    sen[i] = spell.correction(w)
                    checked_words.append(sen[i])
        token_counts.update(checked_words)
        self._vague_words = freqFilter(token_counts)
        return token_counts

    def _updateSeries(self, column, sentences): 
        """ 
        Subtitutes 'mispelled' words (those that are in 
        `self._vague_words`) in the `column` with the 
        `sentences` given
        """
        mended = copy.deepcopy(sentences)
        if self._vague_words:
            pbar = tqdm(sentences, "Removing 'bad samples'")
            k = 0
            for i, sen in enumerate(pbar):
                flag = True
                for w in sen:
                    if w in self._vague_words:
                        self.df.drop(i, inplace=True)
                        del mended[k]
                        flag = False
                        break
                if flag:
                    k += 1

        self.df[column] = mended
        self.df.index = np.arange(len(self.df))
