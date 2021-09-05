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
    Remove from the database (json file specified by `path` argument)
    label categories which are not in `templates` list. The new file
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


class TextProcessor:
    def __init__(
            self, path, cache, mode='toy',
            check_spell=True, min_freq=2,
            glove_folder='../embeddings',
            emb_size=50, glove_filtration=True):
        self._mode = mode
        self._path = Path(path)
        self._cache = Path(cache)

        self._emb_size = emb_size
        self._min_freq = min_freq
        self._check_spell = check_spell
        self._save_df_as = f'{self._path.stem}.pkl'
        self._save_t2i_as = f'{mode}_vocab.pkl'
        self._tqdmBF = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
        self._GF = glove_filtration

        if mode not in ['toy', 'simple', 'casual']:
            raise TypeError("mode must be 'toy', 'simple' or 'casual'")
        if mode == 'casual':
            raise NotImplemented('This feature is not added')

        self._prepareGloveDict(glove_folder, emb_size)
        self._doTextPart()

    def getGloveEmbeddings(self, monitor_unk=False):
        """
        Prepares word embedding matrix of width `emb_size`
        with `self.t2i` vocab. from the corresponding file in the `folder`.
        Caches the result at `self.cache`
        """
        if (self._cache/'emb_matrix.npy').exists():
            return np.load(self._cache/'emb_matrix.npy')

        not_found = 0
        emb_matrix = np.empty((len(self.t2i), self._emb_size), 'float32')
        for t, i in self.t2i.items():
            try:
                emb_matrix[i] = self._glove[t]
            except KeyError:
                if monitor_unk: print('UNK:', t)
                emb_matrix[i] = .6 * np.random.randn(self._emb_size)
                not_found += 1
        emb_matrix[0] = 0

        if not_found:
            print('No of missed tokens in glove dict:', not_found)
        np.save(self._cache/'emb_matrix', emb_matrix)
        return emb_matrix

    def _prepareGloveDict(self, folder, emb_size):
        """
        Constructs GloVe dictionary. Cashes the result
        """
        save_as = self._cache/f'glove.{emb_size}d.pkl'
        if save_as.exists():
            with open(save_as, 'rb') as fp:
                self._glove = pickle.load(fp)
            return

        folder = Path(folder)
        with open(folder/f'glove.6B.{emb_size}d.txt') as fp:
            raw_data = fp.readlines()

        pbar = tqdm(
            raw_data, 'Reading glove embeddings',
            bar_format=self._tqdmBF)

        self._glove = {}
        for line in pbar:
            t, *v = line.split()
            self._glove[t] = np.array(v, 'float32')

        with open(save_as, 'wb') as fp:
            pickle.dump(self._glove, fp)

    def _doTextPart(self):
        """
        Does all the work related to text part:
        filtration by word frequency, spell correction,
        data frame columns update. Also, caches the results
        """
        if ((self._cache/self._save_t2i_as).exists() and
                (self._cache/self._save_df_as).exists()):
            self.df = pd.read_pickle(self._cache/self._save_df_as)
            with open(self._cache/self._save_t2i_as, 'rb') as fp:
                self.t2i = pickle.load(fp)
                self._max_len, self._act_max_len = pickle.load(fp)
            return

        ext = self._path.suffix
        if ext == '.json':
            self.df = pd.read_json(self._path)
        elif ext == '.pkl':
            self.df = pd.read_pickle(self._path)
        else:
            raise TypeError("'json' and 'pkl' are only supported")

        tokens = ['PAD']
        self._extractObjects(self._mode, tokens)
        self._extractActions(tokens)
        self.df.label = self.df.apply(
                lambda x: re.sub('something',
                    ' '.join(x.placeholders),
                    ' '.join(x.template)), axis=1)
        self.df.label = self.df.label.map(str.split)
        self._act_max_len = max(map(len, self.df.template))
        if self._mode == 'toy': self._max_len = self._act_max_len
        else: self._max_len = max(map(len, self.df.label))

        self.t2i = {t: i for i, t in enumerate(tokens)}
        self._cache.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(self._cache/self._save_df_as)
        with open(self._cache/self._save_t2i_as, 'wb') as fp:
            pickle.dump(self.t2i, fp)
            pickle.dump((self._max_len, self._act_max_len), fp)

    def _extractActions(self, tokens):
        """
        Updates the `self.df.template` series and  collects tokens
        encountered in the column into `tokens`
        """
        self.df.template = self.df.template.map(
                #lambda x: re.sub('\[|\]', '', x))
                lambda x: re.sub('\[.*?\]', 'something', x))
                # there are strings where there are more than just
                # word 'something'. In this case, the words characterize
                # the object better, however, it may not help to
                # train the network
        self.df.template = self.df.template.map(str.lower)
        sentences = self.df.template.unique()
        if self._GF:
            bad_sens, bad_ids = [], []
            for i, sen in enumerate(sentences):
                for w in sen.split():
                    if w not in self._glove:
                        bad_sens.append(sen)
                        bad_ids.append(i)
                        break
            mask = self.df.template.isin(bad_sens)
            self.df = self.df[~mask]
            sentences = np.delete(sentences, bad_ids)
            self.df.index = np.arange(len(self.df))
        self.df.template = self.df.template.map(wordpunct_tokenize)
        token_counts = Counter(
                np.concatenate(list(map(str.split, sentences))))
        tokens += list(token_counts.keys())

    def _extractObjects(self, mode, tokens):
        """
        Updates the `self.df.placeholders` series according to the processing
        `mode` policy. Collects tokens encountered in the column into `tokens`
        """
        sentences = self.df.placeholders
        if mode in ['simple', 'toy']:
            mask = (sentences.map(len) == 1)
            sentences = sentences[mask]
            self.df = self.df[mask]
            if mode == 'toy':
                mask = (sentences.map(
                    lambda x: len(wordpunct_tokenize(x[0])) == 1))
                sentences = sentences[mask]
                self.df = self.df[mask]
            sentences = sentences.map(lambda x: x[0])
            mask = sentences.map(lambda x: 'something' not in x)
            sentences = sentences[mask]
            self.df = self.df[mask]
            self.df.index = np.arange(mask.sum())

        sentences = list(sentences.map(wordpunct_tokenize).values)
        # << converting to list to be able to delete elements
        token_counts = self._findTypos(sentences)
        self._updateObjects(sentences)

        for w in self._vague_words:
            token_counts.pop(w)
        for w in token_counts:
            if w in self._glove:
                tokens.append(w)

    def _findTypos(self, sentences):
        """
        Colects suspicious (rare) words in `self._vague_words`.
        `self._min_freq` defines the rarity extent.
        If `self._check_spell` == True, then the words found
        will be corrected with pyspellchecker

        Returns counts of the words in the sentences
        """
        def freqFilter(C):
            return [] if self._min_freq < 1 else [
                    x for x in C.keys() if C[x] <= self._min_freq]

        token_counts = Counter(np.concatenate(sentences))
        self._vague_words = freqFilter(token_counts)

        if not (self._check_spell and self._vague_words):
            return token_counts

        spell = SpellChecker()
        checked_words = []
        pbar = tqdm(
            sentences,
            'Spell-check',
            bar_format=self._tqdmBF)

        for sen in pbar:
            for i, w in enumerate(sen):
                if w in self._vague_words:
                    sen[i] = spell.correction(w)
                    checked_words.append(sen[i])
        token_counts.update(checked_words)
        self._vague_words = freqFilter(token_counts)
        return token_counts

    def _updateObjects(self, sentences):
        """
        Subtitutes 'mispelled' words (those that are in `self._vague_words`)
        in the `self.df.placeholders` with the `sentences` given
        """
        def to_remove(word):
            vague = word in self._vague_words
            if self._GF:
                not_found = word not in self._glove
                return vague or not_found
            return vague

        mended = copy.deepcopy(sentences)
        if self._vague_words or self._GF:
            pbar = tqdm(
                sentences,
                "Removing 'bad samples'",
                bar_format=self._tqdmBF)

            k = 0
            for i, sen in enumerate(pbar):
                flag = True
                for w in sen:
                    if to_remove(w):
                        self.df.drop(i, inplace=True)
                        del mended[k]
                        flag = False
                        break
                if flag:
                    k += 1

        self.df.placeholders = mended
        self.df.index = np.arange(len(self.df))
