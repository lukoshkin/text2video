import cv2
import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path
from smart_open import open
from torch.utils.data import Dataset
from text_processing2 import TextProcessor


class LabeledVideoDataset(TextProcessor, Dataset):
    """
    Args:
    -----
    path2lables - first required positional argument.
    path2videos - second required positional argument.
    cache - folder to preserve intermediate and final results.
    video_shape=(D, H, W, C), where D is the number of frames.
    step - if a video has multiple of D frames, then this var
           is applied to reduce their number.
    mode - the way of screening samples based a sentence object.
           If 'toy', only sentences with single word objects are selected.
           If 'simple' - with single-word and hyphenated compound word objects.
           If 'casual' - any, including those presented by multiple word objects.
    check_spell - whether or not to correct words spelling.
    min_word_freq - maximum word frequency of rare words. Sentences
                    containing them will be discarded from the dataset.
    glove_folder - path to GloVe embeddings (default: ../embeddings).
    emb_size - dimensionality of GloVE embeddings.
    glove_filtration - remove samples (sentences) containing
                       'out-of-GloVe-dictionary' words.
    transform - function which should be applied on a video.
    """
    def __init__(
            self, path2labels, path2videos, cache,
            video_shape=(32, 32, 32, 3), step=2,
            mode='toy', check_spell=True, min_word_freq=2,
            glove_folder='../embeddings', emb_size=50,
            glove_filtration=True, transform=None):

        super().__init__(
                path2labels, cache, mode,
                check_spell, min_word_freq,
                glove_folder, emb_size, glove_filtration)

        self.transform = transform if transform else lambda x: x
        self._save_db_as = Path(f'{self._path.stem}.db')
        self._video_shape = video_shape
        self._step = step

        if (cache/self._save_db_as).exists():
            with open(cache/self._save_db_as, 'rb') as fp:
                self.data = pickle.load(fp)
                self._i2i = pickle.load(fp)
        else:
            self._i2i = {}
            self.data = {'major': [], 'minor': []}

            self._prepareDatabase(Path(path2videos))
            print('Caching database to', self._save_db_as)
            with open(cache/self._save_db_as, 'wb') as fp:
                pickle.dump(self.data, fp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self._i2i, fp)
            print('Done!')

    def __len__(self):
        return len(self.data['minor'])

    def __getitem__(self, index):
        """
        'minor' is a set of extracted and encoded
          object, action performed on the object,
          and the number of tokens in the sentence
          presenting this action.

        'major' is a set of video, its description,
          and the number of words in the latter.
        """
        lbl, lbl_len, video = self.data['major'][index]
        obj_vec, act_vec, act_len = self.data['minor'][index]

        return {'major':
                    {'label': lbl,
                     'lbllen': lbl_len,
                     'video': self.transform(video)},
                'minor':
                    {'object': obj_vec,
                     'action': act_vec,
                     'actlen': act_len}}

    def getById(self, video_ids):
        """
        Extracts the samples' data by their video ids
        """
        ids = list(map(self._i2i.get, video_ids))
        selected_major = np.take(self.data['major'], ids)
        selected_minor = np.take(self.data['minor'], ids)
        return np.rec.array(selected_major), np.rec.array(selected_minor)

    def sen2vec(self, sen, mode):
        """
        Converts a sentence to a sequence of positive integers
        according to 't2i' dictionary. Depending on the mode,
        the result may be padded to the required length.

        Output type: int64 - necessary for nn.Embedding
        """
        if mode == 'toy':
            return self.t2i[sen[0]]

        filling = [self.t2i[w] for w in sen]
        if mode == 'simple':
            return np.array(filling)

        max_len = self._max_len if mode == 'action' else self._act_max_len
        numerated = np.zeros(max_len, 'int')
        numerated[:len(filling)] = filling
        return numerated

    def _prepareDatabase(self, path2videos):
        """
        Fetches videos and corresponding text representations to `self.data`
        Prepares `self._i2i` for the later use by the `self.getById` method
        """
        D, H, W, C = self._video_shape
        new_index, corrupted, mult = 0, 0, []
        pbar = tqdm(
            self.df.iterrows(), "Preparing dataset",
            len(self.df), bar_format=self._tqdmBF)

        for old_index, sample in pbar:
            video = path2videos/f"{sample.id}.webm"
            ViCap = cv2.VideoCapture(str(video))
            _D = ViCap.get(cv2.CAP_PROP_FRAME_COUNT)
            if int(_D) <  D:
                self.df.drop(old_index, inplace=True)
                continue

            mult.append(int(_D) // D)
            frames, CNT = self._extractFrames(ViCap, D*mult[-1])
            if CNT == D * mult[-1]:
                frames = np.array(frames, 'f4').transpose(3,0,1,2)
                frames = frames[:, ::self._step*mult[-1]] / 255
                self._processSample(frames, sample)
                self._i2i[sample.id] = new_index
                new_index += 1
            else:
                corrupted += 1
                mult.pop()
                self.df.drop(old_index, inplace=True)
        self.df.index = np.arange(len(self.df))
        print('No of corrupted videos:', corrupted)
        self.data['major'] = np.array(
                self.data['major'],
                [('', 'i8', self._max_len), ('', 'i8'),
                 ('', 'f4', (C, D//self._step, H, W))])
        self.data['minor'] = np.array(
                self.data['minor'],
                [('', 'O'), ('', 'i8', self._act_max_len), ('', 'i8')])

    def _extractFrames(self, ViCap, length):
        """
        Retrieves cropped to the required shape frames from the stream
        `ViCap`. The maximum number of extracted frames is limited
        to the `length` value. In addition to the list of frames,
        it returns the length of the resulting video - `CNT`
        """
        CNT = 0
        frames = []
        success = True
        while success and (CNT < length):
            success, image = ViCap.read()
            if success:
                image = cv2.resize(
                        image, self._video_shape[1:-1],
                        interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames += [image]
                CNT += 1

        ViCap.release()
        cv2.destroyAllWindows()
        return frames, CNT

    def _processSample(self, frames, sample):
        """
        Obtains objects, action and the entire sentence (all vectorized)
        from the corresponding to frames row in `self.df`. Adds the earlier
        extracted attributes to `self.data`
        """
        lbl, obj, act = sample[1:4]
        obj_len, act_len = map(len, [obj, act])
        lbl_len = act_len + obj_len - 1

        obj_vec = self.sen2vec(obj, self._mode)
        act_vec = self.sen2vec(act, 'action')
        lbl_vec = self.sen2vec(lbl, 'label')

        self.data['minor'].append((obj_vec, act_vec, act_len))
        self.data['major'].append((lbl_vec, lbl_len, frames))
