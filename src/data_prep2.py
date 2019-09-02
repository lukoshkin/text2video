import cv2
import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path
from smart_open import open
from torch.utils.data import Dataset
from text_processing2 import TextProcessor


class LabeledVideoDataset(TextProcessor, Dataset):
    def __init__(
            self, path, cache,
            video_shape=(32, 64, 64, 3), step=2,
            mode='toy', min_word_freq = 2, transform=None):
        self.transform = transform if transform else lambda x: x
        super().__init__(path, cache, mode, min_word_freq)
        self._save_db_as = Path(f'{self.path.stem}.db')
        self._video_shape = video_shape
        self._step = step

        if (self.cache/self._save_db_as).exists():
            with open(self.cache/self._save_db_as, 'rb') as fp:
                self.data = pickle.load(fp)
                self._i2i = pickle.load(fp)
        else:
            self._i2i = {}
            self.data = {'major': [], 'minor': []}

            self.doTextPart()
            self._prepareDatabase()
            print('Caching database to', self._save_db_as)
            with open(cache/self._save_db_as, 'wb') as fp:
                pickle.dump(self.data, fp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self._i2i, fp)
            print('Done!')

    def __len__(self):
        return len(self.data['minor'])

    def __getitem__(self, index):
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
        ids = map(self._i2i.get, video_ids)
        selected = np.take(self.data, list(ids))
        return np.rec.array(selected)

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

    def _prepareDatabase(self):
        """
        Fetches videos and corresponding text representations to `self.data`
        Prepares `self._i2i` for the later use by the `self.getById` method
        """
        D, H, W, C = self._video_shape
        index, corrupted, mult = 0, 0, []
        folder = self.path.parents[1]/'video'
        pbar = tqdm(self.df.iterrows(), "Preparing dataset", len(self.df))

        for _, sample in pbar:
            video = folder/f"{sample.id}.webm"
            ViCap = cv2.VideoCapture(str(video))
            _D = ViCap.get(cv2.CAP_PROP_FRAME_COUNT)
            if int(_D) <  D: continue

            mult.append(int(_D) // D)
            frames, CNT = self._extractFrames(ViCap, D*mult[-1])
            if CNT == D * mult[-1]:
                frames = np.array(frames, 'f4').transpose(3,0,1,2)
                frames = frames[:, ::self._step*mult[-1]] / 255
                self._processSample(frames, sample)
                self._i2i[sample.id] = index
                index += 1
            else:
                corrupted += 1
                mult.pop()
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

        obj_vec = self.sen2vec(obj, self.mode)
        act_vec = self.sen2vec(act, 'action')
        lbl_vec = self.sen2vec(lbl, 'label')

        self.data['minor'].append((obj_vec, act_vec, act_len))
        self.data['major'].append((lbl_vec, lbl_len, frames))
