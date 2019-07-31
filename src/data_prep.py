import cv2
import pickle
import numpy as np

from tqdm import tqdm
from pathlib import Path
from smart_open import open
from torch.utils.data import Dataset
from text_processing import doTextPart, sen2vec


class LabeledVideoDataset(Dataset):
    def __init__(
            self, path, cache, 
            video_shape=(32, 64, 64, 3), step=2, 
            min_word_freq = 2, check_spell=False, 
            transform=None, ext='webm'):
        self.transform = transform if transform else lambda x: x

        path = Path(path)
        file_name = path.stem

        cache = Path(cache)
        if (cache / f"{file_name}.db").exists():
            with open(cache / f"{file_name}.db", 'rb') as fp:
                self.data = pickle.load(fp)
                self.i2i = pickle.load(fp)
        else:
            self.data = []
            cache.mkdir(parents=True, exist_ok=True)

            max_len, t2i, df = doTextPart (
                            path, cache, 
                            min_word_freq, 
                            check_spell
                        )
            index = 0
            self.i2i = {}

            mult = []
            corrupted = 0
            D, H, W, C = video_shape
            folder = path.parents[1]

            pbar = tqdm(df.iterrows(), "Preparing dataset", len(df))
            for _, sample in pbar:
                video = folder / 'video' / f"{sample['id']}.{ext}"
                ViCap = cv2.VideoCapture(str(video))
                _D = ViCap.get(cv2.CAP_PROP_FRAME_COUNT)

                if int(_D) <  D :  continue
                mult.append(int(_D) // D) 

                CNT = 0
                frames = []
                success = True
                while success and (CNT < D * mult[-1]):
                    success, image = ViCap.read()
                    if success:
                        image = cv2.resize(
                                image, (H, W), 
                                interpolation=cv2.INTER_AREA)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        frames += [image]
                        CNT += 1

                ViCap.release()
                cv2.destroyAllWindows()

                if CNT == D * mult[-1]:
                    frames = np.array(
                        frames, 'float32').transpose(3,0,1,2) / 255
                    sen_len = len(sample['label'])
                    numerated = sen2vec(sample['label'], t2i, max_len)
                    self.data.append(
                        (sen_len, numerated, frames[:, ::step * mult[-1]]))
                    self.i2i[sample['id']] = index
                    index += 1
                else:
                    corrupted += 1
                    mult.pop()

            self.data = np.array(
                    self.data,
                    [('', 'int64'),
                     ('', 'int64', max_len),
                     ('', 'float32', (C, D//step, H, W))])

            # save maximum sen. length to use it further (in 'main.py')
            print('No of corrupted videos', corrupted)
            print(f'Caching database to {file_name}.db')
            with open(cache / f'{file_name}.db', 'wb') as fp:
                pickle.dump(self.data, fp)
                pickle.dump(self.i2i, fp)
            print('Done!')

    def __getitem__(self, index):
        sen_len, label, video = self.data[index]

        return {'sen_len': sen_len,
                'label': label,
                'video': self.transform(video)}

    def __len__(self):
        return len(self.data)

    def getById(self, video_ids):
        ids = map(self.i2i.get, video_ids)
        selected = np.take(self.data, list(ids))

        return np.rec.array(selected)
