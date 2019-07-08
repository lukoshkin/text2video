import json
import pickle

import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, path, cache, 
                 video_shape=(32, 64, 64, 3), 
                 step=2, transform=None, ext='webm'):
        self.transform = transform

        path = Path(path)
        file_name = path.stem.split('.')[0]

        cache = Path(cache)
        if (cache / f"{file_name}.db").exists():
            with open(cache / f"{file_name}.db", 'rb') as fp:
                self.data = pickle.load(fp)
        else:
            self.data = []
            cache.mkdir(parents=True, exist_ok=True)

            with open(path) as fp:
                raw_data = json.load(fp)
            
            mult = []
            corrupted = 0
            D, H, W, C = video_shape
            folder = path.parents[0]
            for sample in tqdm(raw_data, "Preparing dataset"):
                video = folder / f"{sample['id']}.{ext}"
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
                        image = cv2.resize(image, (H, W))
                        frames += [image]
                        CNT += 1

                ViCap.release()
                cv2.destroyAllWindows()

                if CNT == D * mult[-1]:
                    frames = np.array(frames, 'uint8')
                    self.data.append (
                        (  sample['label'], 
                           frames[::step * mult[-1]]  )
                    )
                else:
                    corrupted += 1
                    mult.pop()

            print('No of corrupted videos', corrupted)
            with open(cache / f'{file_name}.db', 'wb') as fp:
                pickle.dump(self.data, fp)

    def __getitem__(self, index):
        label, video = self.data[index]

        return {'label' : label, 
                'video' : video}

    def __len__(self):
        return len(self.data)
