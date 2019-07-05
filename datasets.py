import json
import pickle

import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, path, cache, vlen, 
                 step=1, transform=None):
        self.transform = transform

        path = Path(path)
        file_name = path.stem.split('.')[0]

        cache = Path(cache)
        if (cache / file_name).exists():
            with open(cache / file_name, 'rb') as fh:
                self.data = pickle.load(fh)
        else:
            self.data = []
            durations = []
            cache.mkdir(parents=True, exist_ok=True)

            with open(path) as fh:
                raw_data = json.load(fh)
            
            folder = path.parents[0]
            for sample in tqdm(raw_data, "Preparing dataset"):
                video = folder / f"{sample['id']}.webm"
                ViCap = cv2.VideoCapture(video)

                frames = []
                success = True
                while success:
                    success, image = ViCap.read()
                    if success:
                        frames += [image]

                ViCap.release()
                cv2.destroyAllWindows()
                frames = np.array(frames, 'uint8')

                assert len(frames) > 0, \
                "Something went wrong, no frames were extracted"
                durations.append(len(frames)) 

                self.data.append([sample['label'], frames])
            
            with open(cache / f'{file_name}.db', 'wb') as fh:
                pickle.dump(self.data, fh)
                pickle.dump(durations, fh)

    def __getitem__(self, index):
        label, frames = self.data[index]
        return {'label'  : label, 
                'values' : self.transform(frames[::step])}

    def __len__(self):
        return len(self.data)
        
        

class ImagesFromVideoDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'rb') as fh:
            self.data = pickle.load(fh)
            self.scan_r = np.cumsum(pickle.load(fh))
            self.scan_l = np.r_[0, self.scan_r[:-1]]    

        self.transform = transform if transform else lambda x: x

    def __getitem__(self, index):
        vi_no = np.searchsorted(self.scan_r, index, 'right')
        im_no = index - self.scan_l[vi_no] 
        label, frames = self.data  [vi_no]

        # further, label may be changed to placeholders
        return {'label'  : label, 
                'values' : self.transform(frames[im_no])}

    def __len__(self):
        return self.scan_r[-1]

