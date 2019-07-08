import json

import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, path, cache, 
                 step=1, transform=None, ext='webm'):
        self.transform = transform

        path = Path(path)
        file_name = path.stem.split('.')[0]

        cache = Path(cache)
        if (cache / file_name).exists():
            self.data = np.load(cache / file_name)
        else:
            self.data = []
            durations = []
            cache.mkdir(parents=True, exist_ok=True)

            with open(path) as fh:
                raw_data = json.load(fh)
            
            folder = path.parents[0]
            for sample in tqdm(raw_data, "Preparing dataset"):
                video = folder / f"{sample['id']}.{ext}"
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
                self.data.append((sample['label'], frames))
            
            durations = np.array(durations, 'uint16')

            vlen = durations.min()
            N = vlen // step + 1
            self.data = np.rec.array (
                [(x, y[:vlen:step]) for x, y in self.data],
                [('', 'O'), ('', 'uint8', (N, *image.shape)]
            )
            np.savez (
                cache / f'{file_name}.db', 
                self.data, durations
            )

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
        


class ImagesFromVideoDataset(Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'rb') as fh:
            self.data = np.load(fh)
            self.scan_r = np.cumsum(np.load(fh))
        self.scan_l = np.r_[0, self.scan_r[:-1]]    

        self.transform = transform if transform else lambda x: x

    def __getitem__(self, index):
        vi_no = np.searchsorted(self.scan_r, index, 'right')
        im_no = index - self.scan_l[vi_no] 
        label, frames = self.data[vi_no]

        # further, label may be changed to placeholders
        return {'label'  : label, 
                'values' : self.transform(frames[im_no])}

    def __len__(self):
        return self.scan_r[-1]

