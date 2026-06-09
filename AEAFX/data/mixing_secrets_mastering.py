import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from . import fx

import numpy as np
import os
import json
import random
import librosa


class MSMastering_Dataset(Dataset):
    def __init__(
        self, root_dir: str, aligned_dir="aligned", audio_length=20, samplerate=44100
    ):
        super(MSMastering_Dataset, self).__init__()
        self.root_dir = root_dir
        self.samplerate = samplerate
        self.audio_length_sp=audio_length*self.samplerate

        self.rng = np.random.default_rng()

        fp = open(os.path.join(self.root_dir, "mixing_secrets_mastering_aligned.json"))
        self.metadata = json.load(fp=fp)
        fp.close()

        self.audio_dir = os.path.join(root_dir, aligned_dir)

        self.list_keys = list(self.metadata.keys())

        list_audio=[]
        for key in self.list_keys:
            project_md=self.metadata[key]
            mix_audio_path = os.path.join(self.audio_dir, project_md["unmastered_file"])
            master_audio_path = os.path.join(self.audio_dir, project_md["mastered_file"])

            mix, _ = librosa.load(mix_audio_path, sr=samplerate, mono=True)
            master, _ = librosa.load(master_audio_path, sr=samplerate, mono=True)
            list_audio.append({"mix":mix, "master":master})
        self.list_audio=list_audio

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        mm = self.list_audio[idx]
        mix_audio = mm['mix']
        master_audio = mm['master']

        n = len(mix_audio)
        start = np.random.randint(0, n-self.audio_length_sp)

        mix_audio=torch.Tensor(mix_audio[start:start+self.audio_length_sp]).unsqueeze(0)
        master_audio=torch.Tensor(master_audio[start:start+self.audio_length_sp]).unsqueeze(0)

        mix_audio=mix_audio/mix_audio.abs().amax(dim=1, keepdim=True)
        master_audio=master_audio/master_audio.abs().amax(dim=1, keepdim=True)

        return mix_audio, master_audio, torch.zeros(1)


class MSMastering_Single(Dataset):
    def __init__(
        self,
        root_dir: str,
        aligned_dir="aligned",
        audio_length=10,
        samplerate=22050,
        key_num=0,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.samplerate = samplerate
        self.audio_length = audio_length

        self.rng = np.random.default_rng()

        fp = open(os.path.join(self.root_dir, "mixing_secrets_mastering_aligned.json"))
        self.metadata = json.load(fp=fp)
        fp.close()

        self.audio_dir = os.path.join(root_dir, aligned_dir)

        self.key = list(self.metadata.keys())[key_num]

        self.project_md = self.metadata[self.key]
        print(self.project_md["artist"])
        print(self.project_md["project"])

        self.mix_audio, self.mix_sr = torchaudio.load(
            os.path.join(self.audio_dir, self.project_md["unmastered_file"])
        )

        self.master_audio, self.master_sr = torchaudio.load(
            os.path.join(self.audio_dir, self.project_md["mastered_file"])
        )

        if self.mix_sr != self.samplerate or self.master_sr != self.samplerate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.mix_sr, new_freq=samplerate, lowpass_filter_width=32
            )
            self.mix_audio = self.resampler(self.mix_audio)
            self.master_audio = self.resampler(self.master_audio)
            self.mix_sr = self.samplerate
            self.master_sr = self.samplerate

        audio_length_sp = audio_length * samplerate
        self.num_patches = int(self.mix_audio.size(1) / (audio_length_sp))

        self.patches_list: list[dict] = []

        start_sp = 0
        for i in range(self.num_patches):
            end_sp = start_sp + audio_length_sp
            self.patches_list.append(
                {
                    "mix": self.mix_audio[:, start_sp:end_sp],
                    "master": self.master_audio[:, start_sp:end_sp],
                }
            )
            start_sp += audio_length_sp

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx: int):

        start_sp = int(idx * self.audio_length * self.samplerate)
        end_sp = int(start_sp + self.audio_length * self.samplerate)

        patches_dict = self.patches_list[idx]
        mix_audio = patches_dict["mix"]
        master_audio = patches_dict["master"]

        return mix_audio, master_audio, torch.Tensor([0])


class MSMastering_Single_test(Dataset):
    def __init__(
        self,
        root_dir: str,
        aligned_dir="aligned",
        samplerate=22050,
        key_num=0,
        num_tries: int = 10,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.samplerate = samplerate
        self.num_tries = num_tries
        self.rng = np.random.default_rng()

        fp = open(os.path.join(self.root_dir, "mixing_secrets_mastering_aligned.json"))
        self.metadata = json.load(fp=fp)
        fp.close()

        self.audio_dir = os.path.join(root_dir, aligned_dir)

        self.key = list(self.metadata.keys())[key_num]

        self.project_md = self.metadata[self.key]

        self.mix_audio, self.mix_sr = torchaudio.load(
            os.path.join(self.audio_dir, self.project_md["unmastered_file"])
        )

        self.master_audio, self.master_sr = torchaudio.load(
            os.path.join(self.audio_dir, self.project_md["mastered_file"])
        )

        if self.mix_sr != self.samplerate or self.master_sr != self.samplerate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.mix_sr, new_freq=samplerate, lowpass_filter_width=32
            )
            self.mix_audio = self.resampler(self.mix_audio)
            self.master_audio = self.resampler(self.master_audio)
            self.mix_sr = self.samplerate
            self.master_sr = self.samplerate

        len_mix = self.mix_audio.size(1)
        len_master = self.master_audio.size(1)
        len_min = min(len_mix, len_master)

        self.mix_audio = self.mix_audio[:, :len_min]
        self.master_audio = self.master_audio[:, :len_min]

    def __len__(self):
        return self.num_tries

    def __getitem__(self, idx: int):

        return self.mix_audio, self.master_audio, torch.Tensor([0])


class MSMastering_Dataset_2(Dataset):
    """Mastering class for the mixing secrets mastering dataset.

    Arguments:
    ----------
     -
    """

    def __init__(
        self,
        root_dir,
        list_keys=None,
        aligned_dir="aligned",
        audio_length=10,
        samplerate=22050,
        load_all_audio=True,
        fx: fx.DAFx = None,
    ):
        super(MSMastering_Dataset_2, self).__init__()
        self.root_dir = root_dir
        self.samplerate = samplerate
        self.audio_length = audio_length
        self.list_keys = list_keys
        self.load_all_audio = load_all_audio

        self.fx = fx

        self.do_resample = samplerate != 44100
        if self.do_resample:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=44100, new_freq=samplerate, lowpass_filter_width=32
            )

        fp = open(os.path.join(self.root_dir, "mixing_secrets_mastering_aligned.json"))
        self.metadata = json.load(fp=fp)
        fp.close()

        self.audio_dir = os.path.join(root_dir, aligned_dir)

        if self.list_keys == None:
            self.list_keys = list(self.metadata.keys())

        self.patch_list = []
        for key in self.list_keys:
            p_md = self.metadata[key]
            mix_audio, mix_sr = torchaudio.load(
                os.path.join(self.audio_dir, p_md["unmastered_file"])
            )
            if self.do_resample:
                mix_audio = self.resampler(mix_audio)
                mix_sr = samplerate

            len_sp = mix_audio.size(1)
            len_s = len_sp / self.samplerate
            num_patches = int(len_s / self.audio_length)
            start = 0
            if self.load_all_audio:
                master_audio, master_sr = torchaudio.load(
                    os.path.join(self.audio_dir, p_md["mastered_file"])
                )

                if self.do_resample:
                    master_audio = self.resampler(master_audio)
                    master_sr = samplerate

                for i in range(num_patches):
                    end = start + self.audio_length * self.samplerate

                    mix_crop = mix_audio[:, start:end]
                    master_crop = master_audio[:, start:end]

                    self.patch_list.append(
                        {
                            "mix_audio": mix_crop,
                            "master_audio": master_crop,
                            "mix_sr": mix_sr,
                            "master_sr": master_sr,
                            "mastered_file": p_md["mastered_file"],
                            "unmastered_file": p_md["unmastered_file"],
                            "start_sp": start,
                            "end_sp": start + self.audio_length * self.samplerate,
                        }
                    )
                    start += self.audio_length * self.samplerate

            else:
                for i in range(num_patches):
                    end = start + self.audio_length * self.samplerate
                    self.patch_list.append(
                        {
                            "mastered_file": p_md["mastered_file"],
                            "unmastered_file": p_md["unmastered_file"],
                            "start_sp": start,
                            "end_sp": start + self.audio_length * self.samplerate,
                        }
                    )
                    start += self.audio_length * self.samplerate

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx: int):
        patch_md = self.patch_list[idx]

        if self.load_all_audio:
            mix_audio = patch_md["mix_audio"]
            mix_sr = patch_md["mix_sr"]
            master_audio = patch_md["master_audio"]
            master_sr = patch_md["master_sr"]
        else:
            mix_audio, mix_sr = torchaudio.load(
                os.path.join(self.audio_dir, patch_md["unmastered_file"])
            )

            master_audio, master_sr = torchaudio.load(
                os.path.join(self.audio_dir, patch_md["mastered_file"])
            )

            if self.do_resample:
                mix_audio = self.resampler(mix_audio)
                master_audio = self.resampler(master_audio)
                mix_sr = self.samplerate
                master_sr = self.samplerate

            start_sp = patch_md["start_sp"]
            end_sp = patch_md["end_sp"]

            mix_audio = mix_audio[:, start_sp:end_sp]
            master_audio = master_audio[:, start_sp:end_sp]

        if mix_sr != self.samplerate or master_sr != self.samplerate:
            assert ValueError("Wrong samplerate")

        if self.fx is not None:
            v = np.random.rand(self.fx.num_parameters)
            master_audio = master_audio.squeeze(0).numpy()
            master_audio = self.fx(master_audio, v)
            master_audio = Tensor(master_audio)
            master_audio = master_audio.unsqueeze(0)
        else:
            v = torch.Tensor([0])

        # mix_audio = mix_audio / mix_audio.std(1, keepdim=True)
        # master_audio = master_audio / master_audio.std(1, keepdim=True)

        return mix_audio, master_audio, Tensor([0])


class MSMastering_Dataset_Full(Dataset):
    """Mastering class for the mixing secrets mastering dataset.

    Arguments:
    ----------
     -
    """

    def __init__(
        self,
        root_dir,
        list_keys=None,
        aligned_dir="aligned",
        samplerate=22050,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.samplerate = samplerate
        self.list_keys = list_keys

        self.fx = fx

        self.do_resample = samplerate != 44100

        fp = open(os.path.join(self.root_dir, "mixing_secrets_mastering_aligned.json"))
        self.metadata = json.load(fp=fp)
        fp.close()

        self.audio_dir = os.path.join(root_dir, aligned_dir)

        if self.list_keys == None:
            self.list_keys = list(self.metadata.keys())

        self.patch_list = []
        for key in self.list_keys:
            p_md = self.metadata[key]
            mix_audio, mix_sr = torchaudio.load(
                os.path.join(self.audio_dir, p_md["unmastered_file"]), normalize=True
            )
            if self.do_resample:
                mix_audio = torch.Tensor(
                    librosa.resample(
                        mix_audio.numpy(), orig_sr=mix_sr, target_sr=self.samplerate
                    )
                ).to(mix_audio)

            master_audio, master_sr = torchaudio.load(
                os.path.join(self.audio_dir, p_md["mastered_file"])
            )

            if self.do_resample:
                master_audio = torch.Tensor(
                    librosa.resample(
                        master_audio.numpy(),
                        orig_sr=master_sr,
                        target_sr=self.samplerate,
                    )
                ).to(master_audio)
            
            self.patch_list.append(
                {
                    "mix_audio": mix_audio,
                    "master_audio": master_audio,
                    "mix_sr": mix_sr,
                    "master_sr": master_sr,
                    "mastered_file": p_md["mastered_file"],
                    "unmastered_file": p_md["unmastered_file"],
                }
            )


    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx: int):
        patch_md = self.patch_list[idx]

        mix_audio = patch_md["mix_audio"]
        mix_sr = patch_md["mix_sr"]
        master_audio = patch_md["master_audio"]
        master_sr = patch_md["master_sr"]

        if mix_sr != self.samplerate or master_sr != self.samplerate:
            assert ValueError("Wrong samplerate")

        mix_audio=mix_audio/mix_audio.abs().amax(dim=1, keepdim=True)
        master_audio=master_audio/master_audio.abs().amax(dim=1, keepdim=True)

        return mix_audio, master_audio, Tensor([0])


def get_MSMastering_splits(
    root_dir,
    splits=[0.8, 0.1, 0.1],
    shuffle=True,
    seed=None,
    synfx: fx.DAFx = None,
    **dataset_kwargs
):
    fp = open(os.path.join(root_dir, "mixing_secrets_mastering_aligned.json"))
    metadata = json.load(fp=fp)
    fp.close()
    list_keys = list(metadata.keys())

    if seed is not None:
        random.seed(seed)
    if shuffle:
        random.shuffle(list_keys)

    num_splits = len(splits)
    num_songs = len(list_keys)

    if sum(splits) != 1:
        assert ValueError("The sum of splits should be 1.")

    num_examples_list = []
    end_idx_float = 0
    end_idx_list = []
    for i, split in enumerate(splits):
        end_idx_float += split * num_songs
        end_idx_list.append(int(end_idx_float))

    if end_idx_list[-1] != num_songs:
        end_idx_list[-1] += 1

    prev_idx = 0
    split_keys_list = []
    for end_idx in end_idx_list:
        split_keys_list.append(list_keys[prev_idx:end_idx])
        prev_idx = end_idx

    list_datasets = []
    for i, split_keys in enumerate(split_keys_list):
        if i == 0:

            list_datasets.append(
                MSMastering_Dataset_2(
                    root_dir=root_dir, list_keys=split_keys, fx=synfx, **dataset_kwargs
                )
            )
        else:
            list_datasets.append(
                MSMastering_Dataset_2(
                    root_dir=root_dir, list_keys=split_keys, fx=None, **dataset_kwargs
                )
            )

    return list_datasets
