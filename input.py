import numpy as np
import torch

from config import device
from sequence import ControlSeq, Control


class Input:
    mode_list, note_density = None, None
    reset = False

    def __init__(self, json_input):
        self.mode_list = json_input['Mode']
        self.note_density = json_input['NoteDensity']
        self.reset = json_input['Reset']

    def get_transformed(self):
        if self.mode_list is not None:
            return transform(self.mode_list, self.note_density)
        else:
            return transform([1]*3, self.note_density)

    def should_reset(self):
        return self.reset


def transform(raw_input_string):
    mode, note_density = raw_input_string.split(';')
    mode = list(filter(len, mode.split(',')))  # to mode char list
    return transform(mode, note_density)


def transform(mode, note_density):
    mode = np.array(list(map(float, mode)))  # to mode float nparray
    assert mode.size == 3
    assert np.all(mode >= 0)
    mode = mode / mode.sum() \
        if mode.sum() else np.ones(3) / 3

    note_density = int(note_density)
    assert note_density in range(len(ControlSeq.note_density_bins))
    control = Control(mode, note_density)
    controls = torch.tensor(control.to_array(), dtype=torch.float32)
    controls = controls.repeat(1, 1, 1).to(device)  # 1Xbatch_sizeX controls

    return controls
