import numpy as np
import torch

from config import device
from sequence import ControlSeq, Control


class Input:
    pitch_histogram_list, note_density = None, None
    reset = False

    def __init__(self, json_input):
        self.pitch_histogram_list = json_input['PitchHistogram']
        self.note_density = json_input['NoteDensity']
        self.reset = json_input['Reset']

    def get_transformed(self):
        if self.pitch_histogram_list is not None:
            return transform(self.pitch_histogram_list, self.note_density)
        else:
            return transform([1]*12, self.note_density)

    def should_reset(self):
        return self.reset


def transform(raw_input_string):
    pitch_histogram, note_density = raw_input_string.split(';')
    pitch_histogram = list(filter(len, pitch_histogram.split(',')))  # to pitch_histogram char list
    return transform(pitch_histogram, note_density)


def transform(pitch_histogram, note_density):
    pitch_histogram = np.array(list(map(float, pitch_histogram)))  # to pitch_histogram float nparray
    assert pitch_histogram.size == 12
    assert np.all(pitch_histogram >= 0)
    pitch_histogram = pitch_histogram / pitch_histogram.sum() \
        if pitch_histogram.sum() else np.ones(12) / 12

    note_density = int(note_density)
    assert note_density in range(len(ControlSeq.note_density_bins))
    control = Control(pitch_histogram, note_density)
    controls = torch.tensor(control.to_array(), dtype=torch.float32)
    controls = controls.repeat(1, 1, 1).to(device)  # 1Xbatch_sizeX controls

    return controls
