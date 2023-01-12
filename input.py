import torch

from config import device
from sequence import Control


class Input:
    mode, attack_density, avg_pitches_played, entropy = None, None, None, None
    reset = False
    requested_event_count = None

    def __init__(self, json_input):
        self.mode = json_input['Mode']
        self.attack_density = json_input['AttackDensity']
        self.avg_pitches_played = json_input['AvgPitchesPlayed']
        self.entropy = json_input['Entropy']
        self.reset = json_input['Reset']
        self.requested_event_count = json_input['RequestedEventCount'] if 'RequestedEventCount' in json_input else None

    def get_transformed(self):
        control = Control(self.mode, self.attack_density, self.avg_pitches_played, self.entropy)
        controls = torch.tensor(control.to_array(), dtype=torch.float32)
        controls = controls.repeat(1, 1, 1).to(device)  # 1Xbatch_sizeX controls
        return controls

    def should_reset(self):
        return self.reset

    def get_requested_event_count(self):
        return self.requested_event_count if not None else 50
