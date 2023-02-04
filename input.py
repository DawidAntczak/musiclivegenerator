import torch

from config import device
from sequence import Control


class Input:
    mode, attack_density, avg_pitches_played, entropy = None, None, None, None
    reset = False
    requested_time_length = None
    temperature = None

    def __init__(self, json_input):
        self.mode = json_input['Mode'] if 'Mode' in json_input else None
        self.attack_density = json_input['AttackDensity'] if 'AttackDensity' in json_input else None
        self.avg_pitches_played = json_input['AvgPitchesPlayed'] if 'AvgPitchesPlayed' in json_input else None
        self.entropy = json_input['Entropy'] if 'Entropy' in json_input else None
        self.reset = json_input['Reset']
        self.requested_time_length = json_input['RequestedTimeLength'] if 'RequestedTimeLength' in json_input else 5
        self.temperature = json_input['Temperature'] if 'Temperature' in json_input and json_input['Temperature'] > 0 else 1

    def get_transformed(self):
        control = Control(self.mode, self.attack_density, self.avg_pitches_played, self.entropy)
        controls = torch.tensor(control.to_array(), dtype=torch.float32)
        controls = controls.repeat(1, 1, 1).to(device)
        return controls

    def should_reset(self):
        return self.reset

    def get_requested_time_length(self):
        return self.requested_time_length

    def get_temperature(self):
        return self.temperature
