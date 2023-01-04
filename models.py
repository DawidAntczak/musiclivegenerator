import os


class PreparationMetadata:
    path = None
    name = None
    entropy = None
    pitch_class_distribution_std_dev = None
    note_density = None
    avg_time_between_attacks = None
    avg_pitches_played = None
    primary_key = None
    secondary_key = None

    def __init__(self, preparation_result):
        self.path = preparation_result.path
        self.name = os.path.basename(preparation_result.path)
        self.entropy = preparation_result.entropy
        self.pitch_class_distribution_std_dev = preparation_result.pitch_class_distribution_std_dev
        self.note_density = preparation_result.note_density
        self.avg_time_between_attacks = preparation_result.avg_time_between_attacks
        self.avg_pitches_played = preparation_result.avg_pitches_played
        self.primary_key = KeyMetadata(preparation_result.key_analyze)
        self.secondary_key = KeyMetadata(preparation_result.key_analyze.alternateInterpretations[0])


class KeyMetadata:
    tonic = None
    mode = None
    type = None
    correlation_coefficient = None

    def __init__(self, key_analyze):
        self.tonic = key_analyze.tonic.name
        self.mode = key_analyze.mode
        self.type = key_analyze.type
        self.correlation_coefficient = key_analyze.correlationCoefficient
