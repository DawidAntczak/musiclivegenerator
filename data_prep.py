import functools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import jsonpickle
import music21
import muspy
import pretty_midi
from music21 import *

from progress.bar import Bar

import utils
from models import PreparationMetadata


class PreparationResult:
    path = None
    key_analyze = None
    entropy = None
    pitch_class_distribution_std_dev = None
    note_density = None
    avg_time_between_attacks = None
    avg_pitches_played = None

    def __init__(self, path, key_analyze, entropy, pitch_class_distribution_std_dev, note_density, avg_time_between_attacks, avg_pitches_played):
        self.path = path
        self.key_analyze = key_analyze
        self.entropy = entropy
        self.pitch_class_distribution_std_dev = pitch_class_distribution_std_dev
        self.note_density = note_density
        self.avg_time_between_attacks = avg_time_between_attacks
        self.avg_pitches_played = avg_pitches_played


def transpose_to_common(key_analyze, file_path, save_path):
    target_tonic_pitch = 'C' if key_analyze.mode == 'major' else 'A'
    key_interval = interval.Interval(key_analyze.tonic, pitch.Pitch(target_tonic_pitch))
    semitones = key_interval.semitones + 12 if key_interval.semitones < -6 else key_interval.semitones
    midi = pretty_midi.PrettyMIDI(file_path)
    transpose(midi, semitones)
    midi.write(save_path)


def transpose(mid, semitones):
    if semitones == 0:
        return
    for inst in mid.instruments:
        if not inst.is_drum:    # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def prepare_midi(path, save_dir, collect_only_metadata=False):
    name = os.path.basename(path)
    save_path = os.path.join(save_dir, name)

    midi_stream = converter.parse(path)
    key_analyze = midi_stream.analyze('key')

    if collect_only_metadata is False:
        transpose_to_common(key_analyze, path, save_path)

    midi = pretty_midi.PrettyMIDI(path)
    muspy_music = muspy.inputs.from_pretty_midi(midi)

    entropy = muspy.metrics.pitch_entropy(muspy_music).__float__()

    #fe = music21.features.jSymbolic.PitchClassDistributionFeature(midi_stream)
    #pitch_class_distribution_std_dev = np.std(fe.extract().vector).__float__()

    fe = music21.features.jSymbolic.NoteDensityFeature(midi_stream)
    note_density = fe.extract().vector[0]

    #fe = music21.features.jSymbolic.AverageTimeBetweenAttacksFeature(midi_stream)
    #avg_time_between_attacks = fe.extract().vector[0]

    avg_pitches_played = muspy.metrics.polyphony(muspy_music).__float__()

    return PreparationResult(path, key_analyze, entropy, 0, note_density,
                      0, avg_pitches_played)


def batcher(x, bs):
    return [x[i:i+bs] for i in range(0, len(x), bs)]


def prepare_midi_files_under(midi_root, save_dir, num_workers, batch_size=1000, collect_only_metadata=False):
    #midi_paths = list(filter(lambda x: '-speed_00' in x, utils.find_files_by_extensions(midi_root, ['.mid', '.midi'])))
    #midi_paths = list(filter(lambda x: '-speed_00' in x or '-speed_01' in x or '-speed_03' in x, utils.find_files_by_extensions(midi_root, ['.mid', '.midi'])))
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)

    executor = ProcessPoolExecutor(num_workers)
    batched_midi_paths = batcher(midi_paths, batch_size)
    for i, batch in enumerate(batched_midi_paths):
        futures = []
        for path in batch:
            try:
                partial_prepare_midi = functools.partial(prepare_midi, path, save_dir, collect_only_metadata)
                futures.append((path, executor.submit(partial_prepare_midi)))
            except KeyboardInterrupt:
                print(' Abort')
                return
            except:
                print(' Error')
                continue

        batch_results = []
        for path, future in Bar('Processing').iter(futures):
            try:
                batch_results.append(future.result())
            except Exception as e:
                print(f'Could not process file: {path}. Error {e}')

            metadata = list(map(lambda r: PreparationMetadata(r), batch_results))
            with open(os.path.join(save_dir, f'metadata_{i}.json'), 'w') as f:
                print(jsonpickle.encode(metadata, indent=2), file=f)


if __name__ == '__main__':
    print("Started sleeping at: ", datetime.now())
    #time.sleep(10800)
    print("Ended sleeping at: ", datetime.now())

    midi_root = r'C:\DATA\prep\all-game-piano-music-30s-2'
    save_dir = r'C:\DATA\prep\all-game-piano-music-30s-2-transposed'
    prepare_midi_files_under(
        midi_root=midi_root,
        save_dir=save_dir,
        num_workers=12,
        batch_size=2000,
        collect_only_metadata=False
    )


