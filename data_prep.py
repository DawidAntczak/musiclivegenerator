import functools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import jsonpickle
import pretty_midi
from music21 import *
import csv

from progress.bar import Bar

import utils


class PreparationResult:
    path = None
    key_analyze = None

    def __init__(self, path, key_analyze):
        self.path = path
        self.key_analyze = key_analyze


class KeyMetadata:
    tonic = None
    mode = None
    type = None
    correlation_coefficient = None
    semitones_to_c = None

    def __init__(self, key_analyze):
        self.tonic = key_analyze.tonic.name
        self.mode = key_analyze.mode
        self.type = key_analyze.type
        self.correlation_coefficient = key_analyze.correlationCoefficient
        key_interval = interval.Interval(key_analyze.tonic, pitch.Pitch('C'))
        self.semitones_to_c = key_interval.semitones + 12 if key_interval.semitones < -6 else key_interval.semitones


class PreparationMetadata:
    path = None
    name = None
    primary_key = None
    secondary_key = None

    def __init__(self, preparation_result):
        self.path = preparation_result.path
        self.name = os.path.basename(preparation_result.path)
        self.primary_key = KeyMetadata(preparation_result.key_analyze)
        self.secondary_key = KeyMetadata(preparation_result.key_analyze.alternateInterpretations[0])


def transpose_to_c(key_analyze, file_path, save_path):
    #alt_text = ''
    #for interpretation in key_analyze.alternateInterpretations:
    #for interpretation in key_analyze.alternateInterpretations:
    #    alt_text += str(interpretation) + ' ' + str(interpretation.correlationCoefficient) + ' - '

    #print(str(key_analyze) + ' ' + str(key_analyze.correlationCoefficient) + ' --- ' + alt_text + "\n")
    key_interval = interval.Interval(key_analyze.tonic, pitch.Pitch('C'))
    semitones = key_interval.semitones + 12 if key_interval.semitones < -6 else key_interval.semitones
    midi = pretty_midi.PrettyMIDI(file_path)
    transpose(midi, semitones)
    midi.write(save_path)


def transpose(mid, semitones):
    if semitones == 0:
        return
    for inst in mid.instruments:
        if not inst.is_drum: # Don't transpose drum tracks
            for note in inst.notes:
                note.pitch += semitones


def prepare_midi(path, save_dir, collect_only_metadata=False):
    name = os.path.basename(path)
    save_path = os.path.join(save_dir, name)

    midi_stream = converter.parse(path, quantizePost=False)
    key_analyze = midi_stream.analyze('key')

    if collect_only_metadata is False:
        transpose_to_c(key_analyze, path, save_path)

    return key_analyze


def prepare_midi_files_under(midi_root, save_dir, num_workers, collect_only_metadata=False):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)

    executor = ProcessPoolExecutor(num_workers)
    futures = []
    for path in midi_paths:
        try:
            partial_prepare_midi = functools.partial(prepare_midi, path, save_dir, collect_only_metadata)
            futures.append((path, executor.submit(partial_prepare_midi)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

    results = []
    for path, future in Bar('Processing').iter(futures):
        try:
            results.append(PreparationResult(path, future.result()))
        except Exception as e:
            print(f'Could not process file: {path}. Error {e}')

    return results

if __name__ == '__main__':
    # preprocess_midi_files_under(
    #         midi_root=sys.argv[1],
    #         save_dir=sys.argv[2],
    #         num_workers=int(sys.argv[3]))
    result = prepare_midi_files_under(
        midi_root=r'C:\Repos\EmotionBox\dataset\maestro-v3.0.0\2018',
        save_dir=r'C:\Users\Dawid\AppData\Local\Temp\music21\piano-2018-transposed',
        num_workers=6,
        collect_only_metadata=True
    )

    metadata = list(map(lambda r: PreparationMetadata(r), result))

    with open(r'C:\Repos\EmotionBox\dataset\maestro-v3.0.0\2018\metadata.json', 'w') as f:
        print(jsonpickle.encode(metadata, indent=2, unpicklable=False), file=f)


