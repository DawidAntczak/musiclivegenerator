import functools
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from statistics import mean

import jsonpickle
import music21.features.jSymbolic
import muspy
import numpy as np
import pretty_midi
import torch
from music21 import *
import csv

from progress.bar import Bar

import utils
from models import PreparationMetadata
from sequence import NoteSeq, EventSeq, ControlSeq


class PreparationResult:
    path = None
    key_analyze = None

    def __init__(self, path, key_analyze):
        self.path = path
        self.key_analyze = key_analyze


def prepare_midi(path):
    #name = os.path.basename(path)
    #save_path = os.path.join(save_dir, name)

    midi_stream = converter.parse(path, quantizePost=False)
    key_analyze = midi_stream.analyze('key')
    i = interval.Interval(key_analyze.tonic, pitch.Pitch('D'))
    s_new = midi_stream.transpose(i)

    print(s_new.write('midi'))
    key_analyze = midi_stream.analyze('key')
    return key_analyze


def get_intervals_tops(all_metadata, param_name, intervals_count):
    values = map(lambda m: getattr(m, param_name), all_metadata)
    sorted_values = sorted(values)
    tops = list()
    for i in range(1, intervals_count):
        interval_end_index = ((len(sorted_values) / intervals_count) * i).__int__()
        tops.append(sorted_values[interval_end_index])
    return tops


if __name__ == '__main__':
    # preprocess_midi_files_under(
    #         midi_root=sys.argv[1],
    #         save_dir=sys.argv[2],
    #         num_workers=int(sys.argv[3]))
    #result = prepare_midi(
    #    r'C:\Users\Dawid\AppData\Local\Temp\music21\game-piano-60s-transposed\0fithos_00.midi'
    #)

    #for chord in result.recurse().getElementsByClass('Chord'):
    #    print(chord)

    #ee = pitch_entropy(pretty_midi.PrettyMIDI(r'C:\DATA\prep\game-piano-30s-betterV2\0fithos-speed_00-timesplit_01.midi'))
    #StaccatoIncidenceFeature
    #VariabilityOfNoteDurationFeature
    #RepeatedNotesFeature
    #VariabilityOfNoteDurationFeature
    #midi_stream = converter.parse(r'C:\DATA\prep\game-piano-30s-betterV2\One_Winged_Angel_REBORN-speed_00-timesplit_01.midi', quantizePost=False)

    """
    results = dict()
    midi_paths = list(utils.find_files_by_extensions(r'C:\Repos\EmotionBox\output\2023-01-04 11-07-35', ['.mid', '.midi']))
    for path in midi_paths:
        music = muspy.from_pretty_midi(pretty_midi.PrettyMIDI(path))
        val = muspy.pitch_entropy(music)
        key = 0 if 'ent_0' in path else 1 if 'ent_1' in path else 5
        l = results.get(key, list())
        l.append(val)
        results[key] = l
    for k in results.keys():
        print(f'{k}: {mean(results[k])}')
    """

    #f = fe.extract()
    #print(np.std(f.vector).__float__())

    #path = r'C:\DATA\prep\example\pol2-2-q-speed_00.mid'
    #note_seq, instr_to_note_count = NoteSeq.from_midi_file(path)
    #note_seq.trim_overlapped_notes()
    #with open(os.path.join(r'C:\DATA\prep\example-transposed', 'metadata.json')) as f:
    #    all_metadata = jsonpickle.decode(f.read())

    #metadata = [x for x in all_metadata if x.name == os.path.basename(path)][0]

    #note_seq.adjust_time(-note_seq.notes[0].start) #把起始时间设置到0
    #event_seq = EventSeq.from_note_seq(note_seq)
    #control_seq = ControlSeq.from_event_seq(event_seq, metadata)
    #event_array = event_seq.to_array()

    metadata_paths = list(utils.find_files_by_extensions(r'C:\DATA\prep\maestro-v3.0.0-30s-transposed', ['.json']))
    all_metadata = list()
    for metadata_path in metadata_paths:
        with open(metadata_path) as f:
            metadata_part = jsonpickle.decode(f.read())
            all_metadata.extend(metadata_part)

    intervals_tops = get_intervals_tops(all_metadata, 'note_density', 6)
    print(intervals_tops)
    intervals_tops = get_intervals_tops(all_metadata, 'avg_pitches_played', 3)
    print(intervals_tops)
    intervals_tops = get_intervals_tops(all_metadata, 'entropy', 3)
    print(intervals_tops)
    #midi_stream = converter.parse(r'C:\DATA\prep\example\pol2-2-q-speed_00.mid', quantizePost=True)
    #p = str(midi_stream.write('midi'))
    #midi = pretty_midi.PrettyMIDI(p)
    #print(torch.version.cuda)
