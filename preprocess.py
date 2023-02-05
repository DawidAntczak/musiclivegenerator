import collections
import functools
import json
import os
import time
from datetime import datetime
import faulthandler

import jsonpickle
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
from collections import Counter


def preprocess_midi(path, metadata):
    note_seq, instr_to_note_count = NoteSeq.from_midi_file(path)
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq, metadata)
    event_array = event_seq.to_array()
    return event_array, control_seq.to_compressed_array(), control_seq.metadatas, instr_to_note_count


def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []
    executor = ProcessPoolExecutor(num_workers)
    note_count_dict = dict()
    note_density_bin_dict = dict()
    avg_pitches_played_bin_dict = dict()
    entropy_bin_dict = dict()
    instr_to_note_count_dict = dict()

    metadata_paths = list(utils.find_files_by_extensions(midi_root, ['.json']))
    all_metadata = list()
    for metadata_path in metadata_paths:
        with open(metadata_path) as f:
            metadata_part = jsonpickle.decode(f.read())
            all_metadata.extend(metadata_part)

    all_metadata_dict = {metadata.name: metadata for metadata in all_metadata}

    for path in midi_paths:
        try:
            metadata = all_metadata_dict[os.path.basename(path)]
            partial_preprocess_midi = functools.partial(preprocess_midi, path, metadata)
            results.append((path, executor.submit(partial_preprocess_midi)))
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue
    
    for path, future in Bar('Processing').iter(results):
        print(' ', end='[{}]'.format(path), flush=True)
        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()  # convert path to hashcode
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        try:
            (event_seq, control_seq, metadata_seq, instr_to_note_count) = future.result()
            torch.save((event_seq, control_seq), save_path)
            for metadata in metadata_seq:
                c = note_count_dict.get(metadata.note_count, 0)
                note_count_dict[metadata.note_count] = c + 1
                c = note_density_bin_dict.get(metadata.density_bin, 0)
                note_density_bin_dict[metadata.density_bin] = c + 1
                c = avg_pitches_played_bin_dict.get(metadata.avg_pitches_played_bin, 0)
                avg_pitches_played_bin_dict[metadata.avg_pitches_played_bin] = c + 1
                c = entropy_bin_dict.get(metadata.entropy_bin, 0)
                entropy_bin_dict[metadata.entropy_bin] = c + 1
                instr_to_note_count_dict = dict(Counter(instr_to_note_count_dict) + Counter(instr_to_note_count))
        except Exception as e:
            print(f'Could not process file: {path}. Error {e}')

    return note_count_dict, note_density_bin_dict, avg_pitches_played_bin_dict, entropy_bin_dict, instr_to_note_count_dict


if __name__ == '__main__':
    faulthandler.enable()

    # if you want to run after some time
    # print("Started sleeping at: ", datetime.now())
    # time.sleep(3600)
    # print("Ended sleeping at: ", datetime.now())

    metadata = preprocess_midi_files_under(
        midi_root=r'',  # input
        save_dir=r'dataset/everything-game-30s-transposed',   # output
        num_workers=4)


    print('\nNote counts:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[0].items())), indent=2))
    print('\nNote density bins:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[1].items())), indent=2))
    print('\nAvg pitches played bins:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[2].items())), indent=2))
    print('\nEntropy bins:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[3].items())), indent=2))
    print('\nInstrument to note count (not working properly)')
    print(json.dumps(collections.OrderedDict(sorted(map(lambda kv: (kv[0].__int__(), kv[1].__int__()), metadata[4].items()))), indent=2))
    print('\nDONE')
