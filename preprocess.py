import os
import re
import sys
import torch
import hashlib
from progress.bar import Bar
from concurrent.futures import ProcessPoolExecutor

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
import config
from collections import Counter

def preprocess_midi(path): #midi file->pretty midi(note_seq)->event_seq->control_seq
    note_seq, instr_to_note_count = NoteSeq.from_midi_file(path)
    #note_seq.trim_overlapped_notes()
    note_seq.adjust_time(-note_seq.notes[0].start) #把起始时间设置到0
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    event_array = event_seq.to_array()
    return event_array, control_seq.to_compressed_array(), control_seq.metadatas, instr_to_note_count

def preprocess_midi_files_under(midi_root, save_dir, num_workers):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'
    
    results = []
    executor = ProcessPoolExecutor(num_workers)
    note_count_dict = dict()
    unique_note_count_dict = dict()
    note_density_bin_dict = dict()
    instr_to_note_count_dict = dict()
    note_length_bin_dict = dict()

    for path in midi_paths:
        try:
            results.append((path, executor.submit(preprocess_midi, path)))
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
            (event_seq, control_seq, metadata_seq, instr_to_note_count) = future.result() #event_seq.to_array(), control_seq.to_compressed_array() 组成的元组
            torch.save((event_seq, control_seq), save_path)
            for metadata in metadata_seq:
                c = note_count_dict.get(metadata.note_count, 0)
                note_count_dict[metadata.note_count] = c + 1
                c = unique_note_count_dict.get(metadata.unique_note_count, 0)
                unique_note_count_dict[metadata.unique_note_count] = c + 1
                c = note_density_bin_dict.get(metadata.note_density_bin, 0)
                note_density_bin_dict[metadata.note_density_bin] = c + 1
                instr_to_note_count_dict = dict(Counter(instr_to_note_count_dict) + Counter(instr_to_note_count))
                c = note_length_bin_dict.get(metadata.note_length_bin, 0)
                note_length_bin_dict[metadata.note_length_bin] = c + 1
        except Exception as e:
            print(f'Could not process file: {path}. Error {e}')

    return note_count_dict, unique_note_count_dict, note_density_bin_dict, instr_to_note_count_dict, note_length_bin_dict


if __name__ == '__main__':
    # preprocess_midi_files_under(
    #         midi_root=sys.argv[1],
    #         save_dir=sys.argv[2],
    #         num_workers=int(sys.argv[3]))
    metadata = preprocess_midi_files_under(
        midi_root=r'.\dataset\downloaded_midis\piano',
        save_dir=r'.\dataset\processed-nes-snes-unique-notes-3',
        num_workers=6)

    import json
    import collections
    print('\nNote counts:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[0].items())), indent=2))
    print('\nUnique note counts:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[1].items())), indent=2))
    print('\nNote density bins:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[2].items())), indent=2))
    print('\nInstrument to notes count:')
    print(json.dumps(collections.OrderedDict(sorted(map(lambda kv: (kv[0].__int__(), kv[1].__int__()), metadata[3].items()))), indent=2))
    print('\nNote length bins:')
    print(json.dumps(collections.OrderedDict(sorted(metadata[4].items())), indent=2))
    print('\nDONE')

    # python3
    # preprocess.
    # dataset / midi / NAME
    # dataset / processed / NAME
