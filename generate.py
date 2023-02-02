import torch
import numpy as np
import os
import sys
import optparse
import time

import config
import utils
from config import device, model as model_config
from model import PerformanceRNN
from sequence import EventSeq, Control, ControlSeq

# pylint: disable=E1101,E1102


# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-c', '--control',
                      dest='control',
                      type='string',
                      default=None,
                      help=('control or a processed data file path, '
                            'e.g., "PITCH_HISTOGRAM;NOTE_DENSITY" like '
                            '"2,0,1,1,0,1,0,1,1,0,0,1;4", or '
                            '";3" (which gives all pitches the same probability), '
                            'or "/path/to/processed/midi/file.data" '
                            '(uses control sequence from the given processed data)'))

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=2)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/everything-game-30s-transposed-4.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=5000)

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=1.0)

    parser.add_option('-B', '--beam-size',
                      dest='beam_size',
                      type='int',
                      default=0)

    parser.add_option('-S', '--stochastic-beam-search',
                      dest='stochastic_beam_search',
                      action='store_true',
                      default=False)

    parser.add_option('-T', '--temperature',
                      dest='temperature',
                      type='float',
                      default=1.25)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


opt = getopt()

# ------------------------------------------------------------------------

output_dir = opt.output_dir
output_dir=output_dir+'/'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
sess_path = opt.sess_path
batch_size = opt.batch_size
max_len = opt.max_len
# greedy_ratio = opt.greedy_ratio  调greedy可以避免一直重复
greedy_ratio = 0.8
control = opt.control    #!!!!!!!!!!!!
use_beam_search = opt.beam_size > 0
stochastic_beam_search = opt.stochastic_beam_search
beam_size = opt.beam_size
temperature = opt.temperature
init_zero = opt.init_zero

if use_beam_search:
    greedy_ratio = 'DISABLED'
else:
    beam_size = 'DISABLED'

assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'
# control_dict={'1,0,1,0,1,1,0,1,0,1,0,1':'C大调','3,0,1,0,1,3,0,3,0,1,0,1':'C大调','2,0,1,0,1,2,0,2,0,1,0,1':'C大调',
#               '2,0,1,1,0,2,0,2,1,0,1,0':'C小调','3,0,1,1,0,3,0,3,1,0,1,0':'C小调','2,0,1,1,0,2,0,2,1,0,1,0':'C小调',}
# controls=[';1',';2',';3',';4',';5',';6',';7',';8']
controls=[]
name_dict ={'1,0,0': 'happy', '0,1,0':'sad', '1,1,1':'unspecified'}


for den_num in [1, 2, 3, 4]:
    for pitches_num in [0, 1, 2, '']:
        for entropy in [0, 1, 2, '']:
            controls.append(f'1,0,0;{den_num};{pitches_num};{entropy}')
            controls.append(f'0,1,0;{den_num};{pitches_num};{entropy}')
            controls.append(f'1,1,1;{den_num};{pitches_num};{entropy}')

for control in controls:
    mode, note_density, avg_pitches_count, entropy = control.split(';')

    control_name = name_dict[mode]
    if control is not None: #Get control based on pitch_histogram and note_density
        if os.path.isfile(control) or os.path.isdir(control): # get control info from midi file
            if os.path.isdir(control):
                files = list(utils.find_files_by_extensions(control))
                assert len(files) > 0, f'no file in "{control}"'
                control = np.random.choice(files)
            _, compressed_controls = torch.load(control)
            controls = ControlSeq.recover_compressed_array(compressed_controls)
            if max_len == 0:
                max_len = controls.shape[0]
            controls = torch.tensor(controls, dtype=torch.float32)
            controls = controls.unsqueeze(1).repeat(1, batch_size, 1).to(device)
            control = f'control sequence from "{control}"'

        else:
            mode, note_density, avg_pitches_count, entropy = control.split(';')
            mode = list(filter(len, mode.split(','))) # to pitch_histogram char list
            if len(mode) == 0:
                mode = np.ones(3) / 3
            else:
                mode = np.array(list(map(float, mode))) #to pitch_histogram float nparray
                assert mode.size == 3
                assert np.all(mode >= 0)
                mode = mode / mode.sum() \
                    if mode.sum() else np.ones(3) / 3

            if len(note_density) > 0:
                note_density = int(note_density)
                assert note_density in range(len(ControlSeq.note_density_bins))
            else:
                note_density = (np.ones(len(ControlSeq.note_density_bins)) / len(ControlSeq.note_density_bins)).tolist()

            if len(avg_pitches_count) > 0:
                avg_pitches_count = int(avg_pitches_count)
                assert avg_pitches_count in range(len(ControlSeq.avg_played_pitches_bins))
            else:
                avg_pitches_count = (np.ones(len(ControlSeq.avg_played_pitches_bins)) / len(ControlSeq.avg_played_pitches_bins)).tolist()

            if len(entropy) > 0:
                entropy = int(entropy)
                assert entropy in range(len(ControlSeq.entropy_bins))
            else:
                entropy = (np.ones(len(ControlSeq.entropy_bins)) / len(ControlSeq.entropy_bins)).tolist()

            control = Control(mode.tolist(), note_density, avg_pitches_count, entropy)
            controls = torch.tensor(control.to_array(), dtype=torch.float32)
            controls = controls.repeat(1, batch_size, 1).to(device) # 1Xbatch_sizeX controls
            control = repr(control)

    else:
        controls = None
        control = 'NONE'

    assert max_len > 0, 'either max length or control sequence length should be given'

    # ------------------------------------------------------------------------

    print('-' * 70)
    print('Session:', sess_path)
    print('Batch size:', batch_size)
    print('Max length:', max_len)
    print('Greedy ratio:', greedy_ratio)
    print('Beam size:', beam_size)
    print('Beam search stochastic:', stochastic_beam_search)
    print('Output directory:', output_dir)
    print('Controls:', control)
    print('Temperature:', temperature)
    print('Init zero:', init_zero)
    print('-' * 70)

    # ========================================================================
    # Generating
    # ========================================================================

    state = torch.load(sess_path, map_location=device)
    model = PerformanceRNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])
    model.eval()
    print(model)
    print('-' * 70)

    if init_zero:
        init = torch.zeros(batch_size, model.init_dim).to(device)
    else:
        init = torch.randn(batch_size, model.init_dim).to(device)  #

    with torch.no_grad():
        if use_beam_search:
            outputs = model.beam_search(init, max_len, beam_size,
                                        controls=controls,
                                        temperature=temperature,
                                        stochastic=stochastic_beam_search,
                                        verbose=True)
        else:
            outputs = model.generate(init, max_len,
                                     controls=controls,
                                     greedy=greedy_ratio,
                                     temperature=temperature,
                                     verbose=True)

    outputs = outputs.cpu().numpy().T  # [batch, sample_length(event_num)],T=transport

    # ========================================================================
    # Saving
    # ========================================================================

    os.makedirs(output_dir, exist_ok=True)

    for i, output in enumerate(outputs):
        name = f'output-{i}{control_name}' \
               f'-den_{note_density if type(note_density) == int else "x"}' \
               f'-pit_{avg_pitches_count if type(avg_pitches_count) == int else "x"}' \
               f'-ent_{entropy if type(entropy) == int else "x"}.mid'
        path = os.path.join(output_dir, name)
        n_notes = utils.event_indeces_to_midi_file(output, path)
        print(f'===> {path} ({n_notes} notes)')
