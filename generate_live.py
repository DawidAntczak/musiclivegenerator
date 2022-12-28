import threading

import torch
import numpy as np
import os
import sys
import optparse
import time
import io
import asyncio

from input import transform
from output_handler import OutputHandler
from ws_server import WsServer
from threading import Thread

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
                      default=1)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/train — kopia.sess',
                      help='session file containing the trained model')

    parser.add_option('-o', '--output-dir',
                      dest='output_dir',
                      type='string',
                      default='output/')

    parser.add_option('-l', '--max-length',
                      dest='max_len',
                      type='int',
                      default=500)

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
                      default=1.0)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


def run_server(server, stop_server_event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    loop.run_forever()
    print("Server stopped")


def init(init_zero):
    if init_zero:
        return torch.zeros(batch_size, model.init_dim).to(device)
    else:
        return torch.randn(batch_size, model.init_dim).to(device)


opt = getopt()

# ------------------------------------------------------------------------

output_dir = opt.output_dir
output_dir = output_dir+'/'+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
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
name_dict ={'3,0,1,0,1,2,0,2,0,1,0,1;1':'peaceful', '3,0,1,0,1,2,0,2,0,1,0,1;5':'happy',
             '3,0,1,1,0,2,0,2,1,0,1,0;1':'sad', '3,0,1,1,0,2,0,2,1,0,1,0;5':'tensional'}

for den_num in [1, 5]:
    # controls.append(f'1,0,1,0,1,1,0,1,0,1,0,1;{den_num}')
    # controls.append(f'1,0,1,1,0,1,0,1,0,1,1,0;{den_num}')
    controls.append(f'3,0,1,0,1,2,0,2,0,1,0,1;{den_num}')
    controls.append(f'3,0,1,1,0,2,0,2,1,0,1,0;{den_num}')

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

init = init(init_zero)

output_handler = OutputHandler()

listeners_awaiting_start = set()
server = WsServer(on_start=lambda: listeners_awaiting_start.add(""),
                  on_input_data=output_handler.input_received)
stop_server_event = threading.Event()
thread = Thread(target=lambda: run_server(server, stop_server_event))
thread.start()

output_handler.start(server, model, init, greedy=greedy_ratio, temperature=temperature)

while len(listeners_awaiting_start) < 1:    # hacky way to start server and wait for any listener
    pass

input("Listening to input and generating...")

# ========================================================================
# Generating end
# ========================================================================

#outputs = model.generate(init, max_len,
#                         controls=transform(controls[0]),
#                         greedy=greedy_ratio,
#                         temperature=temperature,
#                         verbose=True)
#outputs = outputs.cpu().numpy().T  # [batch, sample_length(event_num)],T=transport

# ========================================================================
# Saving
# ========================================================================

#os.makedirs(output_dir, exist_ok=True)
#for i, output in enumerate(outputs):
#    name = f'output-{i}.mid'
#    path = os.path.join(output_dir, name)
#    n_notes = utils.event_indeces_to_midi_file(output, path)
#    print(f'===> {path} ({n_notes} notes)')

    # output_handler.collect_to_midi()
    # stream = io.BytesIO()
    # n_notes = utils.event_indeces_to_midi_file(output, stream)
    # stream.seek(0)
    # midiBytes = stream.read()
    # asyncio.run(server.send_data(midiBytes))
#    print(f'===> Sent {n_notes} notes)')
stop_server_event.set()     # not working yet but a nice idea, isn't it?
print("ALL DONE")
