import functools

import torch
import os
import optparse
import time
import asyncio

from output_handler import OutputHandler
from ws_server import WsServer
from threading import Thread

from config import device
from model import PerformanceRNN

# pylint: disable=E1101,E1102


# ========================================================================
# Settings
# ========================================================================

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-b', '--batch-size',
                      dest='batch_size',
                      type='int',
                      default=1)

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default='save/everything-game-30s-transposed.sess',
                      help='session file containing the trained model')

    parser.add_option('-g', '--greedy-ratio',
                      dest='greedy_ratio',
                      type='float',
                      default=0.8)

    parser.add_option('-z', '--init-zero',
                      dest='init_zero',
                      action='store_true',
                      default=False)

    return parser.parse_args()[0]


def run_server(server):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    loop.run_forever()


def init(init_zero):
    if init_zero:
        return torch.zeros(batch_size, model.init_dim).to(device)
    else:
        return torch.randn(batch_size, model.init_dim).to(device)


opt = getopt()

# ------------------------------------------------------------------------

sess_path = opt.sess_path
batch_size = opt.batch_size
greedy_ratio = opt.greedy_ratio
init_zero = opt.init_zero


assert os.path.isfile(sess_path), f'"{sess_path}" is not a file'

# ------------------------------------------------------------------------

print('-' * 70)
print('Session:', sess_path)
print('Batch size:', batch_size)
print('Greedy ratio:', greedy_ratio)
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

init_func = functools.partial(init, init_zero)

output_handler = OutputHandler()

listeners_awaiting_start = set()
server = WsServer(on_start=lambda: listeners_awaiting_start.add(""),
                  on_input_data=output_handler.input_received)
thread = Thread(target=lambda: run_server(server))
thread.start()

output_handler.start(server, model, init_func, greedy=greedy_ratio)

while len(listeners_awaiting_start) < 1:    # hacky way to start server and wait for any listener
    pass

print("Listening to input and generating...")
