import torch
import torch.nn as nn
from torch.distributions import Categorical, Gumbel

import numpy as np
from progress.bar import Bar
from config import device
import config
from sequence import EventSeq


# pylint: disable=E1101,E1102


class PerformanceRNN(nn.Module):
    def __init__(self, event_dim, control_dim, init_dim, hidden_dim,
                 gru_layers=3, gru_dropout=0.3):
        super().__init__()

        self.event_dim = event_dim
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.concat_dim = event_dim + control_dim
        self.input_dim = hidden_dim
        self.output_dim = event_dim

        self.primary_event = self.event_dim - 1

        self.inithid_fc = nn.Linear(init_dim, gru_layers * hidden_dim)
        self.inithid_fc_activation = nn.Tanh()

        self.event_embedding = nn.Embedding(event_dim, event_dim)
        self.concat_input_fc = nn.Linear(self.concat_dim, self.input_dim)
        self.concat_input_fc_activation = nn.LeakyReLU(0.1, inplace=True)

        self.gru = nn.GRU(self.input_dim, self.hidden_dim,
                          num_layers=gru_layers, dropout=gru_dropout)
        self.output_fc = nn.Linear(hidden_dim * gru_layers, self.output_dim)
        self.output_fc_activation = nn.Softmax(dim=-1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.event_embedding.weight)
        nn.init.xavier_normal_(self.inithid_fc.weight)
        self.inithid_fc.bias.data.fill_(0.)
        nn.init.xavier_normal_(self.concat_input_fc.weight)
        nn.init.xavier_normal_(self.output_fc.weight)
        self.output_fc.bias.data.fill_(0.)

    def _sample_event(self, output, greedy=True, temperature=1.0):
        if greedy:
            return output.argmax(-1)
        else:
            output = output / temperature
            probs = self.output_fc_activation(output)
            return Categorical(probs).sample()

    def forward(self, event, control=None, hidden=None):
        assert len(event.shape) == 2
        assert event.shape[0] == 1
        batch_size = event.shape[1]
        event = self.event_embedding(event)

        assert control.shape == (1, batch_size, self.control_dim)

        concat = torch.cat([event, control], -1)
        input = self.concat_input_fc(concat)
        input = self.concat_input_fc_activation(input)

        _, hidden = self.gru(input, hidden)
        output = hidden.permute(1, 0, 2).contiguous()
        output = output.view(batch_size, -1).unsqueeze(0)
        output = self.output_fc(output)
        return output, hidden
    
    def get_primary_event(self, batch_size):
        return torch.LongTensor([[self.primary_event] * batch_size]).to(device)
    
    def init_to_hidden(self, init):
        # [batch_size, init_dim]
        batch_size = init.shape[0]
        out = self.inithid_fc(init)
        out = self.inithid_fc_activation(out)
        out = out.view(self.gru_layers, batch_size, self.hidden_dim)
        return out
    
    def expand_controls(self, controls, steps):
        assert len(controls.shape) == 3
        assert controls.shape[2] == self.control_dim
        if controls.shape[0] > 1:
            assert controls.shape[0] >= steps
            return controls[:steps]
        return controls.repeat(steps, 1, 1)

    def generate(self, init, steps, events=None, controls=None, greedy=1.0,
                 temperature=1.0, teacher_forcing_ratio=1.0, output_type='index', verbose=False):

        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim
        assert steps > 0

        use_teacher_forcing = events is not None
        if use_teacher_forcing:
            assert len(events.shape) == 2
            assert events.shape[0] >= steps - 1
            events = events[:steps - 1]

        event = self.get_primary_event(batch_size)
        use_control = controls is not None
        if use_control:
            controls = self.expand_controls(controls, steps)
        hidden = self.init_to_hidden(init)

        outputs = []
        step_iter = range(steps)
        if verbose:
            step_iter = Bar('Generating').iter(step_iter)

        for step in step_iter:
            control = controls[step].unsqueeze(0) if use_control else None
            output, hidden = self.forward(event, control, hidden)

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy, temperature=temperature)

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.output_fc_activation(output))
            elif output_type == 'logit':
                outputs.append(output)
            else:
                assert False

            if use_teacher_forcing and step < steps - 1:
                if np.random.random() <= teacher_forcing_ratio:
                    event = events[step].unsqueeze(0)

        return torch.cat(outputs, 0)

    def beam_search(self, init, steps, beam_size, controls=None,
                    temperature=1.0, stochastic=False, verbose=False):
        assert len(init.shape) == 2 and init.shape[1] == self.init_dim
        assert self.event_dim >= beam_size > 0 and steps > 0
        
        batch_size = init.shape[0]
        current_beam_size = 1
        
        if controls is not None:
            controls = self.expand_controls(controls, steps)

        # Initial hidden weights
        hidden = self.init_to_hidden(init)
        hidden = hidden[:, :, None, :]
        hidden = hidden.repeat(1, 1, current_beam_size, 1)

        
        # Initial event
        event = self.get_primary_event(batch_size)
        event = event[:, :, None].repeat(1, 1, current_beam_size)

        beam_events = event[0, :, None, :].repeat(1, current_beam_size, 1)

        beam_log_prob = torch.zeros(batch_size, current_beam_size).to(device)
        
        if stochastic:
            beam_log_prob_perturbed = torch.zeros(batch_size, current_beam_size).to(device)
            beam_z = torch.full((batch_size, beam_size), float('inf'))
            gumbel_dist = Gumbel(0, 1)

        step_iter = range(steps)
        if verbose:
            step_iter = Bar(['', 'Stochastic '][stochastic] + 'Beam Search').iter(step_iter)

        for step in step_iter:
            if controls is not None:
                control = controls[step, None, :, None, :]
                control = control.repeat(1, 1, current_beam_size, 1)
                control = control.view(1, batch_size * current_beam_size, self.control_dim)
            else:
                control = None
            
            event = event.view(1, batch_size * current_beam_size)
            hidden = hidden.view(self.gru_layers, batch_size * current_beam_size, self.hidden_dim)

            logits, hidden = self.forward(event, control, hidden)
            hidden = hidden.view(self.gru_layers, batch_size, current_beam_size, self.hidden_dim)
            logits = (logits / temperature).view(1, batch_size, current_beam_size, self.event_dim)
            
            beam_log_prob_expand = logits + beam_log_prob[None, :, :, None]
            beam_log_prob_expand_batch = beam_log_prob_expand.view(1, batch_size, -1)
            
            if stochastic:
                beam_log_prob_expand_perturbed = beam_log_prob_expand + gumbel_dist.sample(beam_log_prob_expand.shape)
                beam_log_prob_Z, _ = beam_log_prob_expand_perturbed.max(-1)

                beam_log_prob_expand_perturbed_normalized = beam_log_prob_expand_perturbed
                
                beam_log_prob_expand_perturbed_normalized_batch = \
                    beam_log_prob_expand_perturbed_normalized.view(1, batch_size, -1)
                _, top_indices = beam_log_prob_expand_perturbed_normalized_batch.topk(beam_size, -1)
                
                beam_log_prob_perturbed = \
                    torch.gather(beam_log_prob_expand_perturbed_normalized_batch, -1, top_indices)[0]

            else:
                _, top_indices = beam_log_prob_expand_batch.topk(beam_size, -1)
            
            beam_log_prob = torch.gather(beam_log_prob_expand_batch, -1, top_indices)[0]
            
            beam_index_old = torch.arange(current_beam_size)[None, None, :, None]
            beam_index_old = beam_index_old.repeat(1, batch_size, 1, self.output_dim)
            beam_index_old = beam_index_old.view(1, batch_size, -1)
            beam_index_new = torch.gather(beam_index_old, -1, top_indices)
            
            hidden = torch.gather(hidden, 2, beam_index_new[:, :, :, None].repeat(4, 1, 1, 1024))
            
            event_index = torch.arange(self.output_dim)[None, None, None, :]
            event_index = event_index.repeat(1, batch_size, current_beam_size, 1)
            event_index = event_index.view(1, batch_size, -1)
            event = torch.gather(event_index, -1, top_indices)
            
            beam_events = torch.gather(beam_events[None], 2, beam_index_new.unsqueeze(-1).repeat(1, 1, 1, beam_events.shape[-1]))
            beam_events = torch.cat([beam_events, event.unsqueeze(-1)], -1)[0]
            
            current_beam_size = beam_size

        best = beam_events[torch.arange(batch_size).long(), beam_log_prob.argmax(-1)]
        best = best.contiguous().t()
        return best

    live_generation_context = None

    def init_live_generation(self, init):
        batch_size = init.shape[0]
        assert init.shape[1] == self.init_dim

        event = self.get_primary_event(batch_size)
        hidden = self.init_to_hidden(init)
        self.live_generation_context = dict()
        self.live_generation_context['event'] = event
        self.live_generation_context['hidden'] = hidden

    def generate_live(self, time_length, controls, greedy, temperature, output_type='index'):
        event, hidden = self.live_generation_context['event'], self.live_generation_context['hidden']
        control = controls[0].unsqueeze(0)
        outputs = []
        current_time_length = 0
        while current_time_length < time_length:
            output, hidden = self.forward(event, control, hidden)
            self.live_generation_context['event'] = event
            self.live_generation_context['hidden'] = hidden

            use_greedy = np.random.random() < greedy
            event = self._sample_event(output, greedy=use_greedy,  temperature=temperature)

            if output_type == 'index':
                outputs.append(event)
            elif output_type == 'softmax':
                outputs.append(self.output_fc_activation(output))
            elif output_type == 'logit':
                outputs.append(output)
            else:
                assert False

            current_time_length += self.get_time_shift(event)

        return torch.cat(outputs, 0)

    def get_current_length(self, outputs):
        outputs = torch.cat(outputs, 0)
        output = outputs.cpu().numpy().T
        assert len(output) == 1
        output = output[0]
        _, time = EventSeq.from_array(output)
        return time

    def get_time_shift(self, event):
        output = event.cpu().numpy().T
        assert len(output) == 1
        output = output[0]
        _, time = EventSeq.from_array(output)
        return time


if __name__ == '__main__':
    model_config = config.model
    model = PerformanceRNN(**model_config).to(device)

    init = torch.randn(64, model.init_dim).to(device)
    events = torch.randn(200, 64).to(device).long()
    controls = torch.randn(200, 64, 15).to(device)
    outputs = model.generate(init, 200, events=events[:-1], controls=controls,
                             teacher_forcing_ratio=1, output_type='logit')

    print('done')
