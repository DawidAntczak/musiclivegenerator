import math

import numpy as np
import copy
import itertools
import collections
from statistics import mean
from pretty_midi import PrettyMIDI, Note, Instrument

from models import PreparationMetadata

# ==================================================================================
# Parameters
# ==================================================================================

# NoteSeq -------------------------------------------------------------------------

DEFAULT_SAVING_PROGRAM = 1
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)
DEFAULT_VELOCITY_RANGE = range(21, 109)
DEFAULT_NORMALIZATION_BASELINE = 60  # C4

# EventSeq ------------------------------------------------------------------------

USE_VELOCITY = False     # TODO co z tym
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 1.15 ** np.arange(32) / 65 #non-linear
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_NOTE_LENGTH = BEAT_LENGTH / 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2

# ControlSeq ----------------------------------------------------------------------

QUANTIZE_NOTE_TIMES = True
DEFAULT_TIME_QUANT = BEAT_LENGTH / 8    # 1/32 s

DEFAULT_WINDOW_SIZE = BEAT_LENGTH * 4
DEFAULT_NOTE_DENSITY_BINS = np.array([0, 1.6175867768595042, 2.8666666666666667, 4.133333333333334, 5.633333333333334, 8.041379310344828]) * 1.1
#DEFAULT_NOTE_DENSITY_BINS = np.array([0, 3.100114678640942, 4.233333333333333, 5.333375279458888, 6.566542750929368, 8.199753494664062])
DEFAULT_AVG_PLAYED_PITCHES_BINS = np.array([0, 1.9161538461538463, 3.0842105263157894]) * 0.7
DEFAULT_ENTROPY_BINS = np.array([0, 3.1804939280290952, 4.0212409983780782]) * 0.7
#DEFAULT_NOTE_DENSITY_BINS = np.arange(12) * 3 + 1 #[1 4 7 10 ..]


# ==================================================================================
# Notes
# ==================================================================================

class NoteSeq:
    @staticmethod
    def from_midi(midi, programs=DEFAULT_LOADING_PROGRAMS):
        instrument_to_notes_count_dict = dict()
        for inst in midi.instruments:
            if not inst.is_drum:
                instrument_to_notes_count_dict[inst.program] = len(inst.notes)
        notes = itertools.chain(*[
            inst.notes for inst in midi.instruments
            if inst.program in programs and not inst.is_drum])
        return NoteSeq(list(notes)), instrument_to_notes_count_dict

    @staticmethod
    def from_midi_file(path, *args, **kwargs):
        midi = PrettyMIDI(path)
        return NoteSeq.from_midi(midi, *args, **kwargs)

    def __init__(self, notes=[]):
        self.notes = []
        if notes:
            for note in notes:
                assert isinstance(note, Note)
            notes = filter(lambda note: note.end >= note.start, notes)
            self.add_notes(list(notes))

    def copy(self):
        return copy.deepcopy(self)

    def to_midi(self, program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
        midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
        inst = Instrument(program, False, 'NoteSeq')
        inst.notes = copy.deepcopy(self.notes)
        midi.instruments.append(inst)
        return midi

    def to_midi_file(self, path, *args, **kwargs):
        self.to_midi(*args, **kwargs).write(path)

    def add_notes(self, notes):
        self.notes += notes
        self.notes.sort(key=lambda note: note.start)

    def adjust_pitches(self, offset):
        for note in self.notes:
            pitch = note.pitch + offset
            pitch = 0 if pitch < 0 else pitch
            pitch = 127 if pitch > 127 else pitch
            note.pitch = pitch

    def adjust_velocities(self, offset):
        for note in self.notes:
            velocity = note.velocity + offset
            velocity = 0 if velocity < 0 else velocity
            velocity = 127 if velocity > 127 else velocity
            note.velocity = velocity

    def adjust_time(self, offset):
        for note in self.notes:
            note.start += offset
            note.end += offset

    def trim_overlapped_notes(self, min_interval=0):
        last_notes = {}
        for i, note in enumerate(self.notes):
            if note.pitch in last_notes:
                last_note = last_notes[note.pitch]
                if note.start - last_note.start <= min_interval:
                    last_note.end = max(note.end, last_note.end)
                    last_note.velocity = max(note.velocity, last_note.velocity)
                    del self.notes[i]
                elif note.start < last_note.end:
                    last_note.end = note.start
            else:
                last_notes[note.pitch] = note


# ==================================================================================
# Events
# ==================================================================================

class Event:
    def __init__(self, type, time, value):
        self.type = type
        self.time = time
        self.value = value

    def __repr__(self):
        return 'Event(type={}, time={}, value={})'.format(
            self.type, self.time, self.value)


class EventSeq:
    pitch_range = DEFAULT_PITCH_RANGE
    velocity_range = DEFAULT_VELOCITY_RANGE
    velocity_steps = DEFAULT_VELOCITY_STEPS
    time_shift_bins = DEFAULT_TIME_SHIFT_BINS

    @staticmethod
    def from_note_seq(note_seq): #从note sequence -> performance
        note_events = []

        if USE_VELOCITY:
            velocity_bins = EventSeq.get_velocity_bins() #21 -106

        for note in note_seq.notes:
            if note.pitch in EventSeq.pitch_range:
                if USE_VELOCITY:
                    velocity = note.velocity
                    velocity = max(velocity, EventSeq.velocity_range.start)
                    velocity = min(velocity, EventSeq.velocity_range.stop - 1)
                    velocity_index = np.searchsorted(velocity_bins, velocity)  # velocity-> velocity index
                    note_events.append(Event('velocity', note.start, velocity_index)) # 记录时间和velocity index

                pitch_index = note.pitch - EventSeq.pitch_range.start  # index=pitch-pitch_min_range
                note_events.append(Event('note_on', note.start, pitch_index))
                note_events.append(Event('note_off', note.end, pitch_index))

        note_events.sort(key=lambda event: event.time)  # stable sort events by time
        events = []

        for i, event in enumerate(note_events): # caculate timeshift event
            events.append(event)

            if event is note_events[-1]:
                break

            interval = note_events[i + 1].time - event.time
            shift = 0

            while interval - shift >= EventSeq.time_shift_bins[0]:
                index = np.searchsorted(EventSeq.time_shift_bins,
                                        interval - shift, side='right') - 1  # time shift-> time shift index 时间精度不够就使用多个timeshift events
                events.append(Event('time_shift', event.time + shift, index))
                shift += EventSeq.time_shift_bins[index]

        return EventSeq(events)

    @staticmethod
    def from_array(event_indeces):  # event num->events
        time = 0
        events = []
        for event_index in event_indeces:
            for event_type, feat_range in EventSeq.feat_ranges().items():
                if feat_range.start <= event_index < feat_range.stop:
                    event_value = event_index - feat_range.start
                    events.append(Event(event_type, time, event_value))
                    if event_type == 'time_shift':
                        time += EventSeq.time_shift_bins[event_value]
                    break

        return EventSeq(events)

    @staticmethod
    def dim():
        return sum(EventSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        feat_dims = collections.OrderedDict()
        feat_dims['note_on'] = len(EventSeq.pitch_range)
        feat_dims['note_off'] = len(EventSeq.pitch_range)
        if USE_VELOCITY:
            feat_dims['velocity'] = EventSeq.velocity_steps
        feat_dims['time_shift'] = len(EventSeq.time_shift_bins)
        return feat_dims

    @staticmethod
    def feat_ranges():  #return a dict {on:0-87,0ff:88-175,velocity:176-207,timeshift:208-240 }
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in EventSeq.feat_dims().items(): # EventSeq.feat_dims() is a order_dict
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def get_velocity_bins():
        n = EventSeq.velocity_range.stop - EventSeq.velocity_range.start
        return np.arange(EventSeq.velocity_range.start,
                         EventSeq.velocity_range.stop,
                         n / (EventSeq.velocity_steps - 1))

    def __init__(self, events=[]):
        for event in events:
            assert isinstance(event, Event)

        self.events = copy.deepcopy(events)

        # compute event times again
        time = 0
        for event in self.events:
            event.time = time
            if event.type == 'time_shift':
                time += EventSeq.time_shift_bins[event.value]

    def to_note_seq(self): #events-> prettymidi Notes
        time = 0
        notes = []

        velocity = DEFAULT_VELOCITY
        velocity_bins = EventSeq.get_velocity_bins()

        last_notes = {}

        for event in self.events:
            if event.type == 'note_on':
                pitch = event.value + EventSeq.pitch_range.start
                note = Note(velocity, pitch, time, None)
                notes.append(note)
                last_notes[pitch] = note

            elif event.type == 'note_off':
                pitch = event.value + EventSeq.pitch_range.start

                if pitch in last_notes:
                    note = last_notes[pitch]
                    note.end = max(time, note.start + MIN_NOTE_LENGTH)
                    del last_notes[pitch]

            elif event.type == 'velocity':
                index = min(event.value, velocity_bins.size - 1)
                velocity = velocity_bins[index]

            elif event.type == 'time_shift':
                time += EventSeq.time_shift_bins[event.value]

        for note in notes:
            if note.end is None:
                note.end = note.start + DEFAULT_NOTE_LENGTH  # if note do not have an end, end it after 0.25s.

            note.velocity = int(note.velocity)

        return NoteSeq(notes)

    def to_array(self): # events-> event nums
        feat_idxs = EventSeq.feat_ranges()
        idxs = [feat_idxs[event.type][event.value] for event in self.events]
        dtype = np.uint8 if EventSeq.dim() <= 256 else np.uint16
        return np.array(idxs, dtype=dtype)


# ==================================================================================
# Controls
# ==================================================================================

class Control:
    def __init__(self, mode, note_density, avg_pitches_played, entropy):
        self.mode = self._ensure_list(mode, 'mode')
        self.note_density = self._ensure_list(note_density, 'note_density')
        self.avg_pitches_played = self._ensure_list(avg_pitches_played, 'avg_pitches_played')
        self.entropy = self._ensure_list(entropy, 'entropy')

    def _ensure_list(self, value, feature_name):
        dim = ControlSeq.feat_dims()[feature_name]

        if value is None:
            return np.ones(dim) / dim

        if type(value) == int or type(value) == np.int64:
            assert 0 <= value < dim
            vals = np.zeros(dim)
            vals[value] = 1.
            return vals

        assert type(value) == list and len(value) == dim
        value = np.array(value)
        assert np.all(value >= 0)
        return value / value.sum() if value.sum() else np.ones(dim) / dim

    def __repr__(self):
        return 'Control(mode={}, note_density={}, avg_pitches_played={}, entropy={})'.format(
            self.mode, self.note_density, self.avg_pitches_played, self.entropy)

    def to_array(self):
        dens = self.note_density
        mode = self.mode
        pitches = self.avg_pitches_played
        entropy = self.entropy
        return np.concatenate([dens, mode, pitches, entropy], 0)


class ControlSeq:
    note_density_bins = DEFAULT_NOTE_DENSITY_BINS
    avg_played_pitches_bins = DEFAULT_AVG_PLAYED_PITCHES_BINS
    entropy_bins = DEFAULT_ENTROPY_BINS
    window_size = DEFAULT_WINDOW_SIZE

    @staticmethod
    def from_event_seq(event_seq, metadata: PreparationMetadata):
        events = list(event_seq.events)
        start, end = 0, 0
        note_count = 0

        controls, metadatas = [], []

        def _mode_index(metadata):
            if metadata.primary_key.correlation_coefficient <= 0.75:
                return 2
            if metadata.primary_key.mode == 'minor':
                return 1
            return 0

        for i, event in enumerate(events):
            while start < i:  #扣除窗左边的音符
                if events[start].type == 'note_on':
                    note_count -= 1.
                start += 1

            while end < len(events): #添加窗左边的音符
                if events[end].time - event.time > ControlSeq.window_size: #在window_size的时间窗内
                    break
                if events[end].type == 'note_on':
                    note_count += 1.
                end += 1

            mode = _mode_index(metadata)

            note_count, avg_pitches_played, entropy = ControlSeq.calculate_metrics(events, start, end)
            note_density_bin = max(np.searchsorted(
                ControlSeq.note_density_bins,
                note_count / ControlSeq.window_size, side='right') - 1, 0) #note_count->index(density)

            avg_played_pitches_bin = max(np.searchsorted(
                ControlSeq.avg_played_pitches_bins,
                avg_pitches_played, side='right') - 1, 0) if avg_pitches_played else None

            entropy_bin = max(np.searchsorted(
                ControlSeq.entropy_bins,
                entropy, side='right') - 1, 0) if entropy else None

            controls.append(Control(mode, note_density_bin, avg_played_pitches_bin, entropy_bin))

            metadatas.append(PreprocessMetadata(note_count.__int__(),
                                                note_density_bin.__int__() if note_density_bin is not None else -1,
                                                avg_played_pitches_bin.__int__() if avg_played_pitches_bin is not None else -1,
                                                entropy_bin.__int__() if entropy_bin is not None else -1
                                                ))

        return ControlSeq(controls, metadatas)

    @staticmethod
    def _quantized_time(time_to_quantize):
        if not QUANTIZE_NOTE_TIMES:
            return time_to_quantize

        quant_times = round(time_to_quantize / DEFAULT_TIME_QUANT)
        return quant_times * DEFAULT_TIME_QUANT

    @staticmethod
    def calculate_metrics(events, window_start, window_end):
        note_on_times = dict()
        pitches = list()
        for i, _ in enumerate(events, window_start):
            if i >= window_end:
                break
            if events[i].type == 'note_on':
                quantized_time = ControlSeq._quantized_time(events[i].time)
                c = note_on_times.get(quantized_time, 0)
                note_on_times[quantized_time] = c + 1
                pitches.append(events[i].value)
        times = list(note_on_times.values())
        entropy = ControlSeq.pitch_entropy(pitches)
        return len(note_on_times), mean(times) if times else None, None if math.isnan(entropy) else entropy

    @staticmethod
    def _entropy(prob):
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.nansum(prob * np.log2(prob))

    @staticmethod
    def pitch_entropy(pitches) -> float:
        counter = np.zeros(128)
        for pitch in pitches:
            counter[pitch] += 1
        denominator = counter.sum()
        if denominator < 1:
            return math.nan
        prob = counter / denominator
        return ControlSeq._entropy(prob)

    @staticmethod
    def dim():
        return sum(ControlSeq.feat_dims().values())

    @staticmethod
    def feat_dims():
        return collections.OrderedDict([
            ('mode', 3),
            ('note_density', len(ControlSeq.note_density_bins)),
            ('avg_pitches_played', len(ControlSeq.avg_played_pitches_bins)),
            ('entropy', len(ControlSeq.entropy_bins))
        ])

    @staticmethod
    def feat_ranges():
        offset = 0
        feat_ranges = collections.OrderedDict()
        for feat_name, feat_dim in ControlSeq.feat_dims().items():
            feat_ranges[feat_name] = range(offset, offset + feat_dim)
            offset += feat_dim
        return feat_ranges

    @staticmethod
    def recover_compressed_array(array):
        feat_dims = ControlSeq.feat_dims()
        assert array.shape[1] == ControlSeq.dim()

        start = 0
        end = feat_dims['note_density']
        dens = array[:, start:end].astype(np.float64)

        start = end
        end += feat_dims['mode']
        mode = array[:, start:end].astype(np.float64)

        start = end
        end += feat_dims['avg_pitches_played']
        pitches = array[:, start:end].astype(np.float64)

        start = end
        end += feat_dims['entropy']
        entropy = array[:, start:end].astype(np.float64)

        return np.concatenate([dens, mode, pitches, entropy], 1)

    def __init__(self, controls, metadatas):
        for control in controls:
            assert isinstance(control, Control)
        for metadata in metadatas:
            assert isinstance(metadata, PreprocessMetadata)
        self.controls = copy.deepcopy(controls)
        self.metadatas = metadatas

    def to_compressed_array(self):
        dens = [control.note_density for control in self.controls]
        dens = np.array(dens).astype(np.uint8)
        
        mode = [control.mode for control in self.controls]
        mode = np.array(mode).astype(np.uint8)
        
        pitches = [control.avg_pitches_played for control in self.controls]
        pitches = np.array(pitches).astype(np.uint8)
        
        entropy = [control.entropy for control in self.controls]
        entropy = np.array(entropy).astype(np.uint8)
        
        return np.concatenate([dens, mode, pitches, entropy], 1)


class PreprocessMetadata:
    def __init__(self, note_count, density_bin, avg_pitches_played_bin, entropy_bin):
        self.note_count = note_count
        self.density_bin = density_bin
        self.avg_pitches_played_bin = avg_pitches_played_bin
        self.entropy_bin = entropy_bin


if __name__ == '__main__':
    import pickle
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/midi/ecomp/BLINOV02.mid'

    print('Converting MIDI to EventSeq')
    es, _ = EventSeq.from_note_seq(NoteSeq.from_midi_file(path))

    print('Converting EventSeq to MIDI')
    EventSeq.from_array(es.to_array()).to_note_seq().to_midi_file('/tmp/test.mid')

    print('Converting EventSeq to ControlSeq')
    cs = ControlSeq.from_event_seq(es)

    print('Saving compressed ControlSeq')
    pickle.dump(cs.to_compressed_array(), open('/tmp/cs-compressed.data', 'wb'))

    print('Loading compressed ControlSeq')
    c = ControlSeq.recover_compressed_array(
        pickle.load(open('/tmp/cs-compressed.data', 'rb')))

    print('Done')
