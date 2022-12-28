from enum import Enum


class UnknownInstrumentException(Exception):
    def __init__(self, instrument):
        super().__init__(f'Unknown instrument {instrument}')


class InstrumentClass(Enum):
    PIANO = 0
    CHROMATIC_PERCUSSION = 1
    ORGAN = 2
    GUITAR = 3
    BASS = 4
    STRINGS = 5
    ENSEMBLE = 6
    BRASS = 7
    REED = 8
    PIPE = 9
    SYNTH_LEAD = 10
    SYNTH_PAD = 11
    SYNTH_EFFECT = 12
    ETHNIC = 13
    PERCUSSIVE = 14
    SOUND_EFFECTS = 15


class Instrument:
    program, instrumentClass = None

    def __init__(self, program):
        self.program = program
        self.instrumentClass = self.determine_instrument_class(program)

    @staticmethod
    def determine_instrument_class(program):
        if 0 <= program <= 7:
            return InstrumentClass.PIANO
        if 8 <= program <= 15:
            return InstrumentClass.CHROMATIC_PERCUSSION
        if 16 <= program <= 23:
            return InstrumentClass.ORGAN
        if 24 <= program <= 31:
            return InstrumentClass.GUITAR
        if 32 <= program <= 39:
            return InstrumentClass.BASS
        if 40 <= program <= 47:
            return InstrumentClass.STRINGS
        if 48 <= program <= 55:
            return InstrumentClass.ENSEMBLE
        if 56 <= program <= 63:
            return InstrumentClass.BRASS
        if 64 <= program <= 71:
            return InstrumentClass.REED
        if 72 <= program <= 79:
            return InstrumentClass.PIPE
        if 80 <= program <= 87:
            return InstrumentClass.SYNTH_LEAD
        if 88 <= program <= 95:
            return InstrumentClass.SYNTH_PAD
        if 96 <= program <= 103:
            return InstrumentClass.SYNTH_EFFECT
        if 104 <= program <= 111:
            return InstrumentClass.ETHNIC
        if 112 <= program <= 119:
            return InstrumentClass.PERCUSSIVE
        if 120 <= program <= 127:
            return InstrumentClass.SOUND_EFFECTS
        raise UnknownInstrumentException
