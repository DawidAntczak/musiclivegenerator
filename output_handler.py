import io

import utils
from input import Input


class OutputHandler:
    server = None
    model = None
    init = None
    greedy = None

    def start(self, server, model, init, greedy=1.0):
        self.server = server
        self.model = model
        self.init = init
        self.model.init_live_generation(init)
        self.greedy = greedy

    async def input_received(self, raw_input_json):
        input = Input(raw_input_json)
        transformed_input = input.get_transformed()
        requested_time_length = input.get_requested_time_length()

        if input.should_reset():
            self.model.init_live_generation(self.init)

        outputs = self.model.generate_live(time_length=requested_time_length, controls=transformed_input,
                                           greedy=self.greedy, temperature=input.get_temperature())
        await self.send_as_midi(outputs)

    async def send_as_midi(self, outputs):
        output = outputs.cpu().numpy().T
        assert len(output) == 1
        output = output[0]
        stream = io.BytesIO()
        utils.event_indeces_to_midi_file(output, stream)
        stream.seek(0)
        midi_bytes = stream.read()
        await self.server.send_data(midi_bytes)
