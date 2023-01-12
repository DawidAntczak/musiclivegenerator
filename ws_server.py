import websockets
import asyncio
import json


class WsServer:
    listeners = set()
    sources = set()
    on_start = None
    on_stop = None
    on_input_data = None

    def __init__(self, on_start=lambda *args: None, on_stop=lambda *args: None, on_input_data=lambda *args: None):
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_input_data = on_input_data

    async def start(self, port=7890):
        server = websockets.serve(self.on_connect, "localhost", port)
        await server
        print("Started server at port " + str(port))

    async def send_data(self, data):
        for listener in self.listeners:
            await listener.send(data)

    async def on_connect(self, websocket, path):
        if self.is_listener_client(path):
            print("Listener connected")
            self.listeners.add(websocket)
        elif self.is_source_client(path):
            print("Source connected")
            self.sources.add(websocket)
        else:
            print("Unknown client connected. Path: " + path)
        try:
            async for message in websocket:
                message_object = json.loads(message)
                if self.is_start_message(message_object):
                    self.on_start()
                elif self.is_stop_message(message_object):
                    self.on_stop()
                elif self.is_input_data_message(message_object):
                    await self.on_input_data(message_object)
                else:
                    print("Received unknown message from client: " + message)
        except websockets.ConnectionClosed as e:
            print(f'Error while disconnecting client: {str(e)}')
        finally:
            print(f'A client (path: {path}) disconnected.')
            if self.is_listener_client(path):
                self.listeners.remove(websocket)
            elif self.is_source_client(path):
                self.sources.remove(websocket)

    def is_listener_client(self, path):
        return path.endswith('listener')

    def is_source_client(self, path):
        return path.endswith('source')

    def is_start_message(self, message):
        return 'Message' in message and message['Message'] == 'START'

    def is_stop_message(self, message):
        return 'Message' in message and message['Message'] == 'STOP'

    def is_input_data_message(self, message):
        return 'Mode' in message \
               and 'AttackDensity' in message \
               and 'AvgPitchesPlayed' in message \
               and 'Entropy' in message


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(WsServer().start())
    asyncio.get_event_loop().run_forever()
