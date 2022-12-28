import websockets
import asyncio


async def listen():
    url = "ws://localhost:2579"
    close_message = "CLOSE"

    async with websockets.connect(url) as ws:
        while msg != close_message:
            msg = await ws.recv()
            print(msg)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(listen())
