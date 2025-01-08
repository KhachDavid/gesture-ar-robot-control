import asyncio
from frameutils import Bluetooth


async def main():
    bluetooth = Bluetooth()
    await bluetooth.connect()

    print(await bluetooth.send_lua("print('hello world')", await_print=True))
    print(await bluetooth.send_lua("print(1 + 2)", await_print=True))

    await bluetooth.disconnect()


asyncio.run(main())
