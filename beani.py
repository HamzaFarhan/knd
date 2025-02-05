import asyncio

from motor.motor_asyncio import AsyncIOMotorClient


async def ping_server():
    # Replace the placeholder with your Atlas connection string
    uri = "mongodb://127.0.0.1:27017"

    # Set the Stable API version when creating a new client
    client = AsyncIOMotorClient(uri)

    # Send a ping to confirm a successful connection
    try:
        await client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)


asyncio.run(ping_server())
