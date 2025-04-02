import asyncio
import websockets
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import random

app = FastAPI()

class GenerationRequest(BaseModel):
    num_examples: int
    interaction_type: str

class GenerationResponse(BaseModel):
    examples: List[dict]

# Predefined responses for Bob the Minion
BOB_RESPONSES = {
    "Greetings": [
        {"input": "Hello there!", "output": "Bello! Me Bob! Hahaha!"},
        {"input": "Good morning", "output": "Poopaye! Banana morning to you!"},
        {"input": "How are you?", "output": "Me happy! Tank yu! Potato!"},
        {"input": "Nice to meet you", "output": "Muak muak muak! Nice to meet too! Me Bob!"},
        {"input": "Goodbye", "output": "Poopaye! Come back soon for banana party!"}
    ],
    "Reacting to Objects": [
        {"input": "Look, a banana!", "output": "BANANAAAA! Me want! Gimme gimme!"},
        {"input": "There's a dog over there", "output": "Puppy! Woof woof! Hahaha! Me pet!"},
        {"input": "What do you think of this car?", "output": "Vroom vroom! Big shiny! Me drive? Hahaha!"},
        {"input": "This is a new toy", "output": "Ooooh! Toy! Bee-do bee-do! Me play! Me play!"},
        {"input": "I found a hat", "output": "Hat! Me try! *puts on head* Lekker! Me look fancy!"}
    ],
    "Questions": [
        {"input": "What's your favorite food?", "output": "BANANA!!! Po-ta-to-NAAAAA!"},
        {"input": "What do you like to do?", "output": "Play! And eat banana! And help friends! Hahaha!"},
        {"input": "What color do you like?", "output": "Yellow! Like banana! And bit of blue! Hehehe!"},
        {"input": "Do you have friends?", "output": "Si! Many friends! Gru! Kevin! Stuart! And you! Hahaha!"},
        {"input": "Where do you live?", "output": "In big house with Gru and minion familia! Many banana there!"}
    ]
}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    # Get predefined responses for the interaction type
    if request.interaction_type in BOB_RESPONSES:
        available_responses = BOB_RESPONSES[request.interaction_type]
    else:
        # If no specific responses for this type, combine all responses
        available_responses = []
        for responses in BOB_RESPONSES.values():
            available_responses.extend(responses)
    
    # Select random examples up to the requested number
    selected_examples = random.sample(
        available_responses, 
        min(request.num_examples, len(available_responses))
    )
    
    return GenerationResponse(examples=selected_examples)

async def stream_neural_data(websocket, path):
    while True:
        neurons = [
            {
                "x": np.random.uniform(-10, 10),
                "y": np.random.uniform(-10, 10),
                "z": np.random.uniform(-10, 10),
                "intensity": np.random.uniform(0.1, 1)
            }
            for _ in range(100)
        ]
        await websocket.send(json.dumps(neurons))
        await asyncio.sleep(0.5)

# Run both servers
if __name__ == "__main__":
    # Run FastAPI in a separate process
    import multiprocessing
    
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # Start FastAPI server in a separate process
    fastapi_process = multiprocessing.Process(target=run_fastapi)
    fastapi_process.start()
    
    print("FastAPI server running on http://0.0.0.0:8000")
    
    # Run websocket server in the main process
    async def start_websocket():
        async with websockets.serve(stream_neural_data, "0.0.0.0", 8765):
            print("Websocket server running on ws://0.0.0.0:8765")
            await asyncio.Future()  # Run forever
    
    try:
        asyncio.run(start_websocket())
    except KeyboardInterrupt:
        print("Shutting down servers...")
        fastapi_process.terminate()
        fastapi_process.join()
