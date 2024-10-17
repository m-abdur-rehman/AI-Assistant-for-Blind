from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
# from aiohttp-ratelimiter import JournalLimiter, RateLimiter
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import aiohttp
import asyncio

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  #gemini-1.5-flash

# Configure Groq LLM
llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

# Load the system prompt
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read().strip()

# Create a conversation template
template = f"""{system_prompt}

Current conversation:
{{history}}
Human: {{input}}
Assistant: """

prompt = ChatPromptTemplate.from_template(template)

# Define a chat message history store
class InMemoryHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def clear(self):
        self.messages = []

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# Create the conversation chain
chain = prompt | llm

# Wrap the chain with message history
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Define trigger words for image capture
TRIGGER_WORDS = [
    "describe the surroundings", "surrounding", "environment", "what is this", "describe the room", 
    "object", "device", "amount", "currency", "cash", "infront", "front", "distance",
    "scan", "analyze", "identify", "location", "situation", "scene", "nearby", "vision",
    "view", "area", "scan object", "recognize", "detail", "explain", "detect", "visualize",
    "look around", "what's here", "spot", "clarify", "size", "measure", 
    "point out", "image", "picture", "this one", "what is this" ,"ceiling" , "floor" , "ground"
]

# Function to process text input
async def process_text_input(text, session_id):
    response = await chain_with_history.ainvoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}}
    )
    if isinstance(response, AIMessage):
        return response.content
    return str(response)

async def process_image(image, prompt, session_id):
    
    response = chain_with_history.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}}
    )
    
    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # async with aiohttp.ClientSession() as session:
    #     async with session.post(
    #         'https://generativeai.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent',
    #         json={
    #             'contents': [
    #                 {'parts': [{'text': prompt}, {'inline_data': {'mime_type': 'image/jpeg', 'data': image_base64}}]}
    #             ]
    #         },
    #         headers={'Authorization': f'Bearer {os.getenv("GOOGLE_API_KEY")}'}
    #     ) as response:
    #         result = await response.json()
    
    # return result['candidates'][0]['content']['parts'][0]['text']
    
    response = gemini_model.generate_content([
        # prompt or "Describe what you see in this image concisely, as if explaining to a blind person.",
        prompt,
        {"mime_type": "image/jpeg", "data": image_base64}
    ])
    return response.text

# limiter = RateLimiter(JournalLimiter(1, 5))  # 1 request per 5 seconds
# class SimpleLimiter:
#     def __init__(self, rate, period):
#         self.rate = rate
#         self.period = period
#         self.tokens = rate
#         self.last_refill = asyncio.get_event_loop().time()

#     async def acquire(self):
#         while True:
#             now = asyncio.get_event_loop().time()
#             time_since_refill = now - self.last_refill
#             if time_since_refill > self.period:
#                 self.tokens = self.rate
#                 self.last_refill = now

#             if self.tokens > 0:
#                 self.tokens -= 1
#                 return
#             else:
#                 await asyncio.sleep(self.period - time_since_refill)

# limiter = SimpleLimiter(1, 5) 

# @limiter.limit("1/5")
async def process_request(request):
    # await limiter.acquire()
    try:
        data = await request.json()
        image_base64 = data.get('image')
        text = data.get('text', '')
        session_id = data.get('session_id', 'default')

        print(f"Received request - Text: {text[:50]}..., Session ID: {session_id}")

        if text:
            text = text.lower()

        is_image_request = any(trigger in text for trigger in TRIGGER_WORDS)

        if is_image_request and image_base64:
            image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_description = await process_image(image, text, session_id)
            print(f"Image processed - Description: {image_description[:50]}...")
            return web.json_response({"response": image_description})

        response = await process_text_input(text, session_id)
        print(f"LLM Response: {response[:50]}...")
        return web.json_response({"response": response})

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        # Ensure resources are released
        if 'image' in locals():
            del image
        if 'image_array' in locals():
            del image_array

# Set up the web server and CORS
app = web.Application()
cors = cors_setup(app, defaults={
    "*": ResourceOptions(
        allow_credentials=True, 
        expose_headers="*",
        allow_headers="*",
    )
})

# Add the POST route for processing
app.router.add_post('/process', process_request)
for route in list(app.router.routes()):
    cors.add(route)

# Run the server
if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8080)