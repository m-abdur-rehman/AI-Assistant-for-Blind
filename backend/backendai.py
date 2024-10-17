from aiohttp import web
from aiohttp_cors import setup as cors_setup, ResourceOptions
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

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

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
    "describe the surroundings","surrounding" ,"environment", "what is this", "describe the room", 
    "object", "device", "amount", "currency", "cash", "infront", "front", "distance",
    "scan", "analyze", "identify", "location", "situation", "scene", "nearby", "vision",
    "view", "area", "scan object", "recognize", "detail", "explain", "detect", "visualize",
    "look around", "what's here", "spot", "clarify" , "size", "measure", 
    "point out"
]

# Function to process text input
def process_text_input(text, session_id):
    response = chain_with_history.invoke(
        {"input": text},
        config={"configurable": {"session_id": session_id}}
    )
    # Extract the content from the AIMessage
    if isinstance(response, AIMessage):
        return response.content
    return str(response)  # Fallback to string representation if not AIMessage

def process_image(image, prompt,session_id):
    response = chain_with_history.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}}
    )
    # if isinstance(response, AIMessage):
    #     return response.content
    # return str(response)

    _, buffer = cv2.imencode(".jpg", image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    response = gemini_model.generate_content([
        # prompt or "Describe what you see in this image concisely, as if explaining to a blind person.",
        prompt,
        {"mime_type": "image/jpeg", "data": image_base64}
    ])

    return response.text



async def process_request(request):
    try:
        data = await request.json()
        image_base64 = data.get('image')
        text = data.get('text', '')
        session_id = data.get('session_id', 'default')  # Get session ID from request

        print(f"Received request - Text: {text[:50]}..., Session ID: {session_id}")

        if text:
            text :str = text.lower()

        is_image_request = any(trigger in text for trigger in TRIGGER_WORDS)

        if is_image_request and image_base64:
            image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_description = process_image(image, text,session_id)
            print(f"Image processed - Description: {image_description[:50]}...")
            
            # Add image description to conversation history
            # process_text_input(f"Image description: {image_description}", session_id)
            
            return web.json_response({"response": image_description})
        
        # if is_image_request and not image_base64:
        #     return web.json_response({"response": "I need an image to describe the surroundings. Please capture an image and try again."})

        # For non-image requests, use the conversation chain with history
        response = process_text_input(text, session_id)

        print(f"LLM Response: {response[:50]}...")

        return web.json_response({"response": response})
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return web.json_response({"error": str(e)}, status=500)
    
# async def process_request(request):
#     try:
#         data = await request.json()
#         image_base64 = data.get('image')
#         text = data.get('text', '')
#         session_id = data.get('session_id', 'default')  # Get session ID from request

#         print(f"Received request - Text: {text[:50]}..., Session ID: {session_id}")

#         if text:
#             text = text.lower()

#         is_image_request = any(trigger in text for trigger in TRIGGER_WORDS)

#         if is_image_request and image_base64:
#             image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             image_description = process_image(image, text)
#             print(f"Image processed - Description: {image_description[:50]}...")
            
#             # Add image description to conversation history
#             process_text_input(f"Image description: {image_description}", session_id)
            
#             return web.json_response({"response": image_description})
        
#         if is_image_request and not image_base64:
#             return web.json_response({"response": "I need an image to describe the surroundings. Please capture an image and try again."})

#         # For non-image requests, use the conversation chain with history
#         response = process_text_input(text, session_id)

#         print(f"LLM Response: {response[:50]}...")

#         return web.json_response({"response": response})
#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

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







# from aiohttp import web
# from aiohttp_cors import setup as cors_setup, ResourceOptions
# import base64
# import cv2
# import numpy as np
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import RunnablewithMessageHistory

# load_dotenv()

# # Configure Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))    
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Configure Groq LLM
# llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

# # Load the system prompt
# with open('system_prompt.txt', 'r') as file:
#     system_prompt = file.read().strip()

# # Create a conversation template
# template = f"""{system_prompt}

# Current conversation:
# {{history}}
# Human: {{input}}
# Assistant: """

# prompt = ChatPromptTemplate.from_template(template)

# # Initialize memory
# memory = ConversationBufferMemory(return_messages=True)

# # Create the conversation chain
# conversation = RunnablewithMessageHistory(
#     llm=llm,
#     memory=memory,
#     prompt=prompt,
#     verbose=True
# )

# # Define trigger words for image capture
# # TRIGGER_WORDS = ["describe the surroundings", "environment", "what is this", "describe the room", "object", "device", "amount", "currency", "cash","infront","front", "distance"]
# TRIGGER_WORDS = [
#     "describe the surroundings", "environment", "what is this", "describe the room", 
#     "object", "device", "amount", "currency", "cash", "infront", "front", "distance",
#     "scan", "analyze", "identify", "location", "situation", "scene", "nearby", "vision",
#     "view", "area", "scan object", "recognize", "detail", "explain", "detect", "visualize",
#     "look around", "what's here", "spot", "clarify" , "size", "measure", 
#     "point out"
# ]

# async def process_request(request):
#     try:
#         data = await request.json()
#         image_base64 = data.get('image')
#         text = data.get('text', '')
#         if text:
#             text = text.lower()

#         print(f"Received request - Text: {text[:50]}...") 

#         is_image_request = any(trigger in text for trigger in TRIGGER_WORDS)

#         if is_image_request and image_base64:
#             image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             # combined_input = f"User said: {text}"
#             image_description = process_image(image,text)
#             print(f"Image processed - Description: {image_description[:50]}...")
#             return web.json_response({"response": image_description})
        
#         if is_image_request and not image_base64:
#             return web.json_response({"response": "I need an image to describe the surroundings. Please capture an image and try again."})

#         # For non-image requests, use the conversation chain
#         # combined_input = f"User said: {text}"
#         combined_input= text 
#         response = conversation.predict(input=combined_input)

#         print(f"LLM Response: {response[:50]}...")

#         return web.json_response({"response": response})
#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# def process_image(image,prompt):
#     _, buffer = cv2.imencode(".jpg", image)
#     image_base64 = base64.b64encode(buffer).decode('utf-8')

#     response = gemini_model.generate_content([
#         prompt or "Describe what you see in this image concisely, as if explaining to a blind person.",
#         {"mime_type": "image/jpeg", "data": image_base64}
#     ])

#     return response.text

# # Set up the web server and CORS
# app = web.Application()
# cors = cors_setup(app, defaults={
#     "*": ResourceOptions(
#         allow_credentials=True,
#         expose_headers="*",
#         allow_headers="*",
#     )
# })

# # Add the POST route for processing
# app.router.add_post('/process', process_request)
# for route in list(app.router.routes()):
#     cors.add(route)

# # Run the server
# if __name__ == '__main__':
#     web.run_app(app, host='0.0.0.0', port=8080)


# from aiohttp import web
# from aiohttp_cors import setup as cors_setup, ResourceOptions
# import base64
# import cv2
# import numpy as np
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import SystemMessage, HumanMessage
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain

# load_dotenv()

# # Configure Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Configure Groq LLM
# llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

# # Load the system prompt
# with open('system_prompt.txt', 'r') as file:
#     system_prompt = file.read().strip()

# # Create a conversation template
# template = f"""{system_prompt}

# Current conversation:
# {{history}}
# Human: {{input}}
# Assistant: """

# prompt = ChatPromptTemplate.from_template(template)

# # Initialize memory
# memory = ConversationBufferMemory(return_messages=True)

# # Create the conversation chain
# conversation = ConversationChain(
#     llm=llm,
#     memory=memory,
#     prompt=prompt,
#     verbose=True
# )

# async def process_request(request):
#     try:
#         data = await request.json()
#         image_base64 = data.get('image')
#         text = data.get('text', '')

#         print(f"Received request - Text: {text[:50]}...")

#         if image_base64:
#             image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             image_description = process_image(image)
#             print(f"Image processed - Description: {image_description[:50]}...")
#         else:
#             image_description = "No image provided"
#             print("No image provided")

#         combined_input = f"Image: {image_description}\nUser said: {text}"

#         # Get the response from the conversation chain
#         response = conversation.predict(input=combined_input)

#         print(f"LLM Response: {response[:50]}...")

#         return web.json_response({"response": response})
#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)

# def process_image(image):
#     _, buffer = cv2.imencode(".jpg", image)
#     image_base64 = base64.b64encode(buffer).decode('utf-8')

#     response = gemini_model.generate_content([
#         "Describe what you see in this image. Be concise.",
#         {"mime_type": "image/jpeg", "data": image_base64}
#     ])

#     return response.text

# # Set up the web server and CORS
# app = web.Application()
# cors = cors_setup(app, defaults={
#     "*": ResourceOptions(
#         allow_credentials=True,
#         expose_headers="*",
#         allow_headers="*",
#     )
# })

# # Add the POST route for processing
# app.router.add_post('/process', process_request)
# for route in list(app.router.routes()):
#     cors.add(route)

# # Run the server
# if __name__ == '__main__':
#     web.run_app(app, host='0.0.0.0', port=8080)

# from aiohttp import web
# from aiohttp_cors import setup as cors_setup, ResourceOptions
# import base64
# import cv2
# import numpy as np
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from langchain_groq import ChatGroq
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import SystemMessage, HumanMessage
# from langchain.schema.runnable import RunnableSequence

# load_dotenv()

# # Configure Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Configure Groq LLM
# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))

# # Load the system prompt
# with open('system_prompt.txt', 'r') as file:
#     system_prompt = file.read().strip()

# # Define the prompt template
# prompt_template = ChatPromptTemplate.from_messages([
#     SystemMessage(content=system_prompt),
#     HumanMessage(content="{text}")
# ])

# # Create a RunnableSequence for conversation handling (without memory in the sequence)
# conversation = RunnableSequence(
#     prompt_template,    # Prompt template (which formats the input)
#     llm                 # LLM to generate a response
# )

# # Manual memory handling using ConversationBufferMemory
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory()  # We will store conversation history here

# # async def process_request(request):
# #     try:
# #         data = await request.json()
# #         image_base64 = data.get('image')
# #         text = data.get('text', '')

# #         print(f"Received request - Text: {text[:50]}...")  # Print first 50 chars of text

# #         # Process image with Gemini
# #         if image_base64:
# #             image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
# #             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# #             image_description = process_image(image)
# #             print(f"Image processed - Description: {image_description[:50]}...")
# #         else:
# #             image_description = "No image provided"
# #             print("No image provided")

# #         # Process text with Groq using RunnableSequence
# #         combined_input = f"Image: {image_description}\nUser said: {text}"

# #         # Add the new user input to memory
# #         memory.chat_memory.add_user_message(combined_input)

# #         # Generate response using the RunnableSequence (no need to await)
# #         llm_response = conversation.invoke({"text": combined_input})  # Remove 'await'

# #         # Add the LLM response to memory as well
# #         memory.chat_memory.add_ai_message(llm_response)
# #         print(f"LLM Response: {llm_response[:50]}...")

# #         return web.json_response({"response": llm_response})
# #     except Exception as e:
# #         print(f"Error processing request: {str(e)}")
# #         return web.json_response({"error": str(e)}, status=500)


# async def process_request(request):
#     try:
#         data = await request.json()
#         image_base64 = data.get('image')
#         text = data.get('text', '')

#         print(f"Received request - Text: {text[:50]}...")

#         if image_base64:
#             image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
#             image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#             image_description = process_image(image)
#             print(f"Image processed - Description: {image_description[:50]}...")
#         else:
#             image_description = "No image provided"
#             print("No image provided")

#         combined_input = f"Image: {image_description}\nUser said: {text}"

#         memory.chat_memory.add_user_message(combined_input)

#         llm_response = conversation.invoke({"text": combined_input})

#         # Extract the content from the AIMessage object
#         response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)

#         memory.chat_memory.add_ai_message(response_content)
#         print(f"LLM Response: {response_content[:50]}...")

#         return web.json_response({"response": response_content})
#     except Exception as e:
#         print(f"Error processing request: {str(e)}")
#         return web.json_response({"error": str(e)}, status=500)
    
# def process_image(image):
#     _, buffer = cv2.imencode(".jpg", image)
#     image_base64 = base64.b64encode(buffer).decode('utf-8')

#     response = gemini_model.generate_content([
#         "Describe what you see in this image. Be concise.",
#         {"mime_type": "image/jpeg", "data": image_base64}
#     ])

#     return response.text

# # Set up the web server and CORS
# app = web.Application()
# cors = cors_setup(app, defaults={
#     "*": ResourceOptions(
#         allow_credentials=True,
#         expose_headers="*",
#         allow_headers="*",
#     )
# })

# # Add the POST route for processing
# app.router.add_post('/process', process_request)
# for route in list(app.router.routes()):
#     cors.add(route)

# # Run the server
# if __name__ == '__main__':
#     web.run_app(app, host='0.0.0.0', port=8080)



# import asyncio
# from aiohttp import web
# import base64
# import cv2
# import numpy as np
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# import azure.cognitiveservices.speech as speechsdk

# load_dotenv()

# # Configure Gemini
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Configure Groq
# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Load system prompt
# with open('system_prompt.txt', 'r') as file:
#     system_prompt = file.read().strip()

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     ("human", "{text}")
# ])

# conversation = LLMChain(
#     llm=llm,
#     prompt=prompt_template,
#     memory=memory
# )

# async def process_request(request):
#     data = await request.json()
#     image_base64 = data.get('image')
#     text = data.get('text', '')

#     # Process image with Gemini
#     if image_base64:
#         image_array = np.frombuffer(base64.b64decode(image_base64), np.uint8)
#         image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
#         image_description = process_image(image)
#     else:
#         image_description = "No image provided"

#     # Process text with Groq
#     combined_input = f"Image: {image_description}\nUser said: {text}"
#     llm_response = conversation.predict(text=combined_input)

#     return web.json_response({"response": llm_response})

# def process_image(image):
#     _, buffer = cv2.imencode(".jpg", image)
#     image_base64 = base64.b64encode(buffer).decode('utf-8')
    
#     response = gemini_model.generate_content([
#         "Describe what you see in this image. Be concise.",
#         {"mime_type": "image/jpeg", "data": image_base64}
#     ])
    
#     return response.text

# app = web.Application()
# app.router.add_post('/process', process_request)

# if __name__ == '__main__':
#     web.run_app(app, port=8080)