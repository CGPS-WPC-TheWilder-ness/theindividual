from json import dumps
from time import time
from flask import request, Response, stream_with_context
from hashlib import sha256
from datetime import datetime
from requests import get
from requests import post 
from json     import loads
from queue import Queue
from typing import Dict, List, Any
import threading
import os
import sys
import copy

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import BaseCallbackHandler, StreamingStdOutCallbackHandler
from langchain.schema import LLMResult

# import local stuff
from server.constants import *
from server.utilities import *

STOP_SEQUENCE = "XXX"

conversations = {}

# specify the embedding function for decoding vector stores
openAI_embedding = OpenAIEmbeddings()

# this store is the primary corpus ... it may be reinstantiated at any time
contextdb = Chroma(persist_directory="./server/.contextdb", embedding_function=openAI_embedding)

# this store is for fine tuning
feedbackdb = Chroma(persist_directory="./server/.feedbackdb", embedding_function=openAI_embedding)

class Backend_Api:
    def __init__(self, app, config: dict) -> None:
        self.app = app
        self.openai_key = os.getenv("OPENAI_API_KEY") or config['openai_key']
        self.openai_api_base = os.getenv("OPENAI_API_BASE") or config['openai_api_base']
        self.proxy = config['proxy']
        self.routes = {
            '/backend-api/v2/conversation': {
                'function': self._conversation,
                'methods': ['POST']
            }
        }

    def _conversation(self):

        conversation_id = request.json["conversation_id"]

        if conversation_id in conversations:
            memory = conversations[conversation_id]
        else:
            init_memory = ConversationBufferWindowMemory(
                k=4,
                ai_prefix="The Individual",
            )
            conversations[conversation_id] = init_memory
            memory = conversations[conversation_id]

        # this is custom handler that plays nice with flask callback
        class StreamingStdOutCallbackHandlerYield(StreamingStdOutCallbackHandler):

            def on_llm_start(
                self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
            ) -> None:
                """Run when LLM starts running."""
                with q.mutex:
                    q.queue.clear()

            def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
                """Run on new LLM token. Only available when streaming is enabled."""
                sys.stdout.write(token)
                sys.stdout.flush()
                q.put(token)

            def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
                """Run when LLM ends running."""
                q.put(STOP_SEQUENCE)
            
        try:
            prompt = request.json['meta']['content']['parts'][0]['content']

            # this takes queue and yields generator
            def stream(rq: Queue, prompt):
                streamed_response = ""
                while True:
                    token = rq.get()
                    streamed_response = streamed_response + token
                    if token == "XXX":
                        if prompt == "wild":
                            print("picle")
                            memory.save_context({"input": "wild"}, {"output": streamed_response})
                        elif prompt == "rephrase":
                            memory.save_context({"input": "rephrase"}, {"output": streamed_response})
                        conversations.update({conversation_id: memory})
                        break
                    yield token

            q = Queue()

            proxies = None
            if self.proxy['enable']:
                proxies = {
                    'http': self.proxy['http'],
                    'https': self.proxy['https'],
                }

            callback_fn = StreamingStdOutCallbackHandlerYield()

            chat_gpt_callback = ChatOpenAI(
                temperature=1,
                model_name="gpt-4o",
                streaming=True,
                callbacks=[callback_fn],
                stop=[STOP_SEQUENCE]
            )

            dummy_memory = ConversationBufferWindowMemory(
                k=4,
                ai_prefix="The Individual",
            )

            split = prompt.split("::")
                    
            if split[0] == "feedback":
                utilities.add_feedback(feedbackdb, split[1], split[2], split[3])

                return self.app.response_class("Feedback successfully added.")

            match prompt:

                case "":
                    return self.app.response_class("You must provide a prompt. Try again.")

                case "intro":
                    return self.app.response_class(text.INTRO)

                case "help":
                    return self.app.response_class(text.HELP)

                case "topics":
                    return self.app.response_class(text.MENU)

                case "menu":
                    return self.app.response_class(text.WELCOME)

                case _:
                    # get context to inject into template
                    context_injection, feedback_injection = utilities.combined_injections(
                        contextdb,
                        feedbackdb,
                        prompt)

                    template = utilities.inject_main(context_injection, feedback_injection)

                    PROMPT = PromptTemplate(
                        input_variables=["history", "input"],
                        template=template
                    )

                    print(PROMPT)

                    def build_and_submit():

                        # because template changes with each prompt (to inject feedback embeddings)
                        # we must reconstruct the chain object for each new prompt
                        conversation = ConversationChain(
                            prompt=PROMPT,
                            llm=chat_gpt_callback,
                            verbose=False,
                            memory=memory
                        )
                        return conversation(prompt)["response"]

                    threading.Thread(target=build_and_submit).start()

            print(utilities.num_tokens(memory.load_memory_variables({})["history"] + PROMPT.template))
            print()
            print()
            return self.app.response_class(stream_with_context(stream(q, prompt)), mimetype="text/event-stream")

        except Exception as e:
            print(e)
            print(e.__traceback__.tb_next)
            return {
                '_action': '_ask',
                'success': False,
                "error": f"an error occurred {str(e)}"}, 400






