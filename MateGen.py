from chatmessage import ChatMessages
from IPython.display import display, Markdown
from response import *


class MateGen:
    def __init__(self,
                 api_key,
                 model='gpt-3.5-turbo-0613',
                 system_content_list=None,
                 project=None,
                 messages=None,
                 available_functions=None):
        """
        Initializes the MateGen class to interact with OpenAI models.

        :param api_key: API key for OpenAI's model access, required to use MateGen.
        :param model: Optional, the specific model to be used. Defaults to 'gpt-3.5-turbo-0613'.
        :param system_content_list: Optional, a list of system messages or documents to be used in the conversation.
        :param project: Optional, an InterProject object for associating the conversation with a specific project.
        :param messages: Optional, a ChatMessages object or a list of dictionaries representing conversation history.
        :param available_functions: Optional, an AvailableFunction object representing external functions for the conversation.
        """

        self.api_key = api_key
        self.model = model
        self.project = project
        self.system_content_list = system_content_list if system_content_list is not None else []

        # Set the token threshold depending on the model
        if '1106' in model:
            self.tokens_thr = 110000
        elif '16k' in model:
            self.tokens_thr = 12000
        elif '4-0613' in model:
            self.tokens_thr = 7000
        else:
            self.tokens_thr = 3000

        # Initialize message history
        self.messages = ChatMessages(system_content_list=self.system_content_list,
                                     tokens_thr=self.tokens_thr)

        # Append any initial messages provided
        if messages:
            self.messages.messages_append(messages)

        self.available_functions = available_functions

    def chat(self, question=None):
        """
        Handles the conversation with the model, supporting both single-round and multi-round interactions.

        :param question: If provided, initiates a single-round conversation. Otherwise, multi-round mode is activated.
        """
        display(Markdown(f"▌ Model set to {self.model}"))

        # Single round mode if a question is provided
        if question:
            self.messages.messages_append({"role": "user", "content": question})
            self.messages = get_chat_response(model=self.model,
                                              messages=self.messages,
                                              available_functions=self.available_functions,)
        else:
            # Multi-round mode for ongoing conversations
            while True:
                self.messages = get_chat_response(model=self.model,
                                                  messages=self.messages,
                                                  available_functions=self.available_functions)

                user_input = input("Do you have any other questions? (Type 'exit' to finish) ")
                if user_input.lower() == "exit":
                    break
                else:
                    self.messages.messages_append({"role": "user", "content": user_input})

    def reset(self):
        """
        Resets the current conversation history.
        """
        self.messages = ChatMessages(system_content_list=self.system_content_list)

    def upload_messages(self):
        """
        Uploads the current conversation messages to the associated project file, if provided.
        """
        if self.project is None:
            print("Please provide a valid InterProject object for the project parameter before uploading messages.")
        else:
            self.project.append_doc_content(content=self.messages.history_messages)


if __name__ == '__main__':
    print("This file contains the MateGen class.")
