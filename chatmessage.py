import tiktoken
import openai
import copy


class ChatMessages:
    """
    The ChatMessages class is designed to construct message objects that can be processed by the Chat model. This class serves as a more structured representation of the original messages that the model receives.
    It accepts a list of dictionaries as one of its attributes, distinguishing between system-related and historical conversation messages. Additionally, it manages token counts and can automatically trim older messages to ensure the conversation fits within the token limit for multi-turn interactions.
    """

    def __init__(self, system_content_list=None, question='Hello', tokens_thr=None, project=None):
        self.system_content_list = system_content_list if system_content_list is not None else []
        system_messages = []  # Stores system-related messages
        history_messages = []  # Stores the user and assistant conversation messages
        messages_all = []  # Combines system and conversation messages
        system_content = ''  # Concatenated system messages string
        history_content = question  # User input as a string
        content_all = ''  # Combines all message content
        num_of_system_messages = 0  # Tracks how many system messages are included
        all_tokens_count = 0  # Total number of tokens in the messages

        # Get the appropriate encoding method for token counting
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # If system content exists, process the system messages
        if self.system_content_list:
            for content in self.system_content_list:
                system_messages.append({"role": "system", "content": content})
                system_content += content

            system_tokens_count = len(encoding.encode(system_content))
            messages_all.extend(system_messages)
            num_of_system_messages = len(self.system_content_list)

            if tokens_thr is not None and system_tokens_count >= tokens_thr:
                print("System messages exceed token limit. Please reduce the number of input documents.")
                system_messages.clear()
                messages_all.clear()
                num_of_system_messages = 0
                system_tokens_count = 0

        all_tokens_count += system_tokens_count

        # Add the initial user message
        history_messages = [{"role": "user", "content": question}]
        messages_all.extend(history_messages)

        user_tokens_count = len(encoding.encode(question))
        all_tokens_count += user_tokens_count

        if tokens_thr is not None and all_tokens_count >= tokens_thr:
            print("User message exceeds token limit. Please adjust the input or external documents.")
            history_messages.clear()
            system_messages.clear()
            messages_all.clear()
            num_of_system_messages = 0
            all_tokens_count = 0

        # Store the messages and attributes
        self.messages = messages_all
        self.system_messages = system_messages
        self.history_messages = history_messages
        self.tokens_count = all_tokens_count
        self.num_of_system_messages = num_of_system_messages
        self.tokens_thr = tokens_thr
        self.encoding = encoding
        self.project = project

    # Method to remove messages manually or based on token limits
    def messages_pop(self, manual=False, index=None):
        def remove_message(index):
            removed_message = self.history_messages.pop(index)
            self.tokens_count -= len(self.encoding.encode(str(removed_message)))

        if self.tokens_thr is not None:
            while self.tokens_count >= self.tokens_thr:
                remove_message(-1)

        if manual:
            if index is None:
                remove_message(-1)
            elif 0 <= index < len(self.history_messages) or index == -1:
                remove_message(index)
            else:
                raise ValueError(f"Invalid index value: {index}")

        # Update the complete messages list
        self.messages = self.system_messages + self.history_messages

    # Method to add new conversation messages
    def messages_append(self, new_messages):
        if isinstance(new_messages, (dict, openai.openai_object.OpenAIObject)):
            self.messages.append(new_messages)
            self.tokens_count += len(self.encoding.encode(str(new_messages)))

        elif isinstance(new_messages, ChatMessages):
            self.messages += new_messages.messages
            self.tokens_count += new_messages.tokens_count

        # Refresh the history messages and handle token constraints
        self.history_messages = self.messages[self.num_of_system_messages:]
        self.messages_pop()

    # Method to create a copy of the message object
    def copy(self):
        system_content_str_list = [msg['content'] for msg in self.system_messages]
        copy_instance = ChatMessages(
            system_content_list=copy.deepcopy(system_content_str_list),
            question=self.history_messages[0]['content'] if self.history_messages else '',
            tokens_thr=self.tokens_thr
        )
        copy_instance.history_messages = copy.deepcopy(self.history_messages)
        copy_instance.messages = copy.deepcopy(self.messages)
        copy_instance.tokens_count = self.tokens_count
        copy_instance.num_of_system_messages = self.num_of_system_messages
        return copy_instance

    # Method to add new system messages
    def add_system_messages(self, new_system_content):
        if isinstance(new_system_content, str):
            new_system_content = [new_system_content]

        self.system_content_list.extend(new_system_content)

        new_system_content_str = ''.join(new_system_content)
        new_token_count = len(self.encoding.encode(new_system_content_str))
        self.tokens_count += new_token_count

        system_messages = [{"role": "system", "content": msg} for msg in self.system_content_list]

        self.system_messages = system_messages
        self.num_of_system_messages = len(self.system_content_list)
        self.messages = self.system_messages + self.history_messages

        self.messages_pop()

    # Method to delete all system messages
    def delete_system_messages(self):
        if self.system_content_list:
            combined_system_content = ''.join(self.system_content_list)
            delete_token_count = len(self.encoding.encode(combined_system_content))

            self.tokens_count -= delete_token_count
            self.num_of_system_messages = 0
            self.system_content_list.clear()
            self.system_messages.clear()
            self.messages = self.history_messages

    # Method to delete function-related messages from history
    def delete_function_messages(self):
        for index in range(len(self.history_messages) - 1, -1, -1):
            msg = self.history_messages[index]
            if msg.get("function_call") or msg.get("role") == "function":
                self.messages_pop(manual=True, index=index)


if __name__ == '__main__':
    print("This file defines the ChatMessages class.")

