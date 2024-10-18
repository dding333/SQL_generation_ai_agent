from planning import *
import openai
import time
import json
from IPython.display import display, Code, Markdown
from openai.error import APIConnectionError
from gptLearning import *


def function_to_call(available_functions, function_call_message):
    """
    Based on a function call message `function_call_message`, return a message with the function's execution result `function_response_messages`.
    :param available_functions: Required parameter, an AvailableFunctions object that describes the basic information of the current external functions.
    :param function_call_message: Required parameter, a message representing an external function call.
    :return: `function_response_messages`, a message consisting of the external function's execution result.
    """

    # Get the name of the function to be called from the function call message
    function_name = function_call_message["function_call"]["name"]

    # Get the corresponding external function object based on the function name
    function_to_call = available_functions.functions_dic[function_name]

    # Extract the function parameters from the function call message
    # This includes the SQL or Python code written by the large model
    function_args = json.loads(function_call_message["function_call"]["arguments"])

    # Pass the parameters to the external function and run it
    try:
        # Add the global variables from the current operation space to the external function
        function_args['g'] = globals()

        # Run the external function
        function_response = function_to_call(**function_args)

    # If the external function encounters an error, extract the error message
    except Exception as e:
        function_response = "The function encountered an error as follows:" + str(e)
        # print(function_response)

    # Create the function_response_messages
    # This message includes information about the successful execution or error of the external function

    function_response_messages = {
        "role": "function",
        "name": function_name,
        "content": function_response,
    }

    return function_response_messages

def get_gpt_response(model,
                     messages,
                     available_functions=None):
    """
    Responsible for calling the Chat model and obtaining the model's response function, and it allows for a temporary pause of 1 minute if a Rate limit issue occurs when calling the GPT model.
    Additionally, for unclear questions, it will prompt the user to modify the input prompt to obtain better model results.
    :param model: Required parameter indicating the name of the large model to be called.
    :param messages: Required parameter, a ChatMessages type object used to store conversation messages.
    :param available_functions: Optional parameter, an AvailableFunctions type object representing the basic information of external functions during the conversation.
    Defaults to None, indicating no external functions.
    :return: Returns the response message from the model.
    """

    # To account for potential communication errors, loop to call the Chat model
    while True:
        try:
            # If no external functions exist
            if available_functions is None:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages)

            # If external functions exist, obtain functions and function_call parameters from the AvailableFunctions object
            else:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages.messages,
                    functions=available_functions.functions,
                    function_call=available_functions.function_call
                )
            break  # Exit the loop if response is successfully obtained

        except APIConnectionError as e:
            # Print the core error information
            print(f"Encountered a connection issue: {str(e)}")
            print("Due to rate limit, pausing for 1 minute and will continue running after 1 minute...")
            time.sleep(60)  # Wait for 1 minute
            print("Waited 60 seconds, starting a new round of questions and answers...")

    return response["choices"][0]["message"]


def get_chat_response(model,
                      messages,
                      available_functions=None,
                      delete_some_messages=False,
                      is_task_decomposition=False):
    """
    Responsible for executing a complete conversation session. Note that a conversation may involve multiple calls to the large model,
    and this function serves as the main function to complete one conversation session.
    The last message in the input messages must be a message that can initiate a conversation.
    This function calls get_gpt_response to obtain the model's output results and then processes the output based on whether it is text or code results,
    by flexibly calling different functions for post-processing.
    :param model: Required parameter indicating the name of the large model to be called.
    :param messages: Required parameter, a ChatMessages type object used to store conversation messages.
    :param available_functions: Optional parameter, an AvailableFunctions type object representing the basic information of external functions during the conversation.
    Defaults to None, indicating no external functions.
    :param delete_some_messages: Optional parameter indicating whether to delete several intermediate messages when concatenating messages, default is False.
    :param is_task_decomposition: Optional parameter indicating whether the current task is task decomposition review, default is False.
    :return: Messages concatenating the final results of this Q&A session.
    """

    # Only when modifying the complex task decomposition result will is_task_decomposition=True occur
    # When is_task_decomposition=True, response_message will not be recreated
    if not is_task_decomposition:
        # First obtain the result of a single large model call
        # At this point, response_message is the message returned by the large model call
        response_message = get_gpt_response(model=model,
                                            messages=messages,
                                            available_functions=available_functions)

    # Complex condition check, if is_task_decomposition = True,
    # or if enhanced mode is enabled and the task involves function response
    # (Note that when is_task_decomposition = True, there is no response_message object)
    if is_task_decomposition:
        # Set is_task_decomposition to True, indicating that the current task is task decomposition
        is_task_decomposition = True
        # In task decomposition, the task decomposition prompt is named text_response_messages
        task_decomp_few_shot = add_task_decomposition_prompt(messages)
        # print("Performing task decomposition, please wait...")
        # Also update response_message; now response_message is the response after task decomposition
        response_message = get_gpt_response(model=model,
                                            messages=task_decomp_few_shot,
                                            available_functions=available_functions)

    # If the current call is generated by modifying conversation requirements, delete several messages from the original messages
    # Note that deleting intermediate messages must be done after creating the new response_message
    if delete_some_messages:
        for i in range(delete_some_messages):
            messages.messages_pop(manual=True, index=-1)

    # If it is a text response task
    if not response_message.get("function_call"):
        text_answer_message = response_message
        messages = is_text_response_valid(model=model,
                                          messages=messages,
                                          text_answer_message=text_answer_message,
                                          available_functions=available_functions,
                                          delete_some_messages=delete_some_messages,
                                          is_task_decomposition=is_task_decomposition)

    # If it is a function response task
    elif response_message.get("function_call"):
        function_call_message = response_message
        messages = is_code_response_valid(model=model,
                                          messages=messages,
                                          function_call_message=function_call_message,
                                          available_functions=available_functions,
                                          delete_some_messages=delete_some_messages)

    return messages


def is_code_response_valid(model,
                           messages,
                           function_call_message,
                           available_functions=None,
                           delete_some_messages=False):
    """
    Responsible for executing an external function call completely. The last message in the input `messages` must be a message containing a function call.
    The function's final task is to pass the code from the function call message to the external function and complete the code execution, supporting both interactive and automated code execution modes.
    :param model: Required parameter indicating the name of the large model to be called.
    :param messages: Required parameter, a ChatMessages type object used to store conversation messages.
    :param function_call_message: Required parameter representing a message containing a function call created by the upper-level function.
    :param available_functions: Optional parameter, an AvailableFunctions type object representing the basic information of external functions during the conversation.
    Defaults to None, indicating no external functions.
    :param delete_some_messages: Optional parameter indicating whether to delete several intermediate messages when concatenating messages, default is False.
    :return: Message containing the latest result from the large model's response.
    """

    # Prepare for printing and modifying code (adding image creation code for the family part)
    # Create a JSON string message object
    code_json_str = function_call_message["function_call"]["arguments"]
    # Convert JSON to a dictionary
    try:
        code_dict = json.loads(code_json_str)
    except Exception as e:
        print("JSON parsing error, recreating code...")
        messages = get_chat_response(model=model,
                                     messages=messages,
                                     available_functions=available_functions,
                                     delete_some_messages=delete_some_messages)

        return messages

    # Create a helper function convert_to_markdown to assist in printing code results
    def convert_to_markdown(code, language):
        return f"```{language}\n{code}\n```"

    # Extract code part parameters
    # If it's SQL, print code in SQL format in Markdown
    if code_dict.get('sql_query'):
        code = code_dict['sql_query']
        markdown_code = convert_to_markdown(code, 'sql')
        print("The following code will be executed:")

    # If it's Python, print code in Python format in Markdown
    elif code_dict.get('py_code'):
        code = code_dict['py_code']
        markdown_code = convert_to_markdown(code, 'python')
        print("The following code will be executed:")

    else:
        markdown_code = code_dict

    display(Markdown(markdown_code))

    function_response_message = function_to_call(available_functions=available_functions,
                                                 function_call_message=function_call_message)

    messages = check_get_final_function_response(model=model,
                                                 messages=messages,
                                                 function_call_message=function_call_message,
                                                 function_response_message=function_response_message,
                                                 available_functions=available_functions,
                                                 delete_some_messages=delete_some_messages)

    return messages


def check_get_final_function_response(model,
                                      messages,
                                      function_call_message,
                                      function_response_message,
                                      available_functions=None,
                                      delete_some_messages=False):
    """
    Responsible for reviewing the results of external function execution.
    If the function_response_message does not contain any error information,
    it will be appended to the message and passed to the get_chat_response function to obtain the next round of conversation results.
    If error information is present in the function_response_message, automatic debug mode will be enabled.
    :param model: Required parameter indicating the name of the large model to be called.
    :param messages: Required parameter, a ChatMessages type object used to store conversation messages.
    :param function_call_message: Required parameter representing a message containing a function call created by the upper-level function.
    :param function_response_message: Required parameter representing a message containing the result of the external function's execution created by the upper-level function.
    :param available_functions: Optional parameter, an AvailableFunctions type object representing the basic information of external functions during the conversation.
    Defaults to None, indicating no external functions.
    :param delete_some_messages: Optional parameter indicating whether to delete several intermediate messages when concatenating messages, default is False.
    :return: Message containing the latest result from the large model's response.
    """

    fun_res_content = function_response_message["content"]

    # If function_response contains errors
    if "error" in fun_res_content:
        print(fun_res_content)

        msg_debug = messages.copy()
        msg_debug.messages_append(function_call_message)
        msg_debug.messages_append(function_response_message)

        messages = msg_debug.copy()

    # If function message does not contain error information
    else:
        print("External function execution complete. Parsing the results...")
        messages.messages_append(function_call_message)
        messages.messages_append(function_response_message)
        messages = get_chat_response(model=model,
                                     messages=messages,
                                     available_functions=available_functions,
                                     delete_some_messages=delete_some_messages)

    return messages


def is_text_response_valid(model,
                           messages,
                           text_answer_message,
                           available_functions=None,
                           delete_some_messages=False,
                           is_task_decomposition=False):
    """
    Responsible for reviewing the creation of text content. The running mode can be either fast mode or manual review mode.
    In fast mode, the model quickly creates text and saves it to the msg object.
    In manual review mode, human confirmation is required before the function saves the text content created by the large model.
    During this process, the model can also be instructed to modify the text based on user input.
    :param model: Required parameter, indicating the name of the large model being called.
    :param messages: Required parameter, a ChatMessages type object used to store conversation messages.
    :param text_answer_message: Required parameter, representing a message containing text content created by the upper-level function.
    :param available_functions: Optional parameter, an AvailableFunctions type object representing the basic information of external functions during the conversation.
    Defaults to None, indicating no external functions.
    :param delete_some_messages: Optional parameter indicating whether to delete several intermediate messages when concatenating messages, default is False.
    :param is_task_decomposition: Optional parameter indicating whether the current task is a review of task decomposition results, default is False.
    :return: Message containing the latest result from the large model's response.
    """

    answer_content = text_answer_message["content"]

    print("Model's Answer:\n")
    display(Markdown(answer_content))

    user_input = None

    # If task decomposition
    if is_task_decomposition:
        user_input = input("Would you like to execute the task according to this process (1),\
        provide modification feedback on the current process (2),\
        ask a new question (3),\
        or exit the conversation (4)?")
        if user_input == '1':
            messages.messages_append(text_answer_message)
            print("Okay, proceeding to execute the process step by step.")
            messages.messages_append({"role": "user", "content": "Very well, please execute the process step by step."})
            is_task_decomposition = False
            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         delete_some_messages=delete_some_messages,
                                         is_task_decomposition=is_task_decomposition)

    if user_input is not None:
        if user_input == '1':
            pass
        elif user_input == '2':
            new_user_content = input("Okay, enter your modification feedback for the model's result:")
            print("Okay, making modifications.")
            messages.messages_append(text_answer_message)
            messages.messages_append({"role": "user", "content": new_user_content})

            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         delete_some_messages=2,
                                         is_task_decomposition=is_task_decomposition)

        elif user_input == '3':
            new_user_content = input("Okay, please ask a new question:")
            messages.messages[-1]["content"] = new_user_content
            messages = get_chat_response(model=model,
                                         messages=messages,
                                         available_functions=available_functions,
                                         delete_some_messages=delete_some_messages,
                                         is_task_decomposition=is_task_decomposition)

        else:
            print("Okay, exiting the current conversation.")

    else:
        messages.messages_append(text_answer_message)

    return messages


if __name__ == '__main__':
    print("This file contains functions to get responses from LLM.")
