def add_task_decomposition_prompt(messages):
    """
    This function is triggered when enhanced mode is active. It inserts task decomposition Few-shot examples
    to help decompose the user's question. It returns an updated message with these examples.

    :param messages: Required parameter, a ChatMessages object containing the conversation history.
    :return: Updated message object with task decomposition Few-shot examples added.
    """

    # Task decomposition Few-shot examples
    # Example 1
    user_question1 = 'What is Google Cloud Email?'
    user_message1_content = f"The current question is: '{user_question1}'. How many steps are necessary to answer this? If no decomposition is required, provide a direct answer."
    assistant_message1_content = (
        'Google Cloud Email is part of Google Workspace (formerly G Suite), commonly known as Gmail. '
        'It provides a secure, user-friendly email service, offering 15GB of free storage, with spam and virus protection. '
        'Gmail can be accessed from any device and includes features like search and labeling to organize your emails.')

    # Example 2
    user_question2 = 'Please introduce OpenAI.'
    user_message2_content = f"The current question is: '{user_question2}'. How many steps are necessary to answer this? If no decomposition is required, provide a direct answer."
    assistant_message2_content = (
        'OpenAI is a company focused on developing artificial intelligence in a way that maximizes societal benefits. '
        'The company aims to ensure the responsible deployment of artificial general intelligence (AGI) and is committed to both humanitarian goals and advanced AI research, such as models like GPT-3.')

    # Example 3
    user_question3 = 'I want to check if there are missing values in the user_payments table in the database.'
    user_message3_content = f"The current question is: '{user_question3}'. How many steps are necessary to answer this? If no decomposition is required, provide a direct answer."
    assistant_message3_content = ('To verify if the user_payments table has missing values, follow these steps: '
                                  '\n\nStep 1: Load the user_payments table using the `extract_data` function. '
                                  '\n\nStep 2: Use Python to check for missing values by running code to detect them in the dataset.')

    # Example 4
    user_question4 = 'I want to find a suitable method for imputing missing values in the user_payments dataset.'
    user_message4_content = f"The current question is: '{user_question4}'. How many steps are necessary to answer this? If no decomposition is required, provide a direct answer."
    assistant_message4_content = ('To find an appropriate method for imputing missing values, follow these steps: '
                                  '\n\nStep 1: Analyze the missing data in the user_payments dataset. Check the missing rates and distribution. '
                                  '\n\nStep 2: Select the imputation strategy, such as using the mode, median, or mean, or building an imputation model. '
                                  '\n\nStep 3: Perform the chosen imputation strategy and verify the results.')

    # Update the conversation with task decomposition examples
    task_decomp_few_shot = messages.copy()
    task_decomp_few_shot.messages_pop(manual=True, index=-1)
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message1_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message1_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message2_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message2_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message3_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message3_content})
    task_decomp_few_shot.messages_append({"role": "user", "content": user_message4_content})
    task_decomp_few_shot.messages_append({"role": "assistant", "content": assistant_message4_content})

    user_question = messages.history_messages[-1]["content"]

    final_question = f"The current question is: '{user_question}'. How many steps are necessary to answer this? If no decomposition is required, provide a direct answer."
    question_message = messages.history_messages[-1].copy()
    question_message["content"] = final_question
    task_decomp_few_shot.messages_append(question_message)

    return task_decomp_few_shot


def modify_prompt(messages, action='add', enable_md_output=True, enable_COT=True):
    """
    This function modifies the conversation prompt by either adding or removing specific prompts, such as
    Chain of Thought (COT) or Markdown formatting, based on user preference.

    :param messages: Required parameter, a ChatMessages object storing the conversation messages.
    :param action: 'add' or 'remove', determines whether to add or remove certain prompts.
    :param enable_md_output: Boolean, indicates whether to include markdown formatting in the prompt.
    :param enable_COT: Boolean, indicates whether to include Chain of Thought (COT) prompts.
    :return: The updated message object.
    """

    # Chain of Thought (COT) prompt
    cot_prompt = "Please think step by step to arrive at a conclusion."
    # Markdown formatting prompt
    md_prompt = "Please format all responses using markdown."

    if action == 'add':
        if enable_COT:
            messages.messages[-1]["content"] += cot_prompt
            messages.history_messages[-1]["content"] += cot_prompt

        if enable_md_output:
            messages.messages[-1]["content"] += md_prompt
            messages.history_messages[-1]["content"] += md_prompt

    elif action == 'remove':
        if enable_md_output:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(md_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(md_prompt, "")

        if enable_COT:
            messages.messages[-1]["content"] = messages.messages[-1]["content"].replace(cot_prompt, "")
            messages.history_messages[-1]["content"] = messages.history_messages[-1]["content"].replace(cot_prompt, "")

    return messages


if __name__ == '__main__':
    print("This file is part of the task decomposition module.")
