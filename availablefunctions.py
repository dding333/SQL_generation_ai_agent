import pandas as pd
import numpy as np
import inspect
import openai
import json
import time
from gptLearning import *


class AvailableFunctions:
    """
    This class manages external function support, facilitating relevant functionality when external functions are invoked.
    It contains attributes for a list of external functions, descriptions of function parameters, and rules for function calls.
    """

    def __init__(self, functions_list=None, functions=None, function_call="auto"):
        # Initialize with empty lists if no functions are provided
        self.functions_list = functions_list if functions_list is not None else []
        self.functions = functions if functions is not None else []
        self.functions_dic = None
        self.function_call = None

        # If functions_list is provided, create a dictionary mapping function names to the actual functions and set function_call rules
        if self.functions_list:
            self.functions_dic = {func.__name__: func for func in self.functions_list}
            self.function_call = function_call

            # Automatically generate function descriptions if none are provided
            if not self.functions:
                self.functions = auto_functions(self.functions_list)

    # Method to add a new external function and optionally update its description and function call rules
    def add_function(self, new_function, function_description=None, function_call_update=None):
        # Add the new function to the list and dictionary
        self.functions_list.append(new_function)
        self.functions_dic[new_function.__name__] = new_function

        # Automatically generate a description if one isn't provided
        if function_description is None:
            new_function_description = auto_functions([new_function])
            self.functions.append(new_function_description)
        else:
            self.functions.append(function_description)

        # Update the function call rules if a new rule is provided
        if function_call_update is not None:
            self.function_call = function_call_update


if __name__ == '__main__':
    print("This file defines the AvailableFunctions class.")

