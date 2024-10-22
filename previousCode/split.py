import numpy as np
# Take input from the user
numbers_input = input("Enter numbers separated by spaces: ")

# Split the input by spaces to get individual numbers as strings
numbers_list = numbers_input.split()

# Convert the list of strings to a numpy array of floats
numbers = np.array([int(num) for num in numbers_list])

# Iterate through each number and format it with a comma
formatted_numbers = [f"{num}, " for num in numbers]

# Join the formatted numbers into a single string
result = ''.join(formatted_numbers)

# Print the result with the comma after each number
print(result)
