import csv
import math


# Define the function f(x, y)
def f(x, y):
    return math.sin(math.pi * 10 * x + 10 / (1 + y ** 2)) + math.log(x ** 2 + y ** 2)



# Running will make a csv named result, only with int inputs, will need to be expanded
# to decimal values of x and y.
if __name__ == "__main__":
    # Open a CSV file for writing
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(['x', 'y', 'result'])

        # Double for-loop for all combinations of x and y
        for x in range(1, 101):
            for y in range(1, 101):
                result = f(x, y)
                writer.writerow([x, y, result])