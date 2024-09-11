import matplotlib.pyplot as plt

# Data
x = [1, 2, 3, 4, 5, 6]
y = [1, 4, 9, 16, 25, 36]

# Create a new figure
plt.figure()

# Plot the data
plt.plot(x, y, marker='o', linestyle='-', color='b', label='y = x^2')

# Add title and labels
plt.title('Plot of y = x^2')
plt.xlabel('x')
plt.ylabel('y')

# Add a legend
plt.legend()

# Show grid
plt.grid(True)

# Display the plot
plt.show()
