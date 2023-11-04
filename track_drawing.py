# Import pygame and initialize it
import pygame
import csv

pygame.init()



# Define a list to store the lists of points
shapes = []

# Define a variable to store the current shape index
shape_index = 0
shapes.append([])

# Define the screen size and color
screen_width = 1000
screen_height = 1000
screen_color = (255, 255, 255)  # White

# Create a screen object
screen = pygame.display.set_mode((screen_width, screen_height))

# Define the line color and width
line_color = (0, 0, 0)  # Black
line_width = 5

# Define a list to store the points
points = []

# Define a boolean variable to indicate if the mouse button is pressed
drawing = False

# Define the main loop
running = True
while running:
    # Fill the screen with the background color
    screen.fill(screen_color)

    # Check for events
    for event in pygame.event.get():
        # If the user clicks the close button, exit the loop
        if event.type == pygame.QUIT:
            running = False

        # If the user presses the mouse button, start drawing and add the first point
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            shapes[shape_index].append(event.pos)

        # If the user moves the mouse while pressing the button, add more points
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                shapes[shape_index].append(event.pos)

        # If the user releases the mouse button, stop drawing and print the points
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                print("starting a new shape")
                shape_index += 1
                shapes.append([])
                print(len(shapes))

            if event.key == pygame.K_RETURN:
                # Open a text file in write mode
                with open("shapes.txt", "w") as f:
                    # Create a csv writer object
                    wr = csv.writer(f)
                    # Write each shape as a row in the file
                    wr.writerows(shapes)
                print(shapes[shape_index])



    # Draw the lines between the points for each shape
    for shape in shapes:
        if len(shape) > 1:
            pygame.draw.lines(screen, line_color, False, shape, line_width)

    # Update the display
    pygame.display.flip()

# Quit pygame and exit the program
pygame.quit()
exit()
