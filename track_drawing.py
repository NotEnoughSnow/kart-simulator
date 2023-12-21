# Import pygame and initialize it
import pygame
import csv
import ast

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
last_line_color = (255, 0, 0)  # Red
track_line_color = (0, 0, 255)  # Blue
line_width = 5

# Define a list to store the points
points = []

# Define a boolean variable to indicate if the mouse button is pressed
drawing = False

mode = "sectors"
print(("working in %s mode" % mode))
shapes_arr =[]

with open("kartSim/resources/shapes.txt", "r") as f:
    reader = csv.reader(f, delimiter=",")
    points = list(reader)
    shapes_arr = [list(map(ast.literal_eval, shape)) for shape in points]


# Define the main loop
running = True
while running:
    # Fill the screen with the background color
    screen.fill(screen_color)

    keys = pygame.key.get_pressed()

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
        #elif event.type == pygame.MOUSEMOTION:
        #    if drawing:
        #        shapes[shape_index].append(event.pos)

        # If the user releases the mouse button, stop drawing and print the points
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                print("starting a new shape")
                shape_index += 1
                shapes.append([])
                print(len(shapes))

            if event.key == pygame.K_z and keys[pygame.K_LCTRL]:
                if len(shapes[shape_index]) > 0:
                    shapes[shape_index].pop()

            if event.key == pygame.K_x and keys[pygame.K_LCTRL]:
                if len(shapes) > 0:
                    shapes.pop()
                    shapes.append([])

            if event.key == pygame.K_RETURN:
                if mode == "track":
                    with open("test_shapes.txt", "w") as f:
                        wr = csv.writer(f)
                        wr.writerows(shapes)
                    print(shapes[shape_index])
                if mode == "sectors":
                    with open("kartSim/resources/sectors.txt", "w") as f:
                        wr = csv.writer(f)
                        wr.writerows(shapes)
                    print(shapes[shape_index])

    # Draw the lines between the points for each shape
    for i in range(len(shapes)):
        if len(shapes[i]) > 2:
            pygame.draw.lines(screen, line_color, False, shapes[i][0:], line_width)
            pygame.draw.lines(screen, last_line_color, False, shapes[i][-2:], line_width)
        elif len(shapes[i]) > 1:
            pygame.draw.lines(screen, line_color, False, shapes[i], line_width)

    if mode == "sectors":
        for shape in shapes_arr:
            pygame.draw.lines(screen, track_line_color, False, shape, line_width)


    # Update the display
    pygame.display.flip()

# Quit pygame and exit the program
pygame.quit()
exit()
