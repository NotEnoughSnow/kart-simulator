import sys
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter

import PySide6

def create_color_grid(scene, grid_size, cell_size):
    """Creates a grid of colored rectangles in the QGraphicsScene."""
    grid = []  # 2D array to store the items
    for row in range(grid_size):
        row_items = []
        for col in range(grid_size):
            # Create a rectangle item
            rect = QGraphicsRectItem(col * cell_size, row * cell_size, cell_size, cell_size)
            rect.setBrush(QColor(row % 256, col % 256, (row + col) % 256))  # Example coloring
            rect.setPen(Qt.NoPen)  # Remove border for a clean look

            # Add the rectangle to the scene and the row array
            scene.addItem(rect)
            row_items.append(rect)
        grid.append(row_items)
    return grid

def main():
    app = QApplication(sys.argv)

    # Create a QGraphicsScene and QGraphicsView
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.setRenderHint(QPainter.Antialiasing)

    # Define grid properties
    grid_size = 100
    cell_size = 10

    # Create the grid and store the items
    color_grid = create_color_grid(scene, grid_size, cell_size)

    # Optional: Adjust the view to fit the grid
    view.setSceneRect(0, 0, grid_size * cell_size, grid_size * cell_size)
    view.setWindowTitle("Pixel Grid Viewer")
    view.resize(grid_size * cell_size + 50, grid_size * cell_size + 50)

    # Show the view
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":

    print(PySide6.__version__)
    main()
