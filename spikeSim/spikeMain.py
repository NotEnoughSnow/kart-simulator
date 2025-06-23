import sys
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QGraphicsRectItem, QMainWindow
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPainter

import PySide6

class LIF_neuron:
    threshold = 5

    def __init__(self, threshold, id):
        self.threshold = threshold
        self.temp_threshold = 0
        self.id = id

    def update(self):
        if self.temp_threshold >= self.threshold:
            self.fire()

    def fire(self):
        print(f"neuron {self.id} fired")


def create_color_grid(scene, grid_size, cell_size):
    grid = []
    for row in range(grid_size):
        row_items = []
        for col in range(grid_size):
            # Create a rectangle item
            rect = QGraphicsRectItem(col * cell_size, row * cell_size, cell_size, cell_size)
            rect.setBrush(QColor(row % 256, col % 256, (row + col) % 256))
            #rect.setPen(Qt.NoPen)

            # Add the rectangle to the scene and the row array
            scene.addItem(rect)
            row_items.append(rect)

        grid.append(row_items)
    return grid

def main():
    app = QApplication([])

    # Create a QGraphicsScene and QGraphicsView
    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    view.resize(800,800)
    view.setRenderHint(QPainter.Antialiasing)

    # Define grid properties
    grid_size = 100
    cell_size = 6

    # Create the grid and store the items
    color_grid = create_color_grid(scene, grid_size, cell_size)


    # Show the view
    view.show()
    sys.exit(app.exec())

if __name__ == "__main__":

    print(PySide6.__version__)
    main()
