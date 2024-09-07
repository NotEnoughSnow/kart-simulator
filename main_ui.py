import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QLineEdit, QFormLayout, QSpinBox, QDoubleSpinBox

class RLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('RL App UI')

        # Main layout
        self.layout = QVBoxLayout()

        # Label for app title
        self.label = QLabel('Select a Mode:')
        self.layout.addWidget(self.label)

        # ComboBox for selecting modes
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(['train', 'play', 'test', 'replay', 'graph', 'optimize'])
        self.mode_selector.currentIndexChanged.connect(self.update_options)  # Update options on selection change
        self.layout.addWidget(self.mode_selector)

        # Placeholder layout for additional mode-specific options
        self.options_layout = QFormLayout()
        self.layout.addLayout(self.options_layout)

        # Start button
        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.run_rl_app)
        self.layout.addWidget(self.start_button)

        # Set the layout
        self.setLayout(self.layout)

    def update_options(self):
        # Clear previous mode-specific options
        for i in reversed(range(self.options_layout.count())):
            self.options_layout.itemAt(i).widget().deleteLater()

        selected_mode = self.mode_selector.currentText()

        if selected_mode == "train":
            self.add_training_options()
        elif selected_mode == "play":
            self.add_play_options()
        elif selected_mode == "test":
            self.add_test_options()

    def add_training_options(self):
        # Example options for training mode
        self.options_layout.addRow("Learning Rate:", QDoubleSpinBox())
        self.options_layout.addRow("Batch Size:", QSpinBox())
        self.options_layout.addRow("Epochs:", QSpinBox())

        # Add more hyperparameters as needed
        self.options_layout.addRow("Gamma:", QDoubleSpinBox())
        self.options_layout.addRow("Clip:", QDoubleSpinBox())

    def add_play_options(self):
        # Example options for play mode
        self.options_layout.addRow("Player Name:", QLineEdit())
        self.options_layout.addRow("Recording:", QComboBox().addItems(["Yes", "No"]))

    def add_test_options(self):
        # Example options for test mode
        self.options_layout.addRow("Test Episode Count:", QSpinBox())
        self.options_layout.addRow("Deterministic:", QComboBox().addItems(["True", "False"]))

    def run_rl_app(self):
        selected_mode = self.mode_selector.currentText()

        # Handle the selected mode accordingly
        if selected_mode == "train":
            self.train()
        elif selected_mode == "play":
            self.play()
        elif selected_mode == "test":
            self.test()

    def train(self):
        # Implement the training logic here
        print("Training mode selected")

    def play(self):
        # Implement the play logic here
        print("Play mode selected")

    def test(self):
        # Implement the test logic here
        print("Test mode selected")

# Main entry point of the PySide6 app
if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = RLApp()
    window.show()

    sys.exit(app.exec_())
