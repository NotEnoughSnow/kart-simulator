import pygame_gui
import pygame
import os
import kartSimulator.sim.LIDAR_vision as vision
from pymunk import Vec2d


ui_start_x = 1000


accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'accelerate.png'))
not_accelerate_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_accelerate.png'))
break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'break.png'))
not_break_image = pygame.image.load(os.path.join('kartSimulator\\resources', 'not_break.png'))


class UImanager:

    def __init__(self,window_surface, window_width, window_length):

        self.window_surface = window_surface

        # UI
        self._guiManager = pygame_gui.UIManager((window_width, window_length))
        self._guiManager.set_visual_debug_mode(True)

        gui_window = pygame_gui.elements.UIWindow(rect=pygame.rect.Rect((ui_start_x, 0), (500, 1000)),
                                                  window_display_title='',
                                                  manager=self._guiManager)

        # next_target
        # goal_distance
        # rew_distance
        # time
        # steering
        # velocity
        # position

        self.text_labels = []
        number_of_text_labels = 20
        self.position_counter = 0

        for i in range(number_of_text_labels):
            self.new_label(gui_window, 25, i)

        self.break_ui = pygame_gui.elements.UIImage(relative_rect=pygame.Rect((50, 0), (100, 100)),
                                                    container=gui_window,
                                                    image_surface=not_break_image,
                                                    manager=self._guiManager)

        self.accelerate_ui = pygame_gui.elements.UIImage(relative_rect=pygame.Rect((250, 0), (100, 100)),
                                                         container=gui_window,
                                                         image_surface=not_accelerate_image,
                                                         manager=self._guiManager)

    def new_label(self, gui_window, gap, i):
        self.text_labels.append(pygame_gui.elements.UILabel(relative_rect=pygame.rect.Rect((25, 100 + i*gap),
                                                                                            (475, 150 + i*gap)),
                                                             text="",
                                                             container=gui_window,
                                                             manager=self._guiManager))

    def draw_vision_points(self, vision_points):
        vision.draw_rays(self.window_surface, Vec2d(ui_start_x + 250, 850), vision_points, 0.3, True, True)

    def draw_vision_cone(self, player_body):
        vision.draw_cone(self.window_surface, player_body)


    def draw_UI_icons(self, acc_value, break_value, steer_right_value, steer_left_value):

        if not break_value:
            self.break_ui.set_image(not_break_image)
        else:
            self.break_ui.set_image(break_image)

        if not acc_value:
            self.accelerate_ui.set_image(not_accelerate_image)
        else:
            self.accelerate_ui.set_image(accelerate_image)

    def update(self, time_delta, background):
        self._guiManager.update(time_delta)
        self._guiManager.draw_ui(background)
        self.position_counter = 0


    def add_ui_text(self, text, variable, precision):
        if self.position_counter > len(self.text_labels) - 1:
            print("requested position :", self.position_counter)
            print("total positions :", len(self.text_labels))
            raise Exception("Increase the text labels")

        self.text_labels[self.position_counter].set_text(f"{text}: {variable:{precision}}")

        self.position_counter += 1



