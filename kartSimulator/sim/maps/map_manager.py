from abc import ABC, abstractmethod

class AbstractMap(ABC):

    @abstractmethod
    def reset(self, player_shape):
        pass

    @abstractmethod
    def create_walls(self):
        pass

    @abstractmethod
    def create_goals(self, mode):
        pass
