import numpy as np
from tensorforce import Environment
import GridController

class params:
  gridSize = 32
  egoSize = 32
  targetSize = 5

class SimpleGrid(Environment):

  def __init__(self):
    super().__init__()
    self.params = params
    self.scene_controller = GridController.SceneController(params)

  def states(self):
    return dict(type='int', shape=(32,32))

  def actions(self):

    return dict(type='int', num_values=4)

  # Optional: should only be defined if environment has a natural fixed
  # maximum episode length; restrict training timesteps via
  #     Environment.create(..., max_episode_timesteps=???)
  def max_episode_timesteps(self):
    return super().max_episode_timesteps()

  # Optional additional steps to close environment
  def close(self):
    super().close()

  def reset(self):
    state = np.random.random(size=(8,))
    return state

  def execute(self, actions):
    next_state = self.scene_controller.EnvironmentStep(actions)      # np.random.random(size=(8,))
    terminal = np.random.random() < 0.5
    reward = np.random.random()
    return next_state, terminal, reward