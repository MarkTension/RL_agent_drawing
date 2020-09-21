import numpy as np
from tensorforce import Environment
import GridController

class params:
  gridSize = 32
  egoSize = 31 # should be uneven (centered around agent)
  targetSize = 16
  stepReward = 0.01
  maxEpisodeTimesteps = 300
  episode = 0


class SimpleGrid(Environment):

  def __init__(self):
    super().__init__()
    self.params = params
    self.scene_controller = GridController.SceneController(params)
    self.reward = 0

  def states(self):
    return dict(type='float', shape=(self.params.egoSize,self.params.egoSize)) # , num_values=3

  def actions(self):

    return {
      "move": dict(type="int", num_values=4),
      "draw": dict(type="int", num_values=2)
    }

  def max_episode_timesteps(self):
    return super().max_episode_timesteps()

  def close(self):
    super().close()

  def reset(self):

    state = np.zeros(shape=(self.params.egoSize,self.params.egoSize)).astype(np.float)

    return state

  def execute(self, actions):
    next_state = self.scene_controller.EnvironmentStep(actions)
    terminal = self.scene_controller.CheckFinished()
    reward = self.scene_controller.stepReward
    self.reward = self.scene_controller.cumreward

    if (terminal):
      self.scene_controller = GridController.SceneController(params)

    return next_state, terminal, reward