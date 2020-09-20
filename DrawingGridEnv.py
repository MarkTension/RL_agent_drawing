import numpy as np
from tensorforce import Environment
import GridController

class params:
  gridSize = 32
  egoSize = 31 # should be uneven (centered around agent)
  targetSize = 5
  stepReward = 0.01
  maxEpisodeTimesteps = 300


class SimpleGrid(Environment):

  def __init__(self):
    super().__init__()
    self.params = params
    self.scene_controller = GridController.SceneController(params)

  def states(self):
    return dict(type='int', shape=(self.params.egoSize,self.params.egoSize))

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
    state = np.random.random(size=(self.params.egoSize,self.params.egoSize))
    return state

  def execute(self, actions):
    next_state = self.scene_controller.EnvironmentStep(actions)
    terminal = self.scene_controller.CheckFinished()
    reward = self.scene_controller.stepReward
    return next_state, terminal, reward