import numpy as np

#  for visualization of the grid
import matplotlib as mpl
from matplotlib import pyplot

import matplotlib.cm as cm


class SceneController:
  """
  Scenecontroller
  - stores the state of the scene
  - makes the agent move in it
  - returns the agent's partial view of the scene

  """
  def __init__(self, params):
    self.params = params

    self.targetSize = params.targetSize
    self.gridSize = params.gridSize
    self.egoSize = params.egoSize

    # initialize state and objective grids, to be filled later
    self.gridState = np.zeros(shape=(self.params.gridSize, self.params.gridSize))
    self.gridState_objective = np.zeros(shape=(self.params.gridSize, self.params.gridSize))
    self.gridState_target = np.zeros(shape=(self.params.gridSize, self.params.gridSize))

    # agent variables
    # agentEgoView is the agent's view: an ego-size slice of the gridState, centered on the agent
    self.agentEgoView = np.zeros(shape=(self.egoSize, self.egoSize))
    self.agentPosition = [np.random.randint(0,self.gridSize-1, dtype=int), np.random.randint(0,self.gridSize-1 , dtype=int)]

    self.stepCount = 0

    # set all episode stuff
    self.OnEpisodeBegin()

  def OnEpisodeBegin(self):

    self.gridState.fill(0)
    self.GenerateTarget()

    self.gridState_objective.fill(1)
    self.gridState_objective -= self.gridState_target
    self.stepCount = 0


  def EnvironmentStep(self, action):
    """
    central part of code that calls actions, and returns state+1
    """
    # reward will be set again in Draw
    self.stepReward = 0

    # do agent actions
    self.Move(action["move"])
    self.Draw(action["draw"])

    # return current state (agent view)
    self.AjustAgentView()

    self.stepCount+=1

    self.VisualizeGrid()

    return self.agentEgoView


  def Move(self, move):
    """
    maniputales the agent's position
    """

    if (move ==0):
      self.agentPosition[0] -= 1
    elif (move == 1):
      self.agentPosition[0] += 1
    elif (move == 2):
      self.agentPosition[1] -= 1
    elif (move == 3):
      self.agentPosition[1] += 1
    else:
      raise ValueError("invalid agent move action")

    self.agentPosition = np.clip(self.agentPosition, 0, self.gridSize-1)


  def Draw(self, draw):

    if (draw == 0):
      pass
    # switch current pixel
    elif (draw == 1):
      if self.gridState[self.agentPosition[0], self.agentPosition[1]] == 0:
        self.gridState[self.agentPosition[0], self.agentPosition[1]] = 1
      else:
        self.gridState[self.agentPosition[0],self.agentPosition[1]] = 0

      # objective
      if self.gridState_objective[self.agentPosition[0], self.agentPosition[1]] == 0:
        self.gridState_objective[self.agentPosition[0], self.agentPosition[1]] = 1
      else:
        self.gridState_objective[self.agentPosition[0],self.agentPosition[1]] = 0

    else:
      raise ValueError("invalid agent plot action")

    # reward
    if draw == 1 and self.gridState[tuple(self.agentPosition)] == self.gridState_target[tuple(self.agentPosition)]:
      self.stepReward = self.params.stepReward
    elif draw == 1 and self.gridState[tuple(self.agentPosition)] != self.gridState_target[tuple(self.agentPosition)]:
      self.stepReward = -self.params.stepReward


    # todo: implement new objective map


  def CheckFinished(self):

    return self.stepCount > self.params.maxEpisodeTimesteps


  def AjustAgentView(self):
    """
    slices part of the objective grid centered on agent position
    before that, pads objective state to make slicing easy

    """

    halfEgoSize = np.floor(self.egoSize*0.5).astype(np.int)

    padded_objective = np.pad(self.gridState_objective, pad_width=halfEgoSize, mode='constant', constant_values=2)

    rows = range(self.agentPosition[0] - halfEgoSize, self.agentPosition[0] + halfEgoSize)
    columns = range(self.agentPosition[1] - halfEgoSize, self.agentPosition[1] + halfEgoSize)

    self.agentEgoView = padded_objective[
                        halfEgoSize + self.agentPosition[0] - halfEgoSize : 1 + halfEgoSize + self.agentPosition[0] + halfEgoSize,
                        halfEgoSize + self.agentPosition[1] - halfEgoSize : 1 + halfEgoSize + self.agentPosition[1] + halfEgoSize].astype(int)
    a = 0

  def GenerateTarget(self):
    """
    samples a quadrant that becomes the target
    :MANIPULATES: target grid
    """
    # reset first
    self.gridState_target.fill(0)

    rand_quadrant = np.random.randint(0,4)

    if (rand_quadrant == 0): # top left
      self.gridState_target[0:self.targetSize, 0:self.targetSize] = 1
    elif (rand_quadrant == 1): # top right
      self.gridState_target[0:self.targetSize, self.targetSize:self.gridSize] = 1
    elif (rand_quadrant == 2): # bottom left
      self.gridState_target[self.targetSize:self.gridSize, 0:self.targetSize] = 1
    elif (rand_quadrant == 3): # bottom rights
      self.gridState_target[self.targetSize:self.gridSize, self.targetSize:self.gridSize] = 1
    else:
      raise ValueError('a quadrant should be colored')

    assert (np.sum(self.gridState_target) >= 1)


  def VisualizeGrid(self):

    pyplot.imshow(self.gridState_objective, extent=(0, self.params.gridSize, self.params.gridSize, 0),
                  interpolation='None', cmap="viridis")
    a = 3
