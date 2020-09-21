import numpy as np

#  for visualization of the grid
import matplotlib as mpl
from matplotlib import pyplot as plt

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

    # sizes
    self.targetSize = params.targetSize
    self.gridSize = params.gridSize
    self.egoSize = params.egoSize
    self.halfEgoSize = np.floor(self.egoSize*0.5).astype(np.int)

    # state and objective grids, to be filled later on
    self.gridState = np.zeros(shape=(self.params.gridSize, self.params.gridSize))
    self.gridState_objective = np.zeros(shape=(self.params.gridSize, self.params.gridSize))
    self.gridState_target = np.zeros(shape=(self.params.gridSize, self.params.gridSize))

    # agent variables
    # agentEgoView is the agent's view: an ego-size slice of the gridState, centered on the agent
    self.agentEgoView = np.zeros(shape=(self.egoSize, self.egoSize))
    self.agentPosition = [np.random.randint(0,self.gridSize-1, dtype=int), np.random.randint(0,self.gridSize-1 , dtype=int)]

    self.stepCount = 0
    self.cumreward = 0

    # set all episode stuff
    self.OnEpisodeBegin()

    # figure plotting
    cmap = cm.get_cmap('viridis', 4)
    plt.figure(figsize=(2, 2))
    plt.title(f"episode {params.episode}")
    self.graph = plt.imshow(self.gridState_objective, extent=(0, self.params.gridSize, self.params.gridSize, 0),
                  interpolation='None', vmin=0, vmax= 2, cmap=cmap)
    plt.colorbar()

  def OnEpisodeBegin(self):

    self.gridState.fill(0)
    self.GenerateTarget()

    self.gridState_objective.fill(0)
    self.gridState_objective = self.gridState_target
    self.stepCount = 0
    self.cumreward = 0


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
    # self.VisualizeGrid()

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
      # else:
      #   self.gridState[self.agentPosition[0],self.agentPosition[1]] = 0

      # objective
      if self.gridState_objective[tuple(self.agentPosition)] == 1:
        self.gridState_objective[tuple(self.agentPosition)] = 0
      else:
        pass

    else:
      raise ValueError("invalid agent plot action")

    # reward
    if draw == 1 and self.gridState[tuple(self.agentPosition)] == self.gridState_target[tuple(self.agentPosition)]:
      self.stepReward = self.params.stepReward
    # elif draw == 1 and self.gridState[tuple(self.agentPosition)] != self.gridState_target[tuple(self.agentPosition)]:
    #   self.stepReward = -self.params.stepReward
    self.cumreward += self.stepReward


  def CheckFinished(self):

    endEpisode = self.stepCount > self.params.maxEpisodeTimesteps
    if (endEpisode):
      plt.close()

    return endEpisode


  def AjustAgentView(self):
    """
    slices part of the objective grid centered on agent position
    before that, pads objective state to make slicing easy

    """

    padded_objective = np.pad(self.gridState_objective, pad_width=self.halfEgoSize, mode='constant', constant_values=2)

    rows = range(self.agentPosition[0] - self.halfEgoSize, self.agentPosition[0] + self.halfEgoSize)
    columns = range(self.agentPosition[1] - self.halfEgoSize, self.agentPosition[1] + self.halfEgoSize)

    self.agentEgoView = padded_objective[
                        self.halfEgoSize + self.agentPosition[0] - self.halfEgoSize : 1 + self.halfEgoSize + self.agentPosition[0] + self.halfEgoSize,
                        self.halfEgoSize + self.agentPosition[1] - self.halfEgoSize : 1 + self.halfEgoSize + self.agentPosition[1] + self.halfEgoSize].astype(int)
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


    # visualization = self.gridState_objective.copy()
    # visualization[tuple(self.agentPosition)] = 0.5
    # plt.imshow(visualization, extent=(0, self.params.gridSize, self.params.gridSize, 0),
    #               interpolation='None', cmap="viridis")

    # self.graph.set_data(visualization)
    visualization2 = self.agentEgoView.copy()
    size = np.floor(self.egoSize/2).astype(np.int)
    visualization2[size, size] = 1.5
    self.graph.set_data(visualization2)
    # self.graphs2.set_data(self.gridState)
    plt.pause(0.0001)
