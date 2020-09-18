import numpy as np

#  for visualization of the grid
import matplotlib as mpl
from matplotlib import pyplot



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

    # agentEgoView is the agent's view: an ego-size slice of the gridState, centered on the agent
    self.agentEgoView = np.zeros(shape=(self.egoSize, self.egoSize))
    self.agentPosition = (np.random.randint(0,self.gridSize-1), np.random.randint(0,self.gridSize-1))

  def Move(self):

  


  def Draw(self):

    raise NotImplementedError


  def GenerateTarget(self):
    """
    samples a quadrant that becomes the target
    :MANIPULATES: target grid
    """

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

    assert (np.sum(self.gridState_target) > 0, "target grid should have content")



  def EnvironmentStep(self, actions):

    raise NotImplementedError

    # return self.gridState




  def VisualizeGrid(self):
    pyplot.figure(figsize=(self.params.egoSize, self.params.egoSize))
    pyplot.imshow(self.gridState)
    pyplot.show()

