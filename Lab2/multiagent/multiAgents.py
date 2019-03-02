# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

inf = float("inf")


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        (newPos_x, newPos_y) = newPos
        width, height = newFood.width, newFood.height

        newGhostPos = [ghostState.configuration.pos for ghostState in newGhostStates]
        newGhostDirection = [ghostState.configuration.direction for ghostState in newGhostStates]
        ghostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPos]
        foodDistances = [ manhattanDistance(newPos, (i,j))**0.1 for i in range(width) for j in range(height) if newFood[i][j]]

        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        score = manhattanDistance(oldPos, newPos) - sum(foodDistances)/max(len(foodDistances), 1)
        if any(ghostdist <= 1 for ghostdist in ghostDistances):
            score -= 100
        if oldFood[newPos_x][newPos_y]:
            score += 20
        return score
        # return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        maxAction = gameState.getLegalActions(0)[0]
        maxValue = -inf
        for action in gameState.getLegalActions(0):
            value = self.minimaxValue(gameState.generateSuccessor(0, action), 0, 1)
            if value > maxValue:
                maxValue = value
                maxAction = action
        return maxAction
        # util.raiseNotDefined()


    def minimaxValue(self, gameState, depth, agent):
        numAgents = gameState.getNumAgents()
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)*1.0
        if agent == 0:
            v1 = -inf
            for action in gameState.getLegalActions(agent):
                v1 = max(v1, self.minimaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1))
            return v1
        elif agent < numAgents - 1:
            v2 = inf
            for action in gameState.getLegalActions(agent):
                v2 = min(v2, self.minimaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1))
            return v2
        elif agent == numAgents - 1:
            v2 = inf
            for action in gameState.getLegalActions(agent):
                v2 = min(v2, self.minimaxValue(gameState.generateSuccessor(agent, action), depth + 1, 0))
            return v2

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actionValues = []
        maxValue = self.minimaxValue(gameState, 0, 0, -inf, +inf, actionValues)

        return gameState.getLegalActions(0)[actionValues.index(max(actionValues))]

    def minimaxValue(self, gameState, depth, agent, alpha, beta, actionValues):
        numAgents = gameState.getNumAgents()
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)*1.0
        if agent == 0:
            v1 = -inf
            for action in gameState.getLegalActions(agent):
                v1 = max(v1, self.minimaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta, actionValues))
                if depth == 0:
                    actionValues.append(v1)
                if v1 > beta:
                    return v1
                alpha = max(alpha, v1)
            return v1
        elif agent < numAgents - 1:
            v2 = inf
            for action in gameState.getLegalActions(agent):
                v2 = min(v2, self.minimaxValue(gameState.generateSuccessor(agent, action), depth, agent + 1, alpha, beta, actionValues))
                if v2 < alpha:
                    return v2
                beta = min(beta, v2)
            return v2
        elif agent == numAgents - 1:
            v2 = inf
            for action in gameState.getLegalActions(agent):
                v2 = min(v2, self.minimaxValue(gameState.generateSuccessor(agent, action), depth + 1, 0, alpha, beta, actionValues))
                if v2 < alpha:
                    return v2
                beta = min(beta, v2)
            return v2


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    oldPos = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    oldGhostStates = currentGameState.getGhostStates()
    oldScaredTimes = [ghostState.scaredTimer for ghostState in oldGhostStates]

    width, height = oldFood.width, oldFood.height

    ghost_score = 0
    for ghost in oldGhostStates:
      distanceGhost = manhattanDistance(oldPos, ghost.getPosition())
      if ghost.scaredTimer > 0:
        ghost_score += distanceGhost**0.5
      else:
        ghost_score -= distanceGhost**0.5


    foodDistances = [ manhattanDistance(oldPos, (i,j))**0.1 for i in range(width) for j in range(height) if oldFood[i][j]]
    score = ghost_score - sum(foodDistances)/max(len(foodDistances), 1) + currentGameState.getScore()

    return score
# Abbreviation
better = betterEvaluationFunction

