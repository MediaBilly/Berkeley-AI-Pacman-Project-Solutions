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
        import sys
        # Calculate distance to closest dot
        minFoodDist = sys.maxint
        for food in newFood.asList():
            minFoodDist = min(manhattanDistance(newPos,food),minFoodDist)
        # Calculate distance to closest ghost
        minGhostDist = sys.maxint
        for ghost in newGhostStates:
            minGhostDist = min(manhattanDistance(ghost.getPosition(),newPos),minGhostDist)
        # If we hit a ghost return the least score possible as it is the worst move
        if minGhostDist == 0:
            return -sys.maxint
        # On the final score we will consider the distances calculated above inversely(the least the better)
        if sum(newScaredTimes) > 0:
            # If ghosts are scared consider positively the distance to closest ghost in order to chase and try to eat it
            score = successorGameState.getScore() + 10 / minFoodDist + 10 / minGhostDist
        else:
            # Otherwise consider it negatively
            score = successorGameState.getScore() + 10 / minFoodDist - 20 / minGhostDist
        return score

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
        def minimax(state,depth,agent):
            if depth <= 0 or len(state.getLegalActions(0)) == 0:
                return (self.evaluationFunction(state),None)
            retAction = None
            # Pacman(maximizer)
            if agent == 0:
                score = float('-inf')
                # Get the move with the maximum score
                for action in state.getLegalActions(agent):
                    tmpScore = minimax(state.generateSuccessor(agent, action),depth,1)[0]
                    if tmpScore > score:
                        score = tmpScore
                        retAction = action
            else: # Ghost(minimizer)
                score = float('inf')
                # Get the move with the minimum score
                for action in state.getLegalActions(agent):
                    # Final ghost so next plays pacman
                    if agent == state.getNumAgents() - 1:
                        tmpScore = minimax(state.generateSuccessor(agent, action),depth-1,0)[0]
                    else: # Next ghost plays
                        tmpScore = minimax(state.generateSuccessor(agent, action), depth, agent + 1)[0]
                    if tmpScore < score:
                        score = tmpScore
                        retAction = action
            return (score,retAction)
        return minimax(gameState,self.depth,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(state,depth,agent,alpha,beta):
            if depth <= 0 or len(state.getLegalActions(0)) == 0:
                return (self.evaluationFunction(state),None)
            retAction = None
            # Pacman(maximizer)
            if agent == 0:
                score = float('-inf')
                # Get the move with the maximum score
                for action in state.getLegalActions(agent):
                    tmpScore = alphabeta(state.generateSuccessor(agent, action),depth,1,alpha,beta)[0]
                    if tmpScore > score:
                        score = tmpScore
                        retAction = action
                    if score > beta:
                        break
                    alpha = max(alpha,score)
            else: # Ghost(minimizer)
                score = float('inf')
                # Get the move with the minimum score
                for action in state.getLegalActions(agent):
                    # Final ghost so next plays pacman
                    if agent == state.getNumAgents() - 1:
                        tmpScore = alphabeta(state.generateSuccessor(agent, action),depth-1,0,alpha,beta)[0]
                    else: # Next ghost plays
                        tmpScore = alphabeta(state.generateSuccessor(agent, action), depth, agent + 1,alpha,beta)[0]
                    if tmpScore < score:
                        score = tmpScore
                        retAction = action
                    if score < alpha:
                        break
                    beta = min(beta,score)
            return (score,retAction)
        return alphabeta(gameState,self.depth,0,float('-inf'),float('inf'))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state,depth,agent):
            if depth <= 0 or len(state.getLegalActions(0)) == 0:
                return (self.evaluationFunction(state),None)
            retAction = None
            # Pacman(maximizer)
            if agent == 0:
                score = float('-inf')
                # Get the move with the maximum score
                for action in state.getLegalActions(agent):
                    tmpScore = expectimax(state.generateSuccessor(agent, action),depth,1)[0]
                    if tmpScore > score:
                        score = tmpScore
                        retAction = action
            else: # Ghost(random)
                score = 0
                # Get the move with the minimum score
                actions = state.getLegalActions(agent)
                numActions = len(actions)
                for action in actions:
                    # Final ghost so next plays pacman
                    if agent == state.getNumAgents() - 1:
                        tmpScore = expectimax(state.generateSuccessor(agent, action),depth-1,0)[0]
                    else: # Next ghost plays
                        tmpScore = expectimax(state.generateSuccessor(agent, action), depth, agent + 1)[0]
                    retAction = action
                    score += tmpScore*(1.0/numActions)
            return (score,retAction)
        return expectimax(gameState,self.depth,0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: same logic with question 1 but this time except for a successor state we are evaluating the current state
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    import sys
    # Calculate distance to closest dot
    minFoodDist = sys.maxint
    for food in food.asList():
        minFoodDist = min(manhattanDistance(pos, food), minFoodDist)
    # Calculate distance to closest ghost
    minGhostDist = sys.maxint
    for ghost in ghostStates:
        minGhostDist = min(manhattanDistance(ghost.getPosition(), pos), minGhostDist)
    # If we hit a ghost return the least score possible as it is the worst move
    if minGhostDist == 0:
        return -sys.maxint
    # On the final score we will consider the distances calculated above inversely(the least the better)
    if sum(scaredTimes) > 0:
        # If ghosts are scared consider positively the distance to closest ghost in order to chase and try to eat it
        score = currentGameState.getScore() + 10 / minFoodDist + 10 / minGhostDist
    else:
        # Otherwise consider it negatively
        score = currentGameState.getScore() + 10 / minFoodDist - 20 / minGhostDist
    return score

# Abbreviation
better = betterEvaluationFunction

