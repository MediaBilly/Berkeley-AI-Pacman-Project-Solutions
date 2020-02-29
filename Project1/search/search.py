# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # fringe is stack
    stack = util.Stack()
    # add the starting state to the fringe and set it's path to empty
    stack.push((problem.getStartState(),[]))
    # keep already expanded nodes in a set in order to avoid duplicate expansion
    checked = set()
    while not stack.isEmpty():
        curState = stack.pop()
        checked.add(curState[0])
        # if popped node is goal state return the path to it
        if problem.isGoalState(curState[0]):
            return curState[1]
        # push all not expanded successors to the frontier with their path
        successors = problem.getSuccessors(curState[0])
        for succ in successors:
            if succ[0] not in checked:
                # construct the successor's path which is the path to the current state + the action to the current successor
                stack.push((succ[0],curState[1] + [succ[1]]))
    # no more remaining nodes to expand and no goal state found so return empty path for no solution
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # fringe is queue
    queue = util.Queue()
    # add the starting state to the fringe and set it's path to empty
    queue.push((problem.getStartState(),[]))
    # keep already expanded nodes in a set in order to avoid duplicate expansion
    checked = set()
    while not queue.isEmpty():
        curState = queue.pop()
        checked.add(curState[0])
        # if popped node is goal state return the path to it
        if problem.isGoalState(curState[0]):
            return curState[1]
        # push all not expanded successors to the frontier with their path
        successors = problem.getSuccessors(curState[0])
        for succ in successors:
            if succ[0] not in checked:
                # construct the successor's path which is the path to the current state + the action to the current successor
                queue.push((succ[0],curState[1] + [succ[1]]))
                checked.add(succ[0])
    # no more remaining nodes to expand and no goal state found so return empty path for no solution
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    # fringe is priority queue
    queue = util.PriorityQueue()
    # add the starting state to the fringe and set it's path to empty and priority to 0
    queue.push((problem.getStartState(),[],0),0)
    # keep already expanded nodes in a set in order to avoid duplicate expansion
    checked = set()
    # keep the priority of expanded nodes in a dictionary because we will need it
    priority = {}
    priority[problem.getStartState()] = 0
    while not queue.isEmpty():
        curState = queue.pop()
        checked.add(curState[0])
        # if popped node is goal state return the path to it
        if problem.isGoalState(curState[0]):
            return curState[1]
        # push all not expanded successors to the frontier with their path
        successors = problem.getSuccessors(curState[0])
        for succ in successors:
            # new not yet expanded node
            if succ[0] not in checked:
                # construct the successor's path which is the path to the current state + the action to the current successor
                # and also it's priority which is the total path cost to the current node + the path cost to the current successor
                queue.push((succ[0],curState[1] + [succ[1]],priority[curState[0]] + succ[2]),priority[curState[0]] + succ[2])
                checked.add(succ[0])
                priority[succ[0]] = priority[curState[0]] + succ[2]
            # already expanded node found again but with smaller evaluation function value f(n)
            elif priority[succ[0]] > priority[curState[0]] + succ[2]:
                # calculate and update new priority
                queue.update((succ[0],curState[1] + [succ[1]],priority[curState[0]] + succ[2]),priority[curState[0]] + succ[2])
                priority[succ[0]] = priority[curState[0]] + succ[2]
    # no more remaining nodes to expand and no goal state found so return empty path for no solution
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    # fringe is priority queue
    queue = util.PriorityQueue()
    # add the starting state to the fringe and set it's path to empty and priority to it's heuristic value
    queue.push((problem.getStartState(),[],heuristic(problem.getStartState(),problem)),heuristic(problem.getStartState(),problem))
    # keep already expanded nodes in a set in order to avoid duplicate expansion
    checked = set()
    # keep the priority and pathcost of expanded nodes in a dictionary because we will need it
    pathcost = {}
    priority = {}
    pathcost[problem.getStartState()] = 0
    priority[problem.getStartState()] = heuristic(problem.getStartState(),problem)
    while not queue.isEmpty():
        curState = queue.pop()
        checked.add(curState[0])
        # if popped node is goal state return the path to it
        if problem.isGoalState(curState[0]):
            return curState[1]
        # push all not expanded successors to the frontier with their path
        successors = problem.getSuccessors(curState[0])
        for succ in successors:
            # new not yet expanded node
            if succ[0] not in checked:
                # construct the successor's path which is the path to the current state + the action to the current successor
                # and also it's priority which is the total path cost to the current node + the path cost to the current successor + the heuristic of the current successor
                pathcost[succ[0]] = pathcost[curState[0]] + succ[2]
                queue.push((succ[0],curState[1] + [succ[1]],pathcost[succ[0]] + heuristic(succ[0],problem)),pathcost[succ[0]] + heuristic(succ[0],problem))
                checked.add(succ[0])
                priority[succ[0]] = pathcost[succ[0]] + heuristic(succ[0],problem)
            # already expanded node found again but with smaller evaluation function value f(n)
            elif priority[succ[0]] > pathcost[curState[0]] + succ[2] + heuristic(succ[0],problem):
                # calculate and update new priority with the new evaluation f(n) = g(n) + h(n)
                pathcost[succ[0]] = pathcost[curState[0]] + succ[2]
                queue.update((succ[0],curState[1] + [succ[1]],pathcost[succ[0]] + heuristic(succ[0],problem)),pathcost[succ[0]] + heuristic(succ[0],problem))
                priority[succ[0]] = pathcost[succ[0]] + heuristic(succ[0],problem)
    # no more remaining nodes to expand and no goal state found so return empty path for no solution
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
