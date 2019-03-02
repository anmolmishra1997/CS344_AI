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
from searchAgents import manhattanHeuristic

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
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# class Stack:
#     "A container with a last-in-first-out (LIFO) queuing policy."
#     def __init__(self):
#         self.list = []

#     def push(self,item):
#         "Push 'item' onto the stack"
#         self.list.append(item)

#     def pop(self):
#         "Pop the most recently pushed item from the stack"
#         return self.list.pop()

#     def isEmpty(self):
#         "Returns true if the stack is empty"
#         return len(self.list) == 0

class Node():
    def __init__(self, state, parent, action, fn, gn, hn):
        self.state = state
        self.parent = parent
        self.action = action
        self.fn = gn + hn
        self.gn = gn
        self.hn = hn

def get_heuristic(problem, state):
	return manhattanHeuristic(state, problem)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    visited = []
    start_state = problem.getStartState()
    current = start_state
    start_node = Node(start_state, 0, 0, 0, 0, get_heuristic(problem, start_state))
    frontier = util.PriorityQueue()
    frontier.push(start_node, start_node.fn)

    while not frontier.isEmpty():
        current = frontier.pop()
        current_state = current.state
        if current_state not in visited:
            visited.append(current_state)

        if problem.isGoalState(current_state):
            break

        for (successor, action, step_cost) in problem.getSuccessors(current_state):
            if successor not in visited:
                gn = current.gn + step_cost
                hn = get_heuristic(problem, current_state)
                successor_node = Node(successor, current, action, gn, gn, hn)
                frontier.push(successor_node, gn + hn)
    
    actions = []
    while current.action != 0:
    	actions.insert(0, current.action)
    	current = current.parent
    print action
    return action

    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
