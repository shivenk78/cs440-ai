# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

from collections import deque, namedtuple
from dataclasses import dataclass, field
import heapq

# Data class for the state Nodes
# Sorted only by g+h (for heapq)
@dataclass (order=True)
class Node:
    pos: tuple = field(compare=False, default=(0,0))
    parent: tuple = field(compare=False, default=None)
    path_cost: int = field(compare=False, default=0)
    g_h: int = 0
    goals: tuple = field(compare=False, default=())

def bfs(maze):

    path = []
    state = Node( maze.start, maze.start, 0 )
    frontier = deque([state])
    explored = {state.pos : state}

    while (frontier):
        state = frontier.popleft()

        if (maze[state.pos] == maze.legend.waypoint):
            #print(f"BREAK!! {state.pos} state // waypoints {maze.waypoints}")
            break
        
        neighbors = maze.neighbors(state.pos[0], state.pos[1])
        for neighbor in neighbors:
            if (neighbor not in explored):
                if (maze.navigable(neighbor[0], neighbor[1])):
                    neigh_state = Node(neighbor, state.pos)
                    frontier.append(neigh_state)
                    explored[neigh_state.pos] =  neigh_state

    while (state.pos != maze.start):
        path.insert(0, state.pos)
        state = explored[state.parent]
    path.insert(0, maze.start)

    # print("\n\nPATH: ")
    # print(path)
    # print("\n\n")

    return path

def manhattan(agent: tuple, waypoint: tuple) -> int:
    return abs(agent[0]-waypoint[0]) + abs(agent[1]-waypoint[1])

def astar_single(maze):
    path = []
    state = Node( maze.start, maze.start, 0 )
    frontier = []
    heapq.heappush(frontier, state)
    explored = {state.pos : state}

    h = manhattan

    while (frontier):
        state = heapq.heappop(frontier)

        if (maze[state.pos] == maze.legend.waypoint):
            #print(f"BREAK!! {state.pos} state // waypoints {maze.waypoints}")
            break
        
        neighbors = maze.neighbors(state.pos[0], state.pos[1])
        for neighbor in neighbors:
            if (neighbor not in explored):
                if (maze.navigable(neighbor[0], neighbor[1])):
                    neigh_state = Node(neighbor, state.pos, state.path_cost + 1, state.path_cost + 1 + h(state.pos, maze.waypoints[0]))
                    heapq.heappush(frontier, neigh_state)
                    explored[neigh_state.pos] =  neigh_state

    while (state.pos != maze.start):
        path.insert(0, state.pos)
        state = explored[state.parent]
    path.insert(0, maze.start)

    # print("\n\nPATH: ")
    # print(path)
    # print("\n\n")

    return path

# Heuristic for astar_corner - finds manhattan to farthest remaining corner!
def manhattan_corner(state, maze):
    return 0
    # dists = []
    # for way in maze.waypoints:
    #     if (way not in state.goals):
    #         dists.append(manhattan(state.pos, way))
    # # if dists:
    #     return max(dists)
    # # return 0

# DictKey = namedtuple('DictKey', ['pos', 'goals'])

def astar_corner(maze):
    path = []
    state = Node( maze.start, maze.start, 0, goals=maze.waypoints)
    print('INIT: ', state)
    frontier = []
    heapq.heappush(frontier, state)
    #explored = {(state.pos, state.goals) : state}
    explored = {}

    h = manhattan_corner

    while (frontier):
        state = heapq.heappop(frontier)

        if (state.pos in state.goals):
            state.goals = tuple(x for x in state.goals if x != state.pos)
            if (len(state.goals) == 0):
                break
        
        neighbors = maze.neighbors(state.pos[0], state.pos[1])
        for neighbor in neighbors:
            if ((neighbor, state.goals) not in explored):
                neigh_state = Node(neighbor, (state.pos, tuple(state.goals)), state.path_cost + 1, state.path_cost + 1 + h(state, maze), tuple(state.goals))
                heapq.heappush(frontier, neigh_state)
                explored[(neigh_state.pos, neigh_state.goals)] = neigh_state
            # elif (state.path_cost + 1 < explored[(neighbor, state.goals)].path_cost):
            #     neigh_state = Node(neighbor, DictKey(state.pos, state.goals), state.path_cost + 1, state.path_cost + 1 + h(state, maze), state.goals)
            #     heapq.heappush(frontier, neigh_state)
            #     explored[DictKey(neigh_state.pos, neigh_state.goals)] = neigh_state


    print("asdfasdf")

    # print(explored)
    # start = maze.start
    # for goal in maze.waypoints:
    #     temp = explored[goal]
    #     print(temp)
    #     curr_path = []
    #     while temp.pos != start:
    #         curr_path.insert(0, temp.pos)
    #         temp = explored[temp.parent]
    #     path.extend(curr_path)
    #     start = goal

    # for key in explored:
        # if (key[0] == (1, 6) or key[0] == (2,6)):
            # if len(key[1]) == 3 :
                # print(f"Key: {key} --- Val: {explored[key]}")

    while (state.pos != maze.start):
        path.insert(0, state.pos)
        #print(f"STATE: {state}")
        state = explored[state.parent]

    # pos=(1, 6), parent=DictKey(pos=(2, 6), goals=((6, 1), (1, 1), (1, 6))), path_cost=19, g_h=19, goals=((6, 1), (1, 1), (1, 6)))


    path.insert(0, maze.start)

    print("\n\nPATH: ")
    print(path)
    print("\n\n")

    return path

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []
    
            
