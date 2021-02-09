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
    state = Node( maze.start, None, 0 )
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

    path.insert(0, state.pos)
    parent = state.parent
    while (parent != None):
        path.insert(0, explored[parent].pos)
        parent = explored[parent].parent

    # print("\n\nPATH: ")
    # print(path)
    # print("\n\n")

    return path

def manhattan(agent: tuple, waypoint: tuple) -> int:
    return abs(agent[0]-waypoint[0]) + abs(agent[1]-waypoint[1])

def astar_single(maze):
    return astar_dest(maze, maze.start, maze.waypoints[0])

def astar_dest(maze, start, dest):
    path = []
    state = Node( start, None, 0 )
    frontier = []
    heapq.heappush(frontier, state)
    explored = {state.pos : state}

    h = manhattan

    while (frontier):
        state = heapq.heappop(frontier)

        if (state.pos == dest):
            #print(f"BREAK!! {state.pos} state // waypoints {maze.waypoints}")
            break
        
        neighbors = maze.neighbors(state.pos[0], state.pos[1])
        for neighbor in neighbors:
            if (neighbor not in explored):
                if (maze.navigable(neighbor[0], neighbor[1])):
                    neigh_state = Node(neighbor, state.pos, state.path_cost + 1, state.path_cost + 1 + h(state.pos, maze.waypoints[0]))
                    heapq.heappush(frontier, neigh_state)
                    explored[neigh_state.pos] =  neigh_state

    path.insert(0, state.pos)
    parent = state.parent
    while (parent != None):
        path.insert(0, explored[parent].pos)
        parent = explored[parent].parent

    # print("\n\nPATH: ")
    # print(path)
    # print("\n\n")

    return path

# Heuristic for astar_corner - finds manhattan to farthest remaining corner!
def manhattan_corner(state, maze):
    dists = []
    for way in maze.waypoints:
        if (way not in state.goals):
            dists.append(manhattan(state.pos, way))
    if dists:
        return max(dists)
    return maze.size.x + maze.size.y

def manhattan_corner_min(state, maze):
    dists = []
    for way in state.goals:
        dists.append(manhattan(state.pos, way))
    if dists:
        return min(dists)
    return 0

def zero(state, maze):
    return 0

# DictKey = namedtuple('DictKey', ['pos', 'goals'])

def astar_corner(maze):
    return astar_multiple(maze)
    # path = []
    # state = Node( maze.start, None, 0, goals=maze.waypoints)
    # frontier = []
    # heapq.heappush(frontier, state)
    # #explored = {(state.pos, state.goals) : state}
    # explored = {}

    # h = manhattan_corner_min #manhattan_corner if maze.size.x > 10 else zero

    # while (frontier):
    #     state = heapq.heappop(frontier)

    #     if (state.pos in state.goals):
    #         state.goals = tuple(x for x in state.goals if x != state.pos)
    #         if (len(state.goals) == 0):
    #             break
        
    #     if ((state.pos, state.goals) not in explored):
    #         explored[(state.pos, state.goals)] = state

    #     neighbors = maze.neighbors(state.pos[0], state.pos[1])
    #     for neighbor in neighbors:
    #         if ((neighbor, state.goals) not in explored):
    #             neigh_state = Node(neighbor, (state.pos, tuple(state.goals)), state.path_cost + 1, state.path_cost + 1 + h(state, maze), tuple(state.goals))
    #             # print("STATE: ", state)
    #             # print("NEIGH: ", neigh_state)
    #             heapq.heappush(frontier, neigh_state)
    #         # elif (state.path_cost + 1 < explored[(neighbor, state.goals)].path_cost):
    #         #     neigh_state = Node(neighbor, (state.pos, state.goals), state.path_cost + 1, state.path_cost + 1 + h(state, maze), state.goals)
    #         #     heapq.heappush(frontier, neigh_state)
    
    # # print(explored)
    # # start = maze.start
    # # for goal in maze.waypoints:
    # #     temp = explored[goal]
    # #     print(temp)
    # #     curr_path = []
    # #     while temp.pos != start:
    # #         curr_path.insert(0, temp.pos)
    # #         temp = explored[temp.parent]
    # #     path.extend(curr_path)
    # #     start = goal

    # # for key in explored:
    #     # if (key[0] == (1, 6) or key[0] == (2,6)):
    #         # if len(key[1]) == 3 :
    #             # print(f"Key: {key} --- Val: {explored[key]}")

    # path.insert(0, state.pos)
    # parent = state.parent
    # while (parent != None):
    #     path.insert(0, explored[parent].pos)
    #     parent = explored[parent].parent

    # # pos=(1, 6), parent=DictKey(pos=(2, 6), goals=((6, 1), (1, 1), (1, 6))), path_cost=19, g_h=19, goals=((6, 1), (1, 1), (1, 6)))


    # path.insert(0, maze.start)

    # # print("\n\nPATH: ")
    # # print(path)
    # # print("\n\n")

    # return path

MST_CACHE = {}
ASTAR_CACHE = {}

def astar_multiple(maze):
    path = []
    state = Node( maze.start, None, 0, goals=maze.waypoints)
    frontier = []
    heapq.heappush(frontier, state)
    explored = {}

    MST_CACHE = {}
    ASTAR_CACHE = {}

    h = MST

    count = 0
    while (frontier):
        state = heapq.heappop(frontier)

        if (state.pos in state.goals):
            state.goals = tuple(x for x in state.goals if x != state.pos)
            if (len(state.goals) == 0):
                break
        
        if ((state.pos, state.goals) not in explored):
            explored[(state.pos, state.goals)] = state

        neighbors = maze.neighbors(state.pos[0], state.pos[1])
        for neighbor in neighbors:
            if ((neighbor, state.goals) not in explored):
                neigh_state = Node(neighbor, (state.pos, tuple(state.goals)), state.path_cost + 1, state.path_cost + 1 + h(state, maze), tuple(state.goals))
                heapq.heappush(frontier, neigh_state)

    path.insert(0, state.pos)
    parent = state.parent
    while (parent != None):
        path.insert(0, explored[parent].pos)
        parent = explored[parent].parent

    # print("\n\nPATH: ")
    # print(path)
    # print("\n\n")

    return path

def MSTDist(a, b, maze):
    if ((a, b) in ASTAR_CACHE):
        return ASTAR_CACHE[(a, b)]
    if ((b, a) in ASTAR_CACHE):
        return ASTAR_CACHE[(b, a)]
    edge = len(astar_dest(maze, a, b))
    ASTAR_CACHE[(a, b)] = edge
    return edge

import copy

# Data class for the edges for Prim's MST
# Sorted only by len (for heapq)
@dataclass (order=True)
class Edge:
    a: tuple = field(compare=False, default=(0,0))
    b: tuple = field(compare=False, default=(0,0))
    size: int = 0

# goals is a tuple of tuples
def MST(state, maze):
    goals = state.goals

    if (goals in MST_CACHE):
        return MST_CACHE[goals]

    length = 0
    all = []

    for i in range(0, len(goals)):
        for j in range(i, len(goals)):
            if i != j:
                all.append(Edge(goals[i], goals[j], MSTDist(goals[i], goals[j], copy.copy(maze))))

    all.sort()
    tree = set()
    tree.add(goals[0])
    outside = set(goals[1:])

    while (len(tree) < len(goals)):
        toPop = 0;
        for i in range(0, len(all)):
            edge = all[i]
            #print(edge)
            if (edge.a in tree and edge.b in outside):
                tree.add(edge.b)
                outside.remove(edge.b)
                length += edge.size
                toPop = i
                break
            elif (edge.b in tree and edge.a in outside):
                tree.add(edge.a)
                outside.remove(edge.a)
                length += edge.size
                toPop = i
                break
        all.pop(toPop)

    return length

def fast(maze):
    return []
    
            
