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

from collections import deque
from dataclasses import dataclass

@dataclass (frozen=True, eq=True)
class Node:
    pos: tuple = (0, 0)
    parent: tuple = None
    cost: int = 0

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
                    neigh_state = Node(neighbor, state.pos, state.cost + 0)
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

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    return []

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
    
            
