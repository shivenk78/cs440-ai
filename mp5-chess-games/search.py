import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    
    moves = [ move for move in generateMoves(side, board, flags)]
    moveList = []
    moveTree = {}
    if len(moves) == 0 or depth == 0:
      return evaluate(board), moveList, moveTree
    
    best_val = None
    best_movelist = []
    if side: # Min
      best_val = math.inf
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        poss_val, poss_movelist, poss_movetree = minimax(newside, newboard, newflags, depth-1)
        
        moveTree[encode(*move)] = poss_movetree

        if poss_val < best_val:
          best_val = poss_val
          best_movelist = [move] + poss_movelist

    else: # Max
      best_val = -math.inf
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        poss_val, poss_movelist, poss_movetree = minimax(newside, newboard, newflags, depth-1)

        moveTree[encode(*move)] = poss_movetree
        
        if poss_val > best_val:
          best_val = poss_val
          best_movelist = [move] + poss_movelist

    moveList = best_movelist
    return best_val, moveList, moveTree

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    
    moves = [ move for move in generateMoves(side, board, flags)]
    moveTree = {}
    if len(moves) == 0 or depth == 0:
      return evaluate(board), [], moveTree
    
    best_val = None
    best_movelist = []
    if side: # Min
      best_val = math.inf
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        poss_val, poss_movelist, poss_movetree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)

        moveTree[encode(*move)] = poss_movetree

        if poss_val < best_val:
          best_val = poss_val
          best_movelist = [move] + poss_movelist

        beta = min(beta, best_val)
        if beta <= alpha:
          return best_val, best_movelist, moveTree

    else: # Max
      best_val = -math.inf
      for move in moves:
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        poss_val, poss_movelist, poss_movetree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)

        moveTree[encode(*move)] = poss_movetree

        if poss_val > best_val:
          best_val = poss_val
          best_movelist = [move] + poss_movelist

        alpha = max(alpha, best_val)
        if alpha >= beta:
          return best_val, best_movelist, moveTree

    return best_val, best_movelist, moveTree

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''

    moves = [move for move in generateMoves(side, board, flags)]
    moveList = []
    moveTree = {}
    if len(moves) == 0 or depth == 0 or breadth == 0:
      return evaluate(board), moveList, moveTree

    init_values = []
    init_move_lists = []
    for move in moves:
      init_side, init_board, init_flags = makeMove(side, board, move[0], move[1], flags, move[2])
      init_movetree = {}

      value_sum = 0
      rand_movelist = None
      for i in range(breadth):
        val, rand_movelist, rand_movetree = stoch_path(init_side, init_board, init_flags, depth-1, chooser)
        value_sum += val
        init_movetree.update(rand_movetree)
      
      init_values.append(value_sum/breadth)
      init_move_lists.append([move])
      moveTree[encode(*move)] = init_movetree
    
    if side: # Min
      min_val = math.inf
      min_path = None
      for i in range(len(init_values)):
        if init_values[i] < min_val:
          min_val = init_values[i]
          min_path = init_move_lists[i]
      
      return min_val, min_path, moveTree

    else: # Max
      max_val = -math.inf
      max_path = None
      for i in range(len(init_values)):
        if init_values[i] > max_val:
          max_val = init_values[i]
          max_path = init_move_lists[i]
      
      return max_val, max_path, moveTree

def stoch_path(side, board, flags, depth, chooser):
  moves = [ move for move in generateMoves(side, board, flags) ]
  moveList = []
  moveTree = {}
  if depth == 0 or len(moves) == 0:
    return evaluate(board), moveList, moveTree
  
  rand_move = chooser(moves)
  newside, newboard, newflags = makeMove(side, board, rand_move[0], rand_move[1], flags, rand_move[2])
  path_val, path_list, path_tree = stoch_path(newside, newboard, newflags, depth-1, chooser)

  moveTree[encode(*rand_move)] = path_tree
  moveList = [rand_move] + path_list

  return path_val, moveList, moveTree
