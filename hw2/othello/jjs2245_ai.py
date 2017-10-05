#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
COMS W4701 Artificial Intelligence - Programming Homework 2

An AI player for Othello. This is the template file that you need to
complete and submit.

@author: Julian Silerio
         jjs2245
"""

import random, sys, time, math
from heapq import heappush, heappop

states = {}

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

def compute_utility(board, color):
    dark, light = get_score(board)
    if color is 1:
        return dark - light
    elif color is 2:
        return light - dark

############ MINIMAX ###############################

def minimax_min_node(board, color):
    moves = get_possible_moves(board, color)
    nodes = []

    if not moves:
        return compute_utility(board, color)
    else:
        min = math.inf
        for move in moves:
            new_board = play_move(board, color, move[0], move[1])
            heappush(nodes, (compute_utility, new_board))

        while nodes:
            new_board = heappop(nodes)[1]
            if new_board not in states:
                score = minimax_max_node(new_board, color)
                states[new_board] = score
            else:
                score = states[new_board]
            if min > score:
                min = score
        return min


def minimax_max_node(board, color):
    moves = get_possible_moves(board, color)
    nodes = []

    if not moves:
        return compute_utility(board, color)
    else:
        max = -math.inf
        for move in moves:
            new_board = play_move(board, color, move[0], move[1])
            heappush(nodes, (compute_utility, new_board))

        while nodes:
            new_board = heappop(nodes)[1]
            if new_board not in states:
                score = minimax_min_node(new_board, color)
                states[new_board] = score
            else:
                score = states[new_board]
            if max < score:
                max = score
        return max


def select_move_minimax(board, color):
    """
    Given a board and a player color, decide on a move.
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.
    """
    moves = get_possible_moves(board,color)
    best_score = -math.inf
    best_move = None
    nodes = []

    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        heappush(nodes, (compute_utility, new_board))

    while nodes:
        new_board = heappop(nodes)[1]
        if new_board not in states:
            score = minimax_min_node(new_board, color)
            states[new_board] = score
        else:
            score = states[new_board]
        if score > best_score:
            best_score = score
            best_move = move

    sys.stderr.write('best score {}\n'.format(best_score))
    return (best_move)

############ ALPHA-BETA PRUNING #####################

#alphabeta_min_node(board, color, alpha, beta, level, limit)
def alphabeta_min_node(board, color, alpha, beta, level, limit):
    moves = get_possible_moves(board, color)
    nodes = []

    if not moves or level == limit:
        return compute_utility(board, color)
    else:
        min = math.inf
        for move in moves:
            new_board = play_move(board, color, move[0], move[1])
            heappush(nodes, (compute_utility, new_board))

        while nodes:
            new_board = heappop(nodes)[1]
            if new_board not in states:
                score = alphabeta_max_node(new_board, color, alpha, beta, level + 1, limit)
                states[new_board] = score
            else:
                score = states[new_board]

            if score < min:
                min = score
                if min <= alpha:
                    return score
                if min < beta:
                    beta = min
        return min

#alphabeta_max_node(board, color, alpha, beta, level, limit)
def alphabeta_max_node(board, color, alpha, beta, level, limit):
    moves = get_possible_moves(board, color)
    nodes = []

    if not moves or level == limit:
        return compute_utility(board, color)
    else:
        max = -math.inf
        for move in moves:
            new_board = play_move(board, color, move[0], move[1])
            heappush(nodes, (compute_utility, new_board))

        while nodes:
            new_board = heappop(nodes)[1]
            if new_board not in states:
                score = alphabeta_min_node(new_board, color, alpha, beta, level + 1, limit)
                states[new_board] = score
            else:
                score = states[new_board]
            if score > max:
                max = score
                if max >= beta:
                    return max
                if max > alpha:
                    alpha = max
        return max

def select_move_alphabeta(board, color):
    moves = get_possible_moves(board, color)
    nodes = []

    best_score = -math.inf
    best_move = None

    limit = 15 #change this to change limit

    alpha = -math.inf
    beta = math.inf

    for move in moves:
        new_board = play_move(board, color, move[0], move[1])
        heappush(nodes, (compute_utility, new_board))

    while nodes:
        new_board = heappop(nodes)[1]
        if new_board not in states:
            score = alphabeta_min_node(new_board, color, alpha, beta, 1, limit)
            states[new_board] = score
        else:
            score = states[new_board]
        if score > best_score:
            best_score = score
            best_move = move
    sys.stderr.write('best score {}\n'.format(best_score))
    return (best_move)


####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Julian's AI") # First line is the name of this AI
    color = int(input()) # Then we read the color: 1 for dark (goes first),
                         # 2 for light.

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            #movei, movej = select_move_minimax(board, color)
            movei, movej = select_move_alphabeta(board, color)
            print("{} {}".format(movei, movej))


if __name__ == "__main__":
    run_ai()
