"""
COMS W4701 Artificial Intelligence - Programming Homework 1

In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now).
See Courseworks for detailed instructions.

@author: Julian (jjs2245)
"""

import time

def state_to_string(state):
    row_strings = [" ".join([str(cell) for cell in row]) for row in state]
    return "\n".join(row_strings)


def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped.
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]

    new_state = []
    for row in range(len(state)):
        new_row = []
        for column in range(len(state[row])):
            if row == i1 and column == j1:
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else:
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)


def get_successors(state):
    """
    This function returns a list of possible successor states resulting
    from applicable actions.
    The result should be a list containing (Action, state) tuples.
    For example [("Up", ((1, 4, 2),(0, 5, 8),(3, 6, 7))),
                 ("Left",((4, 0, 2),(1, 5, 8),(3, 6, 7)))]
    """

    child_states = []

    for row in range(len(state)):
        for col in range(len(state[row])):
            if state[row][col] == 0:
                #left
                if col + 1 < len(state[row]):
                    left_state = []
                    left_state.append("Left")
                    left_state.append(swap_cells(state, row, col, row, col + 1))
                    child_states.append(left_state)
                #right
                if col - 1 > -1:
                    right_state = []
                    right_state.append("Right")
                    right_state.append(swap_cells(state, row, col, row, col - 1))
                    child_states.append(right_state)
                #up
                if row + 1 < len(state):
                    up_state = []
                    up_state.append("Up")
                    up_state.append(swap_cells(state, row, col, row + 1, col))
                    child_states.append(up_state)
                #down
                if row - 1 > -1:
                    down_state = []
                    down_state.append("Down")
                    down_state.append(swap_cells(state, row - 1, col, row, col))
                    child_states.append(down_state)

    return child_states


def goal_test(state):
    """
    Returns True if the state is a goal state, False otherwise.
    """
    if(state == ((0,1,2),(3,4,5),(6,7,8))):
        return True
    return False


def bfs(state):
    """
    Breadth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0

    frontier = []

    frontier = [state]
    explored = set()
    seen = set()

    while frontier:
        n = frontier.pop(0)
        explored.add(n)
        states_expanded += 1

        if goal_test(n):
            max_frontier = len(frontier) + 1 # add 1 for current state
            path = get_path(n, parents, actions)
            return path, states_expanded, max_frontier

        for action, successor in get_successors(n):
            if successor not in explored and successor not in seen:
                parents[successor] = n
                actions[successor] = action
                frontier.append(successor)
                seen.add(successor)

    #  return solution, states_expanded, max_frontier
    return None, states_expanded, max_frontier # No solution found

def get_path(state, parents, actions):
    path = []
    while state in parents:
        path = [actions[state]] + path
        state = parents[state]
    return path

def dfs(state):
    """
    Depth first search.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """

    parents = {}
    actions = {}

    states_expanded = 0
    max_frontier = 0

    frontier = []

    frontier = [state]
    explored = set()
    seen = set()

    while frontier:
        n = frontier.pop()
        explored.add(n)
        states_expanded += 1

        if goal_test(n):
            max_frontier = len(frontier) + 1 # add 1 for current state
            path = get_path(n, parents, actions)
            return path, states_expanded, max_frontier

        for action, successor in get_successors(n):
            if successor not in explored and successor not in seen:
                parents[successor] = n
                actions[successor] = action
                frontier.append(successor)
                seen.add(successor)


    return None, states_expanded, max_frontier # No solution found


def misplaced_heuristic(state):
    """
    Returns the number of misplaced tiles.
    """
    total = 0
    counter = 0

    for row in state:
        for col in row:
            if col != counter and col > 0:
                total += 1
            counter += 1

    return total


def manhattan_heuristic(state):
    """
    For each misplaced tile, compute the manhattan distance between the current
    position and the goal position. Then sum all distances.
    """

    total = 0
    counter = 0

    goal_state = []

    for x in range(len(state)):
        for y in range(len(state[x])):
            goal_state.append((x, y))

    for row in range(len(state)):
        for col in range(len(state[row])):
            if state[row][col] != counter and state[row][col] > 0:
                total += abs(goal_state[state[row][col]][0] - row) + abs(goal_state[state[row][col]][1] - col)
            counter += 1
    return total


def best_first(state, heuristic = misplaced_heuristic):
    """
    Breadth first search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """

    # You might want to use these functions to maintain a priority queue
    from heapq import heappush
    from heapq import heappop

    parents = {}
    actions = {}

    frontier = []
    heappush(frontier, (0,state))

    states_expanded = 0
    max_frontier = 0

    explored = set()
    seen = set()

    while frontier:
        cost, n = heappop(frontier)
        explored.add(n)
        states_expanded += 1

        if goal_test(n):
            max_frontier = len(frontier) + 1 # add 1 for current state
            path = get_path(n, parents, actions)
            return path, states_expanded, max_frontier

        for action, successor in get_successors(n):
            if successor not in explored and successor not in seen:
                parents[successor] = n
                actions[successor] = action
                heappush(frontier, (heuristic(successor), successor))
                seen.add(successor)

    return None, 0, 0


def astar(state, heuristic = misplaced_heuristic):
    """
    A-star search using the heuristic function passed as a parameter.
    Returns three values: A list of actions, the number of states expanded, and
    the maximum size of the frontier.
    """
    # You might want to use these functions to maintain a priority queue

    from heapq import heappush
    from heapq import heappop

    parents = {}
    actions = {}
    costs = {}
    costs[state] = 0

    frontier = []
    heappush(frontier, (costs[state], state))

    states_expanded = 0
    max_frontier = 0

    explored = set()
    seen = set()

    while frontier:
        cost, n = heappop(frontier)
        explored.add(n)
        states_expanded += 1

        if goal_test(n):
            max_frontier = len(frontier) + 1 # add 1 for current state
            path = get_path(n, parents, actions)
            return path, states_expanded, max_frontier

        for action, successor in get_successors(n):
            if successor not in seen and successor not in explored:
                parents[successor] = n
                actions[successor] = action
                costs[successor] = cost + heuristic(successor)
                heappush(frontier, (costs[successor], successor))
                seen.add(successor)

    return None, states_expanded, max_frontier # No solution found


def print_result(solution, states_expanded, max_frontier):
    """
    Helper function to format test output.
    """
    if solution is None:
        print("No solution found.")
    else:
        print("Solution has {} actions.".format(len(solution)))
    print("Total states expanded: {}.".format(states_expanded))
    print("Max frontier size: {}.".format(max_frontier))


if __name__ == "__main__":

    #Easy test case
    #test_state = ((1, 4, 2),
    #              (0, 5, 8),
    #              (3, 6, 7))

    #More difficult test case
    test_state = ((7, 2, 4),
                  (5, 0, 6),
                  (8, 3, 1))

    #print(state_to_string(test_state))
    #print()


    print("====BFS====")
    start = time.time()
    solution, states_expanded, max_frontier = bfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    if solution is not None:
        print(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====DFS====")
    start = time.time()
    solution, states_expanded, max_frontier = dfs(test_state)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

    print()
    print("====Greedy Best-First (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = best_first(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))


    print()
    print("====A* (Misplaced Tiles Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))


    print()
    print("====A* (Total Manhattan Distance Heuristic)====")
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))

