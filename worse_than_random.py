#!/usr/bin/env python3
import chess
import chess.svg
import sys
import networkx as nx
import matplotlib.pyplot as plt

board = chess.Board()
STEPS_AHEAD = 3

# Opening Move Lists
OPENINGS = {
    "qgd": "d4 d5 c4 e6",
    "ruy_lopez": "e4 e5 Nf3 Nc6 Bb5 a6",
    "four_knights": "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6",
    "vienna_game": "e4 e5 Nc3 Nf6 Bc4 Nxe4"
}

# Constants for piece colors
WHITE = chess.WHITE
BLACK = chess.BLACK

# Evaluation Constants
CHECKMATE = 300000
PHASE_CONSTANT = 256

# Material value of different pieces
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: CHECKMATE
}

# Piece-square tables
# representing the positional value of the given type of pieces on the board
pawn_pos_white = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

# reverse the position value for the opponent pieces
pawn_pos_black = list(reversed(pawn_pos_white))

knight_pos_white = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

knight_pos_black = list(reversed(knight_pos_white))

bishop_pos_white = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

bishop_pos_black = list(reversed(bishop_pos_white))

rook_pos_white = [
    [0, 0, 0, 5, 5, 0, 0, 0],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [0, 5, 5, 5, 5, 5, 5, 0]
]

rook_pos_black = list(reversed(rook_pos_white))

queen_pos_white = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

queen_pos_black = list(reversed(queen_pos_white))

king_pos_white = [
    [20, 30, 10, 0, 0, 10, 30, 20],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30]
]

king_pos_black = list(reversed(king_pos_white))

king_pos_white_end = [
    [-50, -30, -30, -30, -30, -30, -30, -50],
    [-30, -30, 0, 0, 0, 0, -30, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 30, 40, 40, 30, -10, -30],
    [-30, -10, 20, 30, 30, 20, -10, -30],
    [-30, -20, -10, 0, 0, -10, -20, -30],
    [-50, -40, -30, -20, -20, -30, -40, -50]
]

king_pos_black_end = list(reversed(king_pos_white_end))


def evaluate_board(board):
    '''
    Evaluate the current state of the board with the status of the game,
    material value and position value.
    Parameters:
        board: The current state of the board.
    Returns:
        The evaluation score of the board.
    '''
    # Give Checkmate a high value
    if board.is_checkmate():
        return -CHECKMATE if board.turn else CHECKMATE
    # Give any draws a value of 0
    elif board.is_stalemate() or board.is_insufficient_material() \
            or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    # initialize the score for white and black
    white_score = 0
    black_score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # get the value of the piece if there is a piece on the square
            piece_value = get_piece_value(piece)

            # add the material value and positional value of the piece
            if piece.color == chess.WHITE:
                white_score += piece_value
                white_score += \
                    get_positional_value(piece, square, chess.WHITE, board)
            else:
                black_score += piece_value
                black_score += \
                    get_positional_value(piece, square, chess.BLACK, board)

    return white_score - black_score


def get_piece_value(piece):
    '''
    Get the value of a piece based on its type.
    Parameters:
        piece: The piece to get the value of.
    Returns:
        The value of the piece.
    '''
    return PIECE_VALUES[piece.piece_type]


def get_positional_value(piece, square, color, board):
    '''
    Get the positional value of a piece based on its type and position.
    Parameters:
        piece: The piece to get the positional value of.
        square: The square the piece is on.
        color: The color of the piece.
        board: The current state of the board.
    Returns:
        The positional value of the piece.
    '''
    row = chess.square_rank(square)
    col = chess.square_file(square)
    is_endgame = is_endgame_phase(board)

    # return the positional value of a given piece
    # accoring to its postion on the board, type and color
    if piece.piece_type == chess.PAWN:
        return pawn_pos_white[row][col] if color == chess.WHITE \
            else pawn_pos_black[row][col]
    elif piece.piece_type == chess.KNIGHT:
        return knight_pos_white[row][col] if color == chess.WHITE \
            else knight_pos_black[row][col]
    elif piece.piece_type == chess.BISHOP:
        return bishop_pos_white[row][col] if color == chess.WHITE \
            else bishop_pos_black[row][col]
    elif piece.piece_type == chess.ROOK:
        return rook_pos_white[row][col] if color == chess.WHITE \
            else rook_pos_black[row][col]
    elif piece.piece_type == chess.QUEEN:
        return queen_pos_white[row][col] if color == chess.WHITE \
            else queen_pos_black[row][col]
    elif piece.piece_type == chess.KING:
        # if it is in endgame status
        # use the endgame positional value for the King
        if is_endgame:
            return king_pos_white_end[row][col] if color == chess.WHITE \
                else king_pos_black_end[row][col]
        else:
            return king_pos_white[row][col] if color == chess.WHITE \
                else king_pos_black[row][col]


def is_endgame_phase(board):
    '''
    Check if the game is in the endgame phase.
    Parameters:
        board: The current state of the board.
    Returns:
        True if the game is in the endgame phase, False otherwise.
    '''
    # initialize the total phase value
    total_phase = 16 * PHASE_CONSTANT
    # initialize the phase value for calculation
    phase = total_phase

    # when there are fewer pieces, minus points from our current phase value
    # different pieces have different values to subtract
    phase -= len(board.pieces(chess.KNIGHT, chess.WHITE)) * 1 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.KNIGHT, chess.BLACK)) * 1 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.BISHOP, chess.WHITE)) * 1 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.BISHOP, chess.BLACK)) * 1 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.ROOK, chess.WHITE)) * 2 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.ROOK, chess.BLACK)) * 2 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.QUEEN, chess.WHITE)) * 4 * PHASE_CONSTANT
    phase -= len(board.pieces(chess.QUEEN, chess.BLACK)) * 4 * PHASE_CONSTANT

    # if there are less than half of the initial phase value
    # we enter end game phase
    return phase < total_phase / 2


def find_moves(board, depth=STEPS_AHEAD):
    '''
    Find the top 3 moves with the highest evaluation score
    from all legal moves.
    Parameters:
        board: The current state of the board.
        depth: The depth (steps ahead) to search for the best moves.
    Returns:
        A list of the top 3 moves with the highest evaluation score.
    '''
    all_moves = list(board.legal_moves)
    # Based on whose turn it is,
    # select the top 3 moves or the bottom 3 moves
    # to maximize or minimize the evaluation score
    moves = sorted(all_moves, key=lambda move: evaluate_move(board, move),
                   reverse=board.turn)[:depth]
    return moves


def evaluate_move(board, move):
    '''
    Evaluate the board after making a move.
    Parameters:
        board: The current state of the board.
        move: The move to evaluate.
    Returns:
        The evaluation score of the board after making the move.
    '''
    board.push(move)
    score = evaluate_board(board)
    # Undo the move after evaluation
    board.pop()
    return score


def minimax_ab_pruning(G, board, node, current_depth, max_depth,
                       alpha, beta, maximizing_player, move_sequence=''):
    '''
    Recursively implement the minimax algorithm with alpha-beta pruning
    and build the game tree.
    Parameters:
        G: The game tree graph.
        board: The current state of the board.
        node: The current node in the tree.
        current_depth: The current depth of the tree.
        max_depth: The maximum depth of the tree.
        alpha: The alpha value for pruning.
        beta: The beta value for pruning.
        maximizing_player: A boolean value indicating
                           if the player is maximizing.
        move_sequence: The sequence of moves leading to the current state.
    Returns:
        The max or min evaluation score of the node.
    '''
    # base case: when we have reached the maximum depth
    # or the game is over(terminal state)
    if current_depth == max_depth or board.is_game_over():
        return evaluate_board(board)

    moves = find_moves(board, depth=STEPS_AHEAD)

    # a flag for pruning, used when we want to draw the edges pruned
    do_pruning = False

    if maximizing_player:
        value = float('-inf')
        for move in moves:
            # draw the pruned edges if the flag is set
            if do_pruning:
                child = f"{move_sequence} {move.uci()} (pruned)"
                mark_descendants_pruned(G, board, node, child, move,
                                        current_depth, max_depth)
                continue
            # push one move to check the status of that move
            # and build nodes, edges for the tree
            board.push(move)
            child = f"{move_sequence} {move.uci()}"
            G.add_node(child, utility=None)
            G.add_edge(node, child, move=move.uci(), pruned=False)
            eval = minimax_ab_pruning(G, board, child, current_depth + 1,
                                      max_depth, alpha, beta, False,
                                      f"{move_sequence} {move.uci()}")
            value = max(value, eval)
            G.nodes[child]['utility'] = eval
            alpha = max(alpha, value)
            board.pop()
            # alpha-beta pruning
            if beta <= alpha:
                # update pruning flag
                do_pruning = True
                continue
        G.nodes[node]['utility'] = value
        return value
    else:
        value = float('inf')
        for move in moves:
            if do_pruning:
                child = f"{move_sequence} {move.uci()} (pruned)"
                mark_descendants_pruned(G, board, node, child, move,
                                        current_depth, max_depth)
                continue
            board.push(move)
            child = f"{move_sequence} {move.uci()}"
            G.add_node(child, utility=None)
            G.add_edge(node, child, move=move.uci(), pruned=False)
            eval = minimax_ab_pruning(G, board, child, current_depth + 1,
                                      max_depth, alpha, beta, True,
                                      f"{move_sequence} {move.uci()}")
            value = min(value, eval)
            G.nodes[child]['utility'] = eval
            beta = min(beta, value)
            board.pop()
            if beta <= alpha:
                do_pruning = True
                continue
        G.nodes[node]['utility'] = value
        return value


def build_game_tree_with_minimax_ab_pruning(board, depth):
    '''
    Build a game tree graph up to a given depth with the Minimax algorithm
    and Alpha-Beta Pruning.
    Parameters:
        board: The current state of the board.
        depth: The depth of the game tree to build.
    Returns:
        A tuple:
            the root node of the tree.
            the created game tree graph.
    '''
    G = nx.DiGraph()
    # initialize the root node
    # root node do not have a move
    # root node have the utility value of the current board
    root = f"{''}"
    G.add_node(root)
    G.nodes[root]['utility'] = evaluate_board(board)
    minimax_ab_pruning(G, board, root, 0, depth,
                       float('-inf'), float('inf'), board.turn)
    return root, G


def mark_descendants_pruned(G, board, node, pruned_node,
                            move, current_depth, max_depth):
    '''
    Create node and edges for those have been pruned.
    Parameters:
        G: The game tree graph.
        board: The current state of the board.
        node: The current node in the tree.
        pruned_node: The pruned node in the tree.
        move: The move that leads to the pruned node.
        current_depth: The current depth of the tree.
        max_depth: The maximum depth of the tree.
    '''
    # add prunded node as X for pruned
    G.add_node(pruned_node, utility="X")
    G.add_edge(node, pruned_node, move=move.uci(), pruned=True)
    # recursively call mark_descendants_pruned to mark all
    # pruned nodes and edges until reach to the leaf nodes
    if current_depth + 1 < max_depth:
        moves = find_moves(board, depth=STEPS_AHEAD)
        for move in moves:
            child = f"{pruned_node} {move.uci()} (pruned)"
            G.add_node(child, utility="X")
            G.add_edge(pruned_node, child, move=move.uci(), pruned=True)
            mark_descendants_pruned(G, board, pruned_node, child, move,
                                    current_depth + 1, max_depth)


def select_best_move(board, depth=STEPS_AHEAD):
    '''
    Select the best move from the game tree with Minimax Alpha-Beta Pruning.
    Parameters:
        board: The current state of the board.
        depth: The depth of the tree.
    Returns:
        A tuple:
            the root node of the tree.
            the best move selected by the bot.
            the game tree graph.
    '''
    # Build game tree with Minimax and Alpha-Beta Pruning
    root, updated_tree = build_game_tree_with_minimax_ab_pruning(board, depth)
    # initialize the best move and the best value
    best_move = None
    best_value = -float('inf') if board.turn else float('inf')

    # check the move and the utility value of every node
    for child in updated_tree.successors(root):
        move = updated_tree[root][child]['move']
        child_value = updated_tree.nodes[child]['utility']
        # Select the move based on turn
        if board.turn:
            if child_value > best_value:
                best_value = child_value
                best_move = move
        else:
            if child_value < best_value:
                best_value = child_value
                best_move = move
    return root, best_move, updated_tree


def visualize_game_tree(root, best_move, G):
    '''
    Visualize the game tree with the best move as title.
    Parameters:
        root: The root node of the game tree.
        best_move: The best move selected by the bot.
        G: The game tree graph.
    '''
    # setting up the visualized graph
    # layers can be treated as depths
    # green ones are maximizing nodes, red ones are minimizing nodes
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = layer
    layer_colors = {1: 'red', 2: 'green', 3: 'red'}
    # root node blue
    node_colors = [layer_colors.get(G.nodes[node]["layer"], 'blue')
                   for node in G.nodes]
    pos = nx.multipartite_layout(G, subset_key="layer", scale=5)

    # setting up nodes' format in the graph
    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True,
                     labels={node: G.nodes[node]['utility']
                             for node in G.nodes},
                     font_size=8,
                     node_color=node_colors)

    # add colors to different type of edges
    # blue for best move edge
    # red for pruned edges
    edge_colors = []
    for u, v in G.edges:
        if G.edges[u, v].get('pruned', True):
            edge_colors.append('red')
        elif u == root and G.edges[u, v]['move'] == best_move:
            edge_colors.append('blue')
        else:
            edge_colors.append('black')
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors)

    # add move labels to all edges in tree
    edge_labels = {(u, v): G.edges[u, v]['move'] for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

    # title and layout
    ax.set_title(f"Best Move From QGD: {best_move}")
    fig.tight_layout()
    plt.show()


def uci(msg: str):
    '''Returns result of UCI protocol given passed message'''
    if msg == "uci":
        print("id name Worse Than Random")
        print("id author Xin Qi")
        print("uciok")
    elif msg == "isready":
        print("readyok")
    elif msg.startswith("position startpos moves"):
        board.clear()
        board.set_fen(chess.STARTING_FEN)
        moves = msg.split()[3:]
        for move in moves:
            board.push(chess.Move.from_uci(move))
    elif msg.startswith("position fen"):
        fen = msg.removeprefix("position fen ")
        board.set_fen(fen)
    elif msg.startswith("go"):
        # modify this line to select the best move based on implementation
        root, best_move, game_tree = select_best_move(board, 3)
        print(f"bestmove {best_move}")
    elif msg == "quit":
        sys.exit(0)
    return


# functons below are not used in the bot
def expand_tree(G, board, parent, current_depth, max_depth, move_sequence=''):
    '''
    This function is not used in the bot.
    When I started to work on the assignment,
    this is a easy starter generating a tree.
    Also, you can use it to build a tree and compare it
    with other algorithms.

    Expand the game tree graph up to a given depth.
    Parameters:
        G: The game tree graph.
        board: The current state of the board.
        parent: The parent node of the tree or subtrees.
        current_depth: The current depth of the tree.
        max_depth: The maximum depth of the tree.
        move_sequence: The sequence of moves leading to the current state.
    '''

    # base case: when we have reached the maximum depth
    if current_depth == max_depth:
        return

    # find all the eligible moves for the current state
    moves = find_moves(board, STEPS_AHEAD)

    for move in moves:
        board.push(move)
        # after one move, we create a new child node
        # and add edges between the parent and the child
        child = f"{move_sequence} {move.uci()}"
        G.add_node(child, utility=evaluate_board(board))
        G.add_edge(parent, child, move=move.uci(), pruned=False)
        # recursively call expand_tree to build the game tree
        expand_tree(G, board, child, current_depth + 1, max_depth,
                    f"{move_sequence} {move.uci()}")
        # undo the move for a new iteration
        board.pop()


def build_game_tree(board, depth):
    '''
    This function is not used in the bot.
    When I started to work on the assignment,
    this is a easy starter generating a tree.
    Also, you can use it to build a tree and compare it
    with other algorithms.

    Build a game tree graph up to a given depth.
    Parameters:
        board: The current state of the chess board.
        depth: The depth of the game tree to build.
    Returns:
        A tuple:
            the root node of the tree.
            a directed graph representing the game tree.
    '''
    # create a graph
    G = nx.DiGraph()
    # add the current board status as the root node
    root = f"{''}"
    G.add_node(root)
    G.nodes[root]['utility'] = evaluate_board(board)
    # recursively call expand_tree to build the game tree
    expand_tree(G, board, root, 0, depth)
    return root, G


def minimax(G, node, depth, maximizing_player):
    '''
    This function is not used in the bot.
    When I started to work on the assignment,
    this is a easy starter updating the generated tree.
    Also, you can use it to update a tree and compare it
    with Alpha-Beta Pruning.

    Implement the minimax algorithm to the game tree.
    Parameters:
        G: The game tree graph.
        node: The current node in the tree.
        depth: The depth of the tree.
        maximizing_player: A boolean value indicating
                           if the player is maximizing.
    Returns:
        The max or min evaluation score of the node.
    '''
    # base case: when depth is 0 or end state for the game
    if depth == 0 or board.is_game_over():
        return G.nodes[node]['utility']

    # Player's turn, maximizing player
    if maximizing_player:
        max_eval = float('-inf')
        for child in G.successors(node):
            eval = minimax(G, child, depth - 1, False)
            max_eval = max(max_eval, eval)
        G.nodes[node]['utility'] = max_eval
        return max_eval
    # Opponent's turn, minimizing player
    else:
        min_eval = float('inf')
        for child in G.successors(node):
            eval = minimax(G, child, depth - 1, True)
            min_eval = min(min_eval, eval)
        G.nodes[node]['utility'] = min_eval
        return min_eval


def alpha_beta_pruning(G, node, depth, alpha, beta, maximizing_player):
    '''
    This function is not used due to the difficulty of
    implementing alpha-beta pruning after a minimax update.

    Implement the alpha-beta pruning algorithm to the tree
    updated by minimax.
    Parameters:
        G: The game tree graph.
        node: The current node in the tree.
        depth: The depth of the tree.
        alpha: The alpha value for pruning.
        beta: The beta value for pruning.
        maximizing_player: A boolean value indicating
                           if the player is maximizing.
    '''
    # base case: when depth is 0 or end state for the game
    if depth == 0 or board.is_game_over():
        return G.nodes[node]['utility']

    if maximizing_player:
        value = float('-inf')
        for child in G.successors(node):
            eval = alpha_beta_pruning(G, child, depth - 1, alpha, beta, False)
            value = max(value, eval)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = float('inf')
        for child in G.successors(node):
            eval = alpha_beta_pruning(G, child, depth - 1, alpha, beta, True)
            value = min(value, eval)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def update_tree_with_minimax(board, depth):
    '''
    This function is not used due to the difficulty of
    implementing alpha-beta pruning after a minimax update.

    Update the original game tree with the Minimax algorithm.
    Parameters:
        board: The current state of the board.
        depth: The depth of the tree.
    Returns:
        A tuple:
            the root node of the tree.
            the updated game tree graph.
    '''
    root, game_tree = build_game_tree(board, depth)
    _ = minimax(game_tree, root, depth, board.turn)
    return root, game_tree


def update_tree_with_alpha_beta_pruning(board, depth):
    '''
    This function is not used due to the difficulty of
    implementing alpha-beta pruning after a minimax update.

    Update the game tree with the Alpha-Beta Pruning algorithm.
    The game tree is already updated by the Minimax algorithm.
    Parameters:
        board: The current state of the board.
        depth: The depth of the tree.
    Returns:
        A tuple:
            the root node of the tree.
            the updated game tree graph.
    '''
    root, game_tree = update_tree_with_minimax(board, depth)
    _ = alpha_beta_pruning(game_tree, root, depth,
                           float('-inf'), float('inf'), board.turn)
    return root, game_tree


def select_move_minimax(board, depth=STEPS_AHEAD):
    '''
    This function is used for visualization after minimax
    without alpha beta pruning.

    Select the best move from the updated game tree with minimax.
    Parameters:
        board: The current state of the board.
        depth: The depth of the tree.
    Returns:
        A tuple:
            the root node of the tree.
            the best move selected by the bot.
            the updated game tree graph.
    '''
    # Build and update tree with minimax
    root, updated_tree = update_tree_with_minimax(board, depth)
    # initialize the best move and the best value
    best_move = None
    best_value = -float('inf') if board.turn else float('inf')

    for child in updated_tree.successors(root):
        # check the move and the utility value of every node
        move = updated_tree[root][child]['move']
        child_value = updated_tree.nodes[child]['utility']
        # Select the move based on turn
        if board.turn:
            if child_value > best_value:
                best_value = child_value
                best_move = move
        else:
            if child_value < best_value:
                best_value = child_value
                best_move = move

    # To prevent the bot from making an invalid move
    if best_move is None:
        best_move = find_moves(board, depth=STEPS_AHEAD)[0].uci()

    return root, best_move, updated_tree


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'draw':
        # Start with QGD opening
        opening = OPENINGS["qgd"]
        board = chess.Board()
        for move in opening.split():
            board.push_san(move)
        # visiulaize only with minimax
        # root_1, best_move_1, tree = select_move_minimax(board, 3)
        # visualize_game_tree(root_1, best_move_1, tree)
        root, best_move, game_tree = select_best_move(board, 3)
        visualize_game_tree(root, best_move, game_tree)
    else:
        try:
            while True:
                uci(input())
        except Exception:
            print("Fatal Error")


if __name__ == "__main__":
    main()
