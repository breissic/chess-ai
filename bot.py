import chess
import time
from board import ChessBoard

class ChessBot:
    def __init__(self):
        self.killer_moves = [[None, None] for _ in range(64)] # 2 killer moves / depth
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 350, 
            chess.BISHOP: 350,  
            chess.ROOK: 525,    
            chess.QUEEN: 1000, 
            chess.KING: 20000
        }
        self.pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        25, 25, 25, 25, 25, 25, 25, 25,
        5,  5, 10, 15, 15, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
        ]
    
        self.knight_table = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30, 10, 20, 30, 30, 20, 10,-30,
        -30, 10, 20, 30, 30, 20, 10,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
        ]
    
        self.bishop_table = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10, 10, 10, 20, 20, 10, 10,-10,
        -10, 10, 15, 20, 20, 15, 10,-10,
        -10, 10, 15, 20, 20, 15, 10,-10,
        -10, 10, 10, 20, 20, 10, 10,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
        ]
    
        self.rook_table = [
        0,  0,  0,  5,  5,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0, 10, 10,  0,  0, -5,
        -5,  0,  0, 10, 10,  0,  0, -5,
        -5,  0,  0, 10, 10,  0,  0, -5,
        -5,  0,  0, 10, 10,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
        ]

        self.queen_table = [
        -20,-10,-10,  0,  0,-10,-10,-20,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        0,  5, 10, 15, 15, 10,  5,  0,
        0,  5, 10, 15, 15, 10,  5,  0,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -20,-10,-10,  0,  0,-10,-10,-20
        ]

        self.transposition_table = {}
        self.EXACT = 0
        self.ALPHA = 1
        self.BETA = 2

    def evaluate_position(self, board):
        """
        Evaluates the current board position.
        Positive values favor white, negative values favor black.
        :param board: The current board state.
        :return: The evaluation score.
        """
        if board.is_game_over():
            if board.get_board_state().is_checkmate():
                return -10000 if board.get_board_state().turn else 10000
            return 0 # Draw
        
        score = 0
        board_state = board.get_board_state()

        # First evaluate material with huge weight
        for square in chess.SQUARES:
            piece = board_state.piece_at(square)
            if piece is None:
                continue
            
            # Make material difference the dominant factor
            value = self.piece_values[piece.piece_type] * 10000 * (1 if piece.color else -1)
            score += value

        # Then add positional scores with much lower weight
        for square in chess.SQUARES:
            piece = board_state.piece_at(square)
            if piece is None:
                continue
            
            if piece.piece_type == chess.PAWN:
                pos_value = self.pawn_table[square if piece.color else 63 - square] // 4
            elif piece.piece_type == chess.KNIGHT:
                pos_value = self.knight_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.BISHOP:
                pos_value = self.bishop_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.ROOK:
                pos_value = self.rook_table[square if piece.color else 63 - square]
            elif piece.piece_type == chess.QUEEN:
                pos_value = self.queen_table[square if piece.color else 63 - square]
            else:
                pos_value = 0
            
            score += pos_value if piece.color else -pos_value

        # Minimal weight for structure and mobility
        score += self.evaluate_pawn_structure(board_state) // 16
        score += self.evaluate_mobility(board_state)

        return score

    def evaluate_pawn_structure(self, board_state):
        score = 0

        for color in [True, False]:
            multiplier = 1 if color else -1
            pawns = board_state.pieces(chess.PAWN, color)

            for file in range(8):
                pawns_in_file = sum(1 for square in pawns if chess.square_file(square) == file)
                if pawns_in_file > 1:
                    score -= 5 * multiplier * (pawns_in_file - 1)

            for pawn in pawns:
                file = chess.square_file(pawn)
                adjacent_files = []
                if file > 0:
                    adjacent_files.append(file - 1)
                if file < 7:
                    adjacent_files.append(file + 1)
                
                is_isolated = True
                for adj_file in adjacent_files:
                    if any(chess.square_file(p) == adj_file for p in pawns):
                        is_isolated = False
                        break
                
                if is_isolated:
                    score -= 8 * multiplier

            enemy_pawns = board_state.pieces(chess.PAWN, not color)
            for pawn in pawns:
                file = chess.square_file(pawn)
                rank = chess.square_rank(pawn)
                is_passed = True

                for enemy_pawn in enemy_pawns:
                    e_file = chess.square_file(enemy_pawn)
                    e_rank = chess.square_rank(enemy_pawn)
                    if abs(file - e_file) <= 1:
                        if color and e_rank > rank:
                            is_passed = False
                            break
                        if not color and e_rank < rank:
                            is_passed = False
                            break
                
                if is_passed:
                    score += 15 * multiplier
        
        return score

    def evaluate_mobility(self, board_state):
        """
        Evaluate piece mobility (number of legal moves)
        :param board_state: The current board state.
        :return: The evaluation score.
        """
        score = 0
        original_turn = board_state.turn

        for color in [True, False]:
            board_state.turn = color
            multiplier = 1 if color else -1
            mobility = len(list(board_state.legal_moves))
            score += mobility * multiplier  # Reduced from 2 to 1 point per move

        board_state.turn = original_turn
        
        return score

    def order_moves(self, board, moves, depth):
        """
        Orders moves based on:
        1. MVV - LVA (value of captured piece minus value of the capturing piece)
        2. Killer moves (strong moves that cause beta cutoffs. Probably helpful in other trees)
        3. Bonus to checks
        4. Transposition Table w/ Enhanced Transposition Cutoff to cache previously evaluated positions
        :param depth: The current search depth for killer moves.
        :param board: The current board state.
        :param moves: List of moves to order.
        :return: Ordered list of moves.
        """
        move_scores = []
        board_state = board.get_board_state()
        for move in moves:
            score = 0

            if board_state.is_capture(move):
                captured_piece = board_state.piece_type_at(move.to_square)
                attacker_piece = board_state.piece_type_at(move.from_square)
                if captured_piece and attacker_piece:
                    # Make captures overwhelmingly attractive
                    score = 1000000 + (self.piece_values[captured_piece] * 1000)

            elif move in self.killer_moves[depth]:
                score = 15000

            board.make_move(move)
            if board_state.is_check():
                score += 20000
            board.undo_move()

            to_square_rank = chess.square_rank(move.to_square)
            to_square_file = chess.square_file(move.to_square)
            if 2 <= to_square_rank <= 5 and 2 <= to_square_file <= 5:
                score += 25

            piece = board_state.piece_type_at(move.from_square)
            if piece and piece != chess.PAWN:
                score += 10

            move_scores.append((score, move))

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for score, move in move_scores]

    def add_killer_move(self, move, depth) -> None:
        """
        Adds a killer move to the table. Keeps track of the 2 most recent killer moves at each depth.
        Killer moves are moves that cause beta cutoffs, signaling their strength even if they don't
        directly capture a piece.
        :param move: The move that caused the beta cutoff.
        :param depth: Current search depth.
        :return: None
        """
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth][1] = self.killer_moves[depth][0]
            self.killer_moves[depth][0] = move

    def minimax(self, board, depth, alpha, beta, color=1):
        """
        Minimax algorithm w/ alpha-beta pruning
        :param color: 1 for current player, -1 for opponent.
        :param beta: The maximum score the minimizing player can achieve.
        :param alpha: The maximum score the maximizing player can achieve.
        :param board: The current board state.
        :param depth: How far to look ahead.
        :return: (best_score, best_move)
        """
        alpha_orig = alpha

        board_hash = board.get_board_state().fen()
        tt_entry = self.transposition_table.get((board_hash, depth))
        if tt_entry:
            score, move, flag = tt_entry
            if flag == self.EXACT:
                return score, move
            elif flag == self.ALPHA and score <= alpha:
                return score, move
            elif flag == self.BETA and score >= beta:
                return score, move

        if depth == 0 or board.is_game_over():
            return color * self.evaluate_position(board), None
        
        best_value = float('-inf')
        best_move = None
        for move in self.order_moves(board, board.get_legal_moves(), depth):
            board.make_move(move)
            score, _ = self.minimax(board, depth - 1, -beta, -alpha, -color)
            score = -score
            board.undo_move()
            if score > best_value:
                best_value = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.get_board_state().is_capture(move):
                    self.add_killer_move(move, depth)
                break

        if best_value <= alpha_orig:
            flag = self.ALPHA
        elif best_value >= beta:
            flag = self.BETA
        else:
            flag = self.EXACT
        self.transposition_table[(board_hash, depth)] = (best_value, best_move, flag)

        return best_value, best_move

    def get_move(self, board: ChessBoard):
        """
        Given the current board state, returns the chosen move.
        """
        """
        Dynamic depth calculation based on piece count:
        -Early game: 32 pieces, depth = 4
        -Mid game: 20 pieces, depth = 5
        -Late game: 10 pieces, depth = 6
        -Late late game: <8 pieces, depth = 7
        """
        start_time = time.time()
        board_state = board.get_board_state()
        piece_count = sum(len(board_state.pieces(piece, True)) + len(board_state.pieces(piece, False))
                          for piece in self.piece_values)
        depth = 7 if piece_count < 8 else 6 if piece_count < 12 else 5 if piece_count < 20 else 4
        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for move in self.order_moves(board, board.get_legal_moves(), 0):
            board.make_move(move)
            score, _ = self.minimax(board, depth - 1, -beta, -alpha, -1)
            score = -score
            board.undo_move()
            if score > best_value:
                best_value = score
                best_move = move
            alpha = max(alpha, best_value)
        end_time = time.time()
        print(f"Position evaluation: {best_value}")
        print(f"Search depth: {depth}")
        print(f"Pieces on board: {piece_count}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        return best_move