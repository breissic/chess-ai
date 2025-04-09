import chess
import time
from board import ChessBoard
import random
import threading
import queue
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

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

        # Search statistics
        self.nodes_searched = 0
        self.search_start_time = 0
        self.max_search_time = 2.0  # Reduced from 5.0 to 2.0 seconds
        self.expected_game_time = 1800  # 30 minutes per game
        self.time_remaining = self.expected_game_time

    def evaluate_position(self, board):
        """
        Enhanced evaluation function with more sophisticated positional features.
        Positive values favor white, negative values favor black.
        """
        if board.is_game_over():
            if board.get_board_state().is_checkmate():
                return -10000 if board.get_board_state().turn else 10000
            return 0 # Draw
        
        score = 0
        board_state = board.get_board_state()

        # Material evaluation with piece-square tables
        for square in chess.SQUARES:
            piece = board_state.piece_at(square)
            if piece is None:
                continue
            
            # Material value
            value = self.piece_values[piece.piece_type] * 10000 * (1 if piece.color else -1)
            score += value

            # Piece-square table value
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

        # Pawn structure evaluation
        score += self.evaluate_pawn_structure(board_state) // 16

        # Mobility evaluation
        score += self.evaluate_mobility(board_state)

        # King safety evaluation
        score += self.evaluate_king_safety(board_state)

        # Piece activity evaluation
        score += self.evaluate_piece_activity(board_state)

        # Center control evaluation
        score += self.evaluate_center_control(board_state)

        # Development evaluation
        score += self.evaluate_development(board_state)

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
        """Enhanced move ordering with history heuristic and SEE."""
        move_scores = []
        board_state = board.get_board_state()
        history_table = self.get_history_table(board_state)
        
        for move in moves:
            score = 0

            # 1. MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if board_state.is_capture(move):
                captured_piece = board_state.piece_type_at(move.to_square)
                attacker_piece = board_state.piece_type_at(move.from_square)
                if captured_piece and attacker_piece:
                    score = 1000000 + (self.piece_values[captured_piece] * 1000 - self.piece_values[attacker_piece])

            # 2. Killer moves
            elif move in self.killer_moves[depth]:
                score = 15000

            # 3. History heuristic
            else:
                score = history_table.get(move, 0)

            # 4. Checks (without modifying board state)
            if board_state.gives_check(move):
                score += 20000

            # 5. SEE (Static Exchange Evaluation)
            if board_state.is_capture(move):
                see_score = self.static_exchange_evaluation(board_state, move)
                score += see_score * 100

            # 6. Center control
            to_square_rank = chess.square_rank(move.to_square)
            to_square_file = chess.square_file(move.to_square)
            if 2 <= to_square_rank <= 5 and 2 <= to_square_file <= 5:
                score += 25

            # 7. Piece activity
            piece = board_state.piece_type_at(move.from_square)
            if piece and piece != chess.PAWN:
                score += 10

            move_scores.append((score, move))

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for score, move in move_scores]

    def static_exchange_evaluation(self, board_state, move):
        """Static Exchange Evaluation to estimate the value of a capture sequence."""
        if not board_state.is_capture(move):
            return 0

        captured_piece = board_state.piece_type_at(move.to_square)
        if not captured_piece:
            return 0

        # Create a copy of the board state for SEE
        see_board = board_state.copy()
        gain = [self.piece_values[captured_piece]]
        see_board.push(move)
        
        while True:
            # Find the least valuable attacker
            attackers = []
            for square in chess.SQUARES:
                if see_board.is_attacked_by(not see_board.turn, move.to_square):
                    piece = see_board.piece_type_at(square)
                    if piece:
                        attackers.append((self.piece_values[piece], square))
            
            if not attackers:
                break
                
            # Get the least valuable attacker
            attackers.sort()
            _, attacker_square = attackers[0]
            captured_piece = see_board.piece_type_at(move.to_square)
            
            if not captured_piece:
                break
                
            gain.append(self.piece_values[captured_piece])
            see_board.push(chess.Move(attacker_square, move.to_square))
            
            # If the king is in check, we have to stop
            if see_board.is_check():
                break

        # Calculate the exchange value
        value = 0
        for i, g in enumerate(gain):
            value = g - value

        return value

    def get_history_table(self, board_state):
        """Get the history table for move ordering."""
        history_table = {}
        for move in board_state.legal_moves:
            # Initialize with piece-square bonus
            piece = board_state.piece_type_at(move.from_square)
            if piece:
                history_table[move] = self.piece_values[piece] // 10
                
                # Add bonus for moves to center
                to_square_rank = chess.square_rank(move.to_square)
                to_square_file = chess.square_file(move.to_square)
                if 2 <= to_square_rank <= 5 and 2 <= to_square_file <= 5:
                    history_table[move] += 5
                    
                # Add bonus for moves that attack opponent's pieces
                if board_state.is_attacked_by(not board_state.turn, move.to_square):
                    history_table[move] += 10
                    
        return history_table

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

    def quiescence_search(self, board, alpha, beta, color):
        """Simplified quiescence search."""
        stand_pat = self.evaluate_position(board) * color
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
            
        board_state = board.get_board_state()
        captures = [move for move in board_state.legal_moves if board_state.is_capture(move)]
        captures = self.order_moves(board, captures, 0)
        
        for move in captures:
            board_state.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, -color)
            board_state.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
                
        return alpha

    def null_move_pruning(self, board, depth, beta, color):
        """Null move pruning to speed up search."""
        if depth < 3 or board.get_board_state().is_check():
            return None
            
        board.get_board_state().push(chess.Move.null())
        score = -self.minimax(board, depth - 3, -beta, -beta + 1, -color)
        board.get_board_state().pop()
        
        return score

    def principal_variation_search(self, board, depth, alpha, beta, color):
        """Principal variation search for more efficient alpha-beta pruning."""
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, color)
            
        board_state = board.get_board_state()
        moves = list(board_state.legal_moves)
        if not moves:
            if board_state.is_check():
                return -10000 * color
            return 0
            
        moves = self.order_moves(board, moves, depth)
        
        # First move is searched with full window
        board.make_move(moves[0])
        best_score = -self.principal_variation_search(board, depth - 1, -beta, -alpha, -color)
        board.undo_move()
        
        if best_score >= beta:
            return beta
            
        # Remaining moves are searched with zero window
        for move in moves[1:]:
            board.make_move(move)
            score = -self.principal_variation_search(board, depth - 1, -alpha - 1, -alpha, -color)
            board.undo_move()
            
            if score > best_score:
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
                    best_score = score
                    
        return best_score

    def minimax(self, board, depth, alpha, beta, color=1):
        """Enhanced minimax with improved transposition table and null-move pruning."""
        if depth == 0:
            return self.quiescence_search(board, alpha, beta, color)
            
        # Check transposition table
        board_state = board.get_board_state()
        board_hash = board_state.fen()
        tt_entry = self.transposition_table.get((board_hash, depth))
        if tt_entry:
            score, move, flag = tt_entry
            if flag == self.EXACT:
                return score
            elif flag == self.ALPHA and score <= alpha:
                return alpha
            elif flag == self.BETA and score >= beta:
                return beta
            
        # Null move pruning
        null_score = self.null_move_pruning(board, depth, beta, color)
        if null_score is not None and null_score >= beta:
            return beta
            
        moves = list(board_state.legal_moves)
        if not moves:
            if board_state.is_check():
                return -10000 * color
            return 0
            
        moves = self.order_moves(board, moves, depth)
        best_score = -float('inf')
        best_move = None
        
        for move in moves:
            self.nodes_searched += 1
            board_state.push(move)
            score = -self.minimax(board, depth - 1, -beta, -alpha, -color)
            board_state.pop()
            
            if score >= beta:
                self.add_killer_move(move, depth)
                # Store in transposition table
                self.transposition_table[(board_hash, depth)] = (score, move, self.BETA)
                return beta
                
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    
        # Store in transposition table
        if best_score <= alpha:
            flag = self.ALPHA
        elif best_score >= beta:
            flag = self.BETA
        else:
            flag = self.EXACT
        self.transposition_table[(board_hash, depth)] = (best_score, best_move, flag)
        
        return best_score

    def get_move(self, board: ChessBoard):
        """Get the best move using iterative deepening with time management."""
        self.search_start_time = time.time()
        self.nodes_searched = 0
        best_move = None
        depth = 1
        
        # Calculate time allocation
        self.calculate_time_allocation(board)

        # Start parallel search with reduced workers
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced from 4 to 2 workers
            futures = []
            for _ in range(2):  # Reduced from 4 to 2
                futures.append(executor.submit(self.parallel_search, board, depth))
            
            while True:
                if time.time() - self.search_start_time > self.max_search_time:
                    break
                    
                # Check if any thread has found a move
                for future in futures:
                    if future.done():
                        move, score = future.result()
                        if move:
                            best_move = move
                            break
                
                depth += 1
                
        return best_move

    def parallel_search(self, board, depth):
        """Parallel search implementation with optimized board state handling."""
        board_state = board.get_board_state()
        moves = list(board_state.legal_moves)
        if not moves:
            return None, -float('inf')
            
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        for move in moves:
            self.nodes_searched += 1
            board_state.push(move)
            score = -self.minimax(board, depth - 1, -beta, -alpha, -1)
            board_state.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                    
        return best_move, best_score

    def calculate_time_allocation(self, board):
        """Calculate time allocation based on game phase and remaining time."""
        board_state = board.get_board_state()
        piece_count = len(board_state.piece_map())
        
        # Adjust time based on game phase
        if piece_count > 32:  # Opening
            time_factor = 0.1
        elif piece_count > 20:  # Early middlegame
            time_factor = 0.15
        elif piece_count > 12:  # Late middlegame
            time_factor = 0.2
        else:  # Endgame
            time_factor = 0.25
            
        # Adjust time based on complexity
        legal_moves = len(list(board_state.legal_moves))
        if legal_moves > 30:  # Complex position
            time_factor *= 1.2
        elif legal_moves < 10:  # Simple position
            time_factor *= 0.8
            
        # Calculate maximum search time
        self.max_search_time = min(
            self.time_remaining * time_factor,
            2.0  # Reduced from 5.0 to 2.0 seconds
        )
        
        # Update remaining time
        self.time_remaining -= self.max_search_time

    def evaluate_king_safety(self, board_state):
        """Evaluate king safety based on pawn shield and piece attacks."""
        score = 0
        for color in [True, False]:
            multiplier = 1 if color else -1
            king_square = board_state.king(color)
            if king_square is None:
                continue

            # Pawn shield evaluation
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            pawn_shield = 0

            for file_offset in [-1, 0, 1]:
                file = king_file + file_offset
                if 0 <= file <= 7:
                    for rank_offset in [1, 2]:
                        rank = king_rank + rank_offset if color else king_rank - rank_offset
                        if 0 <= rank <= 7:
                            square = chess.square(file, rank)
                            if board_state.piece_at(square) == chess.Piece(chess.PAWN, color):
                                pawn_shield += 10

            score += pawn_shield * multiplier

            # King attack evaluation
            attackers = 0
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                for square in board_state.pieces(piece_type, not color):
                    if board_state.is_attacked_by(color, square):
                        attackers += 1

            score -= attackers * 5 * multiplier

        return score

    def evaluate_piece_activity(self, board_state):
        """Evaluate piece activity based on control of important squares."""
        score = 0
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            # Knight outposts
            for square in board_state.pieces(chess.KNIGHT, color):
                if not board_state.is_attacked_by(not color, square):
                    score += 15 * multiplier

            # Bishop pair bonus
            if len(board_state.pieces(chess.BISHOP, color)) >= 2:
                score += 30 * multiplier

            # Rook on open/semi-open files
            for square in board_state.pieces(chess.ROOK, color):
                file = chess.square_file(square)
                is_open = True
                for rank in range(8):
                    if board_state.piece_at(chess.square(file, rank)) == chess.Piece(chess.PAWN, color):
                        is_open = False
                        break
                if is_open:
                    score += 15 * multiplier

            # Control of center squares
            for square in center_squares:
                if board_state.is_attacked_by(color, square):
                    score += 5 * multiplier

        return score

    def evaluate_center_control(self, board_state):
        """Evaluate control of the center squares."""
        score = 0
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            for square in center_squares:
                if board_state.is_attacked_by(color, square):
                    score += 10 * multiplier
                if board_state.piece_at(square) == chess.Piece(chess.PAWN, color):
                    score += 20 * multiplier

        return score

    def evaluate_development(self, board_state):
        """Evaluate piece development in the opening."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            # Count developed pieces
            developed_pieces = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board_state.pieces(piece_type, color):
                    if color:
                        if chess.square_rank(square) > 1:  # Not on starting rank
                            developed_pieces += 1
                    else:
                        if chess.square_rank(square) < 6:  # Not on starting rank
                            developed_pieces += 1
            
            score += developed_pieces * 10 * multiplier
            
            # Penalty for moving the same piece multiple times in the opening
            if len(board_state.move_stack) < 20:  # Only in opening
                piece_moves = {}
                for move in board_state.move_stack:
                    piece = board_state.piece_at(move.to_square)
                    if piece and piece.color == color:
                        piece_moves[move.to_square] = piece_moves.get(move.to_square, 0) + 1
                
                for moves in piece_moves.values():
                    if moves > 1:
                        score -= 5 * multiplier

        return score