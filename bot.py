import chess
import time
from board import ChessBoard
import random
import threading
import queue
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
import chess.polyglot

class ChessBot:
    def __init__(self):
        # Constants for transposition table flags
        self.EXACT = 0
        self.ALPHA = 1
        self.BETA = 2
        
        # Initialize opening book with better error handling
        self.opening_book = None
        try:
            self.opening_book = chess.polyglot.open_reader("opening_book.bin")
            print("Opening book loaded successfully")
        except Exception as e:
            print(f"Failed to load opening book: {e}")
            print("Will use pure calculation")
        
        # Enhanced piece values with more balanced scoring
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320, 
            chess.BISHOP: 330,  
            chess.ROOK: 500,    
            chess.QUEEN: 900, 
            chess.KING: 20000
        }
        
        # Opening development priorities
        self.development_squares = {
            chess.KNIGHT: [chess.B1, chess.G1, chess.B8, chess.G8],
            chess.BISHOP: [chess.C1, chess.F1, chess.C8, chess.F8],
            chess.ROOK: [chess.A1, chess.H1, chess.A8, chess.H8],
            chess.QUEEN: [chess.D1, chess.D8]
        }
        
        # Center control squares
        self.center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        # Search parameters
        self.base_max_depth = 4
        self.max_depth = 4  # This will be dynamically adjusted
        self.nodes_searched = 0
        self.search_start_time = 0
        self.max_search_time = 2.0
        self.time_management_factor = 0.85  # Use 85% of max time to avoid timeouts
        self.transposition_table = {}
        self.killer_moves = [[None, None] for _ in range(64)]
        self.history_table = {}
        
        # Prevent king wandering
        self.king_move_penalty = 25
        self.king_move_history = {}
        
        # Piece safety parameters
        self.safety_penalties = {
            chess.PAWN: 50,
            chess.KNIGHT: 200,
            chess.BISHOP: 200,
            chess.ROOK: 300,
            chess.QUEEN: 500,
            chess.KING: 1000
        }
        
        # Position history for draw detection
        self.position_history = {}
        
        # Endgame piece values (different from middlegame)
        self.endgame_piece_values = {
            chess.PAWN: 150,    # Pawns are more valuable in endgame
            chess.KNIGHT: 280,  # Knights slightly less valuable
            chess.BISHOP: 320,  # Bishops maintain value
            chess.ROOK: 600,    # Rooks more valuable
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Material count thresholds for endgame detection
        self.endgame_threshold = 1800  # Total material excluding kings and pawns

    def evaluate_position(self, board):
        """Comprehensive position evaluation with improved strategic assessment."""
        board_state = board.get_board_state()
        
        # If checkmate, return extreme value
        if board_state.is_checkmate():
            return -10000 if board_state.turn else 10000
            
        # Draw detection
        if board_state.is_stalemate() or board_state.is_insufficient_material():
            return 0
        
        white_material = 0
        black_material = 0
        white_mobility = 0
        black_mobility = 0
        white_king_safety = 0
        black_king_safety = 0
        white_pawn_structure = 0
        black_pawn_structure = 0
        white_piece_activity = 0
        black_piece_activity = 0
        
        # Calculate total material to determine game phase
        total_material = sum(len(board_state.pieces(piece_type, color)) * self.piece_values[piece_type]
                            for color in [chess.WHITE, chess.BLACK]
                            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
        
        # Game phase (0 = opening, 1 = endgame)
        is_endgame = total_material < self.endgame_threshold
        
        # Piece values based on game phase
        piece_vals = self.endgame_piece_values if is_endgame else self.piece_values
        
        # Material evaluation
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            white_pieces = board_state.pieces(piece_type, chess.WHITE)
            black_pieces = board_state.pieces(piece_type, chess.BLACK)
            
            white_material += len(white_pieces) * piece_vals[piece_type]
            black_material += len(black_pieces) * piece_vals[piece_type]
            
            # Bishop pair bonus
            if piece_type == chess.BISHOP and len(white_pieces) >= 2:
                white_material += 50
            if piece_type == chess.BISHOP and len(black_pieces) >= 2:
                black_material += 50
                
            # Piece-specific positional evaluation
            for square in white_pieces:
                row, col = 7 - chess.square_rank(square), chess.square_file(square)
                
                # Piece activity - control of center, outposts, etc.
                if piece_type == chess.PAWN:
                    # Passed pawns bonus (increases in endgame)
                    if self.is_passed_pawn(board_state, square):
                        bonus = 20 + (is_endgame * 30)
                        rank_bonus = row * 5 * (1 + is_endgame)  # More value in endgame as pawn advances
                        white_pawn_structure += bonus + rank_bonus
                    
                    # Pawn chain bonus
                    if self.is_pawn_supported(board_state, square):
                        white_pawn_structure += 10
                    
                    # Isolated pawn penalty
                    if self.is_isolated_pawn(board_state, square):
                        white_pawn_structure -= 15
                    
                    # Doubled pawn penalty
                    if self.is_doubled_pawn(board_state, square):
                        white_pawn_structure -= 10
                    
                    # Center pawn bonus
                    if col in [3, 4] and row in [3, 4]:
                        white_piece_activity += 15
                
                elif piece_type == chess.KNIGHT:
                    # Knights on outposts
                    if row in [3, 4, 5] and self.is_supported_by_pawn(board_state, square, chess.WHITE):
                        white_piece_activity += 20
                    
                    # Knights penalized on edges
                    if col == 0 or col == 7 or row == 0 or row == 7:
                        white_piece_activity -= 15
                
                elif piece_type == chess.BISHOP:
                    # Bishops bonus for open diagonals
                    mobility = len(list(board_state.attacks(square)))
                    white_mobility += mobility * 2
                    
                    # Fianchetto bonus
                    if (square == chess.B2 or square == chess.G2) and board_state.piece_at(square + 8) == chess.Piece(chess.PAWN, chess.WHITE):
                        white_piece_activity += 15
                
                elif piece_type == chess.ROOK:
                    # Rook on open file
                    if self.is_open_file(board_state, col):
                        white_piece_activity += 15
                    
                    # Rook on semi-open file
                    elif self.is_semi_open_file(board_state, col, chess.WHITE):
                        white_piece_activity += 10
                    
                    # Rook on 7th rank (or 2nd in endgame)
                    if row == 6 or (is_endgame and row == 1):
                        white_piece_activity += 20
                
                elif piece_type == chess.QUEEN:
                    # Penalize early queen development
                    if not is_endgame and board_state.fullmove_number < 10 and square != chess.D1:
                        white_piece_activity -= 15
                
                elif piece_type == chess.KING:
                    # King safety in opening/middlegame
                    if not is_endgame:
                        # Prefer king castled and with pawn shield
                        if col >= 5 and row == 0:  # Kingside castle position
                            white_king_safety += 60
                            # Check pawn shield
                            for shield_square in [square + 8, square + 9, square + 7]:
                                if (shield_square < 64 and 
                                    board_state.piece_type_at(shield_square) == chess.PAWN and
                                    board_state.color_at(shield_square) == chess.WHITE):
                                    white_king_safety += 10
                        elif col <= 2 and row == 0:  # Queenside castle position
                            white_king_safety += 50
                            # Check pawn shield
                            for shield_square in [square + 8, square + 9, square + 7]:
                                if (shield_square < 64 and 
                                    board_state.piece_type_at(shield_square) == chess.PAWN and
                                    board_state.color_at(shield_square) == chess.WHITE):
                                    white_king_safety += 10
                    else:
                        # King activity in endgame
                        # King centralization in endgame
                        center_distance = abs(3.5 - col) + abs(3.5 - row)
                        white_king_safety += (8 - center_distance) * 10
            
            # Same evaluations for black pieces
            for square in black_pieces:
                row, col = chess.square_rank(square), chess.square_file(square)
                
                if piece_type == chess.PAWN:
                    if self.is_passed_pawn(board_state, square):
                        bonus = 20 + (is_endgame * 30)
                        rank_bonus = (7 - row) * 5 * (1 + is_endgame)
                        black_pawn_structure += bonus + rank_bonus
                    
                    if self.is_pawn_supported(board_state, square):
                        black_pawn_structure += 10
                    
                    if self.is_isolated_pawn(board_state, square):
                        black_pawn_structure -= 15
                    
                    if self.is_doubled_pawn(board_state, square):
                        black_pawn_structure -= 10
                    
                    if col in [3, 4] and row in [3, 4]:
                        black_piece_activity += 15
                
                elif piece_type == chess.KNIGHT:
                    if row in [2, 3, 4] and self.is_supported_by_pawn(board_state, square, chess.BLACK):
                        black_piece_activity += 20
                    
                    if col == 0 or col == 7 or row == 0 or row == 7:
                        black_piece_activity -= 15
                
                elif piece_type == chess.BISHOP:
                    mobility = len(list(board_state.attacks(square)))
                    black_mobility += mobility * 2
                    
                    if (square == chess.B7 or square == chess.G7) and board_state.piece_at(square - 8) == chess.Piece(chess.PAWN, chess.BLACK):
                        black_piece_activity += 15
                
                elif piece_type == chess.ROOK:
                    if self.is_open_file(board_state, col):
                        black_piece_activity += 15
                    elif self.is_semi_open_file(board_state, col, chess.BLACK):
                        black_piece_activity += 10
                    
                    if row == 1 or (is_endgame and row == 6):
                        black_piece_activity += 20
                
                elif piece_type == chess.QUEEN:
                    if not is_endgame and board_state.fullmove_number < 10 and square != chess.D8:
                        black_piece_activity -= 15
                
                elif piece_type == chess.KING:
                    if not is_endgame:
                        if col >= 5 and row == 7:
                            black_king_safety += 60
                            for shield_square in [square - 8, square - 9, square - 7]:
                                if (shield_square >= 0 and 
                                    board_state.piece_type_at(shield_square) == chess.PAWN and
                                    board_state.color_at(shield_square) == chess.BLACK):
                                    black_king_safety += 10
                        elif col <= 2 and row == 7:
                            black_king_safety += 50
                            for shield_square in [square - 8, square - 9, square - 7]:
                                if (shield_square >= 0 and 
                                    board_state.piece_type_at(shield_square) == chess.PAWN and
                                    board_state.color_at(shield_square) == chess.BLACK):
                                    black_king_safety += 10
                    else:
                        center_distance = abs(3.5 - col) + abs(3.5 - row)
                        black_king_safety += (8 - center_distance) * 10
        
        # Sum up all evaluation components
        white_score = (white_material + white_mobility + white_king_safety + 
                       white_pawn_structure + white_piece_activity)
        black_score = (black_material + black_mobility + black_king_safety + 
                       black_pawn_structure + black_piece_activity)
        
        score = white_score - black_score
        
        # Adjust score based on whose turn it is
        return score if board_state.turn == chess.WHITE else -score

    def evaluate_development(self, board_state):
        """Evaluate piece development in the opening."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            # Check for castling
            if board_state.has_castling_rights(color):
                score += 50 * multiplier
            
            # Evaluate piece development
            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board_state.pieces(piece_type, color):
                    # Bonus for developed pieces
                    if color:
                        if chess.square_rank(square) > 1:  # Not on starting rank
                            score += 30 * multiplier
                    else:
                        if chess.square_rank(square) < 6:  # Not on starting rank
                            score += 30 * multiplier
                    
                    # Bonus for center control
                    if square in self.center_squares:
                        score += 20 * multiplier
            
            # Penalty for moving pieces multiple times
            piece_moves = {}
            for move in board_state.move_stack:
                piece = board_state.piece_at(move.to_square)
                if piece and piece.color == color:
                    piece_moves[move.to_square] = piece_moves.get(move.to_square, 0) + 1
            
            for moves in piece_moves.values():
                if moves > 1:
                    score -= 20 * multiplier
        
        return score

    def evaluate_piece_safety(self, board_state):
        """Strict evaluation of piece safety."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color and piece.piece_type != chess.KING:
                    attackers = board_state.attackers(not color, square)
                    defenders = board_state.attackers(color, square)
                    
                    if attackers:
                        # Calculate SEE value
                        see_value = self.static_exchange_evaluation(board_state, chess.Move(square, square))
                        if see_value <= 0:
                            # Heavy penalty for hanging pieces
                            score -= self.piece_values[piece.piece_type] * 2 * multiplier
                        else:
                            # Small bonus for protected pieces
                            score += self.piece_values[piece.piece_type] * 0.1 * multiplier
                    
                    # Check for pins
                    if board_state.is_pinned(color, square):
                        score -= self.piece_values[piece.piece_type] * 0.5 * multiplier
                    
                    # Check for discovered attacks
                    temp_board = board_state.copy()
                    temp_board.remove_piece_at(square)
                    for attacker in attackers:
                        if temp_board.is_attacked_by(not color, attacker):
                            score -= self.piece_values[piece.piece_type] * 0.3 * multiplier
        
        return score

    def get_move(self, board: ChessBoard):
        """Get the best move with improved reliability and error handling."""
        try:
            board_state = board.get_board_state()
            
            # Adjust search depth based on number of pieces
            pieces_count = sum(1 for _ in board_state.piece_map().values())
            # Deeper search when fewer pieces
            if pieces_count <= 6:  # Deep endgame
                self.max_depth = self.base_max_depth + 3
            elif pieces_count <= 10:  # Endgame
                self.max_depth = self.base_max_depth + 2
            elif pieces_count <= 20:  # Late middlegame
                self.max_depth = self.base_max_depth + 1
            else:
                self.max_depth = self.base_max_depth
                
            # Adjust time management to avoid timeouts
            effective_search_time = self.max_search_time * self.time_management_factor
            
            # Get the current king position
            king_square = board_state.king(board_state.turn)
            
            # Track king position to penalize excessive king moves
            if king_square is not None:
                self.king_move_history[king_square] = self.king_move_history.get(king_square, 0) + 1
            
            # Check if the game is in the opening book
            if self.opening_book:
                try:
                    entry = self.opening_book.find(board_state)
                    if entry:
                        print(f"Playing opening book move: {entry.move}")
                        return entry.move
                except Exception as e:
                    # print(f"Opening book error: {e}")
                    print("Played non-book move")
                    
            # Record position for repetition detection
            position_fen = board_state.fen().split(' ')[0]
            self.position_history[position_fen] = self.position_history.get(position_fen, 0) + 1
            
            # Detect game phase
            is_endgame = self.is_endgame(board_state)
            
            # OPENING: Prioritize development in the early game
            if len(board_state.move_stack) < 15 and not is_endgame:
                # Try castling first - a high priority move
                for move in board_state.legal_moves:
                    if board_state.is_castling(move):
                        return move  # Always return castling moves if legal
                
                # Develop knights and bishops to good squares
                for piece_type in [chess.KNIGHT, chess.BISHOP]:
                    for square in board_state.pieces(piece_type, board_state.turn):
                        for move in board_state.legal_moves:
                            if move.from_square == square:
                                # Prioritize center squares
                                if move.to_square in self.center_squares:
                                    if self.is_move_safe(board, move):
                                        return move
                                
                # Generally good opening moves
                for move in board_state.legal_moves:
                    # Control center with pawns
                    if board_state.piece_type_at(move.from_square) == chess.PAWN:
                        if move.to_square in self.center_squares:
                            if self.is_move_safe(board, move):
                                return move
            
            # ENDGAME: Special endgame strategy focusing on promotion and king activity
            if is_endgame:
                # Look for pawn promotions first
                for move in board_state.legal_moves:
                    if move.promotion:
                        return move  # Always return promotion moves
                
                # Advance passed pawns
                for move in board_state.legal_moves:
                    if (board_state.piece_type_at(move.from_square) == chess.PAWN and 
                            self.is_passed_pawn(board_state, move.from_square)):
                        if self.is_move_safe(board, move):
                            return move
                
                # King activity in endgame
                king_square = board_state.king(board_state.turn)
                if king_square is not None:
                    best_king_move = None
                    best_score = -float('inf')
                    
                    for move in board_state.legal_moves:
                        if move.from_square == king_square:
                            # Calculate centralization score
                            to_file = chess.square_file(move.to_square)
                            to_rank = chess.square_rank(move.to_square)
                            centralization = (3.5 - abs(to_file - 3.5)) + (3.5 - abs(to_rank - 3.5))
                            
                            # Score based on distance to enemy king
                            enemy_king = board_state.king(not board_state.turn)
                            score = centralization * 2
                            
                            if enemy_king is not None:
                                enemy_file = chess.square_file(enemy_king)
                                enemy_rank = chess.square_rank(enemy_king)
                                distance = (abs(to_file - enemy_file) + abs(to_rank - enemy_rank))
                                # In endgame, approach the enemy king
                                score += (14 - distance) * 2
                            
                            if score > best_score and self.is_move_safe(board, move):
                                best_score = score
                                best_king_move = move
                    
                    if best_king_move:
                        return best_king_move
            
            # STANDARD SEARCH: Use alpha-beta search for all positions
            
            # First, filter out moves that lead to repeated positions
            non_repetition_moves = []
            for move in board_state.legal_moves:
                temp_board = board_state.copy()
                temp_board.push(move)
                pos = temp_board.fen().split(' ')[0]
                
                # Prefer moves that don't repeat positions
                if pos not in self.position_history or self.position_history[pos] < 1:
                    non_repetition_moves.append(move)
            
            # If we have non-repeating moves, prioritize them
            if non_repetition_moves:
                legal_moves = non_repetition_moves
            
            # Start shallow search and progressively go deeper
            self.search_start_time = time.time()
            self.nodes_searched = 0
            
            best_move = None
            best_score = -float('inf')
            
            # Iterative deepening
            for depth in range(1, self.max_depth + 1):
                # Check if we've exceeded time budget
                if time.time() - self.search_start_time > effective_search_time:
                    break
                
                try:
                    # Run the alpha-beta search at current depth
                    score, move = self.alpha_beta(board, depth, -float('inf'), float('inf'), True)
                    
                    # Update best move if valid
                    if move and move in legal_moves:
                        best_move = move
                        best_score = score
                except Exception as e:
                    print(f"Search error at depth {depth}: {e}")
                    break
            
            # If search found a good move, return it
            if best_move:
                return best_move
                
            # Fallback to safe captures if search failed
            for move in legal_moves:
                if board_state.is_capture(move) and self.is_move_safe(board, move):
                    return move
            
            # Final fallback: any safe move
            safe_moves = [move for move in legal_moves if self.is_move_safe(board, move)]
            if safe_moves:
                return random.choice(safe_moves)
                
            # Last resort: any legal move
            return random.choice(legal_moves)
            
        except Exception as e:
            print(f"Critical error in get_move: {e}")
            # Emergency fallback
            try:
                legal_moves = list(board.get_board_state().legal_moves)
                if legal_moves:
                    return random.choice(legal_moves)
            except:
                pass
            return None

    def is_endgame(self, board_state):
        """Determine if the position is in the endgame phase."""
        try:
            # Count material excluding kings and pawns
            white_material = 0
            black_material = 0
            
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                white_pieces = len(list(board_state.pieces(piece_type, chess.WHITE)))
                black_pieces = len(list(board_state.pieces(piece_type, chess.BLACK)))
                white_material += white_pieces * self.piece_values[piece_type]
                black_material += black_pieces * self.piece_values[piece_type]
            
            total_material = white_material + black_material
            
            # Conditions for endgame:
            # 1. No queens and limited material
            no_queens = len(list(board_state.pieces(chess.QUEEN, chess.WHITE))) == 0 and len(list(board_state.pieces(chess.QUEEN, chess.BLACK))) == 0
            
            # 2. Total material below threshold
            low_material = total_material < self.endgame_threshold
            
            # 3. Only one queen and no other pieces
            one_queen_only = ((len(list(board_state.pieces(chess.QUEEN, chess.WHITE))) + len(list(board_state.pieces(chess.QUEEN, chess.BLACK)))) == 1 and 
                             (white_material + black_material - 900) < 500)
            
            return low_material or no_queens or one_queen_only
        except Exception as e:
            print(f"Error in is_endgame: {e}")
            # Default to midgame if there's an error
            return False

    def is_likely_draw(self, board_state):
        """Check if the position is likely to be a draw based on material or repetition."""
        try:
            # 1. Insufficient material
            if board_state.is_insufficient_material():
                return True
            
            # 2. Check for threefold repetition
            fen = board_state.fen().split(' ')[0]  # Just get the position part, not the full FEN
            repetition_count = self.position_history.get(fen, 0)
            if repetition_count >= 3:
                return True
                
            # 3. Check for fifty-move rule (approaching)
            if board_state.halfmove_clock >= 40:  # Start to avoid repetition before 50-move rule
                return True
                
            # 4. Check for simplified endgame with limited winning chances
            white_pieces = sum(1 for _ in board_state.pieces(chess.KNIGHT, chess.WHITE)) + \
                          sum(1 for _ in board_state.pieces(chess.BISHOP, chess.WHITE)) + \
                          sum(1 for _ in board_state.pieces(chess.ROOK, chess.WHITE)) + \
                          sum(1 for _ in board_state.pieces(chess.QUEEN, chess.WHITE))
                          
            black_pieces = sum(1 for _ in board_state.pieces(chess.KNIGHT, chess.BLACK)) + \
                          sum(1 for _ in board_state.pieces(chess.BISHOP, chess.BLACK)) + \
                          sum(1 for _ in board_state.pieces(chess.ROOK, chess.BLACK)) + \
                          sum(1 for _ in board_state.pieces(chess.QUEEN, chess.BLACK))
            
            white_pawns = len(list(board_state.pieces(chess.PAWN, chess.WHITE)))
            black_pawns = len(list(board_state.pieces(chess.PAWN, chess.BLACK)))
            
            # Few pieces and symmetrical pawn structure often leads to draws
            if white_pieces <= 1 and black_pieces <= 1 and white_pawns == black_pawns:
                return True
                
            return False
        except Exception as e:
            print(f"Error in is_likely_draw: {e}")
            return False

    def evaluate_king_activity(self, board_state):
        """Evaluate king activity in the endgame."""
        score = 0
        
        # In endgame, king should be active and move toward center
        for color in [True, False]:
            multiplier = 1 if color else -1
            king_square = board_state.king(color)
            
            # Distance to center - closer is better
            file, rank = chess.square_file(king_square), chess.square_rank(king_square)
            file_distance = 3.5 - abs(file - 3.5)
            rank_distance = 3.5 - abs(rank - 3.5)
            center_distance = file_distance + rank_distance
            
            # Reward being closer to center
            score += center_distance * 10 * multiplier
            
            # King tropism - in endgame, king should move toward enemy king
            enemy_king = board_state.king(not color)
            if enemy_king:
                enemy_file, enemy_rank = chess.square_file(enemy_king), chess.square_rank(enemy_king)
                file_distance = abs(file - enemy_file)
                rank_distance = abs(rank - enemy_rank)
                king_distance = file_distance + rank_distance
                
                # Reward being closer to enemy king
                score += (14 - king_distance) * 2 * multiplier
        
        return score

    def evaluate_passed_pawns(self, board_state):
        """Evaluate passed pawns in the endgame."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            pawns = board_state.pieces(chess.PAWN, color)
            enemy_pawns = board_state.pieces(chess.PAWN, not color)
            
            for pawn in pawns:
                file = chess.square_file(pawn)
                rank = chess.square_rank(pawn)
                
                # Check if pawn is passed
                is_passed = True
                for enemy_pawn in enemy_pawns:
                    e_file = chess.square_file(enemy_pawn)
                    e_rank = chess.square_rank(enemy_pawn)
                    
                    # If there's an enemy pawn in front or adjacent files, pawn is not passed
                    if abs(file - e_file) <= 1:
                        if (color and e_rank > rank) or (not color and e_rank < rank):
                            is_passed = False
                            break
                
                if is_passed:
                    # Calculate how far advanced the pawn is
                    advancement = rank if color else 7 - rank
                    # Higher bonus for more advanced pawns
                    score += (advancement * 20 + 10) * multiplier
                    
                    # Extra bonus for pawns with clear path to promotion
                    if self.has_clear_promotion_path(board_state, pawn, color):
                        score += 50 * multiplier
        
        return score

    def has_clear_promotion_path(self, board_state, pawn_square, color):
        """Check if a pawn has a clear path to promotion."""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check squares in front of pawn
        if color:  # White
            ranks_to_check = range(rank + 1, 8)
        else:  # Black
            ranks_to_check = range(rank - 1, -1, -1)
            
        for r in ranks_to_check:
            square = chess.square(file, r)
            if board_state.piece_at(square) is not None:
                return False
                
        return True

    def evaluate_material(self, board_state, is_endgame=False):
        """Evaluate material with piece value adjustments based on game phase."""
        score = 0
        piece_values = self.endgame_piece_values if is_endgame else self.piece_values
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                pieces = board_state.pieces(piece_type, color)
                score += len(pieces) * piece_values[piece_type] * multiplier
                
                # Special case for bishops
                if piece_type == chess.BISHOP and len(pieces) >= 2:
                    score += 30 * multiplier  # Bishop pair bonus
        
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
        """Order moves for better pruning efficiency - captures, checks, and history first."""
        move_scores = {}
        board_state = board.get_board_state()
        turn = board_state.turn
        
        for move in moves:
            score = 0
            
            # Prioritize captures by MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
            if board_state.is_capture(move):
                victim_piece = board_state.piece_type_at(move.to_square)
                aggressor_piece = board_state.piece_type_at(move.from_square)
                if victim_piece and aggressor_piece:
                    # MVV-LVA score: value of captured piece minus value of capturing piece / 10
                    score += 10 * self.piece_values.get(victim_piece, 0) - self.piece_values.get(aggressor_piece, 0) / 10
            
            # Prioritize checks
            board_state.push(move)
            if board_state.is_check():
                score += 300  # Bonus for checks
            board_state.pop()
            
            # Prioritize promotions
            if move.promotion:
                score += 900  # Queen promotion is highest
            
            # Use killer moves
            if 0 <= depth < len(self.killer_moves):
                if move == self.killer_moves[depth][0]:
                    score += 200
                elif move == self.killer_moves[depth][1]:
                    score += 150
            
            # Use history heuristic
            score += self.history_table.get(move, 0) / 100
            
            # Center control for early game
            if board_state.fullmove_number < 10:
                if move.to_square in self.center_squares:
                    score += 50
                    
            # Castle bonus
            if board_state.is_castling(move):
                score += 400
                
            # King move penalty in middlegame
            if (board_state.piece_type_at(move.from_square) == chess.KING and 
                not self.is_endgame(board_state) and 
                board_state.fullmove_number < 30):
                # Apply king move penalty if not in endgame and not castling
                if not board_state.is_castling(move):
                    # Stronger penalty for repeated king moves
                    king_square = board_state.king(board_state.turn)
                    if king_square is not None and king_square in self.king_move_history:
                        # Penalize more for repeated king moves
                        score -= self.king_move_penalty * (self.king_move_history[king_square] + 1)
                    else:
                        score -= self.king_move_penalty
            
            move_scores[move] = score
        
        # Sort moves by score in descending order
        sorted_moves = sorted(moves, key=lambda move: move_scores.get(move, 0), reverse=True)
        return sorted_moves

    def static_exchange_evaluation(self, board_state, move):
        """Static Exchange Evaluation with better error handling."""
        try:
            # Make sure the move is legal
            if move not in board_state.legal_moves:
                return -1000  # Very bad score for illegal moves
            
            # If not a capture, no need for SEE
            if not board_state.is_capture(move):
                return 0
    
            captured_piece = board_state.piece_type_at(move.to_square)
            if not captured_piece:
                return 0
    
            # Start with the value of the captured piece
            score = self.piece_values[captured_piece]
            
            # Make the move on a copy
            temp_board = board_state.copy()
            temp_board.push(move)
            
            # If the capturing piece is immediately recaptured, subtract its value
            to_square = move.to_square
            if temp_board.attackers(temp_board.turn, to_square):
                moving_piece = board_state.piece_type_at(move.from_square)
                if moving_piece:
                    score -= self.piece_values[moving_piece]
            
            return score
        except Exception as e:
            # Safely handle any errors
            print(f"Error in static_exchange_evaluation: {e}")
            return -100  # Negative score on error to discourage problematic moves

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

    def evaluate_king_safety(self, board_state):
        """Evaluate king safety with increased emphasis on castling and pawn structure."""
        score = 0
        for color in [True, False]:
            multiplier = 1 if color else -1
            king_square = board_state.king(color)
            if king_square is None:
                continue

            # Castling rights bonus
            if board_state.has_castling_rights(color):
                score += 20 * multiplier

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
                                pawn_shield += 15  # Increased from 10 to 15

            score += pawn_shield * multiplier

            # King attack evaluation
            attackers = 0
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                for square in board_state.pieces(piece_type, not color):
                    if board_state.is_attacked_by(color, square):
                        attackers += 1

            score -= attackers * 8 * multiplier  # Increased from 5 to 8

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

    def evaluate_threats(self, board_state):
        """Evaluate threats and tactical opportunities."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            # Check for hanging pieces
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color:
                    attackers = board_state.attackers(not color, square)
                    defenders = board_state.attackers(color, square)
                    
                    if attackers and len(attackers) > len(defenders):
                        score -= self.piece_values[piece.piece_type] * multiplier
                    
                    # Check for discovered attacks
                    if piece.piece_type != chess.KING:
                        for attacker in attackers:
                            # Check if moving the piece would reveal an attack
                            temp_board = board_state.copy()
                            temp_board.remove_piece_at(square)
                            if temp_board.is_attacked_by(not color, attacker):
                                score -= self.piece_values[piece.piece_type] * multiplier * 0.5
            
            # Check for pins
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color:
                    if board_state.is_pinned(color, square):
                        score -= self.piece_values[piece.piece_type] * multiplier * 0.3
        
        return score

    def evaluate_tactics(self, board_state):
        """Evaluate tactical opportunities and threats."""
        score = 0
        
        for color in [True, False]:
            multiplier = 1 if color else -1
            
            # Check for hanging pieces
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color:
                    attackers = board_state.attackers(not color, square)
                    defenders = board_state.attackers(color, square)
                    
                    if attackers and len(attackers) > len(defenders):
                        # Hanging piece bonus
                        score -= self.piece_values[piece.piece_type] * multiplier * 1.5
                    
                    # Check for discovered attacks
                    if piece.piece_type != chess.KING:
                        for attacker in attackers:
                            temp_board = board_state.copy()
                            temp_board.remove_piece_at(square)
                            if temp_board.is_attacked_by(not color, attacker):
                                score -= self.piece_values[piece.piece_type] * multiplier
            
            # Check for pins
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color:
                    if board_state.is_pinned(color, square):
                        score -= self.piece_values[piece.piece_type] * multiplier * 0.5
            
            # Check for forks
            for square in chess.SQUARES:
                piece = board_state.piece_at(square)
                if piece and piece.color == color and piece.piece_type in [chess.KNIGHT, chess.QUEEN]:
                    attacked_squares = set()
                    for target in chess.SQUARES:
                        if board_state.is_attacked_by(color, target):
                            attacked_squares.add(target)
                    
                    if len(attacked_squares) >= 2:
                        score += len(attacked_squares) * 10 * multiplier
        
        return score

    def alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        """Enhanced alpha-beta search with better pruning and move ordering."""
        # Check for time limit exceeded
        if time.time() - self.search_start_time > self.max_search_time * self.time_management_factor:
            # Return a sentinel value to indicate timeout
            return (-10000 if maximizing_player else 10000), None
            
        if depth == 0:
            return self.evaluate_position(board), None
        
        board_state = board.get_board_state()
        moves = list(board_state.legal_moves)
        
        if not moves:
            if board_state.is_check():
                return (-10000 if maximizing_player else 10000), None
            return 0, None
        
        # Order moves
        moves = self.order_moves(board, moves, depth)
        
        best_move = None
        if maximizing_player:
            best_score = -float('inf')
            for move in moves:
                board_state.push(move)
                score, _ = self.alpha_beta(board, depth - 1, alpha, beta, False)
                board_state.pop()
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
                
                # Update history table
                self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                
                # Update killer moves
                if depth >= 2 and move not in self.killer_moves[depth]:
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
            
            return best_score, best_move
        else:
            best_score = float('inf')
            for move in moves:
                board_state.push(move)
                score, _ = self.alpha_beta(board, depth - 1, alpha, beta, True)
                board_state.pop()
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
                
                # Update history table
                self.history_table[move] = self.history_table.get(move, 0) + depth * depth
                
                # Update killer moves
                if depth >= 2 and move not in self.killer_moves[depth]:
                    self.killer_moves[depth][1] = self.killer_moves[depth][0]
                    self.killer_moves[depth][0] = move
            
            return best_score, best_move

    def is_move_safe(self, board, move):
        """Strict safety evaluation to prevent hanging pieces."""
        try:
            board_state = board.get_board_state()
            
            # Verify the move is legal in the current position
            if move not in board_state.legal_moves:
                return False
            
            # Make the move on a copy of the board
            temp_board = board_state.copy()
            temp_board.push(move)
            
            # Check if our king is in check after the move
            if temp_board.is_check():
                return False
            
            # Get the square where the piece landed
            to_square = move.to_square
            
            # Check if the moved piece is safe
            moved_piece = temp_board.piece_at(to_square)
            if moved_piece and moved_piece.piece_type != chess.KING:
                attackers = list(temp_board.attackers(not temp_board.turn, to_square))
                defenders = list(temp_board.attackers(temp_board.turn, to_square))
                
                # If there are attackers and they outnumber defenders, the move may be unsafe
                if attackers and len(attackers) > len(defenders):
                    # Simple material exchange evaluation
                    min_attacker_value = min(self.piece_values[temp_board.piece_at(sq).piece_type] 
                                        for sq in attackers if temp_board.piece_at(sq))
                    
                    # If the attacker value is less than our piece value, the exchange is bad
                    if min_attacker_value < self.piece_values[moved_piece.piece_type]:
                        return False
            
            # Quick check for obvious hanging pieces after move
            for square in chess.SQUARES:
                piece = temp_board.piece_at(square)
                if piece and piece.color == temp_board.turn and piece.piece_type != chess.KING:
                    attackers = list(temp_board.attackers(not temp_board.turn, square))
                    defenders = list(temp_board.attackers(temp_board.turn, square))
                    
                    # If a piece is attacked with no defenders, it's hanging
                    if attackers and not defenders:
                        return False
            
            return True
        except Exception as e:
            print(f"Error in is_move_safe: {e}")
            # If there's an error in evaluation, play it safe
            return False

    def would_hang_piece(self, board, move):
        """Check if a move would hang a piece."""
        board_state = board.get_board_state()
        temp_board = board_state.copy()
        temp_board.push(move)
        
        # Check if any piece is hanging after the move
        for square in chess.SQUARES:
            piece = temp_board.piece_at(square)
            if piece and piece.color == temp_board.turn and piece.piece_type != chess.KING:
                attackers = temp_board.attackers(not temp_board.turn, square)
                defenders = temp_board.attackers(temp_board.turn, square)
                
                if attackers and len(attackers) > len(defenders):
                    return True
        
        return False

    def is_passed_pawn(self, board_state, square):
        """Check if a pawn is passed (no opposing pawns ahead on same or adjacent files)"""
        color = board_state.color_at(square)
        if not color or board_state.piece_type_at(square) != chess.PAWN:
            return False
            
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Direction to advance (up for white, down for black)
        direction = 1 if color == chess.BLACK else -1
        
        # Check if there are any opposing pawns ahead on same or adjacent files
        for r in range(rank + direction, 8 if direction > 0 else -1, direction):
            for f in range(max(0, file - 1), min(8, file + 2)):
                test_square = chess.square(f, r)
                if (board_state.piece_type_at(test_square) == chess.PAWN and 
                    board_state.color_at(test_square) != color):
                    return False
        
        return True
        
    def is_pawn_supported(self, board_state, square):
        """Check if a pawn is supported by another friendly pawn"""
        color = board_state.color_at(square)
        if not color or board_state.piece_type_at(square) != chess.PAWN:
            return False
            
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check if there's a friendly pawn supporting this one
        support_rank = rank - 1 if color == chess.WHITE else rank + 1
        if support_rank < 0 or support_rank > 7:
            return False
            
        for support_file in [file - 1, file + 1]:
            if support_file < 0 or support_file > 7:
                continue
                
            support_square = chess.square(support_file, support_rank)
            if (board_state.piece_type_at(support_square) == chess.PAWN and 
                board_state.color_at(support_square) == color):
                return True
        
        return False
        
    def is_isolated_pawn(self, board_state, square):
        """Check if a pawn is isolated (no friendly pawns on adjacent files)"""
        color = board_state.color_at(square)
        if not color or board_state.piece_type_at(square) != chess.PAWN:
            return False
            
        file = chess.square_file(square)
        
        # Check adjacent files for friendly pawns
        for adjacent_file in [file - 1, file + 1]:
            if adjacent_file < 0 or adjacent_file > 7:
                continue
                
            for rank in range(8):
                adjacent_square = chess.square(adjacent_file, rank)
                if (board_state.piece_type_at(adjacent_square) == chess.PAWN and 
                    board_state.color_at(adjacent_square) == color):
                    return False
        
        return True
        
    def is_doubled_pawn(self, board_state, square):
        """Check if a pawn is doubled (another friendly pawn on the same file)"""
        color = board_state.color_at(square)
        if not color or board_state.piece_type_at(square) != chess.PAWN:
            return False
            
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        # Check other squares on the same file
        for other_rank in range(8):
            if other_rank == rank:
                continue
                
            other_square = chess.square(file, other_rank)
            if (board_state.piece_type_at(other_square) == chess.PAWN and 
                board_state.color_at(other_square) == color):
                return True
        
        return False
        
    def is_open_file(self, board_state, file):
        """Check if a file is completely open (no pawns)"""
        for rank in range(8):
            square = chess.square(file, rank)
            if board_state.piece_type_at(square) == chess.PAWN:
                return False
        return True
        
    def is_semi_open_file(self, board_state, file, color):
        """Check if a file has no friendly pawns but might have enemy pawns"""
        for rank in range(8):
            square = chess.square(file, rank)
            if (board_state.piece_type_at(square) == chess.PAWN and 
                board_state.color_at(square) == color):
                return False
        return True
        
    def is_supported_by_pawn(self, board_state, square, color):
        """Check if a square is supported by a friendly pawn"""
        # Where would supporting pawns be located?
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        support_rank = rank - 1 if color == chess.WHITE else rank + 1
        if support_rank < 0 or support_rank > 7:
            return False
            
        for support_file in [file - 1, file + 1]:
            if support_file < 0 or support_file > 7:
                continue
                
            support_square = chess.square(support_file, support_rank)
            if (board_state.piece_type_at(support_square) == chess.PAWN and 
                board_state.color_at(support_square) == color):
                return True
        
        return False

    def update_king_move_history(self, board_state, move):
        """Update the king move history after making a move."""
        # Only track if it's a king move
        if board_state.piece_type_at(move.from_square) == chess.KING:
            self.king_move_history[move.to_square] = self.king_move_history.get(move.to_square, 0) + 1