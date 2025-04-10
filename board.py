import chess

class ChessBoard:
    def __init__(self):
        self.board = chess.Board()
        self.move_history = []
    
    def get_legal_moves(self):
        """Returns a list of legal moves in the current position."""
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        """
        Attempts to make a move on the board.
        Returns True if successful, False if illegal.
        """
        try:
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move)
                return True
            return False
        except Exception as e:
            print(f"Error making move: {e}")
            return False
    
    def is_game_over(self):
        """Returns True if the game is over."""
        # Check for standard termination conditions
        if self.board.is_game_over():
            return True
            
        # Check for threefold repetition
        if self.board.is_repetition(3):
            return True
            
        # Check for fifty-move rule
        if self.board.halfmove_clock >= 100:
            return True
            
        # Check for insufficient material
        if self.board.is_insufficient_material():
            return True
            
        return False
    
    def get_board_state(self):
        """Returns the current board state."""
        return self.board

    def get_result(self):
        """Returns the game result if the game is over."""
        if self.is_game_over():
            # Check for checkmate
            if self.board.is_checkmate():
                return "Checkmate - " + ("Black wins" if self.board.turn == chess.WHITE else "White wins")
                
            # Check for stalemate
            if self.board.is_stalemate():
                return "Stalemate - Draw"
                
            # Check for threefold repetition
            if self.board.is_repetition(3):
                return "Threefold repetition - Draw"
                
            # Check for fifty-move rule
            if self.board.halfmove_clock >= 100:
                return "Fifty-move rule - Draw"
                
            # Check for insufficient material
            if self.board.is_insufficient_material():
                return "Insufficient material - Draw"
                
            return self.board.outcome()
        return None

    def undo_move(self):
        """Undoes the last move made."""
        if len(self.move_history) > 0:
            self.move_history.pop()
        return self.board.pop()
        
    def get_move_count(self):
        """Returns the number of half-moves played."""
        return len(self.board.move_stack)
        
    def get_repeated_positions_count(self, fen=None):
        """
        Returns the number of times the current position has been repeated.
        Optionally takes a FEN string to check a specific position.
        """
        if fen is None:
            # Get current position FEN (excluding move counters and en passant square)
            fen = self.board.fen().split(' ')[0]
            
        position_count = 0
        # Create a copy of the board to navigate through move history
        temp_board = chess.Board()
        
        # Count occurrences
        for i in range(len(self.board.move_stack)):
            temp_board_fen = temp_board.fen().split(' ')[0]
            if temp_board_fen == fen:
                position_count += 1
                
            if i < len(self.board.move_stack):
                temp_board.push(self.board.move_stack[i])
                
        # Check current position
        if temp_board.fen().split(' ')[0] == fen:
            position_count += 1
            
        return position_count
