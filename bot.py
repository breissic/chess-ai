import chess
from board import ChessBoard

class ChessBot:
    def __init__(self):
        pass
    
    def get_move(self, board: ChessBoard):
        """
        Given the current board state, returns the chosen move.
        This is the main function students will implement.
        """
        # For now, just return the first legal move
        legal_moves = board.get_legal_moves()
        if legal_moves:
            return legal_moves[0]
        return None