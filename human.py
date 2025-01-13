import chess
import pygame

class HumanPlayer:
    def __init__(self, color):
        self.color = color
        self.selected_square = None
        
    def get_square_from_coords(self, x, y, flipped=False):
        """Convert screen coordinates to chess square."""
        file_idx = x * 8 // 600
        rank_idx = y * 8 // 600
        if flipped:
            file_idx = 7 - file_idx
            rank_idx = 7 - rank_idx
        else:
            rank_idx = 7 - rank_idx
        return chess.square(file_idx, rank_idx)

    def is_promotion_move(self, board, from_square, to_square):
        """Check if the move would be a pawn promotion."""
        piece = board.get_board_state().piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(to_square)
            return (self.color == chess.WHITE and rank == 7) or \
                   (self.color == chess.BLACK and rank == 0)
        return False

    def get_promotion_choice(self):
        """Get the promotion piece choice from the player."""
        # Create a simple text-based menu for promotion choices
        pygame.font.init()
        font = pygame.font.Font(None, 36)
        screen = pygame.display.get_surface()
        
        pieces = {
            'Q': (chess.QUEEN, "Queen"), 
            'R': (chess.ROOK, "Rook"), 
            'B': (chess.BISHOP, "Bishop"), 
            'K': (chess.KNIGHT, "Knight")
        }
        choices = []
        y = 250
        
        # Draw promotion options
        for key, piece in pieces.items():
            text = font.render(f"Press {key} for {piece[1]}", True, (255, 255, 255))
            rect = text.get_rect(center=(300, y))
            screen.blit(text, rect)
            choices.append((piece, rect))
            y += 50
        
        pygame.display.flip()
        
        # Wait for valid choice
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                for key, piece_type in [
                    (pygame.K_q, chess.QUEEN),
                    (pygame.K_r, chess.ROOK),
                    (pygame.K_b, chess.BISHOP),
                    (pygame.K_k, chess.KNIGHT)
                ]:
                    if event.key == key:
                        return piece_type
        
    def get_move(self, board):
        """Get move from human player through GUI interaction."""
        pygame.event.clear()
        
        while True:
            event = pygame.event.wait()
            
            if event.type == pygame.QUIT:
                return None
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                square = self.get_square_from_coords(x, y, self.color == chess.BLACK)
                
                if self.selected_square is None:
                    # First click - select piece
                    piece = board.get_board_state().piece_at(square)
                    if piece and piece.color == self.color:
                        self.selected_square = square
                else:
                    # Second click - try to make move
                    from_square = self.selected_square
                    to_square = square
                    
                    # Check if this is a promotion move
                    if self.is_promotion_move(board, from_square, to_square):
                        promotion_piece = self.get_promotion_choice()
                        if promotion_piece is None:
                            self.selected_square = None
                            continue
                        move = chess.Move(from_square, to_square, promotion=promotion_piece)
                    else:
                        move = chess.Move(from_square, to_square)
                    
                    # Check if move is legal
                    if move in board.get_legal_moves():
                        self.selected_square = None
                        return move
                    
                    # If illegal move, clear selection
                    self.selected_square = None
        
        return None