# game.py
import chess
import chess.svg
from board import ChessBoard
from bot import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image

IS_BOT = False  # Set to False for human vs bot, True for bot vs bot

class ChessGame:
    def __init__(self):
        self.board = ChessBoard()
        
        # Initialize players based on IS_BOT flag
        if IS_BOT:
            self.white_player = ChessBot()
            self.black_player = ChessBot()
        else:
            self.white_player = HumanPlayer(chess.WHITE)
            self.black_player = ChessBot()
        
        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")
        
    def svg_to_pygame_surface(self, svg_string):
        """Convert SVG string to Pygame surface"""
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        return pygame.image.fromstring(data, size, mode)

    def display_board(self, last_move=None):
        """Display the current board state"""
        # Create SVG with highlighted last move if exists
        svg = chess.svg.board(
            board=self.board.get_board_state(),
            lastmove=last_move,
            size=self.WINDOW_SIZE
        )
        
        # Convert SVG to Pygame surface and display
        py_image = self.svg_to_pygame_surface(svg)
        self.screen.blit(py_image, (0, 0))
        pygame.display.flip()

    def play_game(self):
        """Main game loop"""
        last_move = None
        
        while not self.board.is_game_over():
            # Display current board
            self.display_board(last_move)
            
            # Determine current player
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            
            # Get player's move
            move = current_player.get_move(self.board)
            
            if move is None:
                print("Game ended by player")
                break
                
            # Make the move
            if not self.board.make_move(move):
                print(f"Illegal move attempted: {move}")
                break
                
            print(f"Move played: {move}")
            last_move = move
            
            # Add delay only for bot moves
            if isinstance(current_player, ChessBot):
                pygame.time.wait(1000)  # 1 second delay
            
        # Display final position
        self.display_board(last_move)
        result = self.board.get_result()
        print(f"Game Over! Result: {result}")
        
        # Keep window open until closed
        while True:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                break
        
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.play_game()