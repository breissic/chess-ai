import chess
import chess.svg
from board import ChessBoard
from bot import ChessBot
from human import HumanPlayer
import pygame
import cairosvg
import io
from PIL import Image
import time
import random
import argparse
import os
from load_bot import load_bot_from_params

IS_BOT = False  # Set to False for human vs bot, True for bot vs bot
MAX_MOVES = 200  # Maximum number of moves before forcing a draw
BOT_MOVE_TIMEOUT = 15.0  # Increased timeout from 5 to 15 seconds

class ChessGame:
    def __init__(self, white_bot=None, black_bot=None, human_player=True):
        self.board = ChessBoard()
        
        # Initialize players with the updated human_player parameter
        if human_player and white_bot is None:
            # Human as white vs bot as black
            self.white_player = HumanPlayer(chess.WHITE, self)
            self.black_player = black_bot if black_bot else ChessBot()
        elif human_player and black_bot is None:
            # Human as black vs bot as white
            self.white_player = white_bot if white_bot else ChessBot()
            self.black_player = HumanPlayer(chess.BLACK, self)
        else:
            # Bot vs bot
            self.white_player = white_bot if white_bot else ChessBot()
            self.black_player = black_bot if black_bot else ChessBot()
        
        # Initialize Pygame
        pygame.init()
        self.WINDOW_SIZE = 600
        self.screen = pygame.display.set_mode((self.WINDOW_SIZE, self.WINDOW_SIZE))
        pygame.display.set_caption("Chess Game")
        
        # Game state variables
        self.move_count = 0
        self.last_capture_or_pawn_move = 0  # For tracking potential stalemates
        self.position_history = {}  # For tracking repeated positions
        
    def svg_to_pygame_surface(self, svg_string):
        """Convert SVG string to Pygame surface"""
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((self.WINDOW_SIZE, self.WINDOW_SIZE))
        mode = image.mode
        size = image.size
        data = image.tobytes()
        return pygame.image.fromstring(data, size, mode)

    def display_board(self, last_move=None, selected_square=None):
        """Display the current board state"""
        # Build highlight dictionary for the selected square
        highlight_squares = None
        if selected_square is not None:
            highlight_squares = {
                selected_square: {"fill": "#FFFF00", "stroke": "none"}
            }

        # Create SVG with highlighted last move and selected square
        svg = chess.svg.board(
            board=self.board.get_board_state(),
            lastmove=last_move,
            squares=highlight_squares,     # colored square highlight
            size=self.WINDOW_SIZE
        )
        
        # Convert SVG to Pygame surface and display
        py_image = self.svg_to_pygame_surface(svg)
        self.screen.blit(py_image, (0, 0))
        pygame.display.flip()

    def display_result(self, result):
        """Display the game result on the screen"""
        # Create a semi-transparent overlay
        overlay = pygame.Surface((self.WINDOW_SIZE, self.WINDOW_SIZE), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Semi-transparent black
        self.screen.blit(overlay, (0, 0))
        
        # Create text
        font = pygame.font.SysFont('Arial', 36)
        text = font.render(str(result), True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.WINDOW_SIZE/2, self.WINDOW_SIZE/2))
        self.screen.blit(text, text_rect)
        
        # Add instruction to quit
        font_small = pygame.font.SysFont('Arial', 24)
        quit_text = font_small.render("Click to exit", True, (255, 255, 255))
        quit_rect = quit_text.get_rect(center=(self.WINDOW_SIZE/2, self.WINDOW_SIZE/2 + 50))
        self.screen.blit(quit_text, quit_rect)
        
        pygame.display.flip()

    def check_draw_conditions(self):
        """Check for various draw conditions"""
        board_state = self.board.get_board_state()
        
        # Check for stalemate or insufficient material
        if board_state.is_stalemate() or board_state.is_insufficient_material():
            return True
            
        # Check for threefold repetition
        position_fen = board_state.fen().split(' ')[0]  # Just the board position
        self.position_history[position_fen] = self.position_history.get(position_fen, 0) + 1
        if self.position_history[position_fen] >= 3:
            return True
            
        # Check for fifty-move rule
        if board_state.halfmove_clock >= 50:
            return True
            
        # Check for move limit reached
        if self.move_count >= MAX_MOVES:
            return True
            
        return False

    def play_game(self):
        """Main game loop"""
        last_move = None
        consecutive_bad_moves = 0  # To detect stuck states
        last_positions = []  # Track recent positions to detect cycles
        
        while not self.board.is_game_over():
            # Check for user quitting
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Get current player for selected square highlighting
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            selected_square = getattr(current_player, 'selected_square', None)
            
            # Display current board with highlights
            self.display_board(last_move, selected_square)
            
            # Check for draw by repetition or other draw conditions
            if self.check_draw_conditions():
                print("Game drawn by special condition")
                break
                
            # Safety check for infinite loops - store last 6 positions
            current_fen = self.board.get_board_state().fen().split(' ')[0]
            last_positions.append(current_fen)
            if len(last_positions) > 6:
                last_positions.pop(0)
                # Check if we're in a 2-move cycle (4 positions alternating)
                if len(last_positions) == 6 and last_positions[0] == last_positions[4] and last_positions[1] == last_positions[5]:
                    print("Detected move cycle - ending game")
                    break
            
            # Determine current player
            current_player = self.white_player if self.board.get_board_state().turn else self.black_player
            
            # Set a timeout for bot moves (increased from 5 to 15 seconds)
            start_time = time.time()
            move = None
            
            if isinstance(current_player, ChessBot):
                try:
                    # Use a timeout for bot moves
                    move = current_player.get_move(self.board)
                    # If the bot takes too long, interrupt
                    if time.time() - start_time > BOT_MOVE_TIMEOUT:
                        print("Bot timeout - using best move found so far")
                        # If we have a move despite timeout, use it
                        # Otherwise, find a good move
                        if not move:
                            board_state = self.board.get_board_state()
                            legal_moves = list(board_state.legal_moves)
                            
                            # Try to find a capture or check first
                            safe_moves = []
                            for m in legal_moves:
                                if board_state.is_capture(m) or board_state.gives_check(m):
                                    safe_moves.append(m)
                            
                            # If no good moves, use any legal move
                            if not safe_moves:
                                safe_moves = legal_moves
                                
                            if safe_moves:
                                move = random.choice(safe_moves)
                except Exception as e:
                    print(f"Bot error: {e}")
                    # If bot fails, make a random move
                    legal_moves = self.board.get_legal_moves()
                    if legal_moves:
                        move = random.choice(legal_moves)
            else:
                # Human player
                move = current_player.get_move(self.board)
            
            if move is None:
                print("Game ended - no valid move found")
                break
                
            # Make the move
            if not self.board.make_move(move):
                print(f"Illegal move attempted: {move}")
                consecutive_bad_moves += 1
                if consecutive_bad_moves >= 3:
                    print("Too many illegal moves - ending game")
                    break
                continue
            
            # Reset bad move counter after successful move
            consecutive_bad_moves = 0
            
            # Track move count
            self.move_count += 1
            
            # Track captures and pawn moves for fifty-move rule
            board_state = self.board.get_board_state()
            if board_state.is_capture(move) or board_state.piece_type_at(move.to_square) == chess.PAWN:
                self.last_capture_or_pawn_move = self.move_count
            
            print(f"Move played: {move}")
            last_move = move
            
            # Add delay only for bot moves
            if isinstance(current_player, ChessBot):
                pygame.time.wait(1000)  # 1 second delay
            
        # Display final position
        self.display_board(last_move)
        result = self.board.get_result()
        print(f"Game Over! Result: {result}")
        
        # Display the result in GUI
        self.display_result(result)
        
        # Keep window open until closed
        waiting_for_close = True
        while waiting_for_close:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting_for_close = False
                    break
        
        pygame.quit()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Play chess against AI or watch AI vs AI")
    parser.add_argument("--bot-vs-bot", action="store_true", help="Bot vs Bot mode")
    parser.add_argument("--white-bot", type=str, help="Path to parameters file for white bot")
    parser.add_argument("--black-bot", type=str, help="Path to parameters file for black bot")
    parser.add_argument("--play-as-black", action="store_true", help="Play as black against the bot")
    args = parser.parse_args()
    
    # Set bot mode based on arguments
    is_human_game = not args.bot_vs_bot
    
    # Load bots from parameter files if specified
    white_bot = None
    black_bot = None
    
    if args.white_bot:
        if os.path.exists(args.white_bot):
            print(f"Loading white bot from {args.white_bot}")
            white_bot = load_bot_from_params(args.white_bot)
        else:
            print(f"Warning: White bot parameter file {args.white_bot} not found, using default bot")
            white_bot = ChessBot()
    
    if args.black_bot:
        if os.path.exists(args.black_bot):
            print(f"Loading black bot from {args.black_bot}")
            black_bot = load_bot_from_params(args.black_bot)
        else:
            print(f"Warning: Black bot parameter file {args.black_bot} not found, using default bot")
            black_bot = ChessBot()
    
    # Create and run game
    if args.play_as_black:
        # Human plays as black
        game = ChessGame(white_bot=white_bot, black_bot=None, human_player=True)
    elif args.white_bot and args.black_bot:
        # Bot vs Bot when both are specified
        game = ChessGame(white_bot=white_bot, black_bot=black_bot, human_player=False)
    elif args.bot_vs_bot:
        # Bot vs Bot (default bots if not specified)
        game = ChessGame(white_bot=white_bot or ChessBot(), 
                         black_bot=black_bot or ChessBot(),
                         human_player=False)
    else:
        # Default: Human as white vs bot as black
        game = ChessGame(white_bot=None, black_bot=black_bot, human_player=True)
    
    game.play_game()

if __name__ == "__main__":
    main()