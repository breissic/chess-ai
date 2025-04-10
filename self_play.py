import chess
import random
import time
import json
import copy
import os
import signal
import sys
from board import ChessBoard
from bot import ChessBot

# Increase timeout from 5 to 15 seconds
BOT_MOVE_TIMEOUT = 15.0

# Flag to track if an interruption has occurred
interrupted = False

# Signal handler for graceful interruption
def signal_handler(sig, frame):
    global interrupted
    print("\nTraining will complete the current iteration and then exit...")
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

class SelfPlayTrainer:
    def __init__(self, bot_constructor, games_per_iteration=50, iterations=10, start_iteration=0):
        """
        Create a self-play trainer for chess bots
       
        Args:
            bot_constructor: Function that returns a new ChessBot instance
            games_per_iteration: Number of games to play in each training iteration
            iterations: Number of training iterations to run
            start_iteration: Iteration to start from (for resuming training)
        """
        self.bot_constructor = bot_constructor
        self.games_per_iteration = games_per_iteration
        self.iterations = iterations
        self.start_iteration = start_iteration
        self.best_bot = bot_constructor()
        self.game_history = []
        self.performance_history = []
        
        # Load existing performance history if resuming
        if self.start_iteration > 0:
            self._load_history()
       
    def train(self, output_base="trained_bot"):
        """
        Run the complete self-play training process
        
        Args:
            output_base: Base filename for saving checkpoints
        """
        global interrupted
        
        # Start from the specified iteration
        for iteration in range(self.start_iteration, self.iterations):
            print(f"Starting training iteration {iteration+1}/{self.iterations}")
           
            # Create challenger bot with variations
            challenger_bot = self.create_challenger()
           
            # Play training games
            results = self.play_match(self.best_bot, challenger_bot)
           
            # Analyze results
            analysis = self.analyze_results(results)
            print(f"Iteration {iteration+1} results: {analysis}")
           
            # Update the best bot if challenger performed better
            self.update_best_bot(challenger_bot, analysis)
           
            # Store performance metrics
            self.performance_history.append({
                "iteration": iteration + 1,
                "timestamp": time.time(),
                "metrics": analysis
            })
            
            # Save checkpoint after each iteration
            self._save_checkpoint(iteration + 1, output_base)
            
            # Check if we were interrupted
            if interrupted:
                print(f"Training interrupted after iteration {iteration+1}")
                print(f"To resume training from this point, use: --load {output_base}_params.json --start-iteration {iteration+1}")
                break
           
        return self.best_bot
    
    def _save_checkpoint(self, current_iteration, output_base):
        """Save checkpoint after each iteration"""
        # Save bot parameters
        params_file = f"{output_base}_params.json"
        self.save_best_bot(params_file)
        
        # Save history
        history_file = f"{output_base}_history.json"
        self.save_training_history(history_file)
        
        # Save current iteration for resume
        resume_file = f"{output_base}_resume.json"
        with open(resume_file, 'w') as f:
            json.dump({
                "current_iteration": current_iteration,
                "total_iterations": self.iterations
            }, f, indent=2)
            
        print(f"Checkpoint saved after iteration {current_iteration}")
        
    def _load_history(self):
        """Load existing performance history when resuming training"""
        try:
            history_file = "trained_bot_history.json"
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get("performance_history", [])
                    self.game_history = data.get("game_history", [])
                print(f"Loaded existing history with {len(self.performance_history)} previous iterations")
        except Exception as e:
            print(f"Error loading history: {e}")
   
    def create_challenger(self):
        """Create a challenger bot with slightly modified parameters"""
        challenger = self.bot_constructor()
       
        # Modify piece values with more focused randomization
        if hasattr(challenger, 'piece_values'):
            for piece in challenger.piece_values:
                # Smaller random adjustments for more stable training
                challenger.piece_values[piece] *= random.uniform(0.97, 1.03)
        
        # Modify endgame piece values
        if hasattr(challenger, 'endgame_piece_values'):
            for piece in challenger.endgame_piece_values:
                challenger.endgame_piece_values[piece] *= random.uniform(0.97, 1.03)
        
        # Modify search parameters
        if hasattr(challenger, 'max_depth'):
            # Occasionally adjust search depth
            if random.random() < 0.2:
                challenger.max_depth = max(3, min(7, challenger.max_depth + random.choice([-1, 1])))
        
        # Modify endgame threshold
        if hasattr(challenger, 'endgame_threshold'):
            challenger.endgame_threshold *= random.uniform(0.95, 1.05)
            
        # Occasionally adjust time management factor
        if hasattr(challenger, 'time_management_factor'):
            if random.random() < 0.15:
                challenger.time_management_factor = min(0.95, max(0.7, challenger.time_management_factor + random.uniform(-0.05, 0.05)))
        
        return challenger
   
    def play_match(self, bot1, bot2):
        """Play a series of games between two bots"""
        results = []
       
        for game_num in range(self.games_per_iteration):
            # Alternate colors for fairness
            if game_num % 2 == 0:
                white_bot, black_bot = bot1, bot2
                color_map = {1: "bot1", -1: "bot2", 0: "draw"}
            else:
                white_bot, black_bot = bot2, bot1
                color_map = {-1: "bot1", 1: "bot2", 0: "draw"}
           
            # Play a game
            board = ChessBoard()
            moves = []
            result_code = self.play_game(board, white_bot, black_bot, moves)
           
            # Record result
            winner = color_map[result_code]
            results.append({
                "game_num": game_num,
                "winner": winner,
                "moves": moves,
                "result_code": result_code
            })
           
            if (game_num + 1) % 5 == 0:
                print(f"Played {game_num + 1}/{self.games_per_iteration} games")
       
        return results
   
    def play_game(self, board, white_bot, black_bot, moves):
        """Play a single game between two bots, return the result code"""
        move_count = 0
        position_history = {}
       
        while not board.is_game_over():
            # Track positions for threefold repetition detection
            board_fen = board.get_board_state().fen().split(' ')[0]  # Just piece positions
            position_history[board_fen] = position_history.get(board_fen, 0) + 1
           
            # Detect draws by repetition or 50-move rule
            if position_history[board_fen] >= 3 or board.get_board_state().halfmove_clock >= 100:
                return 0  # Draw
           
            # Get current bot's move
            current_bot = white_bot if board.get_board_state().turn == chess.WHITE else black_bot
            
            # Set a timeout for bot moves (increased from 5 to 15 seconds)
            start_time = time.time()
            try:
                move = current_bot.get_move(board)
                
                # Check if move calculation took too long
                if time.time() - start_time > BOT_MOVE_TIMEOUT:
                    print("Bot timeout - using best move found so far")
                    # If the bot returned a move despite timeout, use it
                    # Otherwise fall back to a safe legal move
                    if not move:
                        legal_moves = board.get_legal_moves()
                        safe_moves = []
                        
                        # Try to find a capture or check
                        board_state = board.get_board_state()
                        for m in legal_moves:
                            if board_state.is_capture(m) or board_state.gives_check(m):
                                safe_moves.append(m)
                        
                        # If no captures or checks, use any legal move
                        if not safe_moves:
                            safe_moves = legal_moves
                            
                        if safe_moves:
                            move = safe_moves[0]
                        else:
                            return 0  # Draw if no legal moves
                
                if move:
                    moves.append(move)
                    if not board.make_move(move):
                        print(f"Illegal move attempted: {move}")
                        # Forfeit the game if an illegal move is attempted
                        return -1 if board.get_board_state().turn == chess.WHITE else 1
                else:
                    # No legal moves (should be caught by is_game_over, but just in case)
                    break
            except Exception as e:
                print(f"Error during move calculation: {e}")
                # Forfeit the game if an error occurs
                return -1 if board.get_board_state().turn == chess.WHITE else 1
           
            move_count += 1
           
            # Implement a move limit to prevent infinite games
            if move_count > 200:
                return 0  # Draw by excessive moves
       
        # Determine game result
        if board.get_board_state().is_checkmate():
            return -1 if board.get_board_state().turn == chess.WHITE else 1  # Winner is opposite of current turn
        else:
            return 0  # Draw by stalemate, insufficient material, etc.
   
    def analyze_results(self, results):
        """Analyze the match results"""
        bot1_wins = sum(1 for r in results if r["winner"] == "bot1")
        bot2_wins = sum(1 for r in results if r["winner"] == "bot2")
        draws = sum(1 for r in results if r["winner"] == "draw")
       
        # Calculate performance metrics
        total_games = len(results)
        bot1_win_rate = bot1_wins / total_games if total_games > 0 else 0
        bot2_win_rate = bot2_wins / total_games if total_games > 0 else 0
        draw_rate = draws / total_games if total_games > 0 else 0
       
        # Calculate average game length
        game_lengths = [len(r["moves"]) for r in results]
        avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0
       
        return {
            "bot1_wins": bot1_wins,
            "bot2_wins": bot2_wins,
            "draws": draws,
            "bot1_win_rate": bot1_win_rate,
            "bot2_win_rate": bot2_win_rate,
            "draw_rate": draw_rate,
            "avg_game_length": avg_game_length
        }
   
    def update_best_bot(self, challenger, analysis):
        """Update the best bot if the challenger performed better"""
        # If challenger (bot2) won more than current best (bot1), adopt its parameters
        if analysis["bot2_win_rate"] > analysis["bot1_win_rate"]:
            print("Challenger performed better - updating best bot")
           
            # Copy challenger parameters to best bot
            if hasattr(challenger, 'piece_values') and hasattr(self.best_bot, 'piece_values'):
                self.best_bot.piece_values = copy.deepcopy(challenger.piece_values)
            
            if hasattr(challenger, 'endgame_piece_values') and hasattr(self.best_bot, 'endgame_piece_values'):
                self.best_bot.endgame_piece_values = copy.deepcopy(challenger.endgame_piece_values)
            
            if hasattr(challenger, 'max_depth') and hasattr(self.best_bot, 'max_depth'):
                self.best_bot.max_depth = challenger.max_depth
            
            if hasattr(challenger, 'endgame_threshold') and hasattr(self.best_bot, 'endgame_threshold'):
                self.best_bot.endgame_threshold = challenger.endgame_threshold
            
            # Add a small log to show what parameters changed
            self._log_parameter_changes(self.best_bot, challenger)
   
    def _log_parameter_changes(self, best_bot, challenger):
        """Log the parameter changes between best bot and challenger"""
        if hasattr(best_bot, 'piece_values') and hasattr(challenger, 'piece_values'):
            for piece, value in challenger.piece_values.items():
                print(f"Piece value {piece}: {value}")
        
        if hasattr(best_bot, 'max_depth') and hasattr(challenger, 'max_depth'):
            print(f"Search depth: {challenger.max_depth}")
        
        if hasattr(best_bot, 'endgame_threshold') and hasattr(challenger, 'endgame_threshold'):
            print(f"Endgame threshold: {challenger.endgame_threshold}")
   
    def save_best_bot(self, filepath):
        """Save the best bot's parameters to a file"""
        params = {}
       
        # Save piece values if they exist
        if hasattr(self.best_bot, 'piece_values'):
            # Convert chess.PIECE constants to strings for JSON serialization
            piece_values = {}
            for piece, value in self.best_bot.piece_values.items():
                piece_values[str(piece)] = value
            params['piece_values'] = piece_values
        
        # Save endgame piece values if they exist
        if hasattr(self.best_bot, 'endgame_piece_values'):
            endgame_piece_values = {}
            for piece, value in self.best_bot.endgame_piece_values.items():
                endgame_piece_values[str(piece)] = value
            params['endgame_piece_values'] = endgame_piece_values
        
        # Save other numeric parameters
        if hasattr(self.best_bot, 'max_depth'):
            params['max_depth'] = self.best_bot.max_depth
        
        if hasattr(self.best_bot, 'max_search_time'):
            params['max_search_time'] = self.best_bot.max_search_time
        
        if hasattr(self.best_bot, 'endgame_threshold'):
            params['endgame_threshold'] = self.best_bot.endgame_threshold
       
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
   
    def save_training_history(self, filepath):
        """Save the training history to a file"""
        history = {
            "performance_history": self.performance_history,
            "game_history": self.game_history[-100:]  # Save only the last 100 games to save space
        }
       
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2) 