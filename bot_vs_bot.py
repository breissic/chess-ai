import chess
import time
import random
import argparse
from board import ChessBoard
from bot import ChessBot
from load_bot import load_bot_from_params

# Increase timeout from 5 to 15 seconds
BOT_MOVE_TIMEOUT = 15.0

def play_game(white_bot, black_bot, max_moves=200, verbose=True):
    """Play a game between two bots without visualization"""
    board = ChessBoard()
    moves = []
    move_count = 0
    position_history = {}
    start_time = time.time()
    
    if verbose:
        print("Starting new game")
    
    while not board.is_game_over():
        # Check for draw conditions
        board_fen = board.get_board_state().fen().split(' ')[0]
        position_history[board_fen] = position_history.get(board_fen, 0) + 1
        
        if position_history[board_fen] >= 3:
            if verbose:
                print("Draw by repetition")
            return 0, moves  # Draw
        
        if board.get_board_state().halfmove_clock >= 100:
            if verbose:
                print("Draw by fifty-move rule")
            return 0, moves  # Draw
        
        # Get current bot
        current_bot = white_bot if board.get_board_state().turn else black_bot
        
        # Set timeout for bot's move (increased from 5 to 15 seconds)
        move_start = time.time()
        try:
            move = current_bot.get_move(board)
            
            # Enforce timeout
            if time.time() - move_start > BOT_MOVE_TIMEOUT:
                if verbose:
                    print("Bot timeout - using best move found so far")
                
                # If we have a move despite timeout, use it
                # Otherwise find a good move
                if not move:
                    board_state = board.get_board_state()
                    legal_moves = list(board_state.legal_moves)
                    
                    # Try to find captures or checks first
                    safe_moves = []
                    for m in legal_moves:
                        if board_state.is_capture(m) or board_state.gives_check(m):
                            safe_moves.append(m)
                    
                    # If no good moves found, use any legal move
                    if not safe_moves:
                        safe_moves = legal_moves
                        
                    if safe_moves:
                        move = safe_moves[0]  # Use first move as default
                    else:
                        if verbose:
                            print("No legal moves")
                        break
            
            if move:
                if verbose:
                    print(f"Move: {move}")
                moves.append(move)
                if not board.make_move(move):
                    if verbose:
                        print(f"Illegal move attempted: {move}")
                    return -1 if board.get_board_state().turn else 1  # Current player loses
            else:
                if verbose:
                    print("No move returned")
                break
        except Exception as e:
            if verbose:
                print(f"Error during move calculation: {e}")
            return -1 if board.get_board_state().turn else 1  # Current player loses
        
        move_count += 1
        
        # Implement move limit
        if move_count > max_moves:
            if verbose:
                print(f"Draw by move limit ({max_moves})")
            return 0, moves  # Draw
    
    # Game is over
    game_time = time.time() - start_time
    
    if board.get_board_state().is_checkmate():
        result = -1 if board.get_board_state().turn else 1
        if verbose:
            winner = "Black" if board.get_board_state().turn else "White"
            print(f"Checkmate! {winner} wins in {move_count} moves ({game_time:.1f}s)")
        return result, moves
    
    if verbose:
        print(f"Draw in {move_count} moves ({game_time:.1f}s)")
    return 0, moves  # Draw

def run_multiple_games(white_bot, black_bot, num_games=10, max_moves=200, verbose=True):
    """Run multiple games between two bots and report statistics"""
    white_wins = 0
    black_wins = 0
    draws = 0
    total_moves = 0
    start_time = time.time()
    
    for game_num in range(num_games):
        if verbose:
            print(f"\nGame {game_num+1}/{num_games}")
        
        # Alternate colors every other game for fairness
        if game_num % 2 == 0:
            result, moves = play_game(white_bot, black_bot, max_moves, verbose)
        else:
            result, moves = play_game(black_bot, white_bot, max_moves, verbose)
            result = -result  # Invert result since colors are swapped
        
        if result == 1:
            white_wins += 1
        elif result == -1:
            black_wins += 1
        else:
            draws += 1
        
        total_moves += len(moves)
    
    total_time = time.time() - start_time
    avg_moves = total_moves / num_games if num_games > 0 else 0
    
    print("\n===== Results =====")
    print(f"Games played: {num_games}")
    print(f"White wins: {white_wins} ({white_wins/num_games*100:.1f}%)")
    print(f"Black wins: {black_wins} ({black_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print(f"Average moves per game: {avg_moves:.1f}")
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Average time per game: {total_time/num_games:.1f} seconds")
    
    return {
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_moves": avg_moves,
        "total_time": total_time
    }

def main():
    parser = argparse.ArgumentParser(description="Run multiple bot vs bot chess games")
    parser.add_argument("--white", type=str, help="Path to parameters file for white bot")
    parser.add_argument("--black", type=str, help="Path to parameters file for black bot")
    parser.add_argument("--games", type=int, default=10, help="Number of games to play")
    parser.add_argument("--moves", type=int, default=200, help="Maximum moves per game")
    parser.add_argument("--quiet", action="store_true", help="Minimal output mode")
    args = parser.parse_args()
    
    # Load bots
    white_bot = load_bot_from_params(args.white) if args.white else ChessBot()
    black_bot = load_bot_from_params(args.black) if args.black else ChessBot()
    
    verbose = not args.quiet
    
    # Run games
    run_multiple_games(white_bot, black_bot, args.games, args.moves, verbose)

if __name__ == "__main__":
    main() 