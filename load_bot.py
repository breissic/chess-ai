import json
import chess
import argparse
from bot import ChessBot

def load_bot_from_params(params_file):
    """Load a chess bot with parameters from a JSON file"""
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Create a bot with default parameters
        bot = ChessBot()
        
        # Update piece values
        if 'piece_values' in params:
            # Convert string keys back to chess piece constants
            for piece_str, value in params['piece_values'].items():
                # Convert string representation to chess.PIECE constant
                piece = int(piece_str)
                bot.piece_values[piece] = value
                
        # Update endgame piece values
        if 'endgame_piece_values' in params:
            for piece_str, value in params['endgame_piece_values'].items():
                piece = int(piece_str)
                bot.endgame_piece_values[piece] = value
                
        # Update other parameters
        if 'max_depth' in params:
            bot.max_depth = params['max_depth']
            
        if 'max_search_time' in params:
            bot.max_search_time = params['max_search_time']
            
        if 'endgame_threshold' in params:
            bot.endgame_threshold = params['endgame_threshold']
        
        return bot
    except Exception as e:
        print(f"Error loading bot parameters: {e}")
        return ChessBot()  # Return default bot if loading fails

def main():
    parser = argparse.ArgumentParser(description="Load and display trained chess bot parameters")
    parser.add_argument("--params", type=str, required=True, help="Path to parameters JSON file")
    args = parser.parse_args()
    
    bot = load_bot_from_params(args.params)
    
    print("=== Loaded Chess Bot Parameters ===")
    
    # Display key parameters
    if hasattr(bot, 'max_depth'):
        print(f"Search depth: {bot.max_depth}")
        
    if hasattr(bot, 'max_search_time'):
        print(f"Max search time: {bot.max_search_time} seconds")
        
    if hasattr(bot, 'endgame_threshold'):
        print(f"Endgame threshold: {bot.endgame_threshold}")
    
    if hasattr(bot, 'piece_values'):
        print("\nPiece values:")
        piece_names = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King"
        }
        for piece, value in bot.piece_values.items():
            piece_name = piece_names.get(piece, f"Piece {piece}")
            print(f"  {piece_name}: {value:.1f}")
    
    if hasattr(bot, 'endgame_piece_values'):
        print("\nEndgame piece values:")
        for piece, value in bot.endgame_piece_values.items():
            piece_name = piece_names.get(piece, f"Piece {piece}")
            print(f"  {piece_name}: {value:.1f}")
    
    print("\nBot loaded successfully!")

if __name__ == "__main__":
    main() 