import argparse
import time
import os
import json
from bot import ChessBot
from self_play import SelfPlayTrainer
from load_bot import load_bot_from_params

def create_bot(params_file=None):
    """Create a new chess bot instance with default or loaded parameters"""
    if params_file:
        # Load parameters from a file
        return load_bot_from_params(params_file)
    # Create with default parameters
    return ChessBot()

def get_resume_info(output_base):
    """Check if there's a resume file for continuing training"""
    resume_file = f"{output_base}_resume.json"
    if os.path.exists(resume_file):
        try:
            with open(resume_file, 'r') as f:
                data = json.load(f)
                return data.get("current_iteration", 0)
        except:
            pass
    return 0

def main():
    parser = argparse.ArgumentParser(description="Train chess bot using self-play")
    parser.add_argument("--games", type=int, default=20, help="Number of games per iteration")
    parser.add_argument("--iterations", type=int, default=5, help="Number of training iterations")
    parser.add_argument("--output", type=str, default="trained_bot", help="Base filename for output files")
    parser.add_argument("--load", type=str, help="Load parameters from this file to continue training")
    parser.add_argument("--start-iteration", type=int, help="Starting iteration for resumed training")
    args = parser.parse_args()
    
    # Check if we can auto-resume
    start_iteration = 0
    if args.start_iteration is not None:
        start_iteration = args.start_iteration
    elif args.load:
        # Try to auto-detect resume point if loading parameters
        detected_iteration = get_resume_info(args.output)
        if detected_iteration > 0:
            start_iteration = detected_iteration
            print(f"Auto-detected resume point at iteration {start_iteration}")
    
    print(f"=== Chess Bot Self-Play Training ===")
    print(f"Games per iteration: {args.games}")
    print(f"Training iterations: {args.iterations}")
    print(f"Output file base: {args.output}")
    if args.load:
        print(f"Loading parameters from: {args.load}")
    if start_iteration > 0:
        print(f"Resuming from iteration: {start_iteration}/{args.iterations}")
    print("=" * 40)
    
    # Create bot constructor function that loads from file if specified
    def bot_constructor():
        return create_bot(args.load)
    
    # Create the trainer with resume support
    trainer = SelfPlayTrainer(
        bot_constructor=bot_constructor,
        games_per_iteration=args.games,
        iterations=args.iterations,
        start_iteration=start_iteration
    )
    
    # Start training
    start_time = time.time()
    print("Starting self-play training...")
    best_bot = trainer.train(output_base=args.output)
    end_time = time.time()
    
    # Save results
    bot_params_file = f"{args.output}_params.json"
    history_file = f"{args.output}_history.json"
    
    trainer.save_best_bot(bot_params_file)
    trainer.save_training_history(history_file)
    
    # Show summary
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Total games played: {args.games * (args.iterations - start_iteration)}")
    print(f"Best bot parameters saved to: {bot_params_file}")
    print(f"Training history saved to: {history_file}")
    
    # Show final parameters
    print("\nFinal parameters:")
    if hasattr(best_bot, 'max_depth'):
        print(f"Search depth: {best_bot.max_depth}")
    if hasattr(best_bot, 'endgame_threshold'):
        print(f"Endgame threshold: {best_bot.endgame_threshold}")
    if hasattr(best_bot, 'piece_values'):
        print("Piece values:")
        for piece, value in best_bot.piece_values.items():
            print(f"  {piece}: {value:.1f}")
            
    print("\nTo play against the trained bot:")
    print(f"python game.py --white --black-bot {bot_params_file}  # Play as white")
    print(f"python game.py --black --white-bot {bot_params_file}  # Play as black")

if __name__ == "__main__":
    main() 