# main_v2.py
import torch
import logging
import os
import sys
import argparse

# Use the new, modified modules
from config_v2 import get_experiments, get_file_paths
from train_v2 import run_all_experiments
from utils_v2 import setup_logger, set_plot_style


def main():
    """Main execution function for the improved training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run the gesture recognition model training and evaluation pipeline.")
    parser.add_argument('--log-dir', type=str, default=None, help='Directory to store log files, overriding config.')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'], default=None,
                        help='Manually specify device (cuda, mps, cpu).')
    args = parser.parse_args()

    print("\n" + "=" * 25)
    print(">>> Running 【V16 Production Version】 Program <<<")
    print("=" * 25 + "\n")

    try:
        file_paths = get_file_paths()
        log_dir = args.log_dir if args.log_dir else file_paths.get("log_dir", "logs")
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        setup_logger(log_dir=log_dir)
        set_plot_style()

        if args.device:
            device = torch.device(args.device)
            logging.info(f"User has manually specified device: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info("CUDA backend detected. Using NVIDIA GPU for acceleration.")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("MPS backend detected. Using Apple Silicon GPU for acceleration.")
        else:
            device = torch.device("cpu")
            logging.info("No GPU detected. Using CPU for execution.")

        experiments, base_config, common_params = get_experiments()
        logging.info("Successfully loaded all experiment configurations and file paths.")

        # The main function now handles K-fold CV, TTA, and ensembling internally
        run_all_experiments(experiments, base_config, common_params, file_paths, device)

        logging.info("All experiments completed successfully.")

    except (FileNotFoundError, ValueError) as config_error:
        logging.error(f"Configuration or file path error: {config_error}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Specifically check for CUDA OOM error for a more informative message
        if 'CUDA out of memory' in str(e):
            logging.error(
                f"Critical Error: CUDA Out of Memory. This can happen with large models or long sequence lengths. "
                "Try reducing 'batch_size' or 'embed_dim' in 'config_v2.py'.", exc_info=True)
        else:
            logging.error(f"An uncaught critical error occurred during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()