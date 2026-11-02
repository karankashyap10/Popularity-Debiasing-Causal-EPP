
import argparse

from config import Config
from data.preprocess import prepare_ciao
from train import train


def main():
    parser = argparse.ArgumentParser(
        description="CausalEPP on Ciao + LightGCN"
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess data/Ciao/ratings.csv"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )

    args = parser.parse_args()
    config = Config()

    # Step 1: Update epochs if provided
    if args.epochs is not None:
        config.epochs = args.epochs

    # Step 2: Preprocess data if requested
    if args.preprocess:
        print("Starting preprocessing...")

        prepare_ciao(
            raw_file=config.raw_file,
            out_dir=config.data_dir,
            min_user=config.min_user_interactions,
            min_item=config.min_item_interactions,
            num_time_bins=config.num_time_bins,
            seed=config.seed,
            popularity_quantile=config.high_popularity_quantile
        )

        print("Preprocessing completed.")

    # Step 3: Train the model
    print("Starting CausalEPP training...")

    (
        model,
        test_metrics,
        global_popularity,
        expected_popularity
    ) = train(config)

    # Step 4: Display returned results
    print("\nTraining completed successfully!")

    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    print("\nGlobal item popularity:")
    print("Shape:", global_popularity.shape)
    print("First 10 values:", global_popularity[:10])

    print("\nExpected local popularity:")
    print("Shape:", expected_popularity.shape)
    print("First 10 values:", expected_popularity[:10])

    print("\nOutput directory:", config.output_dir)


if __name__ == "__main__":
    main()

