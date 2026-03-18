import csv
from bayes_opt import BayesianOptimization

def get_next_parameters(history_file):
    #Define the range of parameters to optimize
    pbounds = {
        "mix_ratio": (0.1, 0.9),
        "mix_time": (10, 60)
    }

    #Create Opimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )
    

    with open(history_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            params = {
                "mix_ratio": float(row["mix_ratio"]),
                "mix_time": float(row["mix_time"])
            }

            score = float(row["score"])

            optimizer.register(
                params=params,
                target=score
            )

    # Ask BO for next experiment
    next_point = optimizer.suggest()
    print("\nLoaded previous experiments.")
    next_parameters = next_point

    print("\nNext suggested parameters:")
    print("mix_ratio =", next_point["mix_ratio"])
    print("mix_time =", next_point["mix_time"])
    return next_point