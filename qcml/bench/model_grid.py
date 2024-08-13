model_grid = {
    'SVC': {
        'C': [0.1, 1, 10, 100],  # Regularization parameter
    },
    'KernelPerceptron': {
        'n_iter': [-1],  # Number of iterations
        'max_iter': [1000],  # Maximum number of iterations
        'shuffle': [True],  # Whether to shuffle the training data before each iteration
        'tol': [1e-3],  # Tolerance for stopping criteria
    }
}
