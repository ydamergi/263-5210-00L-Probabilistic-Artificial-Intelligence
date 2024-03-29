import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
import matplotlib.pyplot as plt
from matplotlib import cm

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
COST_W_UNDERPREDICT = 25.0
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 10.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
        self.random_state = 0
        # self.kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        # + WhiteKernel(noise_level=0.0, noise_level_bounds=(1e-01, 10.0))

        self.grp00 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)
        self.grp01 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)
        self.grp02 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)
        self.grp10 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)
        self.grp11 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)
        self.grp12 = GaussianProcessRegressor(kernel=self.kernel, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                                              n_restarts_optimizer=5,random_state= self.random_state)

        # TODO: Add custom initialization for your model here if necessary

    def make_predictions(self, test_features: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param test_features: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        print('Predicting on test features...')
        # TODO: Use your GP to estimate the posterior mean and stddev for each location here
        gp_mean = np.zeros(test_features.shape[0], dtype=float)
        gp_std = np.zeros(test_features.shape[0], dtype=float)

        for i in range(len(test_features)):
            if (test_features[i, 0] <= 1/3 )and (test_features[i, 1] <= 0.5):
                results = self.grp00.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]
                
            if ( 1/3 < test_features[i, 0] <= 2/3 ) and (test_features[i, 1] <= 0.5):
                results = self.grp01.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]
                
            if ( 2/3 < test_features[i, 0] <= 1.0 ) and (test_features[i, 1] <= 0.5):
                results = self.grp02.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]
                
            if (test_features[i, 0] <= 1/3 ) and (test_features[i, 1] > 0.5):
                results = self.grp10.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]
                 
            if ( 1/3 < test_features[i, 0] <= 2/3 ) and (test_features[i, 1] > 0.5):
                results = self.grp11.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]
                 
            if ( 2/3 < test_features[i, 0] <= 1.0 ) and (test_features[i, 1] > 0.5):
                results = self.grp12.predict(test_features[i].reshape(1, -1), True, False)
                gp_mean[i] = results[0][0]
                gp_std[i] = results[1][0]

        # gp_mean,gp_std = self.grp.predict(test_features,True,False)
        # print("newgpstd is")
        # print(gp_std)
        # print("newgpmean")
        # print(gp_mean)

        # TODO: Use the GP posterior to form your predictions here
        predictions = gp_mean + 0.2 * gp_std
        print("Prediction finished.")

        return predictions, gp_mean, gp_std

    def fitting_model(self, train_GT: np.ndarray, train_features: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_features: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_GT: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """

        print('Training model...')
        # TODO: Fit your model here
        train_features00 = []
        train_features01 = []
        train_features02 = []
        train_features10 = []
        train_features11 = []
        train_features12 = []
        
        train_GT00 = []
        train_GT01 = []
        train_GT02 = []
        train_GT10 = []
        train_GT11 = []
        train_GT12 = []

        for i in range(len(train_GT)):
            if (train_features[i, 0] <= 1/3) and (train_features[i, 1] <= 0.5):
                train_features00.append(train_features[i])
                train_GT00.append(train_GT[i])
                
            if (1/3 <train_features[i, 0] <= 2/3) and (train_features[i, 1] <= 0.5):
                train_features01.append(train_features[i])
                train_GT01.append(train_GT[i])
                
            if (2/3 < train_features[i, 0] <= 1.0) and (train_features[i, 1] <= 0.5):
                train_features02.append(train_features[i])
                train_GT02.append(train_GT[i])
                
            if (train_features[i, 0] <= 1/3) and (train_features[i, 1] > 0.5):
                train_features10.append(train_features[i])
                train_GT10.append(train_GT[i])
                
            if (1/3 <train_features[i, 0] <= 2/3) and (train_features[i, 1] > 0.5):
                train_features11.append(train_features[i])
                train_GT11.append(train_GT[i])
                
            if (2/3 < train_features[i, 0] <= 1.0) and (train_features[i, 1] > 0.5):
                train_features12.append(train_features[i])
                train_GT12.append(train_GT[i])

        self.grp00 = self.grp00.fit(train_features00, train_GT00)
        self.grp01 = self.grp01.fit(train_features01, train_GT01)
        self.grp02 = self.grp02.fit(train_features02, train_GT02)

        self.grp10 = self.grp10.fit(train_features10, train_GT10)
        self.grp11 = self.grp11.fit(train_features11, train_GT11)
        self.grp12 = self.grp12.fit(train_features12, train_GT12)
        
        print('Training finished.')
        pass


def cost_function(ground_truth: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask_1 = predictions < ground_truth
    weights[mask_1] = COST_W_UNDERPREDICT

    # Case ii): significant overprediction
    mask_2 = (predictions >= 1.2 * ground_truth)
    weights[mask_2] = COST_W_OVERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.make_predictions(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)

    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=cm.get_cmap()(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def main():
    # Load the training dateset and test features
    print("Loading data...")
    train_features = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_GT = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_features = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)
    print("Data loading finished.")

    # Fit the model
    model = Model()
    model.fitting_model(train_GT, train_features)

    # Predict on the test features
    predictions = model.make_predictions(test_features)
    print(predictions[1])

    if EXTENDED_EVALUATION:
        perform_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
