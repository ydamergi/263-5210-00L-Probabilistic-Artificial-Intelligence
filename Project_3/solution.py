import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
import scipy.stats as stat

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0
np.random.seed(42)

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.data_points =np.array([]).reshape(-1, domain.shape[0])
        self.data_points_x = np.array([]).reshape(-1, domain.shape[0])
        self.data_points_f = np.array([]).reshape(-1, domain.shape[0]) 
        self.data_points_v = np.array([]).reshape(-1, domain.shape[0])
        self.kernel_f = 0.5*kernels.Matern(length_scale=0.5,length_scale_bounds=(1e-07, 100000.0), nu=2.5)
        self.kernel_v = kernels.ConstantKernel(constant_value=1.5, constant_value_bounds = (1e-07, 100000.0)) + np.sqrt(2)*kernels.Matern(length_scale=0.5, length_scale_bounds=(1e-07, 100000.0), nu=2.5)
        self.gp_f_template = GaussianProcessRegressor(kernel=self.kernel_f, alpha=0.15**2, random_state=0)
        self.gp_v_template = GaussianProcessRegressor(kernel=self.kernel_v, alpha=0.0001**2, random_state=0)
        self.kappa = 2
        
        
    def get_initial_safe_point():
        """Return initial safe point"""
        x_domain = np.linspace(*domain[0], 4000)[:, None]
        c_val = np.vectorize(v)(x_domain)
        x_valid = x_domain[c_val > SAFETY_THRESHOLD]
        np.random.seed(SEED)
        np.random.shuffle(x_valid)
        x_init = x_valid[0]
        return x_init
        


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if(self.data_points_x.size == 0):
            initial_guess = get_initial_safe_point()
            next_recommendation = initial_guess
        else:
            next_recommendation = self.optimize_acquisition_function()
            if((next_recommendation==0.0) or (next_recommendation == 5.0)):
                luck = np.random.random_sample()
                if(luck > 0.6):
                    next_recommendation = 5*(np.random.random_sample())
            luck2 = np.random.random_sample()
            if(luck2>0.8):
                next_recommendation = 5*(np.random.random_sample())
        print("the next reco is")
        print(next_recommendation)
        return next_recommendation
        


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) *  np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(func =objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x): #implementing UCB as acquisition function
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        meanf, stdf = self.gp_f_template.predict(x.reshape(-1, domain.shape[0]), return_std=True)
        meanv, stdv = self.gp_v_template.predict(x.reshape(-1, domain.shape[0]), return_std=True)
        
        if(stdv == 0):
            if(meanv >= SAFETY_THRESHOLD):
                constrain = 0.95*(meanv - SAFETY_THRESHOLD)
            else :
                constrain = 0.05*(SAFETY_THRESHOLD - meanv)
        else:
            constrain = 1 - stat.norm.cdf(SAFETY_THRESHOLD, meanv, stdv)
            
            
        return float((meanf + self.kappa*stdf)*constrain)
        


    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        
        #append the new point to the list of data and update the gp.

        self.data_points_x = np.vstack((self.data_points_x, x))
        self.data_points_f = np.vstack((self.data_points_f, f))
        self.data_points_v = np.vstack((self.data_points_v, v))
       
        
        self.gp_f_template.fit(self.data_points_x, self.data_points_f)
        self.gp_v_template.fit(self.data_points_x, self.data_points_v)
        

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        #TODO: enter your code here
        print("la data est")
        print(self.data_points_x)
        point_with_correct_speed = []
        for i in range(len(self.data_points_x)):
            if(self.data_points_v[i]>SAFETY_THRESHOLD):
                point_with_correct_speed.append([self.data_points_x[i],self.data_points_f[i]]) 
        print("la data avec correct speed est")
        print(point_with_correct_speed)
        temp = 0
        indice = 0
        for i in range(len(point_with_correct_speed)):
            if(point_with_correct_speed[i][1] > temp):
                indice = i
                temp = point_with_correct_speed[i][1]
        return point_with_correct_speed[indice][0]


        


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0

def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
    
