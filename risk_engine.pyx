# risk_engine.pyx
# Cython implementation of TVP-EVT/QuantileRegression model core computations

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, exp, sqrt, fabs, pow, sin, cos, INFINITY
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.string cimport memset, memcpy

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t INT_t

# Enable boundscheck=False and wraparound=False for performance
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class TVPEVTCore:
    """
    High-performance Cython implementation of TVP-EVT model core algorithms.
    """
    
    # Class variables
    cdef double innovation_variance_xi
    cdef double innovation_variance_beta  
    cdef double persistence_beta
    cdef int n_particles
    cdef double threshold_percentile
    
    # State arrays
    cdef double[:] xi_history
    cdef double[:] beta_history
    cdef double[:] threshold_history
    cdef double[:] returns_data
    cdef int n_obs
    cdef bint fitted
    
    def __init__(self, 
                 double innovation_variance_xi=0.01,
                 double innovation_variance_beta=0.01, 
                 double persistence_beta=0.95,
                 int n_particles=1000,
                 double threshold_percentile=0.95):
        
        self.innovation_variance_xi = innovation_variance_xi
        self.innovation_variance_beta = innovation_variance_beta
        self.persistence_beta = persistence_beta
        self.n_particles = n_particles
        self.threshold_percentile = threshold_percentile
        self.fitted = False
        self.n_obs = 0
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit(self, cnp.ndarray[DTYPE_t, ndim=1] returns):
        """
        Fit TVP-EVT model to returns data with optimized computations.
        """
        cdef int n = returns.shape[0]
        self.n_obs = n
        
        # Initialize memory views
        self.returns_data = returns
        self.xi_history = np.zeros(n, dtype=np.float64)
        self.beta_history = np.zeros(n, dtype=np.float64)
        self.threshold_history = np.zeros(n, dtype=np.float64)
        
        # Minimum window for stable estimation
        cdef int window_size = max(250, n // 4)
        cdef int t, start_idx, n_exceedances
        cdef double threshold, xi_t, beta_t
        cdef double[:] window_data
        cdef double[:] exceedances
        
        for t in range(window_size, n):
            start_idx = max(0, t - window_size)
            
            # Extract window data
            window_data = returns[start_idx:t]
            
            # Select threshold using Hill estimator
            threshold = self._select_threshold_hill(window_data)
            self.threshold_history[t] = threshold
            
            # Extract exceedances
            exceedances = self._extract_exceedances(window_data, threshold)
            n_exceedances = exceedances.shape[0]
            
            if n_exceedances < 10:
                # Use previous parameters or defaults
                if t > window_size:
                    xi_t = self.xi_history[t-1]
                    beta_t = self.beta_history[t-1]
                else:
                    xi_t = 0.1
                    beta_t = self._compute_std(window_data)
            else:
                # Fit GPD using maximum likelihood
                xi_t, beta_t = self._fit_gpd_mle(exceedances)
                
                # Apply time-varying evolution
                if t > window_size:
                    xi_t = self._evolve_xi(self.xi_history[t-1])
                    beta_t = self._evolve_beta(self.beta_history[t-1], beta_t)
            
            self.xi_history[t] = xi_t
            self.beta_history[t] = beta_t
        
        self.fitted = True
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _select_threshold_hill(self, double[:] data):
        """
        Select threshold using Hill estimator - optimized Cython version.
        """
        cdef int n = data.shape[0]
        cdef int min_k = max(10, int(n * (1 - self.threshold_percentile)))
        cdef int max_k = int(n * 0.05)
        
        # Sort data (use quickselect for better performance)
        cdef double[:] sorted_data = self._quicksort(data)
        
        # Find optimal k using Hill estimator stability
        cdef int k_opt = min_k + (max_k - min_k) // 2  # Simple heuristic
        cdef double threshold = sorted_data[n - k_opt - 1]
        
        return threshold
    
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    cdef double[:] _extract_exceedances(self, double[:] data, double threshold):
        """
        Extract exceedances above threshold - optimized.
        """
        cdef int n = data.shape[0]
        cdef int i, count = 0
        
        # Count exceedances first
        for i in range(n):
            if data[i] > threshold:
                count += 1
        
        # Allocate and fill exceedances array
        cdef cnp.ndarray[DTYPE_t, ndim=1] exceedances = np.zeros(count, dtype=np.float64)
        cdef double[:] exc_view = exceedances
        cdef int idx = 0
        
        for i in range(n):
            if data[i] > threshold:
                exc_view[idx] = data[i] - threshold
                idx += 1
        
        return exc_view
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (double, double) _fit_gpd_mle(self, double[:] exceedances):
        """
        Fit Generalized Pareto Distribution using Maximum Likelihood - optimized.
        """
        cdef int n = exceedances.shape[0]
        if n < 5:
            return 0.1, 1.0
        
        # Method of moments initial estimates
        cdef double mean_exc = self._compute_mean(exceedances)
        cdef double var_exc = self._compute_variance(exceedances, mean_exc)
        
        cdef double xi_init = -0.5 + mean_exc * mean_exc / var_exc
        cdef double beta_init = mean_exc * (1.0 + xi_init)
        
        # Constrain initial estimates
        xi_init = self._clip(xi_init, -0.5, 0.5)
        beta_init = max(beta_init, 0.001)
        
        # Newton-Raphson optimization
        cdef double xi_opt, beta_opt
        xi_opt, beta_opt = self._newton_raphson_gpd(exceedances, xi_init, beta_init)
        
        return xi_opt, beta_opt
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef (double, double) _newton_raphson_gpd(self, double[:] exceedances, 
                                            double xi_init, double beta_init):
        """
        Newton-Raphson optimization for GPD parameters.
        """
        cdef int n = exceedances.shape[0]
        cdef double xi = xi_init
        cdef double beta = beta_init
        cdef double tol = 1e-6
        cdef int max_iter = 50
        cdef int iter_count = 0
        
        cdef double loglik, loglik_prev = -INFINITY
        cdef double grad_xi, grad_beta, hess_xi_xi, hess_beta_beta, hess_xi_beta
        cdef double det, delta_xi, delta_beta
        cdef int i
        cdef double y_i, z_i
        
        while iter_count < max_iter:
            # Compute log-likelihood and derivatives
            loglik = 0.0
            grad_xi = 0.0
            grad_beta = 0.0
            hess_xi_xi = 0.0
            hess_beta_beta = 0.0
            hess_xi_beta = 0.0
            
            for i in range(n):
                y_i = xi * exceedances[i] / beta
                
                # Check validity
                if y_i <= -1.0:
                    return xi_init, beta_init  # Return initial values if invalid
                
                z_i = 1.0 + y_i
                
                # Log-likelihood
                loglik -= log(beta) + (1.0 + 1.0/xi) * log(z_i)
                
                # First derivatives
                grad_xi += (1.0/(xi*xi)) * log(z_i) - (1.0 + 1.0/xi) * exceedances[i] / (beta * z_i)
                grad_beta += -1.0/beta + (1.0 + 1.0/xi) * xi * exceedances[i] / (beta * beta * z_i)
                
                # Second derivatives (Hessian)
                hess_xi_xi += -2.0/(xi*xi*xi) * log(z_i) + 2.0/(xi*xi) * exceedances[i]/(beta * z_i) + (1.0 + 1.0/xi) * exceedances[i]*exceedances[i]/(beta*beta * z_i*z_i)
                hess_beta_beta += 1.0/(beta*beta) - 2.0*(1.0 + 1.0/xi)*xi*exceedances[i]/(beta*beta*beta * z_i) + (1.0 + 1.0/xi)*xi*xi*exceedances[i]*exceedances[i]/(beta*beta*beta*beta * z_i*z_i)
                hess_xi_beta += exceedances[i]/(beta*beta * z_i) - (1.0 + 1.0/xi)*xi*exceedances[i]*exceedances[i]/(beta*beta*beta * z_i*z_i)
            
            # Check convergence
            if fabs(loglik - loglik_prev) < tol:
                break
            
            loglik_prev = loglik
            
            # Newton-Raphson update
            det = hess_xi_xi * hess_beta_beta - hess_xi_beta * hess_xi_beta
            
            if fabs(det) < 1e-12:
                break  # Singular Hessian
            
            delta_xi = (hess_beta_beta * grad_xi - hess_xi_beta * grad_beta) / det
            delta_beta = (hess_xi_xi * grad_beta - hess_xi_beta * grad_xi) / det
            
            # Update with step size control
            xi -= 0.5 * delta_xi  # Conservative step size
            beta -= 0.5 * delta_beta
            
            # Ensure constraints
            xi = self._clip(xi, -0.5, 0.5)
            beta = max(beta, 0.001)
            
            iter_count += 1
        
        return xi, beta
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evolve_xi(self, double xi_prev):
        """
        Evolve xi parameter using random walk.
        """
        cdef double innovation = self._normal_random() * sqrt(self.innovation_variance_xi)
        cdef double xi_new = xi_prev + innovation
        return self._clip(xi_new, -0.5, 0.5)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _evolve_beta(self, double beta_prev, double beta_mle):
        """
        Evolve beta parameter using AR(1) in log space.
        """
        cdef double log_beta_prev = log(beta_prev)
        cdef double log_beta_mle = log(beta_mle)
        cdef double innovation = self._normal_random() * sqrt(self.innovation_variance_beta)
        
        cdef double log_beta_new = (self.persistence_beta * log_beta_prev + 
                                   (1.0 - self.persistence_beta) * log_beta_mle + 
                                   innovation)
        
        return exp(log_beta_new)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def forecast(self, int horizon=1, list confidence_levels=[0.95, 0.99]):
        """
        Generate forecasts using latest parameters.
        """
        # Declare all cdef variables at the beginning
        cdef double xi_current, beta_current, threshold_current
        cdef double alpha, var_forecast, es_forecast
        
        if not self.fitted or self.n_obs == 0:
            raise ValueError("Model must be fitted before forecasting")
        
        # Use latest parameters
        xi_current = self.xi_history[self.n_obs - 1]
        beta_current = self.beta_history[self.n_obs - 1] 
        threshold_current = self.threshold_history[self.n_obs - 1]
        
        # Compute forecasts for each confidence level
        forecasts = {}
        
        for conf_level in confidence_levels:
            alpha = 1.0 - conf_level
            
            var_forecast = self._compute_var(xi_current, beta_current, threshold_current, alpha)
            es_forecast = self._compute_expected_shortfall(xi_current, beta_current, threshold_current, alpha)
            
            forecasts[f'var_{int(conf_level*100)}'] = var_forecast
            forecasts[f'es_{int(conf_level*100)}'] = es_forecast
        
        # Add parameter forecasts
        forecasts['xi_forecast'] = xi_current
        forecasts['beta_forecast'] = beta_current
        forecasts['threshold'] = threshold_current
        
        return forecasts
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_var(self, double xi, double beta, double threshold, double alpha):
        """
        Compute Value at Risk using GPD.
        """
        cdef double var_val
        
        if fabs(xi) < 1e-6:  # Exponential case
            var_val = threshold - beta * log(1.0 - alpha)
        else:
            var_val = threshold + (beta / xi) * (pow(1.0 - alpha, -xi) - 1.0)
        
        return var_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_expected_shortfall(self, double xi, double beta, double threshold, double alpha):
        """
        Compute Expected Shortfall using GPD.
        """
        cdef double var_val = self._compute_var(xi, beta, threshold, alpha)
        cdef double es_val
        
        if fabs(xi) < 1e-6:  # Exponential case
            es_val = var_val + beta
        else:
            if xi >= 1.0:
                return INFINITY  # ES doesn't exist
            es_val = (var_val + beta - xi * threshold) / (1.0 - xi)
        
        return es_val
    
    # Utility functions
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:] _quicksort(self, double[:] data):
        """
        Optimized quicksort implementation.
        """
        cdef int n = data.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=1] sorted_array = np.copy(data)
        cdef double[:] sorted_view = sorted_array
        
        self._quicksort_inplace(sorted_view, 0, n-1)
        return sorted_view
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _quicksort_inplace(self, double[:] arr, int low, int high):
        """
        In-place quicksort implementation.
        """
        cdef int pi
        
        if low < high:
            pi = self._partition(arr, low, high)
            self._quicksort_inplace(arr, low, pi - 1)
            self._quicksort_inplace(arr, pi + 1, high)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _partition(self, double[:] arr, int low, int high):
        """
        Partition function for quicksort.
        """
        cdef double pivot = arr[high]
        cdef int i = low - 1
        cdef int j
        cdef double temp
        
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                temp = arr[i]
                arr[i] = arr[j]
                arr[j] = temp
        
        temp = arr[i + 1]
        arr[i + 1] = arr[high]
        arr[high] = temp
        
        return i + 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_mean(self, double[:] data):
        """
        Compute mean of array.
        """
        cdef int n = data.shape[0]
        cdef double sum_val = 0.0
        cdef int i
        
        for i in range(n):
            sum_val += data[i]
        
        return sum_val / n
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_variance(self, double[:] data, double mean_val):
        """
        Compute variance of array given mean.
        """
        cdef int n = data.shape[0]
        cdef double sum_sq = 0.0
        cdef double diff
        cdef int i
        
        for i in range(n):
            diff = data[i] - mean_val
            sum_sq += diff * diff
        
        return sum_sq / (n - 1)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_std(self, double[:] data):
        """
        Compute standard deviation of array.
        """
        cdef double mean_val = self._compute_mean(data)
        cdef double var_val = self._compute_variance(data, mean_val)
        return sqrt(var_val)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _clip(self, double value, double min_val, double max_val):
        """
        Clip value to range [min_val, max_val].
        """
        if value < min_val:
            return min_val
        elif value > max_val:
            return max_val
        else:
            return value
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _normal_random(self):
        """
        Generate normal random variable using Box-Muller transform.
        """
        cdef double u1, u2, z0
        
        u1 = (<double>rand()) / RAND_MAX
        u2 = (<double>rand()) / RAND_MAX
        
        z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265359 * u2)
        
        return z0
    
    def get_parameters(self):
        """
        Get current model parameters (Python interface).
        """
        if not self.fitted:
            return {}
        
        return {
            'xi': np.array(self.xi_history),
            'beta': np.array(self.beta_history), 
            'threshold': np.array(self.threshold_history),
            'n_observations': self.n_obs
        }
    
    def get_latest_parameters(self):
        """
        Get latest model parameters (Python interface).
        """
        if not self.fitted or self.n_obs == 0:
            return {}
        
        return {
            'xi': self.xi_history[self.n_obs - 1],
            'beta': self.beta_history[self.n_obs - 1],
            'threshold': self.threshold_history[self.n_obs - 1]
        }

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class QuantileRegressionCore:
    """
    High-performance Cython implementation of quantile regression optimization.
    """
    
    cdef double l1_regularization
    cdef int max_iter
    cdef double tolerance
    
    def __init__(self, double l1_regularization=0.01, int max_iter=1000, double tolerance=1e-6):
        self.l1_regularization = l1_regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def fit_quantile(self, cnp.ndarray[DTYPE_t, ndim=2] X, 
                     cnp.ndarray[DTYPE_t, ndim=1] y, 
                     double quantile):
        """
        Fit quantile regression using optimized interior point method.
        """
        cdef int n = X.shape[0]
        cdef int p = X.shape[1]
        
        # Add intercept column
        cdef cnp.ndarray[DTYPE_t, ndim=2] X_aug = np.column_stack([np.ones(n), X])
        cdef double[:, :] X_view = X_aug
        cdef double[:] y_view = y
        
        # Initialize parameters
        cdef cnp.ndarray[DTYPE_t, ndim=1] beta = np.zeros(p + 1, dtype=np.float64)
        cdef double[:] beta_view = beta
        
        # Run optimization
        self._interior_point_optimize(X_view, y_view, beta_view, quantile)
        
        return np.array(beta)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _interior_point_optimize(self, double[:, :] X, double[:] y, 
                                     double[:] beta, double quantile):
        """
        Interior point optimization for quantile regression.
        """
        cdef int n = X.shape[0]
        cdef int p = X.shape[1]
        cdef int iter_count = 0
        cdef double objective, objective_prev = 1e10
        
        # Working arrays
        cdef cnp.ndarray[DTYPE_t, ndim=1] residuals = np.zeros(n, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] weights = np.zeros(n, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] gradient = np.zeros(p, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=2] hessian = np.zeros((p, p), dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] delta_beta = np.zeros(p, dtype=np.float64)
        
        cdef double[:] res_view = residuals
        cdef double[:] w_view = weights
        cdef double[:] grad_view = gradient
        cdef double[:, :] hess_view = hessian
        cdef double[:] delta_view = delta_beta
        
        cdef int i, j, k
        cdef double step_size, max_step
        
        while iter_count < self.max_iter:
            # Compute residuals and objective
            objective = self._compute_objective(X, y, beta, res_view, quantile)
            
            # Check convergence
            if fabs(objective - objective_prev) < self.tolerance:
                break
            
            objective_prev = objective
            
            # Compute weights for IRLS
            self._compute_weights(res_view, w_view, quantile)
            
            # Compute gradient and Hessian
            self._compute_gradient_hessian(X, y, beta, res_view, w_view, 
                                         grad_view, hess_view, quantile)
            
            # Solve for Newton step
            if not self._solve_newton_step(hess_view, grad_view, delta_view):
                break  # Singular Hessian
            
            # Line search for step size
            step_size = self._line_search(X, y, beta, delta_view, quantile)
            
            # Update parameters
            for j in range(p):
                beta[j] -= step_size * delta_view[j]
            
            iter_count += 1
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _compute_objective(self, double[:, :] X, double[:] y, double[:] beta,
                                 double[:] residuals, double quantile):
        """
        Compute quantile regression objective function.
        """
        cdef int n = X.shape[0]
        cdef int p = X.shape[1]
        cdef int i, j
        cdef double pred, res, obj = 0.0
        
        # Compute residuals and quantile loss
        for i in range(n):
            pred = 0.0
            for j in range(p):
                pred += X[i, j] * beta[j]
            
            res = y[i] - pred
            residuals[i] = res
            
            # Quantile loss
            if res >= 0:
                obj += quantile * res
            else:
                obj += (quantile - 1.0) * res
        
        # Add L1 regularization (excluding intercept)
        for j in range(1, p):
            obj += self.l1_regularization * fabs(beta[j])
        
        return obj
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_weights(self, double[:] residuals, double[:] weights, double quantile):
        """
        Compute weights for iteratively reweighted least squares.
        """
        cdef int n = residuals.shape[0]
        cdef int i
        cdef double res, eps = 1e-6
        
        for i in range(n):
            res = residuals[i]
            if fabs(res) < eps:
                weights[i] = 1.0 / eps  # Avoid division by zero
            else:
                if res > 0:
                    weights[i] = quantile / fabs(res)
                else:
                    weights[i] = (1.0 - quantile) / fabs(res)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _compute_gradient_hessian(self, double[:, :] X, double[:] y, double[:] beta,
                                      double[:] residuals, double[:] weights,
                                      double[:] gradient, double[:, :] hessian, 
                                      double quantile):
        """
        Compute gradient and Hessian for Newton step.
        """
        cdef int n = X.shape[0]
        cdef int p = X.shape[1]
        cdef int i, j, k
        cdef double w_i, x_ij, x_ik
        
        # Reset gradient and Hessian
        for j in range(p):
            gradient[j] = 0.0
            for k in range(p):
                hessian[j, k] = 0.0
        
        # Compute weighted gradient and Hessian
        for i in range(n):
            w_i = weights[i]
            
            for j in range(p):
                x_ij = X[i, j]
                
                # Gradient
                if residuals[i] >= 0:
                    gradient[j] -= quantile * x_ij
                else:
                    gradient[j] -= (quantile - 1.0) * x_ij
                
                # Hessian (approximate)
                for k in range(j, p):
                    x_ik = X[i, k]
                    hessian[j, k] += w_i * x_ij * x_ik
                    if j != k:
                        hessian[k, j] = hessian[j, k]  # Symmetric
        
        # Add L1 regularization to gradient and Hessian
        for j in range(1, p):  # Skip intercept
            if beta[j] > 0:
                gradient[j] += self.l1_regularization
            elif beta[j] < 0:
                gradient[j] -= self.l1_regularization
            
            # Add small diagonal term for numerical stability
            hessian[j, j] += 1e-8
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _solve_newton_step(self, double[:, :] hessian, double[:] gradient, 
                               double[:] delta):
        """
        Solve Hessian * delta = -gradient using Cholesky decomposition.
        """
        cdef int p = hessian.shape[0]
        cdef int i, j, k
        cdef double sum_val, diag_val
        cdef int max_iter = 10
        cdef int iter_count = 0
        cdef double residual_norm, old_delta
        
        # Simple diagonal preconditioning for numerical stability
        for i in range(p):
            if hessian[i, i] < 1e-12:
                hessian[i, i] = 1e-6
        
        # Use simplified Gauss-Seidel iteration for robustness
        # Initialize delta
        for i in range(p):
            delta[i] = 0.0
        
        iter_count = 0
        while iter_count < max_iter:
            residual_norm = 0.0
            
            for i in range(p):
                sum_val = 0.0
                for j in range(p):
                    if i != j:
                        sum_val += hessian[i, j] * delta[j]
                
                old_delta = delta[i]
                delta[i] = (-gradient[i] - sum_val) / hessian[i, i]
                
                residual_norm += (delta[i] - old_delta) * (delta[i] - old_delta)
            
            if sqrt(residual_norm) < 1e-6:
                break
            
            iter_count += 1
        
        return True
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _line_search(self, double[:, :] X, double[:] y, double[:] beta,
                           double[:] delta, double quantile):
        """
        Backtracking line search for step size.
        """
        cdef double alpha = 1.0
        cdef double rho = 0.5
        cdef double c1 = 1e-4
        cdef int max_iter = 20
        cdef int iter_count = 0
        
        cdef int p = beta.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=1] beta_new = np.zeros(p, dtype=np.float64)
        cdef cnp.ndarray[DTYPE_t, ndim=1] residuals_dummy = np.zeros(X.shape[0], dtype=np.float64)
        cdef double[:] beta_new_view = beta_new
        cdef double[:] res_dummy_view = residuals_dummy
        
        cdef double obj_current = self._compute_objective(X, y, beta, res_dummy_view, quantile)
        cdef double obj_new
        cdef int j
        
        while iter_count < max_iter:
            # Compute new beta
            for j in range(p):
                beta_new_view[j] = beta[j] - alpha * delta[j]
            
            # Compute new objective
            obj_new = self._compute_objective(X, y, beta_new_view, res_dummy_view, quantile)
            
            # Check Armijo condition (simplified)
            if obj_new <= obj_current + c1 * alpha * self._dot_product(delta, delta):
                break
            
            alpha *= rho
            iter_count += 1
        
        return alpha
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double _dot_product(self, double[:] a, double[:] b):
        """
        Compute dot product of two vectors.
        """
        cdef int n = a.shape[0]
        cdef double result = 0.0
        cdef int i
        
        for i in range(n):
            result += a[i] * b[i]
        
        return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef class FeatureEngineeringCore:
    """
    High-performance feature engineering for quantile regression.
    """
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_rolling_features(self, cnp.ndarray[DTYPE_t, ndim=1] returns, 
                                int window_size):
        """
        Compute rolling statistical features efficiently.
        """
        cdef int n = returns.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=2] features = np.zeros((n, 6), dtype=np.float64)
        cdef double[:, :] feat_view = features
        cdef double[:] ret_view = returns
        
        cdef int i, j, start_idx
        cdef double window_mean, window_var, window_skew, window_kurt
        cdef double sum_val, sum_sq, sum_cube, sum_quad
        cdef double centered, std_val
        
        for i in range(n):
            start_idx = max(0, i - window_size + 1)
            
            # Compute basic statistics
            sum_val = 0.0
            for j in range(start_idx, i + 1):
                sum_val += ret_view[j]
            
            window_mean = sum_val / (i - start_idx + 1)
            
            # Compute higher moments
            sum_sq = 0.0
            sum_cube = 0.0 
            sum_quad = 0.0
            
            for j in range(start_idx, i + 1):
                centered = ret_view[j] - window_mean
                sum_sq += centered * centered
                sum_cube += centered * centered * centered
                sum_quad += centered * centered * centered * centered
            
            window_var = sum_sq / (i - start_idx + 1)
            std_val = sqrt(window_var)
            
            if std_val > 1e-10:
                window_skew = (sum_cube / (i - start_idx + 1)) / (std_val * std_val * std_val)
                window_kurt = (sum_quad / (i - start_idx + 1)) / (std_val * std_val * std_val * std_val)
            else:
                window_skew = 0.0
                window_kurt = 3.0
            
            # Store features
            feat_view[i, 0] = window_mean
            feat_view[i, 1] = std_val * sqrt(252.0)  # Annualized volatility
            feat_view[i, 2] = window_skew
            feat_view[i, 3] = window_kurt
            feat_view[i, 4] = fabs(ret_view[i])  # Absolute return
            feat_view[i, 5] = ret_view[i] * ret_view[i]  # Squared return
        
        return features
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_seasonal_features(self, int n_obs, int n_harmonics):
        """
        Compute harmonic seasonal features efficiently.
        """
        cdef cnp.ndarray[DTYPE_t, ndim=2] seasonal = np.zeros((n_obs, 2 * n_harmonics), dtype=np.float64)
        cdef double[:, :] seas_view = seasonal
        
        cdef int i, k, feat_idx
        cdef double t_norm, freq
        cdef double pi = 3.14159265359
        
        for i in range(n_obs):
            t_norm = <double>i / 252.0  # Normalize by trading year
            
            feat_idx = 0
            for k in range(1, n_harmonics + 1):
                freq = 2.0 * pi * k * t_norm
                seas_view[i, feat_idx] = sin(freq)
                seas_view[i, feat_idx + 1] = cos(freq)
                feat_idx += 2
        
        return seasonal
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_autoregressive_features(self, cnp.ndarray[DTYPE_t, ndim=1] returns, 
                                       int max_lag):
        """
        Compute autoregressive features efficiently.
        """
        cdef int n = returns.shape[0]
        cdef cnp.ndarray[DTYPE_t, ndim=2] ar_features = np.zeros((n, max_lag + 2), dtype=np.float64)
        cdef double[:, :] ar_view = ar_features
        cdef double[:] ret_view = returns
        
        cdef int i, lag
        
        for i in range(n):
            # Lagged returns
            for lag in range(1, max_lag + 1):
                if i >= lag:
                    ar_view[i, lag - 1] = ret_view[i - lag]
            
            # Lagged absolute return
            if i >= 1:
                ar_view[i, max_lag] = fabs(ret_view[i - 1])
            
            # Lagged squared return
            if i >= 1:
                ar_view[i, max_lag + 1] = ret_view[i - 1] * ret_view[i - 1]
        
        return ar_features