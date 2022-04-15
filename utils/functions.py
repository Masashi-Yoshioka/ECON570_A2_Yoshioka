import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, t
from tqdm import tqdm
import statsmodels.api as sm


###################################################
######## Helper functions for Assignment 2 ########
###################################################


def fn_generate_corrmat(dim, corr):
    '''
    Generate a matrix with 1 in diagonal and with corr in off-diagonal
    
    dim: number of dimensions (int)
    corr: correlation coefficient (float)
    '''
    
    acc  = []
    for i in range(dim):
        row = np.ones([1, dim]) * corr
        row[0][i] = 1
        acc.append(row)
    
    return np.concatenate(acc, axis = 0)


def fn_generate_multnorm(nobs, nvar, corr):
    '''
    Generate variables from a multivariate normal distribution
    
    nobs: number of observations (int)
    nvar: number of variables (int)
    corr: correlation between variables (float)
    '''
    
    mu = np.zeros(nvar)
    std = (np.abs(np.random.normal(loc = 1., scale = .5, size = (nvar, 1))))**(1/2)
    
    # Generate random normal distribution
    acc = []
    for i in range(nvar):
        acc.append(np.random.normal(mu[i], std[i], nobs).reshape(nobs, -1))
    
    normvars = np.concatenate(acc, axis = 1)

    cov = fn_generate_corrmat(nvar, corr)
    L = np.linalg.cholesky(cov)

    X = np.transpose(np.dot(L, np.transpose(normvars)))

    return X


class my_DGP():
    '''
    class to generate data
    '''
    
    
    def __init__(self, nobs, conf = False, selec = False, overrep = False):
        '''
        nobs: number of observations (int)
        conf: if True, treatment is confounded (bool)
        selec: if True, there are selection biases (bool)
        overrep: if True, outcome is overrepresented at zero (bool)
        '''
        self.nobs = nobs
        self.conf = conf
        self.selec = selec
        self.overrep = overrep
    
    
    def random_treat(self, p):
        '''
        Generate a treatment dummy with random assignment
        
        p: proportion of treatment (float)
        '''
        
        nobs = self.nobs
        treated = random.sample(range(nobs), round(nobs * p))
        T = np.array([(1 if i in treated else 0) for i in range(nobs)]).reshape(-1, 1)

        return T
    
    
    def conf_treat(self, C, p):
        '''
        Generate a treatment dummy that depends on C (confounder)
        Use a sigmoid function to associate C with T
        
        C: matrix of confounders (array_like)
        p: probability threshold for being treated (float)
        '''
        
        # Normalize C
        C_scaled = StandardScaler().fit_transform(C)
        nobs, nvar = C_scaled.shape

        # Pick up coefficients of C from uniform distribution
        gamma = np.random.uniform(low = 0., high = 1., size = (nvar, 1))
        
        # Disturbance
        e = np.random.normal(loc = 0., scale = 1., size = (nobs, 1))
        
        # Define T using a sigmoid function
        p_logit = np.exp(C_scaled @ gamma + e)/(1. + np.exp(C_scaled @ gamma + e))
        T = (p_logit > p) * 1

        return T
    
    
    def generate(self, tau, nvar_cov = 10, corr_cov = 0.5, nvar_conf = 1, corr_conf = 0.5,
                 nvar_selec = 1, corr_selec = 0.5, p = 0.5, intercept = 1., output = False, file_name = None):
        '''
        Generate Y (outcome), T (treatment), X (covariates), C (confounders), S (selection biases) and G (group dummy; for Question 4)
        
        tau: treatment effect (float)
        nvar_cov: number of covariates (int)
        corr_cov: correlation between covariates (float)
        
        nvar_conf: number of confounders (int)
        corr_conf: correlation between confounders (float)
        
        nvar_selec: number of selection biases (int)
        corr_selec: correlation between selection biases (float)
        
        p: proportion of treatment under random assignment
           or probability threshold for being treated under confounded treatment (float)
        intercept: intercept in the equation for outcome (float)
        
        output: If True, the data will be downloaded as a csv file (bool)
        file_name: name of csv file (str)
        '''

        nobs = self.nobs; conf = self.conf; selec = self.selec; overrep = self.overrep
        
        # When outcomes are not overrepresented at zero
        if overrep == False:

            # Generate covariates
            X = fn_generate_multnorm(nobs, nvar_cov, corr_cov)

            # With confounders
            if conf == True:

                # Generate confounders
                C = fn_generate_multnorm(nobs, nvar_conf, corr_conf)

                # Generate a confounded treatment
                T = self.conf_treat(C, p)

                # Combine covariates and confounders
                X_all = np.concatenate([X, C], axis = 1)

            # Without confounders
            elif conf == False:

                # No confounders
                C = None

                # Generate a treatment under random assignment
                T = self.random_treat(p)

                # Store just covariates since there is no confounder
                X_all = X.copy()

            # With selection biases
            if selec == True:

                # Generate selection biases by S = gamma * T + epsilon where gamma is from U(0,1) and epsilon is from N(0,1)
                delta = np.random.uniform(low = 0., high = 1., size = (1, nvar_selec))
                epsilon = np.random.normal(loc = 0., scale = 1., size = (nobs, nvar_selec))
                S = T @ delta + epsilon

                # Compute the term of T and S
                TS = np.concatenate([T, S], axis = 1)
                coef_TS = np.ones([TS.shape[1], 1]) * tau/(np.sum(delta) + 1)
                term_TS = TS @ coef_TS

            # Without selection bias
            elif selec == False:
                S = None
                term_TS = tau * T
            
            # Number of covariates + confounders
            nvar_all = X_all.shape[1]

            # Generate coefficients of covariates and confounders
            beta = np.random.normal(loc = 1., scale = 1., size = (nvar_all, 1))

            # Generate disturbances
            e = np.random.normal(loc = 0., scale = 1., size = (nobs, 1))

            # Generate outcomes Y
            Y = intercept + term_TS + X_all @ beta + e
            
            # There is no group dummy
            G = None
        
        # When outcomes are overrepresenting at zero
        elif overrep == True:
            
            # Randomized treatment
            T = self.random_treat(p)
            
            # Randomly assign samples to 2 groups
            G = self.random_treat(p = 0.5)
            
            # Group 1: Participate regardless of the treatment status
            # Use error following gamma distribution so that Y is always positive
            error1 = np.random.gamma(shape = tau, scale = 1., size = nobs).reshape(-1, 1)
            Y_Group1 = intercept + tau * T + error1 # Note that it is always positive
            
            # Group 2: Participate if and only if treated
            # Use error following uniform distribution so that Y is always positive IF TREATED
            error2 = np.random.uniform(low = -tau * 0.5, high = tau * 0.5, size = nobs).reshape(-1, 1)
            Y_Group2 = (tau + error2) * T # Note that it is zero if and only if T = 0
            
            # Randomly assign samples to two groups
            Y = Y_Group1 * G + Y_Group2 * (1. - G)
            
            # For simplicity, no covariates/confounders/selection biases are assumed
            X = None; C = None; S = None
        
        
        # Download the data as .csv file
        if output == True:
            
            data = np.concatenate([Y, T], axis = 1)
            colnames = ['outcome', 'treatment']
            
            if overrep == False:
                
                # Add covariates
                data = np.concatenate([data, X], axis = 1)
                
                # Add the column names
                for i in range(X.shape[1]):
                    colnames += ['covariate ' + str(i + 1)]
                
                if conf == True:
                    
                    # Add confounders
                    data = np.concatenate([data, C], axis = 1)
                    
                    for i in range(C.shape[1]):
                        colnames += ['confounder ' + str(i + 1)]
                    
                if selec == True:
                    
                    # Add selection biases
                    data = np.concatenate([data, S], axis = 1)
                    
                    for i in range(S.shape[1]):
                        colnames += ['selection bias ' + str(i + 1)]
                    
            elif overrep == True:
                
                # Define a participation
                Z = (Y > 0) * 1

                # Convert group dummy into group number
                Group = 1 * (G == 1) + 2 * (G == 0)

                # Add participation and group number
                data = np.concatenate([data, Z, Group], axis = 1)

                colnames += ['participation', 'group']
            
            # Convert the data into a dataframe
            df = pd.DataFrame(data)
            
            # Change the column names
            df.columns = colnames
            
            # Output the dataframe
            path = './data/' + file_name
            df.to_csv(path, index = False)
            
        
        return Y, T, X, C, S, G
    
    
    def plot_corr(self, tau, nvar_cov = 10, corr_cov = 0.5, nvar_conf = 1, corr_conf = 0.5,
                  nvar_selec = 1, corr_selec = 0.5, p = 0.5, intercept = 1., R = 2000, title_size = 12):
        '''
        Plot correlation coefficients between variables by replicating R times (Monte Carlo simulation)
                
        tau: treatment effect (float)
        nvar_cov: number of covariates (int)
        corr_cov: correlation between covariates (float)
        
        nvar_conf: number of confounders (int)
        corr_conf: correlation between confounders (float)
        
        nvar_selec: number of selection biases (int)
        corr_selec: correlation between selection biases (float)
        
        p: proportion of treatment under random assignment
           or probability threshold for being treated under confounded treatment (float)        
        intercept: intercept in the equation for outcome (float)

        R: number of replications (int)
        title_size: font size of the titles of the plots (float)
        '''     
        
        conf = self.conf; selec = self.selec; overrep = self.overrep
        
        # Store the replicated correlations
        corrYTs = []; corrYXs = []; corrTXs = []; corrXXs = []
        corrTCs = []; corrYCs = []; corrTSs = []; corrYSs = []

        # Replicate data
        for r in tqdm(range(R)):

            # Generate data
            Y, T, X, C, S, _ = self.generate(tau, nvar_cov, corr_cov, nvar_conf, corr_conf, nvar_selec, corr_selec, p, intercept)

            # Calculate correlation coefficients
            corrYT, _ = pearsonr(Y.ravel(), T.ravel())
            corrYTs += [corrYT]

            if overrep == False:
                corrYX, _ = pearsonr(Y.ravel(), X[:, 0].ravel())
                corrYXs += [corrYX]
                corrTX, _ = pearsonr(T.ravel(), X[:, 0].ravel())
                corrTXs += [corrTX]
                corrXX, _ = pearsonr(X[:, 0].ravel(), X[:, 1].ravel())
                corrXXs += [corrXX]

            if conf == True:
                corrTC, _ = pearsonr(T.ravel(), C[:, 0].ravel())
                corrTCs += [corrTC]
                corrYC, _ = pearsonr(Y.ravel(), C[:, 0].ravel())
                corrYCs += [corrYC]
                
            if selec == True:
                corrTS, _ = pearsonr(T.ravel(), S[:, 0].ravel())
                corrTSs += [corrTS]
                corrYS, _ = pearsonr(Y.ravel(), S[:, 0].ravel())
                corrYSs += [corrYS]

        # Overrepresented at zero
        if overrep == True:
            plt.hist(corrYTs)
            plt.title('Correlation between $Y$ and $T$', fontdict = {'fontsize': title_size})
        
        # Without confounders or selection biases
        elif (conf == False) & (selec == False):
            fig, axes = plt.subplots(2, 2, figsize = (12, 8))
            axes[0, 0].hist(corrYTs)
            axes[0, 0].set_title('Correlation between $Y$ and $T$', fontdict = {'fontsize': title_size})
            axes[0, 1].hist(corrYXs)
            axes[0, 1].set_title('Correlation between $Y$ and (one of) $\mathbf{X}$', fontdict = {'fontsize': title_size})
            axes[1, 0].hist(corrTXs)
            axes[1, 0].set_title('Correlation between $T$ and (one of) $\mathbf{X}$', fontdict = {'fontsize': title_size})
            axes[1, 1].hist(corrXXs)
            axes[1, 1].set_title('Correlation within $\mathbf{X}$', fontdict = {'fontsize': title_size})

            # Output the correlations
            Dict = {'corr_YT': corrYTs, 'corr_YX': corrYXs, 'corr_TX': corrTXs, 'corr_XX': corrXXs}
            df = pd.DataFrame(Dict)
            df.to_csv('./data/1_2_Correlations_RA.csv', index = False)
            
        # With confounders but no selection biases
        elif (conf == True) & (selec == False):
            fig, axes = plt.subplots(2, 2, figsize = (12, 8))
            axes[0, 0].hist(corrYTs)
            axes[0, 0].set_title('Correlation between $Y$ and $T$', fontdict = {'fontsize': title_size})
            axes[0, 1].hist(corrYCs)
            axes[0, 1].set_title('Correlation between $Y$ and $C$', fontdict = {'fontsize': title_size})
            axes[1, 0].hist(corrTXs)
            axes[1, 0].set_title('Correlation between $T$ and (one of) $\mathbf{X}$', fontdict = {'fontsize': title_size})
            axes[1, 1].hist(corrTCs)
            axes[1, 1].set_title('Correlation between $T$ and $C$', fontdict = {'fontsize': title_size})

            # Output the correlations
            Dict = {'corr_YT': corrYTs, 'corr_YC': corrYCs, 'corr_TX': corrTXs, 'corr_TC': corrTCs}
            df = pd.DataFrame(Dict)
            df.to_csv('./data/2_2_Correlations_Conf.csv', index = False)
            
        # With selection biases
        elif selec == True:
            fig, axes = plt.subplots(2, 2, figsize = (12, 8))
            axes[0, 0].hist(corrYTs)
            axes[0, 0].set_title('Correlation between $Y$ and $T$', fontdict = {'fontsize': title_size})
            axes[0, 1].hist(corrYXs)
            axes[0, 1].set_title('Correlation between $Y$ and (one of) $\mathbf{X}$', fontdict = {'fontsize': title_size})
            axes[1, 0].hist(corrTSs)
            axes[1, 0].set_title('Correlation between $T$ and $S$', fontdict = {'fontsize': title_size})
            axes[1, 1].hist(corrYSs)
            axes[1, 1].set_title('Correlation between $Y$ and $S$', fontdict = {'fontsize': title_size})

            # Output the correlations
            Dict = {'corr_YT': corrYTs, 'corr_YX': corrYXs, 'corr_TS': corrTSs, 'corr_YS': corrYSs}
            df = pd.DataFrame(Dict)
            df.to_csv('./data/3_2_Correlations_Selec.csv', index = False)
        
        plt.show()


def fn_tauhat(Y, T, X, C, S, X_control, C_control, S_control, COP):
    '''
    Estimate ATE using OLS
    
    Y: outcome (array_like)
    T: treatment (array_like)
    X: covariates (array_like)
    C: confounders (array_like)
    S: selection biases (array_like)
    X_control: If True, X is controlled in the OLS (bool)
    C_control: If True, C is controlled in the OLS (bool)
    S_control: If True, S is controlled in the OLS (bool)
    COP: If True, observations with zero outcomes are discarded (bool)
    '''
    
    # Define explanatory variables
    intercept = np.ones([T.shape[0], 1])
    covars = np.concatenate([intercept, T], axis = 1)
    
    # When controlling for covariates
    if X_control == True:
        covars = np.concatenate([covars, X], axis = 1)
    
    # When controlling for confounders
    if C_control == True: 
        covars = np.concatenate([covars, C], axis = 1)
    
    # When controlling for selection biases
    if S_control == True:
        covars = np.concatenate([covars, S], axis = 1)
    
    # When discarding zero outcomes (conditional on positives)
    if COP == True:
        Z = (Y > 0) * 1
        covars = np.concatenate([covars, Z], axis = 1)
    
    # Run OLS
    n, p = covars.shape
    res = sm.OLS(Y, covars).fit()
    tauhat = res.params[1]
    se_tauhat = res.HC1_se[1]
    dof = n - p

    return tauhat, se_tauhat, dof


def fn_bias_rmse_size(tau, tauhats, se_tauhats, dof, alpha):
    '''
    Calculate bias, RMSE and size from estimates of ATE
    
    tau: true value of ATE (int)
    tauhats: estimates of ATE (array_like)
    se_tauhats: standard errors of estimates of ATE (array_like)
    dof: degrees of freedom for t-test (int)
    alpha: significance level for t-test (float)
    '''
    
    tau0 = tau * np.ones(tauhats.shape) # True value of parameter
    
    # Bias
    b = tauhats - tau0
    bias = np.mean(b)
    
    # RMSE
    rmse = np.sqrt(np.mean(b**2))
    
    # Size
    tval = b/se_tauhats
    cval = np.abs(t.ppf(q = alpha/2, df = dof))
    size = np.mean(1 * (np.abs(tval) > cval))
    
    return bias, rmse, size


def fn_experiment(tau, nvar_cov = 10, corr_cov = 0.5, conf = False, nvar_conf = 1, corr_conf = 0.5,
                  selec = False, nvar_selec = 1, corr_selec = 0.5, overrep = False, p = 0.5, intercept = 1., X_control = False,
                  C_control = False, S_control = False, COP = False, alpha = 0.05, Nrange = range(20, 1002, 2),
                  output = False, file_name = None):
    '''
    Generate data whose sample size ranges from 20 to 1000 (default)
    and get the estimates of ATE with its standard errors and confidence intervals

    tau: treatment effect (float)
    nvar_cov: number of covariates (int)
    corr_cov: correlation between covariates (float)

    conf: if True, treatment is confounded (bool)
    nvar_conf: number of confounders (int)
    corr_conf: correlation between confounders (float)

    selec: if True, there are selection biases (bool)
    nvar_selec: number of selection biases (int)
    corr_selec: correlation between selection biases (float)
    
    overrep: if True, outcome is overrepresented at zero (bool)

    p: proportion of treatment under random assignment
       or probability threshold for being treated under confounded treatment (float)
    intercept: intercept of the equation for outcome (float)

    X_control: If True, X is controlled in the OLS (bool)
    C_control: If True, C is controlled in the OLS (bool)
    S_control: If True, S is controlled in the OLS (bool)
    COP: If True, observations with zero outcomes are discarded (bool)

    alpha: significance level for confidence interval, i.e., (1 - alpha)*100% CI will be returned (float)

    Nrange: range of sample sizes (list)
    
    output: If True, the data will be downloaded as a csv file (bool)
    file_name: name of csv file (str)
    '''

    # Store values as lists
    n_values = []; tauhats = []; sehats = []; lb = []; ub = []

    # Estimate the ATE repeatedly for each samples size in Nrange
    for nobs in tqdm(Nrange):
        
        n_values += [nobs]

        # Generate data
        dgp = my_DGP(nobs, conf, selec, overrep)
        Y, T, X, C, S, _ = dgp.generate(tau, nvar_cov, corr_cov, nvar_conf, corr_conf, nvar_selec, corr_selec, p, intercept)

        # Estimate tauhats
        tauhat, se_tauhat, dof = fn_tauhat(Y, T, X, C, S, X_control, C_control, S_control, COP)
        cval = np.abs(t.ppf(q = alpha/2, df = dof))

        tauhats += [tauhat]
        sehats += [se_tauhat]
        lb += [tauhat - cval * se_tauhat]
        ub += [tauhat + cval * se_tauhat]
    
    # Output the data
    if output == True:
        Dict = {'sample size': n_values, 'tauhat': tauhats, 'sehat': sehats, 'lb': lb, 'ub': ub}
        df = pd.DataFrame(Dict)
        path = './data/' + file_name
        df.to_csv(path, index = False)

    return n_values, tauhats, sehats, lb, ub
    
    
def fn_plot_with_ci(tau, n_values, tauhats, lb, ub, caption, n_lim, alpha = 0.05):
    '''
    Plot the results of ATE estimates

    n_values: list of sample sizes (array_like)
    tauhats: list of estimates (array_like)
    lb: list of lower bounds of confidence intervals (array_like)
    ub: list of upper bounds of confidence intervals (array_like)
    caption: title of the plot (str)
    n_lim: range of sample sizes shown in the plot ((float, float))
    alpha: significance level for confidence interval (float)
    '''
    
    # Trim the values according to n_lim
    n_values_rev = []; tauhats_rev = []; lb_rev = []; ub_rev = []
    
    for i in range(len(n_values)):
        if (n_values[i] >= n_lim[0]) & (n_values[i] <= n_lim[1]):
            n_values_rev += [n_values[i]]
            tauhats_rev += [tauhats[i]]
            lb_rev += [lb[i]]
            ub_rev += [ub[i]]

    # Plot the results
    fig = plt.figure(figsize = (10, 6))
    plt.plot(n_values_rev, tauhats_rev, color = '#2774AE', label = '$\hat{\\tau}$')
    plt.xlabel('Sample Sizes')
    plt.ylabel('ATE Estimates ($\hat{\\tau}$)')
    plt.axhline(y = tau, color = '#990000', linestyle = '-', linewidth = 1,
                label = f'True $\\tau$ $\equiv$ {tau}')
    plt.title(f'{caption}')
    plt.fill_between(n_values_rev, lb_rev, ub_rev, alpha = 0.5, edgecolor = '#FFD100',
                     facecolor = '#FFD100', label = f'{(1 - alpha)*100:.1f}% CI')
    plt.legend()
    plt.show()


def fn_monte_carlo(tau, nvar_cov = 10, corr_cov = 0.5, conf = False, nvar_conf = 1, corr_conf = 0.5,
                   selec = False, nvar_selec = 1, corr_selec = 0.5, overrep = False, p = 0.5, intercept = 1., X_control = False,
                   C_control = False, S_control = False, COP = False, alpha = 0.05,
                   R = 2000, Nrange = (100, 1000), output = False, file_name = None):
    '''
    Implement Monte Carlo simulations and return the bias, RMSE and size
    
    tau: treatment effect (float)
    nvar_cov: number of covariates (int)
    corr_cov: correlation between covariates (float)

    conf: if True, treatment is confounded (bool)
    nvar_conf: number of confounders (int)
    corr_conf: correlation between confounders (float)

    selec: if True, there are selection biases (bool)
    nvar_selec: number of selection biases (int)
    corr_selec: correlation between selection biases (float)
    
    overrep: if True, outcome is overrepresented at zero (bool)

    p: proportion of treatment under random assignment
       or probability threshold for being treated under confounded treatment (float)
    intercept: intercept of the equation for outcome (float)

    X_control: If True, X is controlled in the OLS (bool)
    C_control: If True, C is controlled in the OLS (bool)
    S_control: If True, S is controlled in the OLS (bool)
    COP: If True, observations with zero outcomes are discarded (bool)

    alpha: significance level for confidence interval (float)

    R: number of iterations in Monte Carlo simulation (int)
    Nrange: range of sample sizes (list)
    
    output: If True, the data will be downloaded as a csv file (bool)
    file_name: name of csv file (str)
    '''

    estDict = {}

    for nobs in Nrange:
        
        # Store the estimates, standard errors and biases
        tauhats = []; sehats = []; biases = []
        
        # Monte Carlo simulation
        for r in tqdm(range(R)):
            
            # Generate data
            dgp = my_DGP(nobs, conf, selec, overrep)
            Y, T, X, C, S, _ = dgp.generate(tau, nvar_cov, corr_cov, nvar_conf, corr_conf, nvar_selec, corr_selec, p, intercept)
            
            # Estimate tauhats
            tauhat, se_tauhat, dof = fn_tauhat(Y, T, X, C, S, X_control, C_control, S_control, COP)
            tauhats += [tauhat]
            sehats += [se_tauhat]
            biases += [tauhat - tau]
            
        # Put the values into a dictionary
        estDict[nobs] = {
            'tauhat':np.array(tauhats).reshape(-1, 1),
            'sehat':np.array(sehats).reshape(-1, 1),
            'dof': dof
        }
        
        # Output the data
        if output == True:
            Dict = {'sample size': [nobs] * R, 'iteration': range(1, R + 1), 'tauhat': tauhats, 'sehat': sehats, 'bias': biases}
            df = pd.DataFrame(Dict)
            
            # Append the data
            if nobs == Nrange[0]: df_cum = df.copy()
            else: df_cum = df_cum.append(df, ignore_index = True)
            
    # Show the results
    for nobs, results in estDict.items():
        bias, rmse, size = fn_bias_rmse_size(tau, results['tauhat'], results['sehat'], results['dof'], alpha)
        print(f'N = {nobs}: bias = {bias:.6f}, RMSE = {rmse:.6f}, size = {size:.4f}')
        
    # Output the data
    if output == True:
        path = './data/' + file_name
        df_cum.to_csv(path, index = False)

