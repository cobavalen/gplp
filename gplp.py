from scipy import special
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b as fmin  # minimizar
import time  # medir tiempo
from numba import jit  
from scipy.fftpack import fft
from statsmodels.stats.correlation_tools import cov_nearest
import seaborn as sb
import scipy.signal as sgnl
from scipy import signal
import pandas as pd
import pylab as plot
from sklearn.metrics import mean_squared_error

class GPLP:
    
    # Class Attribute none yet

    # Initializer / Instance Attributes
    
    def __init__(self, space_input, space_output, window, grid_num = 5000):
        #Raw data and important values
        
        #Time domain
        self.offset = np.median(space_input)
        self.x = space_input - self.offset
        self.y = space_output
        self.Nx = len(self.x)
        self.time = np.linspace(np.min(self.x), np.max(self.x), grid_num) 
        self.T = np.abs(np.max(self.x) - np.min(self.x))
        self.grid_num = grid_num
        
        #Freq Domain 
        self.spectra_domain =np.linspace(0.0, self.Nx/(2.0*self.T), int(self.Nx/2) )
        self.spectra = 2.0/self.Nx * np.abs(fft(self.y)[:int(self.Nx/2)])
        
        #Parameters 
        self.window = window        
        self.sigma = 3*np.std(self.y)
        self.gamma = 1/2/((np.max(self.x)-np.min(self.x))/self.Nx)**2
        self.noise = np.std(self.y)/10
                
        #Post training
        
        self.error_bar = None
        self.Covariance = None
        self.filtered = None #Time Domain
        self.filt_spect_dom =np.linspace(0.0, self.grid_num/(2.0*self.T), int(self.grid_num/2) )
        self.filt_spect = None #Freq Domain 
                                                  
        
        #Numerical bullshit
        
        self.min_noise = 0.005
        
    def like_SE(self,theta):
        
        sigma_noise, gamma_1, sig_1 = np.exp(theta)
        Gram = K_SE(
            self.x, self.x, gamma=gamma_1,
            sigma=sig_1) + sigma_noise**2 * np.identity(self.Nx) + self.min_noise*np.identity(self.Nx)
        # inverse with cholesky
        cGg = np.linalg.cholesky(Gram)
        invGram = np.linalg.inv(cGg.T) @ np.linalg.inv(cGg)
        # nll
        nll = 2 * np.log(np.diag(cGg)).sum() + y.T @ (invGram @ y) 
        return 0.5 * nll + 0.5 * len(self.y) * np.log(2 * np.pi)

        
    def train(self):
        # fixed args of function
        args = (self.y, self.x)

        # initial point
        params0 = np.asarray([self.noise, self.gamma, self.sigma])
        X0 = np.log(params0)

        print('Condicion inicial optimizador: ', params0)

        time_GP = time.time()
        X_opt, f_GP, data = fmin(
            self.like_SE,
            X0,
            None, 
            approx_grad=True)
        #   disp=1,
         #   factr=0.00000001 / (2.22E-12),
          #  maxiter=1000)
        time_GP = time.time() - time_GP
        print("Tiempo entrenamiento {:.4f} (s)".format(time_GP))

        sigma_n_GP_opt, gamma_opt, sigma_opt = np.exp(X_opt)
        print('Hiperparametros encontrados: ', np.exp(X_opt), 'NLL: ', f_GP)

        self.sigma =np.exp(X_opt)[2]
        self.gamma= np.exp(X_opt)[1]
        self.noise = np.exp(X_opt)[0] + self.min_noise
        
    
    def kernel_conv(self, flag): #a=1/l , b= factor del sinc, c= |t1-t2| , sigma es una constante multiplicando todo
    #b es window, 1/gamma es l, t es np.abs(np.subtract.outer(self.x, self.x)) y sigma es self.sigma
    
        if type(self.x) == float:
            a = self.gamma
            C = self.sigma*0.5*np.exp(-(self.x**2)*self.gamma)
            if np.abs(self.x) <26*np.sqrt(1/self.gamma):
                return 0
            s1 = (1/(np.sqrt(1/self.gamma)))*(np.pi*self.window*(1/self.gamma) -self.x*1j) 
            s2 = s1.conjugate()
            s3 = C* (special.erf(s1).real + special.erf(s2))
            return s3.real

        else:
            if flag==0:
                t= np.abs(np.subtract.outer(self.x, self.x))
                n = t.shape
            elif flag == 1:
                t= np.abs(np.subtract.outer(self.time, self.time)) 
                n= t.shape
            elif flag == 2:
                t = np.subtract.outer(self.x, self.time)
                n = t.shape
            resp = np.zeros(shape=n)
            C = self.sigma*0.5*np.exp(-(t**2)*self.gamma)
            indexes =  np.abs(t)  <26*np.sqrt(1/self.gamma)

            s1 = (1/(np.sqrt(1/self.gamma)))*(np.pi*self.window*(1/self.gamma) - t[indexes]*1j) 
            s2 = s1.conjugate()
            s3 = C[indexes]* (special.erf(s1) + special.erf(s2))
            resp[indexes] = s3.real

            return resp
        
        
    def gauss_low_filter(self):
        noise_var = 1 #conditioning propblems
        
        K_ry = self.kernel_conv(flag=2)
        K_ry = K_ry.T
        
        K_y = K_SE(self.x, self.x, self.gamma, self.sigma) + self.noise*np.eye(self.Nx) 
        self.filtered= np.matmul(K_ry,np.linalg.inv(K_y)).dot(self.y)
        M = K_ry@np.linalg.solve(K_y+noise_var*np.eye(m), K_ry.T) #Here we use noise_var as there may be conditioning problems
        
        K_rr = self.kernel_conv(flag = 1)
        # print(np.linalg.eigvals(K_rr).min())
        K_rr_parche = cov_nearest(K_rr) +1e-8*np.eye(len(self.time))
        self.Covariance = (K_rr_parche - M)
        self.error_bar =2*np.sqrt(np.diag(self.Covariance))   
        
        self.filt_spect = 2.0/self.grid_num * np.abs(fft(self.filtered)[:int(self.grid_num/2)])                                        
    
    def plot_spectra(self):
        plt.plot(signal.spectra_domain, signal.spectra,'g', label = 'Ground truth spectrum')
        plt.plot(signal.filt_spect_dom, signal.filt_spect,'r', label = 'GPLP')
        
    def plot_signal(self):
        if self.filtered is None:
            plt.plot(self.x, self.y)
        else:
            plt.plot(self.x, self.y)
            plt.plot(self.time, self.filtered)

        
        
        
def K_SE(a, b, gamma=1. / 2, sigma=1):
    """
    Squared Exponential kernel
    Returns the gram matrix given by the kernel
    k(a,b) = sigma**2*exp(-gamma*(a-b)**2)
    Note that: gamma = 1 /(2*lengthscale**2)
    
    Inputs:
    a:(numpy array)   Array length n_a with first input
    b:(numpy array)   Array length n_b with second input
    gamma:(float)     Kernel parameter
    sigma:(float)     Kernel parameter, signal variance

    Returns:
    (numpy array) n_a X n_b gram matrix where element
    [i,j] = k(a[i], b[j])
    """
    # transform to array if a single point
    if np.ndim(a) == 0: a = np.array([a])
    if np.ndim(b) == 0: b = np.array([b])
    # create matrix
    gram = np.zeros((len(a), len(b)))
    # compute
    gram = sigma*np.exp(-gamma * (np.subtract.outer(a,b))**2)
    # condition if a single point
    if (len(a) == 1) or (len(b) == 1):
        return gram.reshape(-1)
    else:
        return gram
		
def synth_data(low_freqs, high_freqs, coefs_low=None, coefs_high= None, L= 40, n=1000, sample = .25, noise_var = 0.01, random_sampling = False): 
    m=int(n*sample) #Puntos a samplear

    delta = int(n/m)


    x = np.linspace(-L,L,n)
    
    if coefs_low is None:
        coefs_low = np.ones(len(low_freqs))
    if coefs_high is None:
        coefs_high = np.ones(len(high_freqs))
    f_baja = np.zeros(n)
    f_alta = np.zeros(n)
    for i in range(len(low_freqs)):
        f_baja += coefs_low[i]*np.cos(low_freqs[i]*2*np.pi*x)
    for i in range(len(high_freqs)):
        f_alta += coefs_high[i]*np.cos(high_freqs[i]*2*np.pi*x)
    f = np.zeros(n)
    f = f_alta + f_baja
    f += np.sqrt(noise_var)*np.random.randn(n)#Señal suma de una frecuancia baja mas una alta 

    #Sampleamos m puntos
    if random_sampling == False:
        y = np.zeros(m)
        positions = np.zeros(m)
        for i in range(m):
            y[i] = f[(delta)*i]
            positions[i] = x[(delta)*i]
    else:
        positions_init = np.arange(n)
        positions_2 = np.random.choice(positions_init, m, replace =False)
        positions_2.sort()
        positions = np.array([x[i] for i in positions_2])
        y = np.array([f[i] for i in positions_2])

    P = np.subtract.outer(x, positions)

    nyq = (m/(2*L))
    
    return (y, positions, P, f, f_alta, f_baja, x,nyq)