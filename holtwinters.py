from math           import sqrt
from numpy          import array, mean, std, sqrt, sum, absolute
from matplotlib     import pyplot as plt
from scipy.optimize import minimize
#from scipy.optimize import fmin_l_bfgs_b
import seaborn      as     sns
import warnings
warnings.filterwarnings('ignore')

class HoltWinters ( object ):

    def __init__(self, seasonality, norm_f): 

        self.seasonality = seasonality
        self.norm_f = norm_f
        self.optimaze  = True
        self.optimazed = False
    ''''   
    seasonallity: 
        {'add','mul'}
        trend component type.
    m:
        season length
    '''

    #--- --- --- --- --- --- --- --- --- --- --- --- ---
    # CORRIGIR FUNÇÃO #-- --- --- --- --- --- --- --- --
    #--- --- --- --- --- --- --- --- --- --- --- --- ---
    # def seasonal_normalizing (self): 
    #     #---------------------------------
    #     # Normalising The Sasonal Indice
    #     #---------------------------------
    #     '''
    #     The seasonal factor can be normalised to sum to zero in the additive case,
    #     or average to one in the multiplicative case
    #     '''
    #     if self.seasonalily == 'add':
    #         init_f_norm = [ (self.init_f[i] - mean(self.init_f))/std(self.init_f) for i in range(self.m)]
            
    #     if self.seasonalily == 'mul':
    #         init_f_norm = [ 1 + (self.init_f[i] - mean(self.init_f))/std(self.init_f) for i in range(self.m)]

    #     return init_f_norm

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def get_param(self):
        return self.alpha, self.beta, self.gamma

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def moving_mean( self, Y, m ):
        #----------------------------------
        # MOVING MEAN ON 10% OF SEASON "m"
        #----------------------------------
        p = int( m * 0.1 )
        result = []
        for i in range(len(Y[0:m])):
            if i < p:
                result.append(mean(Y[0:p-1]))
            else:
                result.append(mean(Y[i:i+(p-1)]))
        return result
    
    #--- --- --- --- --- --- --- --- --- --- --- --- ---
    # CORRIGIR FUNÇÃO MAPE #-- --- --- --- --- --- --- -
    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    # def MAPE ( self ): 
    #     #--------------------------------
    #     # MEAN ABSOLUTE PERCENTAGE ERROR 
    #     #--------------------------------
    #     mape = sum ( absolute( [( m - n ) / m for m, n in zip( self.Y, self.pred[:-1] ) ] ) ) / len(self.Y)
    #     return mape

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def RMSE ( self ):
        #--------------------------
        # ROOT MEAN SQUARED ERROR 
        #--------------------------
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip( self.Y, self.y[:-1])]) / len(self.Y))

        return rmse
        
    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def ObjectFunc(self, params, args):

        alpha, beta, gamma = params

        Y, m = args

        y = self.fit( xt = Y, m = m, alpha = alpha, beta = beta, gamma = gamma )

        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip( Y, y[:-1])]) / len(Y))

        return rmse

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def optmize_param(self):

        initial_values = [0.3, 0.3, 0.3]
        boundaries = [(0, 1), (0, 1), (0, 1)]
        args = [self.Y, self.m]
        self.optimaze = False

        ans = minimize(self.ObjectFunc, x0= initial_values, args= args, bounds=boundaries, method= 'L-BFGS-B')

        alpha, beta, gamma = ans.x
        
        self.fit( self.Y, self.m, alpha = alpha, beta = beta, gamma = gamma )

        return alpha, beta, gamma

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def find_start (self, xt, m):
        '''
        xt: 
            np.array
            the time series.
        m:
            season length
        '''
        self.Y = xt[:]
        y = []
        #-----------------------------------------------------------
        #         SET STAT VALUES: (level, trend, season)
        #-----------------------------------------------------------
        #  SEASONALITY: init_f -> calculate by comparing the 
        #  appropriate observation in the first 'P' with moving mean
        #-----------------------------------------------------------
        mm = self.moving_mean( Y=self.Y, m=m )         # moving mean

        if self.seasonality == 'add':
            init_f = [ ( self.Y[i] - mm[i] )      for i in range(m)] # / mm[i]
            #init_f = [ init_f[i] * ( m / sum( init_f ) )  for i in range(m)]
        
        if self.seasonality == 'mul':
            init_f = [ self.Y[i] / mm[i]                  for i in range(m)]
            init_f = [ init_f[i] * ( m / sum( init_f ) )  for i in range(m)]
        
        #-----------------------------------------------------------
        #             TIME SERIES WITHOUT SEASONALITY
        #----------------------------------------------------------- 
        if self.seasonality == 'add':
            Y_ns = [self.Y[i] - init_f[i] for i in range(m)]
        
        if self.seasonality == 'mul':
            Y_ns = [self.Y[i] / init_f[i] for i in range(m)]

        #-----------------------------------------------------------
        #  TREND: init_b -> from simple linear regression model
        #----------------------------------------------------------- 
        i_2 = sum([ i**2   for i in range(m)])
        x_y = sum([ Y_ns[i]*i for i in range(m)])
        init_b = [( ( m * x_y - (( m+1 ) * ( m/2 ) * sum(Y_ns[0:m-1])) ) / 
                   ( m * i_2 - (( m+1 ) * ( m/2 ))**2) )]
        #-----------------------------------------------------------
        #  LEVEL: init_a -> from simple linear regression model
        #----------------------------------------------------------- 
        init_a = [mean(Y_ns[0:m-1]) - init_b[0] * m/2] 

        #-----------------------------------------------------------
        #  TIME SERIES EQUATION: Y
        #----------------------------------------------------------- 
        if self.seasonality == 'add':            
            # ADDITIVE
            y = [ init_a[0] + init_b[0] + init_f[0] ]

        if self.seasonality == 'mul':    
            # MULTIPLICATIVE
            y = [ (init_a[0] + init_b[0]) * init_f[0] ]

        else:
            exit('seasonal must be add or mul')
        
        return init_a, init_b, init_f, y

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def fit( self, xt, m, alpha, beta, gamma ):
        
        self.pred = []

        self.Y = xt[:]
        self.m = m

        #--------------------------------------
        # DECOMPOSE INTRO LEVEL, TREND, SEASON
        #--------------------------------------
        a, b, f, y = self.find_start( xt, m)

        for i in range(len(self.Y)):

            # ADDITIVE
            if self.seasonality == 'add':
                # LEVEL
                a.append( alpha * (self.Y[i] - f[i] ) + ( 1 - alpha ) * ( a[i] + b[i] ) )
            
                # TREND
                b.append( beta * ( a[i+1] - a[i]) + ( 1 - beta ) * b[i] )
            
                # SEASONAL
                f.append( gamma * ( self.Y[i] / (a[i] + b[i]) ) + ( 1 - gamma ) * f[i] )
                
                # MODELO
                y.append( a[i+1] + b[i+1] + f[i+1])
            
            # MULTIPLICATIVE
            if self.seasonality == 'mul':
                # LEVEL
                a.append( alpha * (self.Y[i] / f[i] ) + ( 1 - alpha ) * ( a[i] + b[i] ) )
            
                # TREND
                b.append( beta * ( a[i+1] - a[i]) + ( 1 - beta ) * b[i] )
            
                # SEASONAL
                f.append( gamma * ( self.Y[i] / (a[i] + b[i]) ) + ( 1 - gamma ) * f[i] ) 
                
                # MODELO
                y.append( ( a[i+1] + b[i+1] ) * f[i+1])
            
            else:
                exit('trend must be add or mul')

        self.a = a;
        self.b = b;
        self.f = f;
        self.y = y;

        if self.optimaze == True:
            self.alpha, self.beta, self.gamma = self.optmize_param()
            
   
        return y;

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def plot(self):
        #------------------------------
        # PLOT LEVEL TREND SEASONALITY
        #------------------------------
        fig, axes = plt.subplots( 4, figsize=(15,10))

        sns.lineplot( data = {'Level'     : self.a },                ax  = axes[0] ).set(title=self.seasonality)
        sns.lineplot( data = {'Trend'     : self.b },                ax  = axes[1] )
        sns.lineplot( data = {'Seasonal'  : self.f },                ax  = axes[2] )
        sns.lineplot( data = {'Real_values':(self.Y).values, 'Fitted Curve': self.y }, ax  = axes[3] )