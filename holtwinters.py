from math            import sqrt
from numpy           import array, mean, std, sqrt, sum, abs
from matplotlib      import pyplot as plt
from scipy.optimize  import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn       as     sns
import warnings
warnings.filterwarnings('ignore')

class HoltWinters ( object ):

    def __init__(self, seasonality):

        self.seasonality = seasonality
        #self.optimaze  = True
        self.optimazed = False
    ''''
    seasonallity:
        {'add','mul'}
        trend component type.
    m:
        season length
    '''

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def get_param(self):
        return self.alpha, self.beta, self.gamma

    #--- --- --- --- --- --- --- --- --- --- --- --- ---
    #==================================================#
    #               ERROR FUNCTION                     #
    #==================================================#
    def mean_percentage_error( self, y, yhat ):
        return mean( ( y - yhat ) / y )

    def mean_absolute_error( self, y, yhat ):
        return mean( abs( ( y - yhat ) ) )

    def mean_absolute_percentage_error( self, y, yhat ):
        return mean( abs( ( y - yhat ) / y ) )

    #==================================================#
    #         VALIDATION - EXPANDING WINDOW            #
    #==================================================#
    def expanding_window( self, data, kfold, t):

        m =  self.m

        mae_list  = []
        mape_list = []
        rmse_list = []
        pred_kfold = {'Kfold': [], 'Pred': []}

        start = 2*m

        for k in range( kfold ):
            self.fit( xt_raw = data[:start+k] , m = self.m,
                                                alpha = 0.0196988284090553,
                                                beta  = 1.0,
                                                gamma = 0.3,
                                                optimize=False )

            self.predict( t = t );

            mae_list.append( self.mean_absolute_error( data[start+k:start+t+k].values, self.pred  ))
            mape_list.append( self.mean_absolute_percentage_error( data[start+k:start+t+k].values, self.pred  ))
            rmse_list.append( sqrt( mean_squared_error( data[start+k:start+t+k].values, self.pred  ) ))

            pred_kfold['Kfold'].append(k)
            pred_kfold['Pred'].append(self.pred)


        sns.distplot( mae_list, kde=True, bins=7 )

        return pd.DataFrame( {'Model Name': 'model_name',
                            'MAE CV':  round( mean( mae_list ) , 2 ).astype( str ) + ' +/- ' + round( std( mae_list ) , 2 ).astype( str ),
                            'MAPE CV': round( mean( mape_list ), 2 ).astype( str ) + ' +/- ' + round( std( mape_list ), 2 ).astype( str ),
                            'RMSE CV': round( mean( rmse_list ), 2 ).astype( str ) + ' +/- ' + round( std( rmse_list ), 2 ).astype( str ) }, index=[0] )

        #return mape_list

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def moving_mean( self, ts, m ):
    #------------------------------------
    # CENTERED MOVING MEAN ON SEASON "m"
    #------------------------------------
        result = []

        for i in range( self.m ):
            result.append(mean(ts[(i):(i+m-1)]))

        return result

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def RMSE ( self, Y, y ):
        #--------------------------
        # ROOT MEAN SQUARED ERROR
        #--------------------------
        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip( Y, y[:-1])]) / len(Y))

        return rmse

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def ObjectFunc(self, params, args):

        alpha, beta, gamma = params

        Y, m = args

        self.fit( xt_raw = Y[:79], m = m, alpha = alpha, beta = beta, gamma = gamma, optimize = False )
        y=self.y

        rmse = sqrt(sum([(m - n) ** 2 for m, n in zip( Y, y[:-1])]) / len(Y))

        return rmse

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def optmize_param(self):

        initial_values = [0.01, 0.01, 0.01]
        boundaries = [(0, 1), (0, 1), (0, 1)]
        self.optimize = False
        args = [self.xt_raw, self.m]


        ans = minimize(self.ObjectFunc, x0= initial_values, args= args, bounds=boundaries, method= 'L-BFGS-B')

        alpha, beta, gamma = ans.x

        self.fit( self.xt_raw, self.m, alpha = alpha, beta = beta, gamma = gamma, optimize = False )

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
        init_a = []
        init_b = []
        init_f = []
        y = []
        #-----------------------------------------------------------
        #         SET STAT VALUES: (level, trend, season)
        #-----------------------------------------------------------
        #  SEASONALITY: init_f -> calculate by comparing the
        #  appropriate observation in the first 'P' with moving mean
        #-----------------------------------------------------------
        mm = self.moving_mean( ts = xt, m = m )         # moving mean
        Y1 = xt[m//2 + 1 :]
        self.mm = mm                                         # To plot

        if self.seasonality == 'add':
            init_f = [ ( Y1[i] - mm[i] )                  for i in range(m)] # / mm[i]

        if self.seasonality == 'mul':
            init_f = [ Y1[i] / mm[i]                      for i in range(m)]
            init_f = [ init_f[i] * ( m / sum( init_f ) )  for i in range(m)]

        #-----------------------------------------------------------
        #             TIME SERIES WITHOUT SEASONALITY
        #-----------------------------------------------------------
        if self.seasonality == 'add':
            Y_ns = [Y1[i] - init_f[i] for i in range(m)]

        if self.seasonality == 'mul':
            Y_ns = [Y1[i] / init_f[i] for i in range(m)]

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

        return init_a, init_b, init_f , y

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def fit( self, xt_raw, m, alpha, beta, gamma, optimize ):
        self.xt_raw = xt_raw
        self.Y = xt_raw[(m//2+1)::]#(m//2+1)+m]
        self.m = m
        self.optimize = optimize


        #--------------------------------------
        # DECOMPOSE INTRO LEVEL, TREND, SEASON
        #--------------------------------------
        a, b, f, y = self.find_start( xt = self.xt_raw, m = m)
        f1 = []

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
                a.append( alpha * (self.Y[i] / f[(i)] ) + ( 1 - alpha ) * ( a[i] + b[i] ) )

                # TREND
                b.append( beta * ( a[i+1] - a[i]) + ( 1 - beta ) * b[i] )

                # SEASONAL #SEM NNORMALIZAÇÃO
                f.append(( gamma * ( self.Y[i] / (a[i]) ) + ( 1 - gamma ) * f[(i)] ) )#* m / (sum(f[-(m-1):]+gamma * ( self.Y[i] / (a[i]) ) + ( 1 - gamma ) * f[-(i)])))

                # MODELO
                y.append( ( a[i+1] + b[i+1] ) * f[i+1])

            else:
                exit('trend must be add or mul')

        self.a = a;
        self.b = b;
        self.f = f; #mean f + f1
        self.y = y[:-1];

        if self.optimize == True:
            self.alpha, self.beta, self.gamma = self.optmize_param()
            self.optimize = False
        else:
            self.alpha = alpha
            self.beta  = beta
            self.gamma = gamma


        return

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def predict( self, t ):
        # t -> forecast horizon
        self.t = t

        pa = [self.a[-1]]
        pb = [self.b[-1]]
        pf = self.f[-m:]
        pred = [self.y[-1]]

        for i in range(t):

            # ADDITIVE
            if self.seasonality == 'add':
                # LEVEL
                pa.append( self.alpha * (pred[i] - pf[i] ) + ( 1 - self.alpha ) * ( pa[i] + pb[i] ) )

                # TREND
                pb.append( self.beta * ( pa[i+1] - pa[i]) + ( 1 - self.beta ) * pb[i] )

                # SEASONAL
                pf.append( self.gamma * ( pred[i] / (pa[i] + pb[i]) ) + ( 1 - self.gamma ) * pf[i] )

                # MODELO
                pred.append( pa[i+1] + pb[i+1] + pf[i+1])

            # MULTIPLICATIVE
            if self.seasonality == 'mul':
                # LEVEL
                pa.append( self.alpha * (pred[i] / pf[i] ) + ( 1 - self.alpha ) * ( pa[i] + pb[i] ) )

                # TREND
                pb.append( self.beta * ( pa[i+1] - pa[i]) + ( 1 - self.beta ) * pb[i] )

                # SEASONAL
                pf.append(( self.gamma * ( pred[i] / (pa[i] + pb[i]) ) + ( 1 - self.gamma ) * pf[i]) )#* m / sum(pf[-(m-1):]))

                # MODELO
                pred.append( ( pa[i+1] + pb[i+1] ) * pf[i+1])

        self.pred = pred[:-1]

        return pred


    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def plot_fit(self):
        #------------------------------
        # PLOT LEVEL TREND SEASONALITY
        #------------------------------
        fig, axes = plt.subplots( 3, figsize=(15,10))

        sns.lineplot( data = {'Level'     : self.a },                ax  = axes[0] ).set(title=self.seasonality)
        sns.lineplot( data = {'Trend'     : self.b },                ax  = axes[1] )
        sns.lineplot( data = {'Seasonal'  : self.f },                ax  = axes[2] )
        #sns.lineplot( data = {'Moving Mean': self.mm },              ax  = axes[3] )
        #sns.lineplot( data = {'Real_values':(self.Y).values, 'Fitted Curve': self.y }, ax  = axes[4] )

    #--- --- --- --- --- --- --- --- --- --- --- --- ---

    def plot_pred( self, test ):
        #------------------------------
        #   PLOT REAL VS PREDICTION
        #------------------------------
        #sns.lineplot( data = { 'Prediction': self.pred } ).set(title= 'Prediction')
        sns.lineplot( data = {'Real_Values':(test).values, 'Prediction': self.pred } ).set(title= 'Prediction')
