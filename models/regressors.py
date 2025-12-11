from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

def get_regressors(n_estimators=100):
    """Return a dictionary of regressors to train."""
    regressors = {
        'random_forest': RandomForestRegressor(n_estimators=n_estimators, random_state=42),
        'xgboost': XGBRegressor(
            n_estimators=n_estimators,
            objective='reg:squarederror',
            random_state=42,
            eval_metric='rmse'
        ),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=n_estimators, random_state=42),
        'svr': SVR(kernel='rbf'),
        'knn': KNeighborsRegressor(n_neighbors=5),
        'mlp': MLPRegressor(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=1.0)
    }
    return regressors
