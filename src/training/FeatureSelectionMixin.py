from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from src.training.BFS import BFS

class FeatureSelectionMixin:
    @staticmethod
    def apply_feature_selection(fs_strategy, X, y):
        rankings = None

        if fs_strategy["name"] == "SelectKBest":
            selector = SelectKBest(mutual_info_regression, k="all").fit(X, y)
            rankings = selector.scores_.argsort()[::-1]
        elif fs_strategy["name"] == "RFE":
            estimator = RandomForestRegressor(n_estimators=5, random_state=42)
            selector = RFE(estimator, n_features_to_select=1).fit(X, y)
            rankings = selector.ranking_.argsort()
        elif fs_strategy["name"] == "BFS":
            selector = BFS()
            selector.fit(X, y)
            rankings = selector.get_feature_rankings()
        else:
            raise ValueError("Invalid feature selection strategy")

        return rankings
