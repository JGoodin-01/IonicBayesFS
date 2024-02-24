from sklearn.neighbors import KNeighborsRegressor
from src.models.BaseModel import BaseModel


class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
