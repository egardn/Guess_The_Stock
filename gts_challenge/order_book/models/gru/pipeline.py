# Pipeline specific to GRU preprocessing
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from gts_challenge.order_book.base.pipeline_interface import PipelineInterface
from gts_challenge.order_book.data.preprocessors import NonPositiveBidSizeFilter, CategoricalEncoder, DataVectorizer, VectorizedSequenceReshaper

class GRUPipeline(PipelineInterface):
    """
    Pipeline for preprocessing order book data for GRU models.
    Uses a single unified pipeline to handle all transformations.
    """
    def __init__(self):
        # Create a single pipeline that handles all transformations
        self.filter = NonPositiveBidSizeFilter()
        
        self.pipeline = Pipeline([
            ('cat_encoder', CategoricalEncoder()),
            ('vectorizer', DataVectorizer()),
            ('reshaper', VectorizedSequenceReshaper())
            ])
    
    def fit(self, X, y=None):
        """Fits the pipeline to the data"""
        self.filter.fit(X)
        X_filtered, y_filtered = self.filter.transform(X, y) 
        self.pipeline.fit(X_filtered)
        return self
    
    def transform(self, X=None, y=None):
        """Transforms data through the unified pipeline"""
        X_filtered, y_filtered = self.filter.transform(X, y)
        
        return self.pipeline.transform(X_filtered), y_filtered
        
    def fit_transform(self, X, y=None):
        """Fits and transforms the data through the unified pipeline"""
        self.filter.fit(X)
        X_filtered, y_filtered = self.filter.transform(X, y)
        
        return self.pipeline.fit_transform(X_filtered), y_filtered


    def save(self, path: str) -> None:
        """Saves the pipeline to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {path}")
        
    @classmethod
    def load(cls, path: str) -> 'GRUPipeline':
        """Loads a pipeline from disk"""
        with open(path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f"Pipeline loaded from {path}")
        return pipeline