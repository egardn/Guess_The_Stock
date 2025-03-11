import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, GRU, Dense, Concatenate, Dropout, Multiply
from tensorflow.keras.optimizers import Adam
import random

# Define embedding dimensions as constants
VENUE_EMBED_DIM = 8
ACTION_EMBED_DIM = 8
TRADE_EMBED_DIM = 8
SEQ_LENGTH = 100

class OrderBookEmbeddingModel(BaseEstimator, TransformerMixin):
    """
    Neural network model for order book classification with proper embeddings
    """
    def __init__(self, n_venues, n_actions, n_categories, epochs, batch_size, learning_rate):
        self.n_venues = n_venues
        self.n_actions = n_actions
        self.n_categories = n_categories
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs  # Add this parameter to store epochs
        self.model = None

    def create_model(self):
        # Create input layers for each feature type
        venue_input = Input(shape=(SEQ_LENGTH,), name='venue_input', dtype='int32')
        action_input = Input(shape=(SEQ_LENGTH,), name='action_input', dtype='int32')
        trade_input = Input(shape=(SEQ_LENGTH,), name='trade_input', dtype='int32')
        numeric_input = Input(shape=(SEQ_LENGTH, 6), name='numeric_input')  # Now 6 features
        
        # Create embedding layers
        venue_embedding = Embedding(self.n_venues, VENUE_EMBED_DIM, name='venue_embedding')(venue_input)
        action_embedding = Embedding(self.n_actions, ACTION_EMBED_DIM, name='action_embedding')(action_input)
        trade_embedding = Embedding(2, TRADE_EMBED_DIM, name='trade_embedding')(trade_input)
        
        # Concatenate embeddings with numeric features
        concatenated = Concatenate(axis=2)([
            venue_embedding,        # (batch, seq_len, 8)
            action_embedding,       # (batch, seq_len, 8)
            trade_embedding,        # (batch, seq_len, 8)
            numeric_input           # (batch, seq_len, 6)
        ])  # Result: (batch, seq_len, 30)
        
        # GRU layers - forward and backward
        forward_gru = GRU(64, return_sequences=False)(concatenated)
        backward_gru = GRU(64, return_sequences=False, go_backwards=True)(concatenated)

        # Concatenate GRU outputs
        concat = Concatenate()([forward_gru, backward_gru])

        # Dense layers
        dense1 = Dense(64, activation='selu')(concat)
        output_layer = Dense(self.n_categories, activation='softmax')(dense1)

        # Create and compile model
        model = Model(
            inputs=[
                venue_input, 
                action_input, 
                trade_input, 
                numeric_input
            ],
            outputs=output_layer
        )
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )

        return model
    
    def fit(self, X, y):
        """Fit the model to the data."""
        # Initialize model if not already done
        if self.model is None:
            self.model = self.create_model()
            self.model.summary()
        
        # Create model inputs - with only 4 required inputs
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Train model
        history = self.model.fit(
            model_inputs, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1
        )
        
        return self

    def transform(self, X):
        # Convert inputs to correct format with same keys as fit() method
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Return predictions
        return self.model.predict(model_inputs, batch_size=self.batch_size)

    def predict(self, X):
        # Convert inputs to correct format with same keys as fit() method
        model_inputs = [
            X['venue_input'], 
            X['action_input'], 
            X['trade_input'],
            X['numeric_input']
        ]
        
        # Return class predictions with consistent batch size
        return np.argmax(self.model.predict(model_inputs, batch_size=self.batch_size), axis=1)