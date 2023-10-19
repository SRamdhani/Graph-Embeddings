from dataclasses import dataclass, field
from .training import Training
import tensorflow as tf
import math

@dataclass(frozen=False, unsafe_hash=True)
class Model(Training):
    batch_size: int = field(init=True, default=int, repr=False, compare=False)
    embedding_size: int = field(init=True, default=int, repr=False, compare=False)
    voc_size: int = field(init=True, default=int, repr=False, compare=False)
    learning_rate: float = field(init=True, default=float, repr=False, compare=False)
    model: tf.keras.Model = field(init=False, default_factory=lambda: tf.keras.Model, repr=False, compare=False)
    opt: tf.keras.optimizers.Adam = field(init=False, default_factory=lambda: tf.keras.optimizers.Adam, repr=False, compare=False)
    nce_weights: tf.Variable = field(init=False, default_factory=lambda: tf.Variable, repr=False, compare=False)
    nce_biases: tf.Variable = field(init=False, default_factory=lambda: tf.Variable, repr=False, compare=False)

    def __post_init__(self) -> None:
        model, opt, nce_weights, nce_biases = Model.g2v(self.batch_size,
                                                        self.embedding_size,
                                                        self.voc_size,
                                                        self.learning_rate)

        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'opt', opt)
        object.__setattr__(self, 'nce_weights', nce_weights)
        object.__setattr__(self, 'nce_biases', nce_biases)

    @staticmethod
    def g2v(batch_size: int, embedding_size: int, voc_size: int, learning_rate: float = 1e-3) -> tuple:
        # Input Data
        train_inputs = tf.keras.Input(batch_size=batch_size, shape=())
        # train_labels = tf.keras.Input(batch_size=batch_size, shape=(1))

        # Look up embeddings for inputs.
        embeddings = tf.keras.layers.Embedding(voc_size, embedding_size, input_length=1)
        embed      = embeddings(train_inputs)

        # Construct the variables for NCE loss.
        initializerRU = tf.keras.initializers.RandomUniform(minval=-1.0/math.sqrt(embedding_size),
                                                            maxval=1.0/math.sqrt(embedding_size))

        initializerZ  = tf.keras.initializers.Zeros()

        nce_weights   = tf.Variable(initializerRU(shape=(voc_size, embedding_size)))
        nce_biases    = tf.Variable(initializerZ(voc_size))

        model         = tf.keras.Model(inputs=train_inputs, outputs=embed)
        opt           = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        return model, opt, nce_weights, nce_biases
