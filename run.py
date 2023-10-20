from graphs.main.model import Model
from graphs.utils.util import Utility
import tensorflow as tf

if __name__ == '__main__':
    # Create simple batcher.
    num_epochs     = 15
    batch_size     = 1000
    embedding_size = 300
    num_sampled    = 30

    tf.config.set_visible_devices([], 'GPU')

    utility = Utility(batch_size=batch_size)

    valid_word = [
        "Matrix, The (1999)",
        "Star Wars: Episode IV - A New Hope (1977)",
        "Lion King, The (1994)",
        "Terminator 2: Judgment Day (1991)",
        "Godfather, The (1972)",
    ]

    valid_examples = utility.getValidExamples(valid_word=valid_word)
    valid_inputs = tf.convert_to_tensor(valid_examples)

    m = Model(num_epochs=num_epochs, num_sampled=num_sampled, batch_size=batch_size,
              embedding_size=embedding_size,voc_size=len(utility.voc_dict.keys()),
              learning_rate=1e-3)

    m.train(top_k=8, g=utility, total_batches=utility.total_batches,
            valid_word=valid_word, valid_inputs=valid_inputs, threshold=2.5,
            voc_dic_reverse=utility.voc_dict_reverse)
