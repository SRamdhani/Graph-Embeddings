from dataclasses import dataclass, field
import tensorflow as tf
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Training:
    num_epochs: int = field(init=True, default=int, repr=False, compare=False)
    total_batches: int = field(init=True, default=int, repr=False, compare=False)
    num_sampled: int = field(init=True, default=int, repr=False, compare=False)

    def train(self, top_k: int,  g: object, valid_word: list,
              voc_dic_reverse: dict, threshold: float, valid_inputs: tf.Tensor) -> None:
        global_loss_val = 1e9

        for epoch in range(self.num_epochs):
            for step in range(self.total_batches): # TODO: REMOVE THIS LATER
                with tf.GradientTape(persistent=True) as tape:
                    batch_inputs, batch_labels = g.generator(step) # TODO: CHANGE THIS LATER
                    model_embedding = self.model(batch_inputs)
                    tape.watch(self.nce_weights)
                    tape.watch(self.nce_biases)

                    loss = tf.reduce_mean(tf.nn.nce_loss(self.nce_weights, self.nce_biases, batch_labels,
                                                         model_embedding, self.num_sampled, self.voc_size))

                grads = tape.gradient(loss, self.model.trainable_weights)+\
                            [tape.gradient(loss, self.nce_weights)] +\
                            [tape.gradient(loss, self.nce_biases)]

                trainables = self.model.trainable_weights + [self.nce_weights] + [self.nce_biases]
                self.opt.apply_gradients(zip(grads, trainables))

                if step % 20 == 0: # TODO: CHANGE THIS TO EPOCH.
                    print('loss: ', loss.numpy())
                    embeddings           = self.model.get_layer('embedding').embeddings
                    norm                 = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1))
                    normalized_embedding = embeddings / tf.reshape(norm, [-1, 1])
                    valid_embeddings     = self.model.get_layer('embedding')(valid_inputs)
                    similarity           = tf.matmul(valid_embeddings, normalized_embedding, transpose_b=True)
                    nearest              = (-1*similarity.numpy()).argsort()[:, 0:(top_k+1)]

                    for i, x in enumerate(nearest):
                        nearest_reverse = [voc_dic_reverse[y] for y in x]
                        log_str = "Nearest to %s: %s" % (valid_word[i], nearest_reverse)
                        print(log_str)
                    print()

                    if loss.numpy() < global_loss_val:
                        global_loss_val = loss.numpy()
                        trained_embeddings = embeddings.numpy()
                        trained_embeddings_norm = normalized_embedding.numpy()
                        np.save('TE_NEW.npy', trained_embeddings)
                        np.save('TE_N_NEW.npy', trained_embeddings_norm)

            if global_loss_val < threshold:
                break