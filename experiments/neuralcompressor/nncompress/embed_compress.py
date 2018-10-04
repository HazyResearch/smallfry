from __future__ import absolute_import, division, print_function

import sys
import os
import time
import math
import logging
import tensorflow as tf
import numpy as np
# Import _linear
if tuple(map(int, tf.__version__.split("."))) >= (1, 6, 0):
    from tensorflow.contrib.rnn.python.ops import core_rnn_cell
    _linear = core_rnn_cell._linear
else:
    from tensorflow.python.ops.rnn_cell_impl import _linear



tf.flags.DEFINE_string('qmats', "data/glove.6B.300d.quant.npy", "output")


class EmbeddingCompressor(object):

    _TAU = 1.0
    _BATCH_SIZE = 64
    _GRAD_CLIP = 0.001
    _LEARNING_RATE = 0.0001

    def __init__(self, n_codebooks, n_centroids, model_path,
                                            tau, # tony line
                                            batch_size, # tony line
                                            learning_rate, # tony line
                                            grad_clip): # tony line
        """
        M: number of codebooks (subcodes)
        K: number of vectors in each codebook
        model_path: prefix for saving or loading the parameters
        """
        self.M = n_codebooks
        self.K = n_centroids
        self._model_path = model_path
        self._TAU = tau # tony line
        self._BATCH_SIZE = batch_size # tony line
        self._GRAD_CLIP = grad_clip # tony line
        self.LEARNING_RATE = learning_rate #tony line

    def _gumbel_dist(self, shape, eps=1e-20):
        U = tf.random_uniform(shape,minval=0,maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _sample_gumbel_vectors(self, logits, temperature):
        y = logits + self._gumbel_dist(tf.shape(logits))
        return tf.nn.softmax( y / temperature)

    def _gumbel_softmax(self, logits, temperature, sampling=True):
        """Compute gumbel softmax.

        Without sampling the gradient will not be computed
        """
        if sampling:
            y = self._sample_gumbel_vectors(logits, temperature)
        else:
            k = tf.shape(logits)[-1]
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def _encode(self, input_matrix, word_ids, embed_size):
        input_embeds = tf.nn.embedding_lookup(input_matrix, word_ids, name="input_embeds")

        M, K = self.M, self.K

        with tf.variable_scope("h"):
            h = tf.nn.tanh(_linear(input_embeds, M * K/2, True))
        with tf.variable_scope("logits"):
            logits = _linear(h, M * K, True)
            logits = tf.log(tf.nn.softplus(logits) + 1e-8)
        logits = tf.reshape(logits, [-1, M, K], name="logits")
        return input_embeds, logits

    def _decode(self, gumbel_output, codebooks):
        return tf.matmul(gumbel_output, codebooks)

    def _reconstruct(self, codes, codebooks):
        return None

    def build_export_graph(self, embed_matrix):
        """Export the graph for exporting codes and codebooks.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        vocab_size = embed_matrix.shape[0]
        embed_size = embed_matrix.shape[1]

        input_matrix = tf.constant(embed_matrix, name="embed_matrix")
        word_ids = tf.placeholder_with_default(
            np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")

        # Define codebooks
        codebooks = tf.get_variable("codebook", [self.M * self.K, embed_size])

        # Coding
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)  # ~ (B, M, K)
        codes = tf.cast(tf.argmax(logits, axis=2), tf.int32)  # ~ (B, M)

        # Reconstruct
        offset = tf.range(self.M, dtype="int32") * self.K
        codes_with_offset = codes + offset[None, :]

        selected_vectors = tf.gather(codebooks, codes_with_offset)  # ~ (B, M, H)
        reconstructed_embed = tf.reduce_sum(selected_vectors, axis=1)  # ~ (B, H)
        return word_ids, codes, reconstructed_embed

    def build_training_graph(self, embed_matrix):
        """Export the training graph.

        Args:
            embed_matrix: numpy matrix of original embeddings
        """
        vocab_size = embed_matrix.shape[0]
        embed_size = embed_matrix.shape[1]

        # Define input variables
        input_matrix = tf.constant(embed_matrix, name="embed_matrix")
        tau = tf.placeholder_with_default(np.array(1.0, dtype='float32'), tuple()) - 0.1
        word_ids = tf.placeholder_with_default(
            np.array([3,4,5], dtype="int32"), shape=[None], name="word_ids")

        # Define codebooks
        codebooks = tf.get_variable("codebook", [self.M * self.K, embed_size])

        # Encoding
        input_embeds, logits = self._encode(input_matrix, word_ids, embed_size)  # ~ (B, M, K)

        # Discretization
        D = self._gumbel_softmax(logits, self._TAU, sampling=True)
        gumbel_output = tf.reshape(D, [-1, self.M * self.K])  # ~ (B, M * K)
        maxp = tf.reduce_mean(tf.reduce_max(D, axis=2))

        # Decoding
        y_hat = self._decode(gumbel_output, codebooks)

        # Define loss
        loss = 0.5 * tf.reduce_sum((y_hat - input_embeds)**2, axis=1)
        loss = tf.reduce_mean(loss, name="loss")

        # Define optimization
        max_grad_norm = self._GRAD_CLIP # tony mod
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        global_norm = tf.identity(global_norm, name="global_norm")
        optimizer = tf.train.AdamOptimizer(self._LEARNING_RATE) # tony mod
        train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op")

        return word_ids, loss, train_op, maxp

    def train(self, embed_matrix, max_epochs=200):
        """Train the model for compress `embed_matrix` and save to `model_path`.

        Args:
            embed_matrix: a numpy matrix
        """
        dca_train_log = [] #tony line
        vocab_size = embed_matrix.shape[0]
        valid_ids = np.random.RandomState(3).randint(0, vocab_size, size=(self._BATCH_SIZE * 10,)).tolist()
        # Training
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph", initializer=tf.random_uniform_initializer(-0.01, 0.01)):
                word_ids_var, loss_op, train_op, maxp_op = self.build_training_graph(embed_matrix)
            # Initialize variables
            tf.global_variables_initializer().run()
            best_loss = 100000
            saver = tf.train.Saver()

            vocab_list = list(range(vocab_size))
            for epoch in range(max_epochs):
                start_time = time.time()
                train_loss_list = []
                train_maxp_list = []
                np.random.shuffle(vocab_list)
                for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                    word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                    loss, _, maxp = sess.run(
                        [loss_op, train_op, maxp_op],
                        {word_ids_var: word_ids}
                    )
                    train_loss_list.append(loss)
                    train_maxp_list.append(maxp)

                # Print every epoch
                time_elapsed = time.time() - start_time
                bps = len(train_loss_list) / time_elapsed

                # Validation
                valid_loss_list = []
                valid_maxp_list = []
                for start_idx in range(0, len(valid_ids), self._BATCH_SIZE):
                    word_ids = valid_ids[start_idx:start_idx + self._BATCH_SIZE]
                    loss, maxp = sess.run(
                        [loss_op, maxp_op],
                        {word_ids_var: word_ids}
                    )
                    valid_loss_list.append(loss)
                    valid_maxp_list.append(maxp)

                # Report
                valid_loss = np.mean(valid_loss_list)
                report_token = ""
                if valid_loss <= best_loss * 0.999:
                    report_token = "*"
                    best_loss = valid_loss
                    saver.save(sess, self._model_path)
                trainloss = float(np.mean(train_loss_list)) #tony line
                validloss = float(np.mean(valid_loss_list)) #tony line
                trainmaxp = float(np.mean(train_maxp_list)) #tony line
                validmaxp = float(np.mean(valid_maxp_list)) #tony line
                t = len(train_loss_list) / time_elapsed # tony line
                log_str = "[epoch{}] trian_loss={:.2f} train_maxp={:.2f} valid_loss={:.2f} valid_maxp={:.2f} bps={:.0f} {}".format(
                    epoch,
                    trainloss, trainmaxp,
                    validloss, validmaxp,
                    t,
                    report_token
                ) #tony lines (log_str)
                dca_train_log.append(           #tony line
                    {'epoch': epoch,            #tony line
                     'trainloss' : trainloss,   #tony line
                     'validloss' : validloss,   #tony line
                     'trainmaxp' : trainmaxp,   #tony line
                     'validmaxp' : validmaxp,   #tony line
                     'time-elapsed': time_elapsed})        #tony line
                logging.info(log_str) #tony mod

        logging.info("Training Done") #tony mod
        return dca_train_log # tony line

    def export(self, embed_matrix, prefix):
        """Export word codes and codebook for given embedding.

        Args:
            embed_matrix: original embedding
            prefix: prefix of saving path
        """
        assert os.path.exists(self._model_path + ".meta")
        vocab_size = embed_matrix.shape[0]
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph"):
                word_ids_var, codes_op, reconstruct_op = self.build_export_graph(embed_matrix)
            saver = tf.train.Saver()
            saver.restore(sess, self._model_path)

            # Dump codebook
            codebook_tensor = sess.graph.get_tensor_by_name('Graph/codebook:0')
            codebook_tensor = sess.run(codebook_tensor) #tony lines
            np.save(prefix + ".codebook", codebook_tensor)# tony modded

            code_rtn = [] #tony lines
            # Dump codes
            with open(prefix + ".codes", "w") as fout:
                vocab_list = list(range(embed_matrix.shape[0]))
                for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                    word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                    codes = sess.run(codes_op, {word_ids_var: word_ids}).tolist()
                    for code in codes:
                        code_rtn.append(code) #tony lines
                        fout.write(" ".join(map(str, code)) + "\n")
            return code_rtn, codebook_tensor #tony lines

    def evaluate(self, embed_matrix):
        assert os.path.exists(self._model_path + ".meta")
        vocab_size = embed_matrix.shape[0]
        with tf.Graph().as_default(), tf.Session() as sess:
            with tf.variable_scope("Graph"):
                word_ids_var, codes_op, reconstruct_op = self.build_export_graph(embed_matrix)
            saver = tf.train.Saver()
            saver.restore(sess, self._model_path)


            vocab_list = list(range(embed_matrix.shape[0]))
            distances = []
            for start_idx in range(0, vocab_size, self._BATCH_SIZE):
                word_ids = vocab_list[start_idx:start_idx + self._BATCH_SIZE]
                reconstructed_vecs = sess.run(reconstruct_op, {word_ids_var: word_ids})
                original_vecs = embed_matrix[start_idx:start_idx + self._BATCH_SIZE]
                distances.extend(np.linalg.norm(reconstructed_vecs - original_vecs, axis=1).tolist())
            frob_error = np.sum([d**2 for d in distances]) #tony line
            return np.mean(distances), frob_error #tony mod
