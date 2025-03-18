import tensorflow as tf

class Decoder(object):
    def __init__(self, config, n_tok, id_end):
        self._config = config
        self._n_tok = n_tok
        self._id_end = id_end
        self._dim_embeddings = config.attn_cell_config.get("dim_embeddings", 80)  # Paper uses 80
        self._dim_hidden = config.attn_cell_config.get("num_units", 512)

    def __call__(self, training, img, formula, dropout):
        batch_size = tf.shape(img)[0]
        E = tf.Variable(tf.random.uniform([self._n_tok, self._dim_embeddings], -1.0, 1.0), name="E")
        E = tf.nn.l2_normalize(E, axis=-1)
        V = tf.reshape(img, [batch_size, -1, img.shape[-1]])  # (N, H' * W', C)

        # Learned initial state (paper: h_0 = tanh(W_h * mean(V) + b_h))
        V_mean = tf.reduce_mean(V, axis=1)  # (N, C)
        W_h0 = tf.Variable(tf.keras.initializers.GlorotUniform()([V.shape[-1], self._dim_hidden]), name="W_h0")
        b_h0 = tf.Variable(tf.zeros([self._dim_hidden]), name="b_h0")
        h_0 = tf.tanh(tf.matmul(V_mean, W_h0) + b_h0)  # (N, 512)
        c_0 = tf.zeros([batch_size, self._dim_hidden])  # Cell state initialized to zero
        initial_state = (c_0, h_0)

        with tf.name_scope("decoder"):
            embeddings = tf.nn.embedding_lookup(E, formula)[:, :-1, :]  # (N, T-1, 80)
            lstm = tf.keras.layers.LSTM(self._dim_hidden, return_sequences=True, return_state=True)
            o_prev = tf.zeros([batch_size, self._dim_hidden])  # Initial o_{t-1}
            lstm_inputs = tf.concat([embeddings, tf.tile(o_prev[:, None, :], [1, tf.shape(embeddings)[1], 1])], axis=-1)
            lstm_outputs, h_final, c_final = lstm(lstm_inputs, initial_state=initial_state)
            train_logits, o_t = self._decode_with_attention(lstm_outputs, V)

        with tf.name_scope("decoder"):
            test_logits = train_logits  # Placeholder for now

        return train_logits, {"logits": test_logits, "ids": None}

    def _decode_with_attention(self, lstm_outputs, V):
        W_h = tf.Variable(tf.keras.initializers.GlorotUniform()([self._dim_hidden, 512]), name="W_h")
        W_v = tf.Variable(tf.keras.initializers.GlorotUniform()([512, 512]), name="W_v")
        W_a = tf.Variable(tf.keras.initializers.GlorotUniform()([512, 1]), name="W_a")
        h_proj = tf.tensordot(lstm_outputs, W_h, [[2], [0]])  # (N, T-1, 512)
        v_proj = tf.tensordot(V, W_v, [[2], [0]])  # (N, H' * W', 512)
        scores = tf.tensordot(tf.tanh(h_proj[:, :, None, :] + v_proj[:, None, :, :]), W_a, [[3], [0]])  # (N, T-1, H' * W', 1)
        alpha = tf.nn.softmax(scores, axis=2)  # (N, T-1, H' * W', 1)
        c_t = tf.reduce_sum(V[:, None, :, :] * alpha, axis=2)  # (N, T-1, 512)

        W_c = tf.Variable(tf.keras.initializers.GlorotUniform()([self._dim_hidden + 512, self._dim_hidden]), name="W_c")
        combined = tf.concat([lstm_outputs, c_t], axis=-1)  # (N, T-1, 1024)
        o_t = tf.tanh(tf.tensordot(combined, W_c, [[2], [0]]))  # (N, T-1, 512)
        W_out = tf.Variable(tf.keras.initializers.GlorotUniform()([self._dim_hidden, self._n_tok]), name="W_out")
        logits = tf.tensordot(o_t, W_out, [[2], [0]])  # (N, T-1, n_tok)
        return logits, o_t