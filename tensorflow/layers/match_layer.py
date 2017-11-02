# -*- coding:utf8 -*-
###############################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
###############################################################################
"""
This module implements the core layer of Match-LSTM and BiDAF

Authors: Yizhong Wang(wangyizhong01@baidu.com)
Date: 2017/09/20 12:00:00
"""

import tensorflow as tf
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the query dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, query_encodes, p_length, q_length):
        """
        Match the passage_encodes with query_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, query_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, query_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-query Attention and Query-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, query_encodes, p_length, q_length):
        """
        Match the passage_encodes with query_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            sim_matrix = tf.matmul(passage_encodes, query_encodes, transpose_b=True)
            context2query_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), query_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            query2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2query_attn,
                                        passage_encodes * context2query_attn,
                                        passage_encodes * query2context_attn], -1)
            return concat_outputs, None