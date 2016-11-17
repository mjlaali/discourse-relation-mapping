import tensorflow as tf
import numpy as np


class Lexconn:

    def __init__(self, emission, entries):
        """ initialize LEXCONN entries
        Arg:
            emission: the emission matrix of discourse connectives for PDTB relations,
                in each cell emission(i, j) = P(PDTB_j|DC_i)
            entries: LEXCONN entries in the formant of (dc_idx, rst_rel_idx)
        """
        self.__emission = emission
        self.__entries = entries
        max_number = np.max(entries, axis=0)
        self.__rst_cnt = max_number[1] + 1
        self.__lexconn_size = entries.shape[0]

        self.__dc_cnt = emission.shape[0]
        self.__pdtb_cnt = emission.shape[1]
        pass

    @staticmethod
    def __one_hot_encoding(ar, ar_range):
        values = ar
        values.shape = (ar.shape[0], 1)
        one_hot = ar == np.arange(ar_range)
        return one_hot.astype(np.float32)

    def get_mapping(self):
        """ find the best mapping according to the given emission and entries matrix
        Return:
            a mapping between PDTB relations and RST relations. Mapping[i, j] = P(RST_j|PDTB_i)
        """
        
        sdc = Lexconn.__one_hot_encoding(self.__entries[:, 0], self.__dc_cnt)
        src = Lexconn.__one_hot_encoding(self.__entries[:, 1], self.__rst_cnt)

        sdc_e = np.dot(sdc, self.__emission) #select emission row for each entry in LEXCONN
        init_value = np.random.rand(self.__pdtb_cnt, self.__rst_cnt).astype(np.float32)
        m_logits = tf.Variable(init_value)  #parameters of mapping probabilities
        m = tf.nn.softmax(m_logits)     #Mapping probabilities
        src_m = tf.matmul(m, src.T)     #select mapping rows for each entry in LEXCONN
        score = tf.matmul(sdc_e, src_m) #calculate the probabilities of all possible combination of connectives and relations
        log_score = tf.log(score)
        sum_log_score = tf.trace(log_score) #we only need those that have the same index in LEXCONN
        optimizer = tf.train.AdamOptimizer(0.01).minimize(tf.neg(sum_log_score))
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(m))
            print(sess.run(sum_log_score))

            # Fit the line.
            np.set_printoptions(precision=4)
            np.set_printoptions(suppress=True)
            for step in range(50000):
                sess.run(optimizer)
                if step % 1000 == 0:
                    print(step, sess.run(sum_log_score))
            return sess.run(m)

if __name__ == "__main__":
    emission = np.loadtxt('../emission.txt')
    lexconn = np.loadtxt('../lexconn.txt')
    print(lexconn.shape)
    print(emission.shape)
    print(np.max(lexconn, axis=0))

    emission = emission.astype(np.float32)
    lexconnEntries = lexconn.astype(np.float32)
    lexconn = Lexconn(emission, lexconnEntries)
    got_mapping = lexconn.get_mapping()
    print(got_mapping)
    np.savetxt('../mapping.txt', got_mapping)

