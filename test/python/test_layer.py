import sys
import os
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))

from singa import layer


def _tuple_to_string(t):
    lt = [str(x) for x in t]
    return '(' + ', '.join(lt) + ')'


class TestPythonLayer(unittest.TestCase):

    def check_shape(self, actual, expect):
        self.assertEqual(actual, expect, 'shape mismatch, actual shape is %s'
                         ' exepcted is %s' % (_tuple_to_string(actual),
                                              _tuple_to_string(expect))
                         )

    def setUp(self):
        self.w = {'init': 'Xavier', 'regularizer': 1e-4}
        self.b = {'init': 'Constant', 'value': 0}
        self.sample_shape = None

    def test_conv2D_shape(self):
        in_sample_shape = (3, 224, 224)
        conv = layer.Conv2D('conv', 64, 3, 1, W_specs=self.w, b_specs=self.b,
                            input_sample_shape=in_sample_shape)
        out_sample_shape = conv.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 224, 224))

    def test_conv2D_forward_backward(self):
        in_sample_shape = (1, 3, 3)
        conv = layer.Conv2D('conv', 1, 3, 1, W_specs=self.w, b_specs=self.b,
                            pad=2, input_sample_shape=in_sample_shape)
        '''
        conv.to_device()
        params = conv.get_params()
        params[0]

        x = np.arange(9) + 1
        w = np.array([1, 1, 0, 0, 0, -1, 0, 1, 0])
        params[0].from_array(w)

        y = conv.forward(x)
        y.to_hos()
        npy = y.to_array()

        out4 = [3, 7, -3, 12]

        dy = np.array([0.1, 0.2, 0.3, 0.4])
        grad.from_array(dy)
        grad.to_device(device)
        ret = conv.backward(grad)
        dx = ret.first.to_array()
        dw = ret.second[0].to_array()
        db = ret.second[1].to_array()
        EXPECT_EQ(dy[0] * wptr[4], dx[0]);
        EXPECT_EQ(dy[0] * wptr[5] + dy[1] * wptr[3], dx[1]);
        EXPECT_EQ(dy[1] * wptr[4], dx[2]);
        EXPECT_EQ(dy[0] * wptr[7] + dy[2] * wptr[1], dx[3]);
        EXPECT_EQ(
            dy[0] * wptr[8] + dy[1] * wptr[6] + dy[2] * wptr[2] + dy[3] * wptr[0],
            dx[4]);
        EXPECT_EQ(dy[1] * wptr[7] + dy[3] * wptr[1], dx[5]);
        EXPECT_EQ(dy[2] * wptr[4], dx[6]);
        EXPECT_EQ(dy[2] * wptr[5] + dy[3] * wptr[3], dx[7]);
        EXPECT_EQ(dy[3] * wptr[4], dx[8]);

        EXPECT_EQ(dy[3] * x[4], dwptr[0]);
        EXPECT_EQ(dy[3] * x[5] + dy[2] * x[3], dwptr[1]);
        EXPECT_EQ(dy[2] * x[4], dwptr[2]);
        EXPECT_EQ(dy[1] * x[1] + dy[3] * x[7], dwptr[3]);
        EXPECT_FLOAT_EQ(dy[0] * x[0] + dy[1] * x[2] + dy[2] * x[6] + dy[3] * x[8],
                        dwptr[4]);
        EXPECT_EQ(dy[0] * x[1] + dy[2] * x[7], dwptr[5]);
        EXPECT_EQ(dy[1] * x[4], dwptr[6]);
        EXPECT_EQ(dy[0] * x[3] + dy[1] * x[5], dwptr[7]);
        EXPECT_EQ(dy[0] * x[4], dwptr[8]);
        '''

    def test_conv1D(self):
        in_sample_shape = (224,)
        conv = layer.Conv1D('conv', 64, 3, 1, W_specs=self.w, b_specs=self.b,
                            pad=1, input_sample_shape=in_sample_shape)
        out_sample_shape = conv.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 224,))

    def test_max_pooling2D(self):
        in_sample_shape = (64, 224, 224)
        pooling = layer.MaxPooling2D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 112, 112))

    def test_max_pooling1D(self):
        in_sample_shape = (224,)
        pooling = layer.MaxPooling1D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (112,))

    def test_avg_pooling2D(self):
        in_sample_shape = (64, 224, 224)
        pooling = layer.AvgPooling2D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64, 112, 112))

    def test_avg_pooling1D(self):
        in_sample_shape = (224,)
        pooling = layer.AvgPooling1D('pool', 3, 2,
                                     input_sample_shape=in_sample_shape)
        out_sample_shape = pooling.get_output_sample_shape()
        self.check_shape(out_sample_shape, (112,))

    def test_batch_normalization(self):
        in_sample_shape = (3, 224, 224)
        bn = layer.BatchNormalization('bn', input_sample_shape=in_sample_shape)
        out_sample_shape = bn.get_output_sample_shape()
        self.check_shape(out_sample_shape, in_sample_shape)

    def test_lrn(self):
        in_sample_shape = (3, 224, 224)
        lrn = layer.LRN('lrn', input_sample_shape=in_sample_shape)
        out_sample_shape = lrn.get_output_sample_shape()
        self.check_shape(out_sample_shape, in_sample_shape)

    def test_dense(self):
        dense = layer.Dense('ip', 32, input_sample_shape=(64,))
        out_sample_shape = dense.get_output_sample_shape()
        self.check_shape(out_sample_shape, (32,))

    def test_dropout(self):
        input_sample_shape = (64, 1, 12)
        dropout = layer.Dropout('drop', input_sample_shape=input_sample_shape)
        out_sample_shape = dropout.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_activation(self):
        input_sample_shape = (64, 1, 12)
        act = layer.Activation('act', input_sample_shape=input_sample_shape)
        out_sample_shape = act.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_softmax(self):
        input_sample_shape = (12,)
        softmax = layer.Softmax('soft', input_sample_shape=input_sample_shape)
        out_sample_shape = softmax.get_output_sample_shape()
        self.check_shape(out_sample_shape, input_sample_shape)

    def test_flatten(self):
        input_sample_shape = (64, 1, 12)
        flatten = layer.Flatten('flat', input_sample_shape=input_sample_shape)
        out_sample_shape = flatten.get_output_sample_shape()
        self.check_shape(out_sample_shape, (64 * 1 * 12, ))

        flatten = layer.Flatten(axis=2, input_sample_shape=input_sample_shape)
        out_sample_shape = flatten.get_output_sample_shape()
        self.check_shape(out_sample_shape, (12,))


if __name__ == '__main__':
    unittest.main()
