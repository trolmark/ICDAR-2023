import tensorflow as tf


class CornersHeatmap:
        
    @staticmethod
    @tf.function
    def tf_heatmap(corners, kernel_sigma, num_corners, **kwargs):
        """
        Parameters
        ----------
        corners: tf.Tensor with shape (num_corners, 2) specifing coords in xy format
        kwargs: "im_w" and "im_h" have to be specified
        """

        kernel_size = 6 * kernel_sigma + 3
        assert kernel_size % 2 != 0

        def gaussian_window2d(size, std):
            a = size // 2
            x_range = tf.range(-a, a+1, dtype=tf.float32)
            window1d = tf.map_fn(lambda x: tf.exp(-0.5 * (x / std) ** 2), x_range)
            return tf.tensordot(window1d, window1d, axes=0)

        def generate_empty_heatmap():
            h = tf.cast(kwargs["im_h"], tf.int64)
            w = tf.cast(kwargs["im_w"], tf.int64)
            heatmap = tf.zeros((h, w))
            return heatmap

        def generate_non_empty_heatmap(x, y, kernel_size):
            kernel = gaussian_window2d(kernel_size, kernel_sigma)
            d = tf.cast(kernel_size // 2, tf.int64)
            h = tf.cast(kwargs["im_h"], tf.int64)
            w = tf.cast(kwargs["im_w"], tf.int64)
          
            left_padding = x - d
            kernel_x0 = -left_padding if left_padding < 0 else tf.cast(0, tf.int64)
            left_padding = tf.math.maximum(left_padding, 0)

            right_padding = w - 1 - x - d
            kernel_x1 = kernel_size+right_padding if right_padding < 0 else tf.cast(kernel_size, tf.int64)
            right_padding = tf.math.maximum(right_padding, 0)

            x_pads = tf.stack([left_padding, right_padding])

            top_padding = y - d
            kernel_y0 = -top_padding if top_padding < 0 else tf.cast(0, tf.int64)
            top_padding = tf.math.maximum(top_padding, 0)

            bottom_padding = h - 1 - y - d
            kernel_y1 = kernel_size+bottom_padding if bottom_padding < 0 else tf.cast(kernel_size, tf.int64)
            bottom_padding = tf.math.maximum(bottom_padding, 0)

            y_pads = tf.stack([top_padding, bottom_padding])
            pads = tf.stack([y_pads, x_pads])

            hm = tf.pad(kernel[kernel_y0:kernel_y1, kernel_x0:kernel_x1], pads)
            return hm

        heatmaps = []

        for i in range(num_corners):
          x = corners[i][0]
          y = corners[i][1]

          heatmap = tf.cond(
            tf.equal(x + y, 0),
            lambda: generate_empty_heatmap(),
            lambda: generate_non_empty_heatmap(x, y, kernel_size)
          )
          heatmaps.append(tf.expand_dims(heatmap, axis=-1))

        heatmap = tf.concat(heatmaps, axis=-1)
        return heatmap
