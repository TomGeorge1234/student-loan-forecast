import base64
import struct
import unittest
import zlib

import numpy as np

from heatmap import smooth_heatmap_image


class SmoothHeatmapTests(unittest.TestCase):
    def test_surface_interpolates_values_and_preserves_axis_direction(self):
        image = smooth_heatmap_image(
            np.array([[0., 0.], [1., 1.]]), (0., 1.), ('#000000', '#ffffff'), size=3,
        )
        png = base64.b64decode(image.split(',', 1)[1])
        self.assertEqual(png[:8], b'\x89PNG\r\n\x1a\n')
        offset, compressed = 8, b''
        while offset < len(png):
            length = struct.unpack('!I', png[offset:offset + 4])[0]
            if png[offset + 4:offset + 8] == b'IDAT':
                compressed += png[offset + 8:offset + 8 + length]
            offset += length + 12
        rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(3, 10)
        np.testing.assert_array_equal(rows[:, 0], 0)
        np.testing.assert_array_equal(rows[:, 1:], np.array([[255] * 9, [128] * 9, [0] * 9]))


if __name__ == '__main__':
    unittest.main()
