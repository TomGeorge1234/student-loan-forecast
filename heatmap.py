"""Smooth raster rendering for a regularly sampled repayment grid."""

import base64
import struct
import zlib

import numpy as np


def smooth_heatmap_image(
    values: np.ndarray,
    domain: tuple[float, ...],
    colours: tuple[str, ...],
    size: int = 260,
) -> str:
    """Bilinearly interpolate numeric samples, then map them to an RGB PNG.

    Rasterizing the colour surface avoids seams between separately drawn cells.
    Values are interpolated before colouring; the original grid remains available
    to the chart as a transparent tooltip layer.
    """
    rows, columns = values.shape
    x = np.linspace(0, columns - 1, size)
    y = np.linspace(0, rows - 1, size)
    horizontal = np.array([np.interp(x, np.arange(columns), row) for row in values])
    smooth = np.array([
        np.interp(y, np.arange(rows), column) for column in horizontal.T
    ]).T
    palette = np.array([
        [int(colour[i:i + 2], 16) for i in (1, 3, 5)] for colour in colours
    ])
    pixels = np.stack([
        np.interp(smooth, domain, palette[:, channel]) for channel in range(3)
    ], axis=-1)
    # Image coordinates run downwards; loan amounts increase upwards.
    pixels = np.rint(pixels[::-1]).astype(np.uint8)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack('!I', len(data)) + kind + data + struct.pack(
            '!I', zlib.crc32(kind + data) & 0xffffffff,
        )

    scanlines = b''.join(b'\x00' + row.tobytes() for row in pixels)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!2I5B', size, size, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(scanlines))
    png += chunk(b'IEND', b'')
    return 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')
