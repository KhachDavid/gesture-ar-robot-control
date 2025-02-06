from PIL import Image
import numpy as np
import struct

class TxSprite:
    def __init__(self, msg_code, image_path):
        self.msg_code = msg_code
        self.image = Image.open(image_path).convert("RGB")

        # Convert to indexed image (16 colors max)
        self.image = self.image.convert("P", palette=Image.ADAPTIVE, colors=16)
        palette = self.image.getpalette()[:48]  # First 16 colors * 3 (RGB)
        
        self.width, self.height = self.image.size
        self.num_colors = len(palette) // 3
        self.pixel_data = list(self.image.getdata())

        # Ensure black is at index 0 and white at index 1
        self.palette_data = self.rearrange_palette(palette)

    def rearrange_palette(self, palette):
        """Ensure black is at the first index and white at second."""
        colors = np.array(palette).reshape(-1, 3)
        
        # Find black (closest to [0,0,0]) and white (closest to [255,255,255])
        black_idx = np.argmin(np.sum(colors ** 2, axis=1))  # Darkest color
        white_idx = np.argmax(np.sum(colors, axis=1))        # Brightest color

        # Swap colors if needed
        colors[[0, black_idx]] = colors[[black_idx, 0]]
        colors[[1, white_idx]] = colors[[white_idx, 1]]

        # Rearrange pixel indices to match new palette ordering
        new_pixel_data = [0 if p == black_idx else 1 if p == white_idx else p for p in self.pixel_data]
        self.pixel_data = new_pixel_data

        return colors.flatten().tolist()

    def pack(self):
        """Pack the sprite into a binary format for Frame."""
        width_msb = self.width >> 8
        width_lsb = self.width & 0xFF
        height_msb = self.height >> 8
        height_lsb = self.height & 0xFF
        bpp = 4  # 16 colors = 4 bits per pixel

        packed_pixels = self.pack_4bit(self.pixel_data)

        # Format: width, height, bpp, numColors, palette data, pixel data
        payload = bytearray([width_msb, width_lsb, height_msb, height_lsb, bpp, self.num_colors])
        payload.extend(self.palette_data)
        payload.extend(packed_pixels)

        return payload

    @staticmethod
    def pack_4bit(pixel_data):
        """Pack 4-bit pixel data into bytes."""
        packed = bytearray()
        for i in range(0, len(pixel_data), 2):
            if i + 1 < len(pixel_data):
                packed.append((pixel_data[i] << 4) | (pixel_data[i + 1] & 0x0F))
            else:
                packed.append(pixel_data[i] << 4)
        return packed
