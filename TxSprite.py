from PIL import Image

class TxSprite:
    def __init__(self, msg_code, image_path):
        self.msg_code = msg_code
        self.num_colors = 2  # 1-bit means 2 colors (black & white)

        # Convert to 1-bit monochrome with dithering
        self.image = Image.open(image_path).convert("1")  # Monochrome + Dithering
        self.width, self.height = self.image.size
        self.bpp = 1  # 1-bit per pixel
        self.pixel_data = list(self.image.getdata())

    def pack(self):
        """Pack the sprite efficiently for 1-bit storage (row-major order)."""
        width_msb = self.width >> 8
        width_lsb = self.width & 0xFF
        height_msb = self.height >> 8
        height_lsb = self.height & 0xFF

        payload = bytearray([width_msb, width_lsb, height_msb, height_lsb, self.bpp, self.num_colors])

        ## Convert pixel data to 1-bit packed format (fix row alignment issue) ##
        packed_pixels = self.pack_1bit_row_major(self.pixel_data, self.width, self.height)
        payload.extend(packed_pixels)

        return payload

    @staticmethod
    def pack_1bit_row_major(pixel_data, width, height):
        """Packs a 1-bit monochrome image ensuring correct row order (left-to-right, top-to-bottom)."""
        packed_data = bytearray()

        for y in range(height):  # Iterate over rows
            byte = 0
            bit_position = 7  # Start with leftmost bit

            for x in range(width):  # Iterate over pixels in a row
                idx = y * width + x  # Correct indexing per row
                pixel = pixel_data[idx]

                if pixel > 127:  # White (1)
                    byte |= (1 << bit_position)

                bit_position -= 1  # Move to next bit

                if bit_position < 0 or x == width - 1:  # Store byte when full or end of row
                    packed_data.append(byte)
                    byte = 0
                    bit_position = 7  # Reset bit position for the next byte

        return packed_data
