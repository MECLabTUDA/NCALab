from pilmoji import Pilmoji  # type: ignore[import-untyped]
from PIL import Image, ImageFont


def get_emoji_image(emoji: str = "🦎", padding: int = 2, size: int = 24):
    """_summary_

    Args:
        emoji (str, optional): String containing a single emoji character. Defaults to "🦎".
        padding (int, optional): Number of pixels to pad to the sides. Defaults to 2.
        size (int, optional): Total image size without padding. Defaults to 24.

    Returns:
        Image: Output PIL.Image containing an emoji on transparent background.
    """
    dims = (padding * 2 + size, padding * 2 + size)
    with Image.new("RGBA", dims, (255, 255, 255, 0)) as image:
        font = ImageFont.truetype("arial.ttf", size)
        with Pilmoji(image) as pilmoji:
            pilmoji.text((padding, padding - size), emoji.strip(), (0, 0, 0), font)
        return image