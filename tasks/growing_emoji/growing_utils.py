from PIL import Image, ImageDraw, ImageFont


def get_emoji_image(emoji: str = "🦎", padding: int = 8, size: int = 24):
    """
    :param emoji: String containing a single emoji character. Defaults to "🦎".
    :type emoji: str
    :param padding: Number of pixels to pad to the sides. Defaults to 2.
    :type padding: int
    :param size: Total image size without padding. Defaults to 24.
    :type size: int
    :returns: Output PIL.Image containing an emoji on transparent background.
    """
    scale = size / 109
    dims = (int(padding * 2 * scale + 128), int(padding * 2 * scale + 128))
    with Image.new("RGBA", dims, (255, 255, 255, 0)) as image:
        font = ImageFont.truetype("NotoColorEmoji.ttf", 109)
        draw = ImageDraw.Draw(image)
        draw.text(
            (padding * scale, padding * scale),
            emoji.strip(),
            font=font,
            embedded_color=True,
        )
        image = image.resize(
            (size + 2 * padding, size + 2 * padding), Image.Resampling.LANCZOS
        )
        return image
