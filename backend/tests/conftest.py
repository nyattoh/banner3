import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_text_image():
    """Create a sample image with text for testing."""
    # Create a 400x200 white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 24)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Add some text
    draw.text((50, 50), "Hello World", fill='black', font=font)
    draw.text((50, 100), "テストテキスト", fill='black', font=font)
    
    return np.array(img)


@pytest.fixture
def sample_banner_image():
    """Create a complex banner image with text, objects, and background."""
    # Create a 600x400 image with gradient background
    img = Image.new('RGB', (600, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Create gradient background
    for y in range(400):
        color_value = int(255 - (y / 400) * 100)
        draw.line([(0, y), (600, y)], fill=(color_value, color_value + 20, color_value + 40))
    
    # Add a product-like rectangle (simulating an object)
    draw.rectangle([150, 100, 250, 200], fill='blue', outline='darkblue', width=2)
    
    # Add text
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 32)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    draw.text((300, 50), "SALE 50% OFF", fill='red', font=font)
    draw.text((300, 300), "Limited Time", fill='black', font=font)
    
    return np.array(img)


@pytest.fixture
def sample_empty_image():
    """Create an empty image for testing edge cases."""
    img = Image.new('RGB', (100, 100), color='white')
    return np.array(img)


@pytest.fixture
def sample_noisy_image():
    """Create a noisy image for testing robustness."""
    # Start with white background
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Add random noise
    noise = np.random.randint(0, 50, (200, 300, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)
    
    # Add some text on the noisy background
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    draw.text((50, 50), "Noisy Text", fill='black', font=font)
    
    return np.array(img_pil)