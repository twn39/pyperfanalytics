import os
import math
from PIL import Image, ImageDraw, ImageFont

def create_gallery(input_dir="images", output_file="images/py_charts_gallery.jpg"):
    # 1. Collect all py_*.png files
    files = [f for f in os.listdir(input_dir) if f.startswith("py_") and f.endswith(".png") and "gallery" not in f]
    files.sort()
    
    if not files:
        print("No Python charts found to combine.")
        return

    # 2. Setup grid parameters (Larger size)
    cols = 4
    rows = math.ceil(len(files) / cols)
    
    # High-resolution dimensions (matches original 1200x800)
    thumb_w = 1200
    thumb_h = 800
    
    # Create master canvas (White background)
    gallery_w = cols * thumb_w
    gallery_h = rows * thumb_h + 200 # Extra space for title
    gallery = Image.new("RGB", (gallery_w, gallery_h), "white")
    draw = ImageDraw.Draw(gallery)
    
    # Larger font for high-res output
    try:
        # Try to find a system font
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 120)
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
    except:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()
        
    draw.text((gallery_w//2, 100), "PyPerfAnalytics Visualization Suite (Plotly)", fill="black", font=font_title, anchor="mm")
    
    # 3. Paste images into grid
    for i, filename in enumerate(files):
        row = i // cols
        col = i % cols
        
        filepath = os.path.join(input_dir, filename)
        img = Image.open(filepath)
        # img is already 1200x800 (or close to it) from write_image, no need to thumbnail if we want max size
        # but we'll ensure it fits the grid cell
        img.thumbnail((thumb_w, thumb_h))
        
        # Center the thumbnail in its cell
        x = col * thumb_w + (thumb_w - img.width) // 2
        y = row * thumb_h + 200 + (thumb_h - img.height) // 2
        
        gallery.paste(img, (x, y))
        
        # Add label to each sub-chart
        label = filename.replace("py_", "").replace(".png", "").replace("_", " ").title()
        draw.text((x + thumb_w//2, y + thumb_h - 40), label, fill="#666666", font=font_label, anchor="mm")

    # 4. Save as JPG with high quality
    gallery.save(output_file, "JPEG", quality=95, optimize=True)
    print(f"High-res Gallery created successfully: {output_file} ({len(files)} charts combined)")

if __name__ == "__main__":
    create_gallery()
