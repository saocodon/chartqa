"""
Bar chart detector - test script.
"""
import logging
from pathlib import Path

from src.detector import BarDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    detector = BarDetector()
    folder = Path("tests")
    
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            print(f"\n{'='*60}")
            print(f"Processing: {file.name}")
            print(f"{'='*60}")
            try:
                result = detector.run(str(file))
                print(f"Title: {result.title}")
                print(f"Direction: {result.direction}")
                print(f"Scale: value = {result.y_axis_scale[0]:.4f} * pixel + {result.y_axis_scale[1]:.4f}")
                print("Data:")
                for label, value in result.data.items():
                    print(f"  {label}: {value:.2f}")
            except Exception as e:
                print(f"ERROR: {e}")

if __name__ == "__main__":
    main()