from pathlib import Path

import pandas as pd

path = Path.home() / "data"

image = []
label = []

for img in path.rglob("*.png"):
    if "abnormal" in str(img):
        y = 1
    else:
        y = 0
    image.append(img)
    label.append(y)

df = pd.DataFrame({"image": image, "label": label})
Path("data").mkdir(exist_ok=True)
df.to_csv("data/data.csv", index=False)
