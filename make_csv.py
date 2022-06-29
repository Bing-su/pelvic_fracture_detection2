from pathlib import Path

import pandas as pd
from typer import Typer

cli = Typer()


@cli.command()
def ai_voucher_only() -> None:
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


@cli.command()
def with_MURA() -> None:
    path = Path.home() / "MURA"

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
    df.to_csv("data/mura.csv", index=False)


if __name__ == "__main__":
    cli()
