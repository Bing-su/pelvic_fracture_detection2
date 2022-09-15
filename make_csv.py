from pathlib import Path

import pandas as pd
from loguru import logger
from typer import Typer

cli = Typer()


@cli.command(no_args_is_help=True)
def make_csv(root: str) -> None:
    path = Path(root)

    image = []
    label = []

    for img in path.rglob("*.png"):
        if "abnormal" in str(img):
            y = 1
        else:
            y = 0
        image.append(str(img.resolve()))
        label.append(y)

    df = pd.DataFrame({"image": image, "label": label})
    logger.debug(f"num of data: {len(df)}")
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/data.csv", index=False)
    logger.info(f"data saved at: {Path('data/data.csv').resolve()}")


if __name__ == "__main__":
    cli()
