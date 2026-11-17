"""Fetch only the 2023 FARS national accident table from NHTSA."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

URL = "https://static.nhtsa.gov/nhtsa/downloads/FARS/2023/National/FARS2023NationalCSV.zip"
ARCHIVE = Path("data/FARS2023NationalCSV.zip")
OUT = Path("data/accident.csv")


def main():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(URL, ARCHIVE)
    with ZipFile(ARCHIVE) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/accident.csv"))
        with archive.open(member) as src, OUT.open("wb") as dst:
            dst.write(src.read())
    print(f"Extracted {OUT} from {URL}")


if __name__ == "__main__":
    main()

