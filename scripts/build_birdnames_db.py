#!/usr/bin/env python3
"""Build birdnames.db from the AIY birds V1 label map CSV.

Downloads the label map from Google and creates a SQLite database
mapping scientific names to common names. Run this once before building
the Docker image, or the Dockerfile entrypoint will run it automatically.

Usage:
    python scripts/build_birdnames_db.py
"""
import csv
import io
import os
import sqlite3
import urllib.request

LABELS_URL = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "birdnames.db")

# Common name overrides / supplements for well-known species
# The label map only has scientific names; these are the most common birds
# in North America. For species not listed here, the scientific name is used.
COMMON_NAMES = {
    "Cardinalis cardinalis": "Northern Cardinal",
    "Cyanocitta cristata": "Blue Jay",
    "Turdus migratorius": "American Robin",
    "Melanerpes carolinus": "Red-bellied Woodpecker",
    "Sitta carolinensis": "White-breasted Nuthatch",
    "Poecile atricapillus": "Black-capped Chickadee",
    "Haemorhous mexicanus": "House Finch",
    "Spinus tristis": "American Goldfinch",
    "Zenaida macroura": "Mourning Dove",
    "Sturnus vulgaris": "European Starling",
    "Passer domesticus": "House Sparrow",
    "Junco hyemalis": "Dark-eyed Junco",
    "Melospiza melodia": "Song Sparrow",
    "Thryothorus ludovicianus": "Carolina Wren",
    "Baeolophus bicolor": "Tufted Titmouse",
    "Sialia sialis": "Eastern Bluebird",
    "Corvus brachyrhynchos": "American Crow",
    "Colaptes auratus": "Northern Flicker",
    "Dryobates pubescens": "Downy Woodpecker",
    "Dryobates villosus": "Hairy Woodpecker",
    "Agelaius phoeniceus": "Red-winged Blackbird",
    "Quiscalus quiscula": "Common Grackle",
    "Molothrus ater": "Brown-headed Cowbird",
    "Mimus polyglottos": "Northern Mockingbird",
    "Dumetella carolinensis": "Gray Catbird",
    "Toxostoma rufum": "Brown Thrasher",
    "Setophaga petechia": "Yellow Warbler",
    "Archilochus colubris": "Ruby-throated Hummingbird",
    "Calypte anna": "Anna's Hummingbird",
    "Pipilo erythrophthalmus": "Eastern Towhee",
    "Zonotrichia albicollis": "White-throated Sparrow",
    "Columba livia": "Rock Pigeon",
    "Cathartes aura": "Turkey Vulture",
    "Buteo jamaicensis": "Red-tailed Hawk",
    "Accipiter cooperii": "Cooper's Hawk",
    "Haliaeetus leucocephalus": "Bald Eagle",
    "Branta canadensis": "Canada Goose",
    "Anas platyrhynchos": "Mallard",
}


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Downloading label map from {LABELS_URL}...")
    response = urllib.request.urlopen(LABELS_URL)
    csv_text = response.read().decode("utf-8")

    conn = sqlite3.connect(OUTPUT_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS birdnames (
            scientific_name TEXT PRIMARY KEY,
            common_name TEXT NOT NULL
        )
    """)

    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader, None)
    count = 0

    for row in reader:
        if len(row) < 2:
            continue
        idx = int(row[0])
        scientific_name = row[1].strip()
        if not scientific_name or idx == 964:  # skip background
            continue

        common_name = COMMON_NAMES.get(scientific_name, scientific_name)
        conn.execute(
            "INSERT OR REPLACE INTO birdnames (scientific_name, common_name) VALUES (?, ?)",
            (scientific_name, common_name),
        )
        count += 1

    conn.commit()
    conn.close()
    print(f"Created {OUTPUT_PATH} with {count} species entries")


if __name__ == "__main__":
    main()
