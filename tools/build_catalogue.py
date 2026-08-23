#!/usr/bin/env python
"""
Build the star catalogue that ships with demeteor, from an HYG release.

    python tools/build_catalogue.py hyg_v44.csv.gz demeteor/catalogue/data/HYG44.tsv

HYG is published as a gzipped CSV of ~120,000 stars with 37 columns, under Git LFS, at
https://codeberg.org/astronexus/hyg -- so a plain download of the blob gives you a 133-byte LFS
pointer rather than the data. Fetch it with `git lfs`, or through the LFS batch API.

What this keeps, and why:

* stars of visual magnitude 6.00 or brighter, which is roughly what an all-sky camera records.
  About 5,000 of them, against 120,000 in the full release;
* six columns of the 37: right ascension, declination, distance, visual and absolute magnitude,
  and the proper name;
* the Sun is dropped. It passes the magnitude cut by a wide margin and is not a useful entry in a
  catalogue of fixed stars.

And the conventions, which are worth stating because they were reverse-engineered from the file
this replaces rather than written down anywhere:

* right ascension is converted from HYG's hours to degrees and rounded to six decimals, which is
  3.6 milliarcseconds -- far finer than any all-sky camera resolves. Rounding is what makes the
  output stable: the unrounded product of a float by 15 runs to 17 digits on a third of the rows;
* every other number is passed through exactly as HYG gives it;
* rows are ordered brightest first;
* a star with no proper name gets an em dash, so that the column is never empty and Catalogue's
  dtype for it stays a string. A dash rather than a word because it is read as a label in a table,
  where "unnamed" is noise repeated four thousand times;
* the first line is a comment, and Catalogue reads the header from the second (`header=1`).

The output is UTF-8. Some names are not ASCII -- Yunü -- and the previous file had that one
mojibaked, which is what happens if this is written in a locale encoding somewhere along the way.
"""
import argparse
import pathlib

import pandas as pd

#: Everything an all-sky camera has a chance of seeing
MAGNITUDE_LIMIT = 6.0

#: What stands in the name column for a star that has no proper name, which is most of them
NO_NAME = '\u2014'


def build(source: pathlib.Path) -> str:
    hyg = pd.read_csv(source)
    stars = hyg[(hyg.mag <= MAGNITUDE_LIMIT) & (hyg.id != 0)].sort_values('mag', kind='stable')

    table = pd.DataFrame({
        'ra': (stars.ra * 15).round(6),
        'dec': stars.dec,
        'dist': stars.dist,
        'vmag': stars.mag,
        'absmag': stars.absmag,
        'name': stars.proper.fillna(NO_NAME),
    })
    return '#\n' + table.to_csv(sep='\t', index=False, lineterminator='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('source', type=pathlib.Path, help="HYG release, .csv or .csv.gz")
    parser.add_argument('target', type=pathlib.Path, help="where to write the catalogue")
    args = parser.parse_args()

    catalogue = build(args.source)
    args.target.write_text(catalogue, encoding='utf-8')
    print(f"{args.target}: {len(catalogue.splitlines()) - 2} stars "
          f"to magnitude {MAGNITUDE_LIMIT}")


if __name__ == '__main__':
    main()
