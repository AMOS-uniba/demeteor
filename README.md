The `demeteor` package is a collection of Python utilities for storing, manipulating (and hopefully later also) visualizing meteor data.
It also contains functions for claculating all-sky projections that are reused across various [AMOS](https://github.com/AMOS-uniba/) projects.

The name is a Slovak pun on "kde (je) meteor?" ("where (is the) meteor?"),
but I thought it also had a nice ring on its own. And it was also free on `pypi`.

Not to be confused with [demetria](https://github.com/sesquideus/demetria/), my own library
for working with 2D scalar and vector fields.

## Licence

demeteor is free software: you may redistribute it and modify it under the terms of the **GNU Lesser
General Public License, version 3 or later**, as published by the Free Software Foundation. The full
texts are in [`COPYING.LESSER`](COPYING.LESSER) and [`COPYING`](COPYING) — the LGPL is a set of
additional permissions on top of the GPL, so both belong here.

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

Being the *Lesser* GPL, importing demeteor puts no licensing obligation on the program that imports
it; modifying demeteor itself does.

**The bundled star catalogue is not covered by that licence.** See below.

## Bundled star catalogue

`Catalogue.bundled()` returns a subset of the [HYG star database](https://codeberg.org/astronexus/hyg)
version 4.4 by **astronexus**, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) — 5070 stars down to visual
magnitude 6. The file is modified from the original (fewer columns, magnitude-limited); the changes
and the full attribution are in
[`demeteor/catalogue/data/README.md`](demeteor/catalogue/data/README.md).

CC BY-SA is a copyleft licence and it attaches to that data file and to anything adapted from it,
not to this library's code.
