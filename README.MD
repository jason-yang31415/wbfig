# wbfig

## installation

The latest wheel file can be downloaded from the [release page](https://github.com/jason-yang31415/wbfig/releases). To install, run

```
pip install wbfig-0.1.0-py3-none-any.whl
```

## usage

```
> py -m wbfig -h
usage: __main__.py [-h] [--edit] [--edit_align] [--edit_crop] [--edit_ladder] layout_file

make western blot figures

positional arguments:
  layout_file         path to layout description file

optional arguments:
  -h, --help          show this help message and exit
  --edit, -e          show prompts to modify all layout parameters
  --edit_align, -ea   show prompts to modify layout parameters for alignment only
  --edit_crop, -ec    show prompts to modify layout parameters for cropping only
  --edit_ladder, -el  show prompts to modify layout parameters for ladder positions only
```

The only required argument is `layout_file`, which is the name of the layout YAML file without the `.yaml` extension (i.e., to generate a figure for `example.yaml`, run `py -m wbfig example`).

## layout specification

The figure to be generated is defined by a layout specification file in YAML format. Keys `stacks`, `crops`, and `figure` are required while keys `ladder` and `format` are optional.

`stacks` is a dictionary mapping stack names to a list of paths to exposure images of the same stack. For example,

```
stacks:
  A:
    - exp0.tif
    - exp1.tif
  B:
    - exp2.tif
    - exp3.tif
```

defines two stacks, `A` and `B`, consisting of two exposures each.

`crops` is a dictionarry mapping crop names to a list of paths to exposure images for which the indicated crop will be used in the figure. For example,

```
crops:
  GeneX:
    - exp0.tif
    - exp2.tif
  GeneY:
    - exp1.tif
    - exp3.tif
```

defines two crops, `GeneX` and `GeneY`, consisting of two exposures each.

`figure` is a list of items specifying the layout of a single column. Keys `header` and `rows` are required for each column while key `format` is optional.

`header` is a list of header rows, each of which is a list of header labels. Header labels have required key `text` and optional keys `span`, `border`, `position`, and `rotation`. Header labels default to span a single lane with no border or rotation. `span` sets the number of lanes the label should span. `border: line` draws a line beneath the label. `position: left` places the label to the left of the header row (this label has span 0). `rotation` sets the rotation of the label in degrees. Each header row must span the same number of lanes.

`rows` is a list of cropped exposures. Rows have required key `crop` and optional keys `exposure`, `ladder`, `label`, and `format`. `crop` indicates which region of the exposure to show. `exposure` indicates which exposure to show and defaults to the first exposure in the list of exposures for that crop. `ladder: yes` indicates that the exposure should have ladder markings. `label` specifies the label to the left of the cropped exposure and has required key `text` and optional keys `span` and `border`. `span` sets the number of rows the label should span. `border: line` draws a line to the right of the label.

For example,

```
figure:
  - header:
      - - text: HEK293T
          span: 6
          border: line
      - - text: "\textit{GENEX}:"
          position: left
        - text: WT
          span: 3
          border: line
        - text: KO
          span: 3
          border: line
  - rows:
      - crop: GeneX
        exposure: exp0.tif
        ladder: yes
        label:
          text: GENEX
          span: 2
          border: line
      - crop: GeneX
        exposure: exp2.tif
        ladder: yes
      - crop: GeneY
        exposure: exp1.tif
        ladder: yes
        label:
          text: GENEY
          span: 2
          border: line
      - crop: GeneY
        ladder: yes
        exposure: exp3.tif
```

describes a single column with two rows of header labels (`HEK293T` spanning all 6 lanes and `WT` and `KO` spanning 3 lanes each) and four rows of cropped exposures (two of `GeneX` and two of `GeneY`).

Note that label text is typeset with LaTeX.

### formatting

Key `format` can be set at the top-level, under each figure column, or under each figure column row to adjust formatting parameters. Formatting parameters are inherited from the top-level to figure columns to figure column rows.

`width`: width (cm) of a figure column; required.

`height`: height (cm) of a cropped exposure; default `null`.

`snap_height`: snap height (cm) of a cropped exposure to the smallest multiple of `snap_height` that fits the bounding box; default `0.5`.

`min_height`: minimum height (cm) of a cropped exposure; default `1`.

`vpad`: vertical padding (cm) to add to top and bottom of bounding box; default `0`.

`hpad`: horizontal padding (cm) to add to left and right of bounding box; default `0.5`.

`vgap`: vertical gap (cm) between figure column rows; value of `vgap` for the first row of a figure column sets the gap below the header; default ``.

`hgap`: horizontal gap (cm) between the cropped exposure and label; default ``.

`linewidth`: line width (cm) for when `border: line` is set; default `0.02`.

`fontsize`: font size of label text; currently does nothing; default `12`.

`font`: font of label text; currently does nothing; default `Helvetica`.

`value_black`: black pixel intensity (0-255) to rescale exposures; default `0`.

`value_white`: white pixel intensity (0-255) to rescale exposures; default `255`.

## controls

### alignment

Exposures to be aligned are overlaid with a reference exposure. Use arrow keys to move the exposure, and [ and ] to rotate the exposure counterclockwise and clockwise, respectively. Hold shift to move/rotate faster. Hold ctrl+left or ctrl+right to view the previous or next exposure. Press enter to finish editing alignment.

### cropping

Right-click four times to mark left, right, top, and bottom anchors to define the bounding box. The first and second anchors mark two lanes (e.g. first and last lanes) to define the left and right edges of the bounding box. Use up and down arrow keys to indicate which lane is being selected. The third and fourth anchors mark the top and bottom edges of the bounding box. Hold ctrl+left or ctrl+right to view the previous or next exposure. Press esc to reset the bounding box. Press enter to finish editing the bounding box.

### ladders

Right-click to mark ladder positions. Use up and down arrow keys to indicate which ladder is being selected. Hold ctrl+left or ctrl+right to view the previous or next exposure. Press esc to reset ladder positions. Press enter to finish editing ladder positions.
