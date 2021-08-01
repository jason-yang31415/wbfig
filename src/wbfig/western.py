from numbers import Number
import numpy as np
import yaml
import os
import pickle
from PIL import Image
import pyx

from .input import Alignment, CropTransform, Ladder
from .input import do_alignment, do_crop, do_ladder


class Formatting:
    """methods for representing formatting inheritance"""

    keys = {
        "width",
        "height",
        "snap_height",
        "min_height",
        "vpad",
        "hpad",
        "vgap",
        "hgap",
        "linewidth",
        "fontsize",
        "font",
        "value_black",
        "value_white",
    }
    default = {
        "width": 4,
        "height": None,
        "snap_height": 0.5,
        "min_height": 1,
        "vpad": 0,
        "hpad": 0.5,
        "vgap": 0.1,
        "hgap": 0.2,
        "linewidth": 0.02,
        "fontsize": 12,
        "font": None,
        "value_black": 0,
        "value_white": 255,
    }

    @classmethod
    def empty(cls):
        return {k: None for k in Formatting.keys}

    @classmethod
    def parse(cls, dictionary):
        format = {}
        for k in Formatting.keys:
            if k in dictionary:
                format[k] = dictionary[k]
            else:
                format[k] = None
        return format

    @classmethod
    def inherit(cls, old, new):
        format = {}
        for k in Formatting.keys:
            if k in new and new[k] is not None:
                format[k] = new[k]
            elif k in old and old[k] is not None:
                format[k] = old[k]
            elif k in Formatting.default:
                format[k] = Formatting.default[k]
        return format


class LayoutSpecError(Exception):
    pass


class Layout:
    """represents how to lay out the figure"""

    opt_keys = {"ladder", "format"}
    req_keys = {"stacks", "crops", "figure"}

    ladders = {"P7719": (10, 17, 26, 34, 43, 55, 72, 95, 130, 180, 250)}
    default_ladder = "P7719"

    @classmethod
    def parse_layout(cls, layout_spec):
        """parse layout specification dictionary"""

        # Check that required keys are present and all keys are required or
        # optional.
        for k in Layout.req_keys:
            if k not in layout_spec:
                raise LayoutSpecError("layout spec missing key: {}".format(k))
        for k in layout_spec:
            if k not in Layout.req_keys and k not in Layout.opt_keys:
                raise LayoutSpecError("layout spec has unknown key: {}".format(k))

        # Set layout ladder. Ladder is optional and defaults to the NEB P7719S.
        if "ladder" in layout_spec:
            layout_ladder = layout_spec["ladder"]
            if isinstance(layout_ladder, str):
                if layout_ladder in Layout.ladders:
                    ladder = Layout.ladders[layout_ladder]
                else:
                    raise LayoutSpecError(
                        "layout spec has unknown ladder: {}".format(layout_ladder)
                    )
            elif isinstance(layout_ladder, list):
                if all(map(lambda x: isinstance(x, Number), layout_ladder)):
                    ladder = layout_ladder
                else:
                    raise LayoutSpecError(
                        "layout spec has unknown ladder: {}".format(layout_ladder)
                    )
        else:
            ladder = Layout.ladders[Layout.default_ladder]

        # file -> Exposure
        exposures = {}
        stacks = {}

        # Set stacks of exposures. Stacks represent groups of exposures of the
        # same membranes.
        # name -> Stack
        layout_stacks = layout_spec["stacks"]
        if isinstance(layout_stacks, dict):
            for name, files in layout_stacks.items():
                if isinstance(files, list) and all(
                    map(lambda x: isinstance(x, str), files)
                ):
                    stacks[name] = Stack.open(name, files)
                    for f in files:
                        exposures[f] = stacks[name].get_exposure(f)
                else:
                    raise LayoutSpecError(
                        "layout spec must have array of files for stack: {}".format(
                            name
                        )
                    )
        else:
            raise LayoutSpecError("layout spec is invalid for key: stacks")

        # Set exposure crops. Crops represent regions of exposures (potentially
        # of different stacks) for the same membrane.
        # name -> Crop
        layout_crops = layout_spec["crops"]
        crops = {}
        if isinstance(layout_crops, dict):
            for name, crop in layout_crops.items():
                if isinstance(layout_crops, str):
                    if crop in exposures:
                        crops[name] = Crop(name, [exposures[crop]])
                    else:
                        raise LayoutSpecError(
                            "layout spec has reference to unknown exposure: {}".format(
                                crop
                            )
                        )
                elif isinstance(crop, list):
                    if all(map(lambda x: isinstance(x, str) and x in exposures, crop)):
                        crops[name] = Crop(name, [exposures[x] for x in crop])
                    else:
                        raise LayoutSpecError(
                            "layout spec has reference to unknown stack(s): {}".format(
                                crop
                            )
                        )
                else:
                    raise LayoutSpecError(
                        "layout spec must have stack or array of stacks for crop: {}".format(
                            name
                        )
                    )
        else:
            raise LayoutSpecError("layout spec is invalid for key: crops")

        # Set formatting.
        if "format" in layout_spec:
            format = Formatting.parse(layout_spec["format"])
        else:
            format = Formatting.empty()

        # Set figure columns. The output figure is organized into columns of
        # crops.
        layout_columns = layout_spec["figure"]
        columns = []
        if isinstance(layout_columns, list):
            for column in layout_columns:
                columns.append(FigureColumn.parse_layout(column, crops, format))
        else:
            raise LayoutSpecError("layout spec is invalid for key: figure")

        return Layout(ladder, exposures, stacks, crops, columns)

    def __init__(self, ladder, exposures, stacks, crops, columns):
        self.ladder = ladder
        self.exposures = exposures
        self.stacks = stacks
        self.crops = crops
        self.columns = columns

    def make_figure(self, path):
        pyx.text.set(pyx.text.LatexEngine)
        pyx.text.preamble(r"\usepackage[scaled]{helvet}")
        pyx.text.preamble(r"\usepackage[T1]{fontenc}")
        pyx.text.preamble(r"\renewcommand\familydefault{\sfdefault}")

        c = pyx.canvas.canvas()
        x = 0
        for column in self.columns:
            dim = column.dim()
            column.make_figure(c, (x, dim[1]))
            x += dim[0]
        c.writeSVGfile(path + ".svg")


class FigureColumn:
    """represents layout of a single column in the figure"""

    req_keys = {"header", "rows"}
    opt_keys = {"format"}
    row_keys = {"label", "crop", "exposure", "ladder"}

    @classmethod
    def parse_layout(cls, layout_spec, crops, inherited_format):
        """parse layout specification dictionary for a single column in the
        figure"""

        column = cls()

        # Check that required keys are present and all keys are required or
        # optional.
        for k in FigureColumn.req_keys:
            if k not in layout_spec:
                raise LayoutSpecError("layout column spec missing key: {}".format(k))
        for k in layout_spec:
            if k not in FigureColumn.req_keys and k not in FigureColumn.opt_keys:
                raise LayoutSpecError(
                    "layout column spec has unknown key: {}".format(k)
                )

        # Set the header for the column. The header consists of rows of lane
        # labels. Labels can span multiple lanes.

        # TODO
        header = layout_spec["header"]
        if isinstance(header, list):
            row_lengths = []
            for row in header:
                if isinstance(row, list):
                    l = 0
                    for item in row:
                        if "position" in item:
                            pass
                        else:
                            l += item["span"] if "span" in item else 1
                    row_lengths.append(l)
                else:
                    raise LayoutSpecError("layout column spec has invalid header")

            if all(map(lambda x: x == row_lengths[0], row_lengths)):
                column.num_lanes = row_lengths[0]
            else:
                raise LayoutSpecError(
                    "layout column spec has inconsistent number of lanes"
                )
        else:
            raise LayoutSpecError("layout column spec has invalid header")
        column.header = header

        # Set formatting for the column.
        if "format" in layout_spec and layout_spec["format"] is not None:
            column.format = Formatting.inherit(
                inherited_format, Formatting.parse(layout_spec["format"])
            )
        else:
            column.format = Formatting.inherit(inherited_format, Formatting.empty())

        # Set rows.
        layout_rows = layout_spec["rows"]
        column.rows = []
        column.row_formats = []
        if isinstance(layout_rows, list):
            for layout_row in layout_rows:
                if "crop" not in layout_row:
                    raise LayoutSpecError("layout column spec row missing crop")
                if layout_row["crop"] not in crops:
                    raise LayoutSpecError(
                        "layout column spec row has unknown crop {}".format(row["crop"])
                    )
                crops[layout_row["crop"]].set_num_lanes(column.num_lanes)

                row = {}
                # TODO: move to class
                for k in FigureColumn.row_keys:
                    if k == "crop":
                        row[k] = crops[layout_row[k]]
                    elif k in layout_row:
                        row[k] = layout_row[k]
                column.rows.append(row)

                if "format" in layout_row:
                    row_format = Formatting.inherit(
                        column.format, Formatting.parse(layout_row["format"])
                    )
                else:
                    row_format = Formatting.inherit(column.format, Formatting.empty())
                column.row_formats.append(row_format)
        else:
            raise LayoutSpecError("layout column spec has invalid rows")

        return column

    def __init__(self):
        pass

    def dim(self):
        crop_dims = [
            row["crop"].dim(format) for row, format in zip(self.rows, self.row_formats)
        ]
        # width, height of crops area of the column in figure units
        crops_width = max([d[0] for d in crop_dims])
        crops_height = sum([d[1] for d in crop_dims]) + sum(
            [format["vgap"] for format in self.row_formats[1:]]
        )
        return crops_width, crops_height

    def _make_header(self, canvas, tl):
        x, y = tl
        vgap = self.row_formats[0]["vgap"]
        y += vgap
        l, r = x + self.format["hpad"], x + self.format["width"] - self.format["hpad"]
        width_per_lane = (r - l) / self.num_lanes
        for row in self.header[::-1]:
            lane = 0
            max_th = 0
            for item in row:
                if "position" in item:
                    if item["position"] == "left":
                        tbox = pyx.text.text(
                            x - self.format["hgap"],
                            y,
                            item["text"],
                            [pyx.text.halign.right],
                        )
                else:
                    span = item["span"] if "span" in item else 1
                    cx = (lane + span / 2) * width_per_lane
                    if "rotation" in item and item["rotation"] != 0:
                        rot = item["rotation"]
                        tbox = pyx.text.text(
                            l + cx,
                            y,
                            item["text"],
                            [pyx.text.halign.left, pyx.trafo.rotate(rot)],
                        )
                    else:
                        tbox = pyx.text.text(
                            l + cx, y, item["text"], [pyx.text.halign.center]
                        )

                    if "border" in item:
                        if item["border"] == "line":
                            canvas.stroke(
                                pyx.path.line(
                                    l + cx - span * width_per_lane / 2 + 0.1,
                                    y - vgap / 2,
                                    l + cx + span * width_per_lane / 2 - 0.1,
                                    y - vgap / 2,
                                ),
                                [pyx.style.linewidth(self.format["linewidth"])],
                            )
                    lane += span

                canvas.insert(tbox)

                th = pyx.unit.tocm(tbox.bbox().height())
                if th > max_th:
                    max_th = th
            y += vgap + max_th

    def make_figure(self, canvas, tl):
        x, y = tl
        self._make_header(canvas, tl)
        for i, (row, format) in enumerate(zip(self.rows, self.row_formats)):
            if i != 0:
                y -= format["vgap"]

            crop = row["crop"]
            dim = crop.dim(format)
            y -= dim[1]

            im = row["exposure"] if "exposure" in row else None
            canvas.insert(
                pyx.bitmap.bitmap(
                    x,
                    y,
                    crop.image(im=im, format=format)[0],
                    width=dim[0],
                    height=dim[1],
                    compressmode="Flate",
                )
            )

            if "label" in row:
                label = row["label"]
                # TODO: move to class
                if "text" not in label:
                    raise LayoutSpecError("label has no text!")

                # gap between exposure crop and label
                label_gap = format["hgap"]

                # text anchor x position
                tx = x - label_gap
                # text box top
                ttop = y + dim[1]
                if "span" not in label or label["span"] == 1:
                    # text box height
                    th = dim[1]
                else:
                    span = label["span"]
                    # text box height
                    th = sum(
                        [
                            r["crop"].dim(f)[1]
                            for (r, f) in zip(
                                self.rows[i : i + span], self.row_formats[i : i + span]
                            )
                        ]
                    ) + sum([f["vgap"] for f in self.row_formats[i + 1 : i + span]])
                canvas.text(
                    tx,
                    ttop - th / 2,
                    label["text"],
                    [pyx.text.valign.middle, pyx.text.halign.boxright],
                )

                if "border" in label:
                    if label["border"] == "line":
                        canvas.stroke(
                            pyx.path.line(
                                tx + label_gap / 2, ttop, tx + label_gap / 2, ttop - th
                            ),
                            [pyx.style.linewidth(format["linewidth"])],
                        )

                # TODO: account for text in column width
                # print(pyx.unit.tocm(pyx.text.text(
                #     0, 0, format["label"]["text"]).bbox().width()))

            if "ladder" in row and row["ladder"]:
                ladder = crop.get_ladder(format)
                for _ly, weight in ladder:
                    # ladder top
                    ltop = y + dim[1]
                    ly = ltop - _ly
                    lx = x + dim[0]
                    canvas.stroke(
                        pyx.path.line(lx, ly, lx + 0.1, ly),
                        [pyx.style.linewidth(format["linewidth"])],
                    )
                    canvas.text(
                        lx + 0.2,
                        ly,
                        str(weight),
                        [pyx.text.valign.middle, pyx.text.halign.boxleft],
                    )


class Stack:
    """represents a stack of exposures with membranes in the same positions"""

    @classmethod
    def open(cls, name, files):
        """open a stack with name `name` and list of exposures `files`"""
        stack = cls(name)
        for f in files:
            stack.append_exposure(f, Exposure.open(f, stack))
        return stack

    def __init__(self, name):
        self.name = name
        # list of exposure files in order
        self._exp_list = []
        # file -> Exposure
        self._exp_dict = {}

        self._alignment = None

    def append_exposure(self, file, exp):
        self._exp_list.append(file)
        self._exp_dict[file] = exp

    def get_reference(self):
        """get the reference exposure for this stack (the first exposure in the
        stack)"""
        return self._exp_dict[self._exp_list[0]]

    def get_exposure(self, file):
        """get the exposure from file"""
        return self._exp_dict[file]

    def load_cache(self, cache):
        if "stacks" in cache and self.name in cache["stacks"]:
            self._alignment = cache["stacks"][self.name]

    def needs_align(self):
        if len(self._exp_list) < 2:
            return False
        if self._alignment is None:
            return True
        if any(map(lambda x: x not in self._alignment, self._exp_list[1:])):
            return True
        if any(
            map(lambda x: x.ref != self.get_reference().name, self._alignment.values())
        ):
            return True
        return False

    def pack_cache(self, cache):
        cache["stacks"][self.name] = self._alignment

    def set_alignment(self, alignment):
        self._alignment = alignment

    def get_alignment(self, file):
        if file == self._exp_list[0]:
            return Alignment(self._exp_dict[self._exp_list[0]].name)
        return self._alignment[file]


class Crop:
    """represents a stack of regions of exposures (possibly from different
    stacks) for a single membrane"""

    def __init__(self, name, exposures):
        self.name = name
        self._exp = exposures
        self._exp_list = [exp.name for exp in exposures]
        self._exp_dict = {exp.name: exp for exp in exposures}

        # reference exposure is the reference for the stack of the first
        # exposure
        self._reference = self._exp[0].stack.get_reference()
        # center for rotation is the image center of the reference exposure
        self._center = (
            self._reference.image().width // 2,
            self._reference.image().height // 2,
        )
        # position of tight crop box
        # {
        #   "rot": rotation, "crop": (first lane, last lane, top, bottom)
        # }
        self._crop_transform = None
        self._alignment = None
        # position of ladders; weight -> y position
        self._ladder = None

        self._num_lanes = None

    def set_num_lanes(self, num_lanes):
        if self._num_lanes is not None and num_lanes != self._num_lanes:
            raise LayoutSpecError(
                "crop {} has different number of lanes".format(self.name)
            )
        self._num_lanes = num_lanes

    def load_cache(self, cache):
        if "crops" in cache and self.name in cache["crops"]:
            if "align" in cache["crops"][self.name]:
                self._alignment = cache["crops"][self.name]["align"]
            if "crop" in cache["crops"][self.name]:
                self._crop_transform = cache["crops"][self.name]["crop"]
            if "ladder" in cache["crops"][self.name]:
                self._ladder = cache["crops"][self.name]["ladder"]

    def needs_align(self):
        if self._alignment is None:
            return True
        if any(map(lambda x: x not in self._alignment, self._exp_list)):
            return True
        if any(
            map(
                lambda x: x.ref
                != self._exp_dict[self._exp_list[0]].stack.get_reference().name,
                self._alignment.values(),
            )
        ):
            return True
        return False

    def needs_crop(self):
        if self._crop_transform is None:
            return True
        if (
            self._crop_transform.ref
            != self._exp_dict[self._exp_list[0]].stack.get_reference().name
        ):
            return True
        return False

    def needs_ladder(self):
        return self._ladder is None

    def pack_cache(self, cache):
        entry = {}
        if self._alignment is not None:
            entry["align"] = self._alignment
        if self._crop_transform is not None:
            entry["crop"] = self._crop_transform
        if self._ladder is not None:
            entry["ladder"] = self._ladder
        if entry:
            cache["crops"][self.name] = entry

    def set_alignment(self, alignment):
        self._alignment = alignment

    def get_ladder(self):
        pass

    def pixel_scale(self, crop_transform=None, format=None):
        if crop_transform is None:
            crop_transform = self._crop_transform
        if format is None:
            format = Formatting.default
        assert format["width"] is not None, "format has no width!"

        # figre width minus padding
        fw_no_padding = format["width"]
        if format["hpad"] is not None:
            fw_no_padding -= 2 * format["hpad"]
        # pixel width minus padding
        pxw_no_padding = crop_transform.r - crop_transform.l
        return pxw_no_padding / fw_no_padding

    def figure_box(self, crop_transform=None, format=None):
        if crop_transform is None:
            crop_transform = self._crop_transform
        if format is None:
            format = Formatting.default
        assert format["width"] is not None, "format has no width!"

        px_scale = self.pixel_scale(crop_transform, format)
        # pixel left, right with padding
        pxl_padding = crop_transform.l - format["hpad"] * px_scale
        pxr_padding = crop_transform.r + format["hpad"] * px_scale

        # pixel height minus padding
        pxh_no_padding = crop_transform.b - crop_transform.t
        # figure height minus padding
        fh_no_padding = pxh_no_padding / px_scale
        # figure height with padding
        fh_no_padding += 2 * format["vpad"]
        if format["height"] is not None:
            fh_padding = format["height"]
        elif format["min_height"] is not None and fh_no_padding < format["min_height"]:
            fh_padding = format["min_height"]
        elif format["snap_height"] is not None:
            fh_padding = (
                np.ceil(fh_no_padding / format["snap_height"]) * format["snap_height"]
            )
        else:
            fh_padding = fh_no_padding
        # pixel height with padding
        pxh_padding = fh_padding * px_scale
        # pixel top, bottom with padding
        pxt_padding = crop_transform.t - (pxh_padding - pxh_no_padding) / 2
        pxb_padding = crop_transform.b + (pxh_padding - pxh_no_padding) / 2
        return (pxl_padding, pxt_padding, pxr_padding, pxb_padding)

    def dim(self, format):
        """calculate dimensions of crop according to format"""
        assert format["width"] is not None, "format has no width!"
        assert self._crop_transform is not None, "cropbox not yet set!"

        px_scale = self.pixel_scale(format=format)
        l, t, r, b = self.figure_box(format=format)
        return ((r - l) / px_scale, (b - t) / px_scale)

    def image(self, im=None, crop_transform=None, format=None):
        """get crop of image `im` according to crop_transform"""
        assert (
            self._crop_transform is not None or crop_transform is not None
        ), "cropbox is not yet set!"
        if im is None:
            assert self._alignment is not None, "alignment is not yet set!"
            im = self._exp_dict[self._exp_list[0]].image(
                self._alignment[self._exp_list[0]]
            )
        elif type(im) is str:
            assert im in self._exp_dict
            im = self._exp_dict[im].image(self._alignment[im])
        if crop_transform is None:
            crop_transform = self._crop_transform
        if format is None:
            format = Formatting.default

        crop = self.figure_box(crop_transform, format)
        if format["value_black"] != 0 or format["value_white"] != 255:
            bl, wh = format["value_black"], format["value_white"]
            array_new = ((np.clip(np.array(im), bl, wh) - bl) / (wh - bl) * 255).astype(
                np.uint8
            )
            im = Image.fromarray(array_new)

        # TODO
        return (
            im.rotate(
                crop_transform.rot, center=self._center, resample=Image.BICUBIC
            ).crop(crop),
            crop,
        )

    def _transform_point(self, xy):
        assert self._crop_transform is not None, "cropbox is not yet set!"
        x, y = xy
        cx, cy = self._center
        # TODO: fix sign
        rot = -self._crop_transform.rot
        s, c = np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))
        new_x, new_y = (x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (
            y - cy
        ) * c + cy
        return new_x, new_y

    def calculate_crop_transform(self, reference, im, position, lanes):
        """convert list of four points (first lane, last lane, top, bottom)
        to cropped coordinates and transform"""
        # calculate rotation
        dx = position[1][0] - position[0][0]
        dy = position[1][1] - position[0][1]
        # TODO: fix sign
        rot = -np.rad2deg(np.arctan(dy / dx))
        s, c = np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))

        # rotate selected points
        def rotate(xy):
            x, y = xy
            cx, cy = self._center
            return ((x - cx) * c - (y - cy) * s + cx, (x - cx) * s + (y - cy) * c + cy)

        position_new = list(map(rotate, position))

        position_width = position_new[1][0] - position_new[0][0]
        width_per_lane = position_width / (lanes[1] - lanes[0])
        left = position_new[0][0] - (lanes[0] - 1 + 0.5) * width_per_lane
        right = position_new[1][0] + (self._num_lanes - lanes[1] + 0.5) * width_per_lane

        # create cropbox transformation with rotation and positions of
        # left, top, right, and bottom boundaries
        crop_transform = CropTransform(
            reference.name, -rot, left, position_new[2][1], right, position_new[3][1]
        )

        # return transformed coordinates and cropbox
        position_new = [
            (x - crop_transform.l, y - crop_transform.t) for (x, y) in position_new
        ]
        # TODO
        return (
            self.image(im=im, crop_transform=crop_transform)[0],
            position_new,
            crop_transform,
        )

    def set_crop_transform(self, crop_transform):
        self._crop_transform = crop_transform

    def set_ladder(self, ladder):
        self._ladder = ladder

    def get_ladder(self, format):
        ladder = []
        px_scale = self.pixel_scale(format=format)
        l, t, r, b = self.figure_box(format=format)
        for x, y, weight in zip(self._ladder.x, self._ladder.y, self._ladder.weight):
            new_x, new_y = self._transform_point((x, y))
            if new_y >= t:
                ladder.append(((new_y - t) / px_scale, weight))
        return ladder


class Exposure:
    @classmethod
    def open(cls, file, stack):
        im = Image.open(file)
        exp = cls(file, im, stack)
        return exp

    def __init__(self, name, im, stack, thumbscale=10):
        self.name = name
        self.stack = stack
        self._im = im
        self._thumbscale = thumbscale
        self._thumb = im.resize((im.width // thumbscale, im.height // thumbscale))

    def thumb(self, transform=None):
        t = self._transform(self._im, transform)
        return t.resize((t.width // self._thumbscale, t.height // self._thumbscale))

    def image(self, transform=None):
        return self._transform(self._im, transform)

    def _transform(self, im, transform=None):
        if transform is None:
            return im
        t = im
        # if "scale" in transform:
        #     t = t.resize(
        #         (int(t.width * transform["scale"]), int(t.height * transform["scale"])))
        t = t.rotate(-transform.rot, translate=(transform.x, transform.y))
        return t


def read_layout_spec(path):
    with open(path + ".yaml", "r") as f:
        layout_spec = yaml.load(f, Loader=yaml.SafeLoader)
    return Layout.parse_layout(layout_spec)


def init_cache():
    return {"stacks": {}, "crops": {}}


def load_cache(path):
    if os.path.exists(path + ".pkl"):
        with open(path + ".pkl", "rb") as f:
            cache = pickle.load(f)
    else:
        cache = init_cache()
    return cache


def save_cache(path, cache):
    with open(path + ".pkl", "wb") as f:
        pickle.dump(cache, f)


def should_edit(name, edit):
    if edit == True:
        return True
    if type(edit) == list and name in edit:
        return True
    return False


def align(layout, cache, save_cache, edit=None):
    """obtain required position, rotation, and cropping information"""
    if edit is None:
        edit = {
            "align": False,
            "crop": False,
            "ladder": False,
        }

    # 1. Align entire exposures within each stack
    for name, stack in layout.stacks.items():
        stack.load_cache(cache)
        if stack.needs_align() or should_edit(name, edit["align"]):
            print(
                f"""
Edit alignment for exposures in stack {name}.
left\tmove exposure left
right\tmove exposure right
up\tmove exposure up
down\tmove exposure down
[\trotate exposure counterclockwise
]\trotate exposure clockwise
shift\thold shift + left, right, up, down, [, or ] to move/rotate faster
ctrl\thold ctrl + left or right to move to previous or next exposure
enter\tfinish editing alignment
            """
            )
            alignment = do_alignment(
                stack.name,
                # use stack reference exposure as reference
                stack.get_reference(),
                # non-reference exposures in stack
                stack._exp_list[1:],
                {f: stack._exp_dict[f] for f in stack._exp_list[1:]},
                # pass init_alignment
                stack._alignment,
            )
            stack.set_alignment(alignment)
        stack.pack_cache(cache)
        save_cache(cache)

    # 2. Fine-tune alignment of exposures for each crop.
    # 3. Define bounding box of each crop, consisting of first lane, last
    #    lane, top, and bottom.
    # 4. Mark ladders if required in figure.
    for name, crop in layout.crops.items():
        crop.load_cache(cache)
        ref = crop._exp[0].stack.get_reference()
        if crop.needs_align() or should_edit(name, edit["align"]):
            print(
                f"""
Edit alignment for exposures in crop {name}.
left\tmove exposure left
right\tmove exposure right
up\tmove exposure up
down\tmove exposure down
[\trotate exposure counterclockwise
]\trotate exposure clockwise
shift\thold shift + left, right, up, down, [, or ] to move/rotate faster
ctrl\thold ctrl + left or right to move to previous or next exposure
enter\tfinish editing alignment
            """
            )
            # initialize alignment for each exposure with its alignment in its
            # stack
            init_alignment = {
                f: crop._alignment[f]
                if (crop._alignment is not None and f in crop._alignment)
                else crop._exp_dict[f].stack.get_alignment(f)
                for f in crop._exp_list
            }
            alignment = do_alignment(
                crop.name,
                crop._exp[0].stack.get_reference(),
                crop._exp_list,
                crop._exp_dict,
                init_alignment,
            )
            crop.set_alignment(alignment)

        # TODO: show reference if not part of crop
        if crop.needs_crop() or should_edit(name, edit["crop"]):
            print(
                f"""
Edit bounding box for crop {name}. Right-click four times to mark left, right,
top, and bottom anchors. For left and right anchors, use up and down arrow keys
to indicate which lane is being selected.
ctrl\thold ctrl + left or right to show previous or next exposure
esc\treset bounding box
enter\tfinish editing bounding box
                """
            )
            crop_transform = do_crop(
                crop.name,
                crop._exp[0].stack.get_reference(),
                crop._exp_list,
                crop._exp_dict,
                crop._alignment,
                lambda r, im, p, l: crop.calculate_crop_transform(r, im, p, l),
            )
            crop.set_crop_transform(crop_transform)

        # TODO: show reference if not part of crop
        if crop.needs_ladder() or should_edit(name, edit["ladder"]):
            print(
                f"""
Edit ladder positions for crop {name}. Right-click to mark ladder positions. Use
up and down arrow keys to indicate which ladder is being selected.
ctrl\thold ctrl + left or right to show previous or next exposure
esc\treset ladder positions
enter\tfinish editing ladder positions
                """
            )
            ladder = do_ladder(
                crop.name,
                crop._exp[0].stack.get_reference(),
                crop._exp_list,
                crop._exp_dict,
                crop._alignment,
                layout.ladder,
            )
            crop.set_ladder(ladder)
        crop.pack_cache(cache)
        save_cache(cache)
