from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton


# disable default matplotlib shortcuts that interfere
plt.rcParams["keymap.back"].remove("left")
plt.rcParams["keymap.forward"].remove("right")


class Alignment(namedtuple("Alignment", ["ref", "rot", "x", "y"], defaults=(None, 0, 0, 0))):
    def __str__(self):
        return f"{self.ref}; {self.rot} deg; ({self.x}, {self.y})"


def blend(im1, im2, alpha):
    """helper method to blend two images of different sizes"""
    def rb_pad(im, width, height):
        padded = Image.new(im.mode, (width, height), 255)
        padded.paste(im, (0, 0))
        return padded

    w, h = max(im1.width, im2.width), max(im1.height, im2.height)
    im1_copy = rb_pad(im1, w, h)
    im2_copy = rb_pad(im2, w, h)
    return Image.blend(im1_copy, im2_copy, alpha=alpha)


def do_alignment(name, reference, exp_list, exp_dict, init_alignment):
    """align given exposures using the matplotlib interface and keyboard
    shortcuts"""

    # initialize alignment to no rotation and no translation unless the
    # exposure has an initial alignment specified in `init_alignment`
    # file -> Alignment
    alignment = {}
    for f in exp_list:
        if init_alignment is not None and f in init_alignment and init_alignment[f].ref == reference.name:
            alignment[f] = init_alignment[f]
        else:
            alignment[f] = Alignment(reference.name)

    fig, ax = plt.subplots()
    im = ax.imshow(exp_dict[exp_list[0]].thumb(), cmap="gray")
    index = 0

    def draw():
        """update matplotlib display"""
        exp_name = exp_list[index]
        # blend together current exposure, with its alignment transform, with
        # the reference exposure
        im.set_data(blend(reference.thumb(), exp_dict[exp_name].thumb(
            alignment[exp_name]), alpha=0.5))
        ax.set_title("{}: {}".format(name, exp_name))

        fig.canvas.draw()

    def on_key(event):
        """handle key events"""
        nonlocal index

        # fast translation or rotation when shift key is pressed
        fast = event.key in {"shift+left", "shift+right",
                             "shift+down", "shift+up", "{", "}"}
        delta_xy = 20 if fast else 2
        delta_rot = 2 if fast else 0.2

        exp_name = exp_list[index]

        if event.key in {"right", "shift+right"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                x=alignment[exp_name].x + delta_xy)
            draw()
        elif event.key in {"left", "shift+left"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                x=alignment[exp_name].x - delta_xy)
            draw()
        elif event.key in {"up", "shift+up"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                y=alignment[exp_name].y - delta_xy)
            draw()
        elif event.key in {"down", "shift+down"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                y=alignment[exp_name].y + delta_xy)
            draw()
        elif event.key in {"[", "{"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                rot=alignment[exp_name].rot - delta_rot)
            draw()
        elif event.key in {"]", "}"}:
            alignment[exp_name] = alignment[exp_name]._replace(
                rot=alignment[exp_name].rot + delta_rot)
            draw()
        # next or previous exposure with ctrl+right and ctrl+left
        elif event.key == "ctrl+right":
            if index < len(exp_list) - 1:
                index += 1
                draw()
        elif event.key == "ctrl+left":
            if index > 0:
                index -= 1
                draw()
        # exit dialogue and continue with enter
        elif event.key == "enter":
            fig.canvas.mpl_disconnect(handle)
            plt.close(fig)

    draw()
    handle = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    return alignment


class CropTransform(namedtuple("CropTransform", ["ref", "rot", "l", "t", "r", "b"])):
    def __str__(self):
        return f"{self.ref}; {self.rot} deg; ({self.l}, {self.t}, {self.r}, {self.b})"


def do_crop(name, reference, exp_list, exp_dict, exp_alignment, crop_fn):
    if reference.name not in exp_list:
        exp_list = [reference.name] + exp_list
        exp_dict = {reference.name: reference, **exp_dict}
        exp_alignment = {reference.name: Alignment(
            reference.name), **exp_alignment}

    # index of exposure to show
    index = 0
    # currently selecting with mouse; ("lane", #) or ("vertical", "top" or
    # "bottom")
    selecting = ("lane", 1)
    # list of coordinates (x, y) selected [first lane, last lane, top,
    # bottom]
    position = []
    # list of lanes selected (e.g. first, last)
    lanes = []

    fig, ax = plt.subplots()
    ax.autoscale(False)
    im = ax.imshow(exp_dict[exp_list[0]].image(
        transform=exp_alignment[exp_list[0]]), cmap="gray")
    points = ax.scatter([], [])
    # final crop transform to return
    crop_transform = None

    current_exp = None

    def draw(reset_axes=False):
        """update matplotlib display"""
        nonlocal current_exp
        nonlocal crop_transform

        # get exposure from index
        exp = exp_dict[exp_list[index]]
        # get aligned image
        exp_im = exp.image(transform=exp_alignment[exp_list[index]])

        if len(position) < 4:
            # before cropbox is defined by 4 points
            if current_exp != exp or reset_axes:
                current_exp = exp
                im.set_data(exp_im)
                im.set_extent((0, exp_im.width, exp_im.height, 0))
                if reset_axes:
                    ax.set_xlim((0, exp_im.width))
                    ax.set_ylim((exp_im.height, 0))

            if position:
                # show points
                points.set_offsets(position)
            else:
                # clear displayed points (matplotlib does not accept empty list)
                points.set_offsets([[None, None]])
        else:
            # show cropped exposure when 4 points have been selected

            # calculate transformed positions and cropbox
            cropped_im, position_new, crop_transform = crop_fn(
                reference, exp_im, position, lanes)
            im.set_data(cropped_im)
            im.set_extent((0, cropped_im.width, cropped_im.height, 0))
            ax.set_xlim((0, cropped_im.width))
            ax.set_ylim((cropped_im.height, 0))

            # move points to transformed positions
            points.set_offsets([(xy[0], xy[1]) for xy in position_new])

        # update title
        if selecting[0] == "lane":
            ax.set_title("{}: lane {}".format(name, selecting[1]))
        elif selecting[0] == "vertical":
            ax.set_title("{}: {}".format(name, selecting[1]))
        elif selecting[0] == "confirm":
            ax.set_title("{}".format(name))

        fig.canvas.draw()

    def on_key(event):
        """handle key events"""
        nonlocal index
        nonlocal selecting
        nonlocal position
        nonlocal lanes

        if event.key == "ctrl+right":
            if index < len(exp_list) - 1:
                index += 1
                draw()
        elif event.key == "ctrl+left":
            if index > 0:
                index -= 1
                draw()
        elif event.key == "up":
            if selecting[0] == "lane":
                selecting = ("lane", selecting[1] + 1)
                draw()
        elif event.key == "down":
            if selecting[0] == "lane":
                selecting = ("lane", selecting[1] - 1)
                draw()
        elif event.key == "escape":
            selecting = ("lane", 1)
            position = []
            lanes = []
            draw(True)
        elif event.key == "enter":
            if len(position) == 4:
                fig.canvas.mpl_disconnect(key_handle)
                fig.canvas.mpl_disconnect(mouse_handle)
                plt.close(fig)

    def on_click(event):
        """handle click events"""
        nonlocal position
        nonlocal selecting

        x, y = event.xdata, event.ydata
        if event.button is MouseButton.RIGHT and len(position) < 4:
            if x and y:
                # add position to list of coordinates
                position.append((x, y))
                if len(position) <= 2:
                    # append associated lane number
                    lanes.append(selecting[1])

                if len(position) == 2:
                    selecting = ("vertical", "top")
                elif len(position) == 3:
                    selecting = ("vertical", "bottom")
                elif len(position) == 4:
                    selecting = ("confirm",)

                if len(position) == 4:
                    draw(True)
                else:
                    draw()

    draw(True)
    key_handle = fig.canvas.mpl_connect("key_press_event", on_key)
    mouse_handle = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    if crop_transform is not None:
        return crop_transform
    else:
        raise RuntimeError("failed to crop")


class Ladder(namedtuple("Ladder", ["ref", "x", "y", "weight"])):
    def __str__(self):
        return f"{self.ref}; {'; '.join([f'{w} ({x}, {y})' for (w, x, y) in zip(self.weight, self.x, self.y)])}"


def do_ladder(name, reference, exp_list, exp_dict, exp_alignment, ladder):
    """specify ladder positions"""
    if reference.name not in exp_list:
        exp_list = [reference.name] + exp_list
        exp_dict = {reference.name: reference, **exp_dict}
        exp_alignment = {reference.name: Alignment(
            reference.name), **exp_alignment}

    # index of exposure to show
    index = 0
    # index of ladder currently selecting
    selecting = 0
    # list of coordinates (x, y) of ladder positions
    position = []
    # list of ladder weights corresponding to position
    weights = []

    fig, ax = plt.subplots()
    ax.autoscale(False)
    im = ax.imshow(exp_dict[exp_list[0]].image(
        transform=exp_alignment[exp_list[0]]), cmap="gray")
    points = ax.scatter([], [])

    current_exp = None

    def draw(reset_axes=False):
        """update matplotlib display"""
        nonlocal current_exp

        # get exposure from index
        exp = exp_dict[exp_list[index]]
        # get aligned image
        exp_im = exp.image(transform=exp_alignment[exp_list[index]])

        if current_exp != exp:
            current_exp = exp
            im.set_data(exp_im)
            im.set_extent((0, exp_im.width, exp_im.height, 0))
            if reset_axes:
                ax.set_xlim((0, exp_im.width))
                ax.set_ylim((exp_im.height, 0))

        if position:
            # show points
            points.set_offsets(position)
        else:
            # clear displayed points (matplotlib does not accept empty list)
            points.set_offsets([[None, None]])

        # update title
        ax.set_title("{}: {} kDa".format(name, ladder[selecting]))
        fig.canvas.draw()

    def on_key(event):
        """handle key events"""
        nonlocal index
        nonlocal selecting
        nonlocal position

        if event.key == "ctrl+right":
            if index < len(exp_list) - 1:
                index += 1
                draw()
        elif event.key == "ctrl+left":
            if index > 0:
                index -= 1
                draw()
        elif event.key == "up":
            if selecting < len(ladder) - 1:
                selecting += 1
                draw()
        elif event.key == "down":
            if selecting > 0:
                selecting -= 1
                draw()
        elif event.key == "escape":
            position = []
            draw()
        elif event.key == "enter":
            fig.canvas.mpl_disconnect(key_handle)
            fig.canvas.mpl_disconnect(mouse_handle)
            plt.close(fig)

    def on_click(event):
        """handle click events"""
        nonlocal position
        nonlocal selecting

        x, y = event.xdata, event.ydata
        if event.button is MouseButton.RIGHT:
            if x and y:
                position.append((x, y))
                weights.append(ladder[selecting])
                draw()

    draw(True)
    key_handle = fig.canvas.mpl_connect("key_press_event", on_key)
    mouse_handle = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    return Ladder(reference.name, tuple([xy[0] for xy in position]), tuple([xy[1] for xy in position]), weights)
