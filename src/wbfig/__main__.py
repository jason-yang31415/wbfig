from wbfig.western import read_layout_spec, align, load_cache, save_cache
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make western blot figures")
    parser.add_argument("layout_file", type=str, help="path to layout description file")
    parser.add_argument(
        "--edit",
        "-e",
        action="store_true",
        help="show prompts to modify all layout parameters",
    )
    parser.add_argument(
        "--edit_align",
        "-ea",
        nargs="?",
        const=True,
        help="show prompts to modify layout parameters for alignment only",
    )
    parser.add_argument(
        "--edit_crop",
        "-ec",
        nargs="?",
        const=True,
        help="show prompts to modify layout parameters for cropping only",
    )
    parser.add_argument(
        "--edit_ladder",
        "-el",
        nargs="?",
        const=True,
        help="show prompts to modify layout parameters for ladder positions only",
    )
    args = parser.parse_args()

    ea = args.edit_align.split(",") if type(args.edit_align) is str else args.edit_align
    ec = args.edit_crop.split(",") if type(args.edit_crop) is str else args.edit_crop
    el = (
        args.edit_ladder.split(",")
        if type(args.edit_ladder) is str
        else args.edit_ladder
    )

    edit = {
        "align": args.edit or ea,
        "crop": args.edit or ec,
        "ladder": args.edit or el,
    }

    layout = read_layout_spec(args.layout_file)
    align(
        layout,
        cache=load_cache(args.layout_file),
        save_cache=lambda c: save_cache(args.layout_file, c),
        edit=edit,
    )
    layout.make_figure(args.layout_file)
