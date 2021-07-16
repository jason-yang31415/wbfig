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
        action="store_true",
        help="show prompts to modify layout parameters for alignment only",
    )
    parser.add_argument(
        "--edit_crop",
        "-ec",
        action="store_true",
        help="show prompts to modify layout parameters for cropping only",
    )
    parser.add_argument(
        "--edit_ladder",
        "-el",
        action="store_true",
        help="show prompts to modify layout parameters for ladder positions only",
    )
    args = parser.parse_args()
    edit = {
        "align": args.edit_align or args.edit,
        "crop": args.edit_crop or args.edit,
        "ladder": args.edit_ladder or args.edit,
    }

    layout = read_layout_spec(args.layout_file)
    align(
        layout,
        cache=load_cache(args.layout_file),
        save_cache=lambda c: save_cache(args.layout_file, c),
        edit=edit,
    )
    layout.make_figure(args.layout_file)
