"""
A script for extra doc generation.
Feel free to extend!

Contents:
1. Create markdown table of show options.
"""
import os

import gustaf as gus

this_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    # create md dir
    os.makedirs(os.path.join(this_dir, "../md"))

    # 1. Show options.
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "../md/show_options.md"), "w") as f:
        derived = gus.helpers.options.ShowOption.__subclasses__()
        for cls in derived:
            f.write(f"## {cls.__qualname__}\n\n")
            for backend, options in cls._valid_options.items():
                f.write(f"### {backend}\n\n")
                for option in options.values():
                    t_str = str(option.allowed_types)
                    t_str = t_str.replace("<class", "").replace("'", "").replace(">", "")
                    f.write(f"<details><summary><strong>{option.key}</strong></summary><p>\n")
                    f.write(f"\n{option.description}  \n")
                    f.write(f"- _allowed types_: {t_str}  \n")
                    f.write(f"</p></details> \n\n")
