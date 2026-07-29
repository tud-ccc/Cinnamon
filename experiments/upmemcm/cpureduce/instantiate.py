#!/usr/bin/env python3
"""Instantiate bench.mlir.tpl by substituting {{M}}, {{T}}, {{2 * T}}."""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("template")
    p.add_argument("--M", type=int, required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    with open(args.template) as f:
        text = f.read()

    text = text.replace("{{M}}", str(args.M))

    with open(args.output, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
