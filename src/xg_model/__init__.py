"""Expected goals (xG) model for NHL unblocked shots, built on puckscrape."""

import argparse
import logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="xg-model", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="download and cache play-by-play")
    c.add_argument("seasons", nargs="+", help="e.g. 2021 2022 2023 2024 (start years)")
    c.add_argument("--workers", type=int, default=8)

    t = sub.add_parser("train", help="fit the model and write reports/")
    t.add_argument("--train", nargs="+", default=["2022", "2023", "2024"])
    t.add_argument("--test", default="2025")
    t.add_argument("--no-moneypuck", action="store_true", help="skip the MoneyPuck benchmark")

    g = sub.add_parser("game", help="xG for one game, straight from the NHL API")
    g.add_argument("game_id", type=int)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s  %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("puckscrape").setLevel(logging.WARNING)

    if args.cmd == "collect":
        from .collect import collect
        for s in args.seasons:
            collect(s, workers=args.workers)
    elif args.cmd == "train":
        from .train import train
        train(args.train, args.test, benchmark=not args.no_moneypuck)
    elif args.cmd == "game":
        from .predict import game_report
        game_report(args.game_id)
