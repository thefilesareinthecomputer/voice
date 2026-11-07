"""voice - local text-to-speech and speech-to-text on Apple Silicon.

    voice speak "text"     read the text aloud (or text on stdin)
    voice speak --stop     stop whatever is playing
    voice voices           list the speaking voices (pick one with speak --voice)
    voice listen           record until a pause, transcribe, print and copy to the clipboard
    voice setup            download the default models once; every other command runs offline

Each command loads its model, does one job and releases it before it exits.
"""

from __future__ import annotations

import argparse
import os
import sys

from voice import models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    speak = sub.add_parser("speak", help="read text aloud")
    speak.add_argument("text", nargs="*", help="the text; read from stdin when absent")
    speak.add_argument("--stop", action="store_true", help="stop whatever is playing")
    speak.add_argument("--voice", help="speaking voice, such as af_heart or bf_emma (see `voice voices`)")
    speak.add_argument("--speed", type=float, help="speaking speed, 1.0 is normal")
    speak.add_argument("--force", action="store_true", help="load even under critical memory pressure")

    sub.add_parser("voices", help="list the speaking voices in the local cache")

    listen = sub.add_parser("listen", help="record, transcribe, print and copy to the clipboard")
    listen.add_argument("--max-seconds", type=float, default=60.0, help="longest recording (default 60)")
    listen.add_argument("--silence", type=float, default=1.5, help="seconds of quiet that end it (default 1.5)")
    listen.add_argument("--no-copy", action="store_true", help="print only")
    listen.add_argument("--force", action="store_true", help="load even under critical memory pressure")

    sub.add_parser("setup", help="download the default models")

    args = parser.parse_args(argv)
    if args.command != "setup":
        # Before anything imports huggingface_hub, which reads this once at import:
        # only `setup` may reach the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if args.command == "speak":
            from voice import speak as cmd

            text = " ".join(args.text) if args.text else (None if args.stop else sys.stdin.read())
            return cmd.run(text, stop=args.stop, force=args.force, voice=args.voice, speed=args.speed)
        if args.command == "voices":
            from voice import speak as cmd

            names = cmd.voices()
            print("\n".join(names) if names else "no voices cached yet; run `voice setup`")
            return 0
        if args.command == "listen":
            from voice import listen as cmd

            return cmd.run(args.max_seconds, args.silence, not args.no_copy, args.force)
        from voice import setup as cmd

        return cmd.run()
    except models.MemoryPressure as exc:
        print(f"voice: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
