"""voice - local text-to-speech and speech-to-text on Apple Silicon.

    voice speak "text"     read the text aloud (or text on stdin)
    voice speak --stop     stop whatever is playing
    voice speak --warm     keep the speaking model loaded in the background, so speech starts at once
    voice speak --release  stop the background speaker and free its memory
    voice voices           list the speaking voices (pick one with speak --voice)
    voice listen           record until a pause, transcribe, print and copy to the clipboard
    voice listen --toggle  start recording in the background, or stop it and copy the text
    voice chat             talk with Claude Code hands-free in this terminal; other options go to claude
    voice setup            download the default models once; every other command runs offline
    voice shrink           write a 16-bit copy of the speech-to-text model: half the size, less memory

Each command loads its model, does one job and releases it before it exits, except the
background speaker and voice chat, which hold theirs until they are stopped.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

from voice import models

MAX_TEXT = 100_000  # characters `speak` reads from stdin; more is an accident, not a reply


def number(low: float, high: float):
    """An argparse type: a finite number from `low` to `high`."""
    def parse(value: str) -> float:
        try:
            n = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value!r} is not a number") from None
        if not (math.isfinite(n) and low <= n <= high):
            raise argparse.ArgumentTypeError(f"{value} is not between {low:g} and {high:g}")
        return n
    return parse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice", description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    speak = sub.add_parser("speak", help="read text aloud")
    speak.add_argument("text", nargs="*", help="the text; read from stdin when absent")
    speak.add_argument("--stop", action="store_true", help="stop whatever is playing")
    speak.add_argument("--voice", help="speaking voice, such as af_heart or bf_emma (see `voice voices`)")
    speak.add_argument("--speed", type=number(0.5, 2.0), help="speaking speed from 0.5 to 2.0; 1.0 is normal")
    speak.add_argument("--force", action="store_true", help="load even under critical memory pressure")
    speak.add_argument("--warm", action="store_true",
                       help="start the resident speaker in the background (about 450 MB), so speech starts at once")
    speak.add_argument("--release", action="store_true", help="stop playback and the resident speaker, freeing it")

    serve = sub.add_parser("serve", help="run the resident speaker in the foreground (speak --warm starts it)")
    serve.add_argument("--force", action="store_true", help="load even under critical memory pressure")

    sub.add_parser("voices", help="list the speaking voices in the local cache")

    listen = sub.add_parser("listen", help="record, transcribe, print and copy to the clipboard")
    listen.add_argument("--toggle", action="store_true",
                        help="start a background recording, or stop the running one; for a keyboard shortcut")
    # --toggle's background half; the value is the descriptor of the pid lock it inherits
    listen.add_argument("--until-stopped", type=int, metavar="FD", help=argparse.SUPPRESS)
    listen.add_argument("--max-seconds", type=number(1, 10_800),
                        help="longest recording in seconds (default 60, or 1800 with --toggle)")
    listen.add_argument("--silence", type=number(0.2, 30),
                        help="seconds of quiet that end it (default 1.5); not with --toggle, which stops on a press")
    listen.add_argument("--no-copy", action="store_true", help="print only; not with --toggle, which never prints")
    listen.add_argument("--paste", action="store_true",
                        help="with --toggle: also paste at the cursor when it stops (the clipboard keeps the text)")
    listen.add_argument("--preload", action="store_true",
                        help="with --toggle: load the model while you talk, for a shorter wait after the stop "
                             "press; holds about 1.7 GB for the whole recording")
    listen.add_argument("--force", action="store_true", help="load even under critical memory pressure")

    chat = sub.add_parser("chat", help="talk with Claude Code hands-free in this terminal",
                          description="Runs claude with a spoken loop. Options voice does not know go to claude.")
    chat.add_argument("--force", action="store_true", help="load even under critical memory pressure")
    chat.add_argument("--silence", type=number(0.2, 30),
                      help="seconds of quiet that end your turn (default 3)")
    chat.add_argument("--toggle", action="store_true",
                      help="turn the loop of the running voice chat off or on; for a global keyboard shortcut")
    chat.add_argument("--hook", action="store_true", help=argparse.SUPPRESS)  # run by Claude Code, not by hand

    sub.add_parser("setup", help="download the models, then check the speaker, microphone and Claude Code")

    shrink = sub.add_parser("shrink", help="write a 16-bit copy of the speech-to-text model")
    shrink.add_argument("--force", action="store_true", help="run even under critical memory pressure")

    args, passthrough = parser.parse_known_args(argv)
    if passthrough and (args.command != "chat" or args.toggle):
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "listen":
        background = args.toggle or args.until_stopped is not None
        if background and args.no_copy:
            listen.error("--no-copy would throw the text away: --toggle never prints it")
        if background and args.silence is not None:
            listen.error("--silence does not apply to --toggle, which stops on the next press")
        for flag in ("paste", "preload"):
            if getattr(args, flag) and not background:
                listen.error(f"--{flag} works only with --toggle")
    if args.command == "speak":
        if (args.stop or args.release) and args.text:
            speak.error("--stop and --release take no text")
        if args.warm and (args.stop or args.release):
            speak.error("--warm starts the speaker; --stop and --release end playback")
    if args.command != "setup":
        # Before anything imports huggingface_hub, which reads this once at import:
        # only `setup` may reach the network.
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        if args.command == "speak":
            from voice import speak as cmd

            text = " ".join(args.text) if args.text else None
            if text is None and not (args.stop or args.release or args.warm):
                if sys.stdin.isatty():
                    speak.print_usage(sys.stderr)
                    print("voice speak: give the text as arguments or on stdin", file=sys.stderr)
                    return 2
                text = sys.stdin.read(MAX_TEXT + 1)
                if len(text) > MAX_TEXT:
                    print(f"voice speak: the text on stdin is over {MAX_TEXT:,} characters", file=sys.stderr)
                    return 2
            return cmd.run(text, stop=args.stop, force=args.force, voice=args.voice, speed=args.speed,
                           warm=args.warm, release=args.release)
        if args.command == "serve":
            from voice import serve as cmd

            return cmd.run(force=args.force)
        if args.command == "voices":
            from voice import speak as cmd

            names = cmd.voices()
            print("\n".join(names) if names else "no voices cached yet; run `voice setup`")
            return 0
        if args.command == "listen":
            from voice import listen as cmd

            copy = not args.no_copy
            if args.toggle or args.until_stopped is not None:
                limit = args.max_seconds if args.max_seconds is not None else 1800.0
                if args.until_stopped is not None:
                    return cmd.run_until_stopped(limit, copy, args.force, args.paste, args.until_stopped, args.preload)
                return cmd.toggle(limit, copy, args.force, args.paste, args.preload)
            return cmd.run(args.max_seconds if args.max_seconds is not None else 60.0,
                           args.silence if args.silence is not None else 1.5, copy, args.force)
        if args.command == "chat":
            from voice import chat as cmd

            if args.hook:
                return cmd.hook()
            if args.toggle:
                return cmd.toggle_running()
            return cmd.run(passthrough, force=args.force, silence=args.silence or cmd.SILENCE)
        if args.command == "shrink":
            from voice import shrink as cmd

            return cmd.run(force=args.force)
        from voice import setup as cmd

        return cmd.run()
    except models.MemoryPressure as exc:
        print(f"voice: {exc}", file=sys.stderr)
        return 3
    except models.ModelMissing as exc:
        print(f"voice: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
