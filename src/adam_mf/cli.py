"""Internal entrypoint, always run using the selected environment's absolute Python."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Use ./adam-mf for deployment and management')
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--new-run', action='store_true')
    sub.add_parser('check')
    pipeline = sub.add_parser('pipeline')
    pipeline.add_argument('--run', type=Path, required=True)
    pipeline.add_argument('--lock-fd', type=int, required=True)
    pipeline.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    from adam_mf.runtime import launcher
    if args.command == 'run':
        return launcher.launch(new_run=args.new_run)
    if args.command == 'check':
        return launcher.check()
    return launcher.pipeline(args.run, args.lock_fd, args.dry_run)


if __name__ == '__main__':
    raise SystemExit(main())
