"""Launch the local natural-language L1 simulation dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import webbrowser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--open-browser', action='store_true')
    parser.add_argument('--fast', action='store_true', help='verification: do not pace simulation')
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')

    from ..assistant.server import DashboardController, DashboardServer
    from .common import setup_logging
    setup_logging(False)
    log_dir = args.log_dir or str(Path(__file__).resolve().parents[5] /
                                  'local_validation' / 'assistant_ui')
    controller = DashboardController(seed=args.seed, realtime=not args.fast, log_dir=log_dir)
    try:
        server = DashboardServer(('127.0.0.1', args.port), controller)
    except OSError:
        controller.close()
        print('Port %s is in use. Open http://127.0.0.1:%s or choose --port.' %
              (args.port, args.port))
        if args.open_browser:
            webbrowser.open('http://127.0.0.1:%s' % args.port)
        return 2
    url = 'http://127.0.0.1:%s' % args.port
    print('L1 simulation dashboard: %s' % url, flush=True)
    print('Keep this window running. Press Ctrl+C to close.', flush=True)
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        controller.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
