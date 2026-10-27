"""python -m retivasc_server <command>

  serve                start the server (settings: "server" in retivasc.local.json)
  adduser NAME [--admin]   add or reset a user (asks for the password)
  deluser NAME         remove a user
  users                list users
  mode [open|accounts] show or switch how people get in: "open" = anyone, each browser a private
                       guest session (public beta); "accounts" = sign in. Restart the server after.
  selftest             run AutoMorph v2 on the bundled sample photos and compare with the reference results
"""

from __future__ import annotations

import argparse
import getpass
import sys

from . import config

from .accounts import Accounts
from .settings import load


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m retivasc_server")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("serve")
    q.add_argument("--port", type=int, help="override the port in retivasc.local.json")
    q = sub.add_parser("adduser")
    q.add_argument("name")
    q.add_argument("--admin", action="store_true")
    q.add_argument("--password", help="for scripts; omit to be asked")
    q = sub.add_parser("deluser")
    q.add_argument("name")
    sub.add_parser("users")
    q = sub.add_parser("mode")
    q.add_argument("value", nargs="?", choices=["open", "accounts"])
    q = sub.add_parser("selftest")
    q.add_argument("--write-reference", action="store_true", help="store this machine's results as the reference")
    a = p.parse_args(argv)
    s = load()
    accounts = Accounts(s.data_dir)
    if a.cmd == "adduser":
        pw = a.password or getpass.getpass(f"password for {a.name}: ")
        accounts.add(a.name, pw, a.admin)
        print(f"user {a.name} saved{' (admin)' if a.admin else ''}")
        return 0
    if a.cmd == "deluser":
        print("removed" if accounts.remove(a.name) else "no such user")
        return 0
    if a.cmd == "users":
        for n in accounts.names():
            print(n, "(admin)" if accounts.is_admin(n) else "")
        return 0
    if a.cmd == "mode":
        if a.value:
            import json

            cfg = config.load()
            cfg.setdefault("server", {})["auth"] = a.value
            config.FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"mode set to {a.value}; restart the server to apply")
        else:
            print(s.auth)
        return 0
    if a.cmd == "selftest":
        from .selftest import run

        return run(s, write_reference=a.write_reference)
    if a.cmd == "serve":
        import uvicorn

        from .app import create_app

        if s.auth == "accounts" and not accounts.names():
            print("no users yet: add one with  python -m retivasc_server adduser NAME --admin", file=sys.stderr)
            return 1
        if a.port:
            s.port = a.port
        app = create_app(s, webui_dist=config.ROOT / "webui" / "dist")
        scheme = "https" if s.ssl_certfile else "http"
        print(f"RetiVASC server on {scheme}://{s.host}:{s.port}/  (data in {s.data_dir})")
        uvicorn.run(app, host=s.host, port=s.port, ssl_certfile=s.ssl_certfile or None, ssl_keyfile=s.ssl_keyfile or None,
                    log_level="warning")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
