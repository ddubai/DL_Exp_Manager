#!/usr/bin/env python3
"""DL Experiment Manager 실행 진입점.

사용법::

    pip install -r requirements.txt
    python main.py                      # 프로젝트 폴더의 experiments.db 사용
    python main.py --db ~/exp/my.db     # DB 경로 지정
    python main.py --sample             # 비어 있으면 샘플 데이터도 함께 생성
"""
from __future__ import annotations

import argparse
import sys

from dl_exp_manager import APP_NAME, ORG_NAME, __version__
from dl_exp_manager.db import default_db_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{__version__}")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help=f"SQLite DB 경로 (기본: {default_db_path()})",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="실행 기록이 하나도 없을 때 예시 데이터를 채워 넣는다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dl_exp_manager.qt import QtWidgets
    from dl_exp_manager.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    window = MainWindow(args.db)

    if args.sample:
        summary = window.db.summary()
        if summary["train"] == 0 and summary["inference"] == 0:
            from dl_exp_manager.sample_data import populate

            populate(window.db)
            window.nav.refresh()
            window.refresh_all()

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
