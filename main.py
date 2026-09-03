#!/usr/bin/env python3
"""DL Experiment Manager 실행 진입점.

사용법::

    pip install -r requirements.txt
    python main.py                      # 프로젝트 폴더의 experiments.db 사용
    python main.py --db ~/exp/my.db     # DB 경로 지정
    python main.py --config my.yaml     # 선택지/컬럼 설정 파일 지정
    python main.py --theme light        # 라이트 테마
    python main.py --sample             # 비어 있으면 샘플 데이터도 함께 생성
"""
from __future__ import annotations

import argparse
import sys

from dl_exp_manager import APP_NAME, ORG_NAME, __version__
from dl_exp_manager.config_store import default_config_path
from dl_exp_manager.db import default_db_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} v{__version__}")
    parser.add_argument(
        "--db",
        default=default_db_path(),
        help=f"SQLite DB path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help=f"Options/columns config YAML path (default: {default_config_path()})",
    )
    parser.add_argument(
        "--theme",
        default="dark",
        choices=["dark", "light"],
        help="UI theme (default: dark)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Populate example data if there are no runs yet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from dl_exp_manager import theme
    from dl_exp_manager.qt import QtWidgets
    from dl_exp_manager.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)
    theme.apply_theme(app, args.theme)

    window = MainWindow(args.db, args.config)

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
