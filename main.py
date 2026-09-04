#!/usr/bin/env python3
"""DL Experiment Manager 실행 진입점.

사용법::

    pip install -r requirements.txt
    python main.py                      # 프로젝트 폴더의 experiments.db 사용
    python main.py --db ~/exp/my.db     # DB 경로 지정
    python main.py --config my.yaml     # 선택지/컬럼 설정 파일 지정
    python main.py --theme light        # 라이트 테마로 강제 실행 (기본: 앱에서 마지막으로 고른 것)
    python main.py --sample             # 비어 있으면 샘플 데이터도 함께 생성

다크/라이트는 실행 중에도 View > Theme 메뉴에서 바로 바꿀 수 있고, 고른 값은 다음 실행에 이어진다.
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
        default=None,
        choices=["dark", "light"],
        help="UI theme (default: last used, via View > Theme in the app, else dark)",
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
    from dl_exp_manager.qt import QtCore, QtWidgets
    from dl_exp_manager.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    # --theme 를 안 주면 앱 안(View > Theme)에서 마지막으로 고른 걸 그대로 이어서 쓴다.
    theme_name = args.theme or QtCore.QSettings(ORG_NAME, APP_NAME).value("theme", "dark")
    if theme_name not in ("dark", "light"):
        theme_name = "dark"
    theme.apply_theme(app, theme_name)

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
