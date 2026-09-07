"""main.py 진입점 - Qt 없이 테스트할 수 있는 부분만.

`main()` 은 실제 QApplication + 이벤트 루프(`app.exec()`)를 돌리므로 여기서는
건드리지 않는다 - 대신 그 안에서 갈리는 판단(`--sample` 을 실행할지 말지)을
`should_populate_sample_data()` 로 빼서 그것만 테스트한다.

이 회귀 테스트가 지키는 버그: `Database.summary()` 가 `inference` 를 `evaluation`
으로 이름을 바꾼 세션에서 여기(그때는 main.py 안의 인라인 코드)를 놓쳐서,
빈 DB 에 `--sample` 을 주면 `KeyError: 'inference'` 로 죽었다.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import parse_args, should_populate_sample_data


def test_should_populate_when_db_is_completely_empty():
    assert should_populate_sample_data({"tasks": 4, "works": 0, "train": 0, "evaluation": 0, "running": 0})


def test_should_not_populate_when_train_runs_already_exist():
    assert not should_populate_sample_data({"train": 2, "evaluation": 0})


def test_should_not_populate_when_evaluation_runs_already_exist():
    """이 케이스가 옛 버그였다 - 예전 코드는 이 키를 'inference' 로 잘못 찾았다."""
    assert not should_populate_sample_data({"train": 0, "evaluation": 3})


def test_should_populate_tolerates_a_summary_missing_the_evaluation_key():
    """`summary()` 의 계약이 바뀌어도 KeyError 로 죽지 않고 "채워도 된다" 쪽으로 판단한다."""
    assert should_populate_sample_data({"train": 0})


def test_parse_args_sample_flag_defaults_to_false():
    args = parse_args([])
    assert args.sample is False
    assert args.theme is None


def test_parse_args_accepts_sample_and_theme_flags():
    args = parse_args(["--sample", "--theme", "light", "--db", "x.db"])
    assert args.sample is True
    assert args.theme == "light"
    assert args.db == "x.db"
