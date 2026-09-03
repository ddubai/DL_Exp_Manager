# DL Experiment Manager

4대의 독립된 학습 서버(Server 1~4)에서 돌린 실험을 **로컬 PC 한 곳에서 아카이빙·검색·비교**하는 PyQt6 데스크톱 애플리케이션입니다.
모든 기록은 로컬 SQLite 파일(`experiments.db`) 하나에 저장되므로 별도 서버나 계정이 필요 없습니다.

```
DL Task (SR / DN / Clustering / Classification)   ← Level 1 : 좌측 트리
   └─ Work ID (SSL2SL, BSR-x4, ...)               ← Level 2 : 좌측 트리 하위
        ├─ Train      탭                          ← Level 3 : 상단 탭
        └─ Inference  탭
```

## 설치 & 실행

```bash
pip install -r requirements.txt

python main.py                    # 프로젝트 폴더의 experiments.db 사용
python main.py --db ~/exp/my.db   # DB 경로 지정
python main.py --sample           # 비어 있으면 예시 데이터까지 생성 (UI 둘러보기용)
```

- Python 3.10+ 권장 (타입 힌트에 `X | None` 문법 사용).
- **PySide6 를 쓰고 싶다면** `requirements.txt` 에서 PyQt6 대신 PySide6 를 설치하기만 하면 됩니다.
  `dl_exp_manager/qt.py` 가 PyQt6 → PySide6 순으로 바인딩을 찾아 API 차이를 흡수합니다.
- Linux 서버 등 GUI 라이브러리가 없는 환경에서는 `libegl1 libgl1 libxkbcommon0` 등이 추가로 필요합니다.

## 화면 구성

| 영역 | 설명 |
|---|---|
| 상단 서버 상태 인디케이터 | Server 1~4 중 현재 학습(`running`) 중인 서버와 모델·경과 시간을 한 줄로 표시. 15초마다 자동 갱신 |
| 좌측 네비게이션 | DL Task ▸ Work ID 드릴다운 트리. Work 별 Train/Inference 건수 표시, 검색·추가·수정·삭제 |
| 중앙 상단 테이블 | 실행 목록. **열 헤더 클릭 시 정렬**, 전 컬럼 검색, 상태 필터, 컬럼 표시/숨김(헤더 우클릭) |
| 중앙 하단 상세 | 선택한 실행의 경로(+📁 폴더 열기), 실행 코드, `config.yml`, Metrics/Notes. 각 항목에 복사 버튼 |
| 우측 입력 폼 | 신규 등록 / 수정. 모델·데이터셋·서버·Optimizer는 **편집 가능한 콤보박스**(목록 선택 + 직접 입력) |

### 주요 기능

- **OS 탐색기 연동** — 경로 옆 `📁 폴더 열기` 버튼이 macOS Finder(`open`), Windows 탐색기(`os.startfile`), Linux(`xdg-open`)를 각각 호출합니다.
  경로가 존재하지 않으면 가장 가까운 상위 폴더를 대신 열고 그 사실을 알려 줍니다. 경로 셀을 더블클릭해도 열립니다.
- **정렬** — 실행 시간·PSNR·Latency 같은 숫자 컬럼은 문자열이 아니라 **실제 크기 순**으로 정렬됩니다(내부적으로 별도의 정렬 Role 사용).
- **동적 메트릭 컬럼** — 평가 지표는 JSON 으로 저장되며, 표에 등장한 지표(PSNR, SSIM, LPIPS, NIQE …)가 자동으로 컬럼이 됩니다. 새 지표를 추가해도 스키마 변경이 필요 없습니다.
- **내보내기** — 현재 필터·정렬이 적용된 표를 CSV(`utf-8-sig`, Excel 한글 안전)로 저장하거나, 선택 행 / 표 전체를 TSV로 클립보드에 복사(엑셀 바로 붙여넣기).
- **실행 시간 입력** — `3h 20m`, `01:30:00`, `5400`(초) 어떤 형식으로 넣어도 파싱됩니다.
- **복제** — 기존 실행을 같은 설정으로 복제(상태는 `queued`)해서 다음 실험 등록을 빠르게.

### 단축키

| 키 | 동작 |
|---|---|
| `F5` | 새로고침 |
| `Ctrl+1` / `Ctrl+2` | Train / Inference 탭 |
| `Ctrl+E` | 현재 탭 CSV 내보내기 |
| `Ctrl+C` / `Ctrl+Shift+C` | 선택 행 / 표 전체 클립보드 복사 |
| `Ctrl+Shift+T` / `Ctrl+Shift+W` | DL Task / Work ID 추가 |
| `Ctrl+O` | 다른 `experiments.db` 열기 |
| `Del` | 선택한 실행 삭제 |

## 프로젝트 구조

```
main.py                        진입점 (--db, --sample)
requirements.txt
dl_exp_manager/
  qt.py                        PyQt6 / PySide6 바인딩 추상화
  constants.py                 서버·Task·모델 프리셋, 상태값, 샘플 config
  db.py                        SQLite 스키마 및 CRUD (Database 클래스)
  models.py                    QAbstractTableModel + 정렬/필터 프록시
  utils.py                     탐색기 열기, 시간/숫자 포맷, CSV·TSV 직렬화
  sample_data.py               예시 실험 데이터
  main_window.py               메인 윈도우, 메뉴, 서버 상태 인디케이터
  widgets/
    common.py                  PathEdit, EditableCombo, MetricsEditor 등 재사용 위젯
    nav_panel.py               Level 1 / Level 2 드릴다운 트리
    run_panel.py               Train / Inference 대시보드 (표 + 상세 + 입력 폼)
tests/test_core.py             GUI 없이 도는 utils / db 테스트
```

## 데이터베이스 스키마

| 테이블 | 주요 컬럼 |
|---|---|
| `servers` | name, host, gpu, note |
| `tasks` | name(UNIQUE), description — **Level 1** |
| `works` | task_id→tasks, name, description, UNIQUE(task_id, name) — **Level 2** |
| `train_runs` | work_id→works, server, model, dataset, dataset_path, result_path, status, started_at, duration_sec, epochs, batch_size, lr, optimizer, metrics_json, exec_command, config_yaml, notes |
| `inference_runs` | work_id→works, server, model, **checkpoint_path**, dataset_path, result_path, device, input_size, **latency_ms**, **throughput_fps**, status, duration_sec, metrics_json, exec_command, config_yaml, notes |

`ON DELETE CASCADE` + `PRAGMA foreign_keys = ON` 이므로 Task 를 지우면 하위 Work 와 모든 실행 기록이 함께 정리됩니다.

## 테스트

```bash
pip install pytest
pytest -q          # 12 passed
```

GUI 위젯은 헤드리스에서 `QT_QPA_PLATFORM=offscreen` 으로 구동해 확인했습니다.
