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
python main.py --config my.yaml   # 선택지/컬럼 설정 파일 지정
python main.py --theme light      # 라이트 테마 (기본: dark)
python main.py --sample           # 비어 있으면 예시 데이터까지 생성 (UI 둘러보기용)
```

- Python 3.10+ 권장 (타입 힌트에 `X | None` 문법 사용).
- 폰트는 시스템에 설치된 것 중 앞 순위를 씁니다(Pretendard → Inter → Noto Sans KR → OS 기본).
  `assets/fonts/` 에 ttf/otf 를 넣어 두면 자동으로 등록해서 함께 후보로 삼습니다.
- **PySide6 를 쓰고 싶다면** `requirements.txt` 에서 PyQt6 대신 PySide6 를 설치하기만 하면 됩니다.
  `dl_exp_manager/qt.py` 가 PyQt6 → PySide6 순으로 바인딩을 찾아 API 차이를 흡수합니다.
- Linux 서버 등 GUI 라이브러리가 없는 환경에서는 `libegl1 libgl1 libxkbcommon0` 등이 추가로 필요합니다.

## 화면 구성

| 영역 | 설명 |
|---|---|
| 상단 서버 상태 패널 | 서버별 GPU 슬롯 점유 현황. 접으면 한 줄, 펴면 실행 중인 코드까지. 15초마다 자동 갱신 |
| 좌측 네비게이션 | DL Task ▸ Work ID 드릴다운 트리. Work 별 Train/Inference 건수 표시, 검색·추가·수정·삭제 |
| 중앙 상단 테이블 | 실행 목록. **열 헤더 클릭 시 정렬**, 전 컬럼 검색, 상태 필터, **Task 별 컬럼 구성**(헤더 우클릭으로 추가/제거/이름변경) |
| 중앙 하단 상세 | 선택한 실행의 경로(+📁 폴더 열기), 실행 코드, `config.yml`, Metrics/Notes. 각 항목에 복사 버튼 |
| 우측 입력 폼 | 신규 등록 / 수정. 모델·데이터셋·서버 등은 **드롭다운 맨 아래 `＋` 로 항목 추가 / 우클릭 삭제**가 되는 콤보박스. GPU는 서버 인벤토리에서 체크박스로 선택 |

### 주요 기능

- **선택지를 설정 파일로 관리** — 콤보박스 항목·평가 지표·표 컬럼을 `config/` 아래에서 관리합니다.
  **기능별로 파일이 나뉘어 있어** SR 을 고치려면 `config/tasks/SR.yaml`(약 17줄) 하나만 열면 됩니다.
  손으로 편집해도 되고 UI 에서 바꿔도 되며, 두 경로가 같은 파일을 씁니다.
  앱에서 바꾼 값은 **그 값이 있던 파일에만** 저장되고, 외부 편집기로 저장하면 앱이 즉시 반영합니다.
- **Task 별 구성** — SR 은 PSNR/SSIM/LPIPS 와 `scale`, Classification 은 Top-1/Top-5 처럼
  Task 마다 선택지·지표·컬럼이 다릅니다. 좌측에서 Task 를 바꾸면 표 컬럼과 폼 필드가 함께 바뀝니다.
- **드롭다운 인라인 항목 관리** — 콤보박스를 펼치면 맨 아래 `＋ 새 항목 추가…` 가 있고,
  기존 항목은 우클릭(또는 `F2`/`Del`)으로 이름 변경·삭제할 수 있습니다.
  추가할 때 *이 Task 전용* / *전체 공통* 을 고를 수 있고, 이름을 바꾸면 기존 기록도 함께 갱신할지 물어봅니다.
- **GPU 단위 서버 상태** — 서버당 GPU N장을 슬롯으로 표시하고, 동시에 도는 학습을 색으로 구분합니다.
  펼치면 각 학습의 GPU 번호·모델·경과 시간·실행 명령어가 나옵니다.
  서로 다른 학습이 같은 GPU 를 잡고 있으면 경고 테두리로 표시합니다.
- **다크 테마** — 색·폰트·간격은 `dl_exp_manager/theme/tokens.py` 한 곳에서 관리합니다
  (규격은 [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)).

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
| `Ctrl+R` | 설정 파일 다시 읽기 |
| `Ctrl+Shift+O` | 지금 보고 있는 Task 의 설정 파일 열기 |
| `F2` | 선택 항목 이름 변경 (트리 / 콤보 항목 / 컬럼 헤더 / 지표 행) |
| `Del` | 선택 항목 삭제 |
| `Ins` | 항목 추가 |

## 프로젝트 구조

```
main.py                        진입점 (--db, --config, --theme, --sample)
requirements.txt
config/
  options.yaml                 진입점 (버전 + 작성법 안내)
  servers.yaml                 서버 & GPU 인벤토리
  defaults.yaml                모든 Task 공통 선택지
  tasks/SR.yaml                Task 별 선택지 · 지표 · 컬럼
  tasks/DN.yaml                (Task 를 추가하면 파일도 함께 생깁니다)
  tasks/...
dl_exp_manager/
  qt.py                        PyQt6 / PySide6 바인딩 추상화
  constants.py                 상태값, 기본 Task, 샘플 config 텍스트
  config_store.py              config/ 의 여러 YAML 을 합쳐 읽고 원래 파일로 되돌려 쓰는 계층
  db.py                        SQLite 스키마, 마이그레이션, CRUD
  models.py                    Task 별 컬럼 구성 + 정렬/필터 프록시
  editing.py                   F2 / Del / 우클릭 편집의 공통 규약
  utils.py                     탐색기 열기, 시간/숫자 포맷, CSV·TSV 직렬화
  sample_data.py               예시 실험 데이터
  main_window.py               메인 윈도우, 메뉴, 설정 파일 감시
  theme/
    tokens.py                  색·치수·폰트 크기의 단일 출처
    dark.qss.tpl               토큰을 치환해 만드는 QSS 템플릿
    fonts.py                   폰트 스택 해석 / 번들 폰트 로드
  widgets/
    common.py                  PathEdit, ManagedCombo, GpuSelector, MetricsEditor
    nav_panel.py               Level 1 / Level 2 드릴다운 트리
    run_panel.py               Train / Inference 대시보드 (표 + 상세 + 입력 폼)
    server_panel.py            GPU 슬롯 기반 서버 상태 패널
tests/                         GUI 없이 도는 것 + offscreen 위젯 테스트
docs/STYLE_GUIDE.md            디자인 토큰과 컴포넌트 규격
docs/ROADMAP.md                설계 배경과 남은 계획
```

## 설정 파일

```
config/
  options.yaml        진입점. 버전과 작성법 안내만 들어 있습니다.
  servers.yaml        서버와 GPU 인벤토리 (index / type / memory_gb)
  defaults.yaml       모든 Task 가 공유하는 기본 선택지
  tasks/
    SR.yaml           Task 별 options · metrics · columns
    DN.yaml
    ...
```

`config/tasks/SR.yaml` 예시 — 이 한 파일이 SR 의 콤보박스, 표 컬럼, 지표 표시를 모두 결정합니다.

```yaml
name: SR
label: Super Resolution
options:
  model: [Restormer, SwinIR, MambaIR, HAT, EDSR, RCAN]
  dataset: [DIV2K, DF2K, Flickr2K, Set5, Set14, Urban100]
  scale: [x2, x3, x4]          # model/dataset/optimizer 외의 이름 = 사용자 정의 필드
metrics:
- {key: PSNR, unit: dB, digits: 2, higher_is_better: true}
- {key: SSIM, digits: 4, higher_is_better: true}
- {key: LPIPS, digits: 3, higher_is_better: false}
columns:
  train: [status, server, gpus, model, dataset, scale, duration, PSNR, SSIM, LPIPS, result_path]
  inference: [status, server, gpus, model, checkpoint_path, dataset, latency_ms, PSNR, SSIM]
```

규칙 몇 가지:

- **상속은 "대체"입니다.** Task 의 `options.model` 이 있으면 `defaults.yaml` 의 `model` 을 덮어씁니다.
  합쳐지지 않으므로 "이 항목이 왜 목록에 있지?" 가 생기지 않습니다.
- **앱이 쓰는 파일은 값이 있던 파일뿐입니다.** SR 모델을 UI 에서 추가하면 `tasks/SR.yaml` 만 바뀝니다.
- **파일 하나가 깨져도 나머지는 삽니다.** `tasks/DN.yaml` 에 문법 오류가 있으면 그 Task 만 빠지고
  상태바에 이유가 뜹니다. 정의가 깨졌을 때 같은 이름의 내장 정의로 덮어쓰지 않습니다(원본 유실 방지).
- **덮어쓰기 전에 `.bak` 을 남깁니다.**
- 예전처럼 `options.yaml` 한 파일에 전부 들어 있으면 첫 실행 때 자동으로 나눠 줍니다(원본은 `.bak`).
- `ruamel.yaml` 을 설치하면 주석과 순서를 보존하며 저장합니다. 없으면 PyYAML 로 동작합니다.

## 데이터베이스 스키마

| 테이블 | 주요 컬럼 |
|---|---|
| `servers` | name, host, gpu, note |
| `tasks` | name(UNIQUE), description — **Level 1** |
| `works` | task_id→tasks, name, description, UNIQUE(task_id, name) — **Level 2** |
| `train_runs` | work_id→works, server, model, dataset, dataset_path, result_path, status, started_at, duration_sec, epochs, batch_size, lr, optimizer, metrics_json, exec_command, config_yaml, notes |
| `inference_runs` | work_id→works, server, model, **checkpoint_path**, dataset_path, result_path, device, input_size, **latency_ms**, **throughput_fps**, status, duration_sec, metrics_json, exec_command, config_yaml, notes |

두 run 테이블은 `gpu_indices`(예: `"0,1"`)와 `extra_json`(Task 별 사용자 정의 필드)도 가집니다.

`ON DELETE CASCADE` + `PRAGMA foreign_keys = ON` 이므로 Task 를 지우면 하위 Work 와 모든 실행 기록이 함께 정리됩니다.
스키마 버전은 `PRAGMA user_version` 으로 관리하며, v1 DB 는 앱 실행 시 자동으로 v2 로 올라갑니다(데이터 보존, 몇 번 실행해도 안전).

## 테스트

```bash
pip install pytest
pytest -q          # 70 passed
```

GUI 위젯은 헤드리스에서 `QT_QPA_PLATFORM=offscreen` 으로 구동해 확인했습니다.
