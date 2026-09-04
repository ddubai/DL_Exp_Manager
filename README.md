# DL Experiment Manager

4대의 독립된 학습 서버(Server 1~4)에서 돌린 실험을 **로컬 PC 한 곳에서 아카이빙·검색·비교**하는 PyQt6 데스크톱 애플리케이션입니다.
모든 기록은 로컬 SQLite 파일(`experiments.db`) 하나에 저장되므로 별도 서버나 계정이 필요 없습니다.

```
DL Task (SR / DN / Clustering / Classification)   ← Level 1 : 좌측 드릴다운
   └─ Work ID (SSL2SL, BSR-x4, ...)               ← Level 2 : 좌측 드릴다운
        ├─ Dataset (이름 + 위치, Work 별 등록)      ← Level 2 화면에 인라인으로
        ├─ Train      탭                          ← Level 3 : 상단 탭 (아이콘 없이 텍스트만)
        └─ Inference  탭
```

## 설치 & 실행

```bash
pip install -r requirements.txt

cp config/servers.template.yaml config/servers.yaml   # 실서버 정보를 직접 채워 넣는다 (gitignore 대상)

python main.py                    # 프로젝트 폴더의 experiments.db 사용
python main.py --db ~/exp/my.db   # DB 경로 지정
python main.py --config my.yaml   # 선택지/컬럼 설정 파일 지정
python main.py --theme light      # 라이트 테마로 처음 실행 (이후엔 앱 안에서 고른 게 우선)
python main.py --sample           # 비어 있으면 예시 데이터까지 생성 (UI 둘러보기용)
```

- **다크/라이트 전환은 앱 메뉴 `View ▸ Theme` 에서도 바로 됩니다.** 재시작 없이 즉시 반영되고
  (좌측 네비게이션 + Train/Inference 표를 새로 그림 - 보던 Task/Work·탭은 그대로 유지),
  고른 값은 다음 실행에도 이어집니다. `--theme` 를 주면 그 실행에서만 강제로 덮어씁니다.
- Python 3.10+ 권장 (타입 힌트에 `X | None` 문법 사용).
- 폰트는 시스템에 설치된 것 중 앞 순위를 씁니다(Pretendard → Apple SD Gothic Neo/Malgun Gothic → Noto Sans KR → OS 기본).
  `assets/fonts/` 에 ttf/otf 를 넣어 두면 자동으로 등록해서 함께 후보로 삼습니다.
- **PySide6 를 쓰고 싶다면** `requirements.txt` 에서 PyQt6 대신 PySide6 를 설치하기만 하면 됩니다.
  `dl_exp_manager/qt.py` 가 PyQt6 → PySide6 순으로 바인딩을 찾아 API 차이를 흡수합니다.
- Linux 서버 등 GUI 라이브러리가 없는 환경에서는 `libegl1 libgl1 libxkbcommon0` 등이 추가로 필요합니다.
- `config/servers.yaml` 을 복사해서 만들지 않아도 앱은 죽지 않습니다 - placeholder 서버 4개로 뜨고
  상태바에 안내가 뜹니다. 실서버를 쓰려면 위 `cp` 명령이나 서버 상태 바의 + 버튼으로 등록하세요.

## 화면 구성

| 영역 | 설명 |
|---|---|
| 상단 서버 상태 바 | 서버 이름 + 사용 중 GPU 비율만 보이는 **한 줄** 표시. 클릭하면 실행 중인 학습(GPU 개수·모델·경과시간·명령어)이 메뉴로, 우클릭하면 서버/GPU 편집 메뉴. 15초마다 자동 갱신 |
| 좌측 네비게이션 | **All Tasks ▸ Task ▸ Work 드릴다운**(트리 아님, 브레드크럼으로 한 번에 한 단계만). Work 까지 들어가면 그 Work 에 등록된 **Dataset(이름 + 위치)** 이 그 자리에 바로 나와 추가/수정/삭제할 수 있습니다. 검색·Task/Work 추가·이름변경·삭제는 그대로 지원 |
| 중앙 상단 테이블 | 실행 목록. 툴바의 **`+ New Run`** 버튼으로 등록, **`⇄ Compare`** 로 2~3개 실행을 지표·config.yaml diff 로 나란히 비교. **열 헤더 클릭 시 정렬**, 전 컬럼 검색, 상태 필터, **Task 별 컬럼 구성**(헤더 우클릭으로 추가/제거/이름변경) |
| 중앙 하단 상세 | **행을 선택했을 때만 나타남.** 경로(+📁 폴더 열기), 실행 코드, `config.yml`, Metrics/Notes, 그리고 그 실행이 **생성/수정/복제될 때마다 기록되는 History** 탭. **🖼 View Image**(결과 폴더의 대표 이미지 한 장) / **📈 Training Curve**(Train 전용, 로그를 파싱해 iteration 별 지표를 그린 라인 차트) 버튼도 여기에 |
| 등록/수정 팝업 | **`+ New Run` 클릭 시에만 뜨는 다이얼로그.** **좌(실행 설정) / 우(경로 + 실행 코드) 2단 분할**이며, 좌측 스크롤과 우측 스크롤이 독립적으로 움직이고 Save/Clear/Cancel 버튼은 스크롤 밖에 고정돼 항상 보입니다. Work ID 는 좌측에서 이미 고른 Work 가 있으면 그 값을 기본으로 채웁니다. Server 는 상단 서버 목록 중에서만 고르고, GPU 는 슬롯 대신 **개수**만 입력합니다. 상태 기본값은 `queued`. **Dataset** 콤보는 그 Work 에 등록된 데이터셋 레지스트리와 바로 연동되어, 고르면 경로(+ Inference 는 Input size 도)가 자동으로 채워집니다(옆 📦 버튼으로 전체 관리). **⇪ Parse** 버튼은 결과 폴더의 `config.yaml` + 학습 로그를 읽어 Model/Dataset/하이퍼파라미터/평가지표/소요시간을 자동으로 채웁니다(자동 로깅). Inference 폼은 GPU 대신 **같은 Work 의 Train Run + Epoch/Iter** 를 먼저 고르는 순서(Server 는 유지) |

### 주요 기능

- **선택지를 설정 파일로 관리** — 콤보박스 항목·평가 지표·표 컬럼을 `config/` 아래에서 관리합니다.
  **기능별로 파일이 나뉘어 있어** SR 을 고치려면 `config/tasks/SR.yaml`(약 17줄) 하나만 열면 됩니다.
  손으로 편집해도 되고 UI 에서 바꿔도 되며, 두 경로가 같은 파일을 씁니다.
  앱에서 바꾼 값은 **그 값이 있던 파일에만** 저장되고, 외부 편집기로 저장하면 앱이 즉시 반영합니다.
- **Task 별 구성** — SR 은 PSNR/SSIM/LPIPS 와 `scale`, Classification 은 Top-1/Top-5 처럼
  Task 마다 선택지·지표·컬럼이 다릅니다. 좌측에서 Task 를 바꾸면 표 컬럼과 폼 필드가 함께 바뀝니다.
  평가 지표는 **같은 Task 안에서 공유**됩니다 — 어느 Run 에서든 새 지표 값을 입력하면 그 Task 의 지표로
  등록되고, 다음 New Run 부터 값 빈 상태로 미리 채워집니다. 지표마다 `higher_is_better`(높을수록/낮을수록
  좋음)를 지정할 수 있고, 같은 Work 안 최고값은 표에서 강조됩니다.
- **Work 별 데이터셋 레지스트리** — 데이터셋을 이름 + (선택) Variant + 경로 + 총 데이터 개수 +
  이미지 크기(예: `256x256`) + 확장자(예: `tiff`)로 등록해 둡니다.
  같은 이름이라도 Variant 를 다르게 두면 "전체 페어"와 "특정 서브셋"을 별개 항목으로 관리할 수 있습니다
  (예: `DIV2K · Full Pair`, `DIV2K · Subset A`). 좌측 네비게이션에서 바로 추가/수정/삭제하고,
  등록/수정 폼의 **Dataset** 콤보 자체가 이 레지스트리와 연동됩니다 — 고르면 경로가 자동으로 채워지고
  (Inference 는 등록해 둔 이미지 크기로 Input size 도 자동 채움, 수정 가능), 드롭다운 맨 아래
  `＋ 새 데이터셋 추가…` 로 바로 등록할 수도 있습니다.
- **train.py 결과에서 자동 채우기** — 결과 폴더의 `config.yaml`(BasicSR 류 스키마 우선 시도)과
  학습 로그(`loss.log` 등)를 파싱해 Model/Dataset/Batch/LR/Optimizer/Epoch, 최근 검증 지표,
  소요 시간을 자동으로 채웁니다. 형식을 못 알아봐도 예외 없이 빈 결과만 돌려주고, 채운 값은 저장 전에
  폼에서 그대로 확인·수정할 수 있습니다.
- **Run 히스토리** — 각 Run 이 언제 생성됐는지, 무엇이 바뀌었는지(필드별 변경 diff), 어느 Run 에서
  복제됐는지를 시각과 함께 기록합니다. 상세 패널의 History 탭에서 확인합니다.
- **드롭다운 인라인 항목 관리** — 콤보박스를 펼치면 맨 아래 `＋ 새 항목 추가…` 가 있고,
  기존 항목은 우클릭(또는 `F2`/`Del`)으로 이름 변경·삭제할 수 있습니다.
  추가할 때 *이 Task 전용* / *전체 공통* 을 고를 수 있고, 이름을 바꾸면 기존 기록도 함께 갱신할지 물어봅니다.
  드롭다운 화살표는 테마 강조색으로 항상 눈에 띄게 그립니다.
- **GPU 단위 서버 상태, 한 줄로** — 상단 바는 서버 이름과 사용 중 GPU **개수**(`● Server 3 (4/4)`)만 보여
  화면을 적게 차지합니다. 클릭하면 각 학습의 GPU 개수·모델·경과 시간·실행 명령어가 메뉴로 나오고,
  한 서버가 가진 GPU 수보다 더 많이 잡혀 있으면(개수 초과) 그 사실이 표시됩니다.
- **다크 테마 + 영문 UI** — 색·폰트·간격은 `dl_exp_manager/theme/tokens.py` 한 곳에서 관리합니다
  (규격은 [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)). 앱 화면의 문구는 모두 영어입니다.
- **OS 탐색기 연동** — 경로 옆 `📁 폴더 열기` 버튼이 macOS Finder(`open`), Windows 탐색기(`os.startfile`), Linux(`xdg-open`)를 각각 호출합니다.
  경로가 존재하지 않으면 가장 가까운 상위 폴더를 대신 열고 그 사실을 알려 줍니다. 경로 셀을 더블클릭해도 열립니다.
- **정렬** — 실행 시간·PSNR·Latency 같은 숫자 컬럼은 문자열이 아니라 **실제 크기 순**으로 정렬됩니다(내부적으로 별도의 정렬 Role 사용).
- **동적 메트릭 컬럼** — 평가 지표는 JSON 으로 저장되며, 표에 등장한 지표(PSNR, SSIM, LPIPS, NIQE …)가 자동으로 컬럼이 됩니다. 새 지표를 추가해도 스키마 변경이 필요 없습니다.
- **내보내기** — 현재 필터·정렬이 적용된 표를 CSV(`utf-8-sig`, Excel 한글 안전)로 저장하거나, 선택 행 / 표 전체를 TSV로 클립보드에 복사(엑셀 바로 붙여넣기). Markdown/HTML 리포트 내보내기도 지원합니다.
- **실행 시간 입력** — `3h 20m`, `01:30:00`, `5400`(초) 어떤 형식으로 넣어도 파싱됩니다.
- **복제** — 기존 실행을 같은 설정으로 복제(상태는 `queued`)해서 다음 실험 등록을 빠르게.
- **즐겨찾기 / 태그 / 실패 사유** — 실패한 실험도 자산입니다. 상태가 `failed` 일 때만 실패 사유 입력란이 나타납니다.

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
| `F2` | 선택 항목 이름 변경 (좌측 Task/Work / 콤보 항목 / 컬럼 헤더 / 지표 행) |
| `Del` | 선택 항목 삭제 |
| `Ins` | 항목 추가 |

## 프로젝트 구조

```
main.py                        진입점 (--db, --config, --theme, --sample)
requirements.txt
config/
  options.yaml                 진입점 (버전 + 작성법 안내)
  servers.yaml                 서버 & GPU 인벤토리 (gitignore 대상, 직접 만들어야 함)
  servers.template.yaml        servers.yaml 을 만들 때 복사하는 예시 (git 추적)
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
  utils.py                     탐색기 열기, 시간/숫자 포맷, CSV·TSV 직렬화, 대표 이미지 탐색
  log_parser.py                train.py 의 config.yaml / 학습 로그(loss.log 등) 파서
  sample_data.py               예시 실험 데이터
  main_window.py               메인 윈도우, 메뉴, 설정 파일 감시
  theme/
    tokens.py                  색·치수·폰트 크기의 단일 출처
    dark.qss.tpl               토큰을 치환해 만드는 QSS 템플릿
    fonts.py                   폰트 스택 해석 / 번들 폰트 로드
    icons.py                   폴더/수정/삭제/추가 벡터 아이콘 (이모지 대체)
  widgets/
    common.py                  PathEdit, ManagedCombo, ServerCombo, GpuSelector, MetricsEditor
    nav_panel.py               Task ▸ Work 드릴다운 + Work 별 Dataset 인라인 표시
    run_panel.py               Train / Inference 대시보드 (표 + 상세 + 입력 폼)
    server_panel.py            GPU 개수 기반 서버 상태 패널
    dataset_dialog.py          Work 별 데이터셋 등록 (이름 + Variant + 경로)
    compare_dialog.py          Run 2~3개 비교 (지표/파라미터 표 + config.yaml diff)
    curve_chart.py             학습 곡선 (커스텀 QPainter 라인 차트, 외부 의존성 없음)
    image_viewer.py            결과 폴더의 대표 이미지 뷰어
    log_viewer.py              로그 tail 뷰어
    search_dialog.py           전역 검색 (Ctrl+K)
tests/                         GUI 없이 도는 것 + offscreen 위젯 테스트
docs/STYLE_GUIDE.md            디자인 토큰과 컴포넌트 규격
docs/ROADMAP.md                설계 배경과 진행 기록
```

## 설정 파일

```
config/
  options.yaml            진입점. 버전과 작성법 안내만 들어 있습니다.
  servers.yaml            서버와 GPU 인벤토리 (index / type / memory_gb) - gitignore 대상
  servers.template.yaml   servers.yaml 예시. git 에는 이것만 들어 있습니다.
  defaults.yaml           모든 Task 가 공유하는 기본 선택지
  tasks/
    SR.yaml               Task 별 options · metrics · columns
    DN.yaml
    ...
```

**`servers.yaml` 은 실서버 IP/구성이 들어가서 git 에 커밋하지 않습니다.** 저장소에는
`servers.template.yaml` 만 들어 있고, 실행 전에 아래처럼 복사해서 직접 채웁니다.

```bash
cp config/servers.template.yaml config/servers.yaml
```

복사하지 않고 바로 실행해도 앱이 죽지는 않습니다 - `servers.template.yaml` 과 같은 내용의
placeholder 서버 4개(Server 1~4)로 뜨고, 상태바에 "복사해서 쓰라"는 안내가 뜹니다. 서버 상태
바의 + 버튼으로 서버를 하나라도 추가하면 그 시점에 `servers.yaml` 이 만들어집니다.

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
| `datasets` | work_id→works, name, variant, path, **sample_count**, **image_size**, **extension**, notes, created_at(등록일 - UI 에 표시), UNIQUE(work_id, name, variant) — Work 별 데이터셋 레지스트리 |
| `train_runs` | work_id→works, server, model, dataset, dataset_path, result_path, status, started_at, duration_sec, epochs, batch_size, crop_size, lr, optimizer, metrics_json, exec_command, config_yaml, notes, favorite, tags, failure_reason |
| `inference_runs` | work_id→works, server, model, **checkpoint_path**, dataset_path, result_path, device, input_size, **latency_ms**, **throughput_fps**, status, duration_sec, metrics_json, exec_command, config_yaml, notes, favorite, tags, failure_reason, **source_train_run_id**, **checkpoint_epoch** |
| `run_history` | run_kind, run_id, action(created/updated/duplicated), detail, created_at — Run 별 변경 이력 |

두 run 테이블은 `gpu_indices`(GPU **개수**, 예: `"2"`. 예전 콤마 인덱스 목록 `"0,1"`도 개수로 읽힙니다)와
`extra_json`(Task 별 사용자 정의 필드)도 가집니다.

`ON DELETE CASCADE` + `PRAGMA foreign_keys = ON` 이므로 Task 를 지우면 하위 Work·Dataset·실행 기록이 함께 정리됩니다.
스키마 버전은 `PRAGMA user_version` 으로 관리하며(현재 v7), 예전 DB 는 앱 실행 시 자동으로 올라갑니다(데이터 보존, 몇 번 실행해도 안전).

## 테스트

```bash
pip install pytest
pytest -q
```

GUI 위젯은 헤드리스에서 `QT_QPA_PLATFORM=offscreen` 으로 구동해 확인했습니다.
