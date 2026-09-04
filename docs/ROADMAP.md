# 개발 계획 (v2)

이 문서는 v2 의 설계안이자 진행 기록입니다.
**R1~R7 은 구현 완료**(§9 참고). **R8(시각화 패널)은 학습 곡선·대표 이미지 뷰어를 축소된 형태로 구현**했고
(§10.4 참고), 나머지(지표 비교 막대/산점도, 성능-속도 산점도, Work 요약 대시보드)는 아직 계획만입니다.
**§7 의 추가 제안 중 2(설정 자동 채우기)·3(Run 비교 뷰)은 구현 완료**, 나머지는 미채택.
§11 에 2026-09 세션에서 새로 추가된 기능(등록 폼 개편, 자동 로깅, Run 히스토리, Work 별 데이터셋 레지스트리,
좌측 네비게이션 드릴다운 개편)을 정리해 뒀습니다.

---

## 1. 확정 요구사항

| # | 요구사항 | 핵심 | 상태 |
|---|---|---|---|
| R1 | 콤보박스 선택지를 config로 관리 | `config/options.yaml`, 코드로도 UI로도 수정 | ✅ |
| R2 | 콤보 드롭다운 맨 아래 `＋` 추가 / 우클릭 삭제 | 인라인 항목 관리 | ✅ |
| R3 | 옵션을 Task 별로 관리 | SR과 Classification의 선택지가 다름 | ✅ |
| R4 | Task 별로 테이블 컬럼을 다르게 | 평가 지표가 Task마다 다름 | ✅ |
| R5 | Server Status에 GPU 단위 표시 | 서버당 N GPU, 동시 학습 다수, GPU 종류(V100/H100) | ✅ |
| R6 | UI 모든 항목 우클릭 추가/제거/수정 + F2 rename | 초반 수동 조정 편의 | ✅ |
| R7 | 다크 테마 + 폰트 | → `docs/STYLE_GUIDE.md` | ✅ |
| R8 | 시각화 패널 | **계획만**, 이번 구현 범위 밖 | 계획만 |

---

## 2. R1·R3 — 설정 파일 설계

### 2.1 파일 위치와 이름

`config/` 아래에 **기능별로 나눠** 둡니다(저장소 추적). 실험용 `config.yml`(입력 폼의 그 필드)과
헷갈리지 않도록 진입점 이름은 **`options.yaml`** 로 구분합니다.

```
config/
  options.yaml        진입점 (버전 + 작성법 안내)
  servers.yaml        서버 & GPU 인벤토리
  defaults.yaml       모든 Task 공통 선택지
  tasks/<이름>.yaml   Task 별 options · metrics · columns
```

처음에는 한 파일로 만들었지만 286줄이 되어 SR 하나 고치려면 전체를 뒤져야 했습니다.
지금은 Task 파일이 약 17줄이라 열면 바로 전체가 보입니다.

읽을 때는 전부 합쳐 하나의 뷰로 보고, **쓸 때는 그 값이 원래 있던 파일에만** 저장합니다.
(SR 모델을 UI 에서 추가하면 `tasks/SR.yaml` 만 바뀝니다.)
주석이 날아가지 않기를 원하면 `ruamel.yaml`(round-trip 로더)을 설치하면 되고,
없으면 PyYAML 로 동작합니다. 스칼라 리스트는 `[a, b, c]` 한 줄로 뽑아 파일을 짧게 유지합니다.

### 2.2 스키마

```yaml
version: 2

# ── 서버 & GPU 인벤토리 (R5) ──────────────────────────
servers:
  - name: Server 1
    host: 192.168.0.101
    gpus:
      - {index: 0, type: H100, memory_gb: 80}
      - {index: 1, type: H100, memory_gb: 80}
  - name: Server 2
    host: 192.168.0.102
    gpus:
      - {index: 0, type: V100, memory_gb: 32}
      - {index: 1, type: V100, memory_gb: 32}

# ── 모든 Task 공통 선택지 ─────────────────────────────
defaults:
  optimizer: [AdamW, Adam, SGD, Lion]
  device: ["cuda:0", "cuda:1", cpu]

# ── Task 별 정의 (R3, R4) ─────────────────────────────
tasks:
  SR:
    label: Super Resolution
    options:                                  # 콤보박스 선택지
      model: [Restormer, SwinIR, MambaIR, HAT, EDSR]
      dataset: [DIV2K, DF2K, Set5, Urban100]
      scale: ["x2", "x3", "x4"]               # 사용자가 만든 커스텀 필드
    metrics:                                  # 표 컬럼 + 폼 프리셋
      - {key: PSNR,  unit: dB, digits: 2, higher_is_better: true}
      - {key: SSIM,  digits: 4, higher_is_better: true}
      - {key: LPIPS, digits: 3, higher_is_better: false}
    columns:
      train:     [status, server, gpus, model, dataset, scale, duration, PSNR, SSIM, LPIPS, result_path]
      inference: [status, server, gpus, model, checkpoint_path, dataset, latency_ms, throughput_fps, PSNR, SSIM]

  Classification:
    label: Image Classification
    options:
      model: [ResNet-50, ViT-B/16, ConvNeXt-T]
      dataset: [ImageNet-1k, CIFAR-100]
    metrics:
      - {key: Top-1, unit: "%", digits: 2, higher_is_better: true}
      - {key: Top-5, unit: "%", digits: 2, higher_is_better: true}
    columns:
      train:     [status, server, gpus, model, dataset, duration, Top-1, Top-5, result_path]
      inference: [status, server, model, checkpoint_path, dataset, latency_ms, Top-1, Top-5]
```

### 2.3 결정한 규칙

- **상속**: `tasks.<T>.options.<field>` 가 있으면 `defaults.<field>` 를 **대체**합니다.
  (합집합 자동 병합은 "왜 이 항목이 여기 있지?"를 만들어서 안 씁니다.)
- **UI `＋` 추가 시 범위**: 다이얼로그에서 `이 Task 전용` / `전체 공통` 을 고르게 합니다.
  기본값은 **이 Task 전용**.
- **metrics 정의가 하는 일 3가지**:
  1. 표의 지표 컬럼 목록
  2. 표시 자릿수/단위 (`32.41 dB`)
  3. `higher_is_better` → 같은 Work 내 최고값 하이라이트 (§7-1)
- **파일이 없거나 깨졌을 때**: 앱은 죽지 않고 내장 기본값으로 뜨며, 상태바에
  "options.yaml 을 읽지 못해 기본값으로 실행 중" 을 표시합니다. 사용자가 손으로 고치는
  파일이므로 문법 오류는 상시 발생한다고 봐야 합니다.
- **저장 시점**: UI 변경 즉시 저장(디바운스 300ms). 저장 전 `.bak` 1개 유지.

### 2.4 구현

```
dl_exp_manager/config_store.py
  load(path) -> OptionsConfig      # 검증 + 기본값 병합, 오류는 수집해서 반환
  OptionsConfig.options_for(task, field) -> list[str]
  OptionsConfig.metrics_for(task) -> list[MetricDef]
  OptionsConfig.columns_for(task, mode) -> list[str]
  OptionsConfig.add_option(task|None, field, value)      # 저장까지
  OptionsConfig.remove_option(...) / rename_option(...)
  OptionsConfig.servers -> list[ServerDef]  (gpus 포함)
```

`QFileSystemWatcher` 로 파일을 감시해서, 외부 편집기로 고치면 앱이 즉시 반영합니다.
(코드로 지정하고 싶다는 요구의 실제 사용감은 여기서 갈립니다.)

---

## 3. R2·R6 — 인라인 항목 관리 / 우클릭 / F2

### 3.1 대상 표면 전수

| 표면 | 추가 | 삭제 | 이름변경(F2) | 저장 위치 |
|---|---|---|---|---|
| 좌측 네비게이션 — DL Task (2026-09 부터 드릴다운, §11.6) | ✅ | ✅ | ✅ | DB `tasks` |
| 좌측 네비게이션 — Work ID | ✅ | ✅ | ✅ | DB `works` |
| 콤보박스 항목 (model/dataset/server/optimizer/device/커스텀) | ✅ 드롭다운 `＋` | ✅ 우클릭 | ✅ | `options.yaml` |
| 테이블 컬럼 (헤더 우클릭) | ✅ | ✅ | ✅ (표시명) | `options.yaml` `columns` |
| 지표 정의 | ✅ | ✅ | ✅ | `options.yaml` `metrics` |
| 폼의 커스텀 필드 | ✅ | ✅ | ✅ | `options.yaml` + DB `extra_json` |
| 서버 / GPU 항목 | ✅ | ✅ | ✅ | `options.yaml` `servers` |

### 3.2 공통 컨트롤러

표면마다 따로 구현하면 동작이 미묘하게 갈립니다. 하나의 프로토콜로 묶습니다.

```python
# dl_exp_manager/editing.py
class ItemEditor(Protocol):
    def can_add(self) -> bool: ...
    def add(self, parent_ctx) -> None: ...
    def rename(self, item_ctx, new_name: str) -> None: ...
    def remove(self, item_ctx) -> None: ...
    def usage_count(self, item_ctx) -> int: ...   # 삭제 전 경고용

def install_item_editing(widget, editor: ItemEditor):
    """우클릭 메뉴 + F2 + Del 을 위젯에 한 번에 부착."""
```

### 3.3 콤보박스 드롭다운 `＋` — 구현 메모

이번 작업에서 **가장 까다로운 부분**이라 미리 적어 둡니다.

- `QComboBox.setView(QListView)` 후, 모델 마지막에 센티넬 행(`＋ 새 항목 추가…`)을 둡니다.
- `activated` 시그널에서 센티넬 인덱스면 → 이전 값 복구 후 추가 다이얼로그.
  (`currentIndexChanged` 가 아니라 `activated` 여야 프로그램적 변경과 구분됩니다.)
- 우클릭: `view().viewport()` 에 이벤트 필터를 걸어 `QEvent.ContextMenu` 를 잡고
  `indexAt(pos)` 로 대상 항목을 찾습니다. 센티넬 행은 메뉴에서 제외.
- F2: 팝업이 열린 상태의 하이라이트 항목 기준. 팝업을 닫고 다이얼로그를 띄운 뒤 재오픈.
- 센티넬은 `Qt.ItemDataRole` 로 표시해 두고, 정렬·자동완성·필터에서 항상 제외합니다.

**폴백 안전장치**: 콤보 옆 `⋯` 버튼 → "항목 관리" 다이얼로그(추가/삭제/순서 변경).
인라인 방식이 플랫폼별로 말썽이면 이쪽이 대안이고, 순서 변경은 어차피 다이얼로그가 필요합니다.

### 3.4 삭제 정책

옵션을 지워도 **기존 run 기록은 건드리지 않습니다**(선택지에서만 제거).
삭제 확인창에 `이 값을 쓰는 실행 N건이 있습니다` 를 표시합니다.
이름 변경 시에는 `기존 기록 N건도 함께 변경할까요?` 를 물어 일괄 UPDATE를 선택할 수 있게 합니다.

---

## 4. R4 — Task 별 테이블 컬럼

### 4.1 컬럼 식별자

- 내장 필드: `status`, `server`, `gpus`, `model`, `dataset`, `dataset_path`, `result_path`,
  `checkpoint_path`, `duration`, `started_at`, `latency_ms`, `throughput_fps`, `notes` …
- 지표: `metrics` 에 정의한 `key` 를 그대로 (`PSNR`, `Top-1`)
- 커스텀: `options.<field>` 로 만든 필드 (`scale`) → DB `extra_json` 에 저장

### 4.2 동작

- 좌측에서 Task를 바꾸면 표 모델이 **컬럼 세트를 갈아끼웁니다**(모델 리셋).
- Task 범위(Work 미선택)에서는 그 Task의 컬럼을 씁니다.
- 컬럼 폭·순서·숨김은 Task별로 `QSettings` 에 기억합니다.
- 헤더 우클릭 메뉴: `컬럼 추가…` / `이 컬럼 숨기기` / `표시명 변경(F2)` / `기본값 복원`.

### 4.3 스키마 변경 (v1 → v2)

`PRAGMA user_version` 기반 마이그레이션을 `db.py` 에 추가합니다. 기존 DB는 보존됩니다.

```sql
ALTER TABLE train_runs      ADD COLUMN gpu_indices TEXT DEFAULT '';   -- "0,1"
ALTER TABLE train_runs      ADD COLUMN extra_json  TEXT DEFAULT '{}'; -- 커스텀 필드
ALTER TABLE inference_runs  ADD COLUMN gpu_indices TEXT DEFAULT '';
ALTER TABLE inference_runs  ADD COLUMN extra_json  TEXT DEFAULT '{}';
PRAGMA user_version = 2;
```

GPU 인벤토리는 DB가 아니라 `options.yaml` 에 둡니다 — 고정 하드웨어 정보이고,
"코드로 지정하고 싶다"는 요구에 그대로 맞습니다.
마이그레이션은 `PRAGMA table_info` 로 컬럼 존재를 먼저 확인해 **몇 번 실행해도 안전**하게 만듭니다.

---

## 5. R5 — GPU 인지형 서버 상태 패널

### 5.1 표시안

```
┌─ 서버 상태 ───────────────────────────────── 3/20 GPU 사용 중 ─┐
│ Server 1  H100 ×8   ██▊▊░░░░  ⌄                                │
│ Server 2  V100 ×4   ░░░░                                       │
│ Server 3  A6000 ×4  ██░░                                       │
│ Server 4  A100 ×4   ░░░░              (offline)                │
└────────────────────────────────────────────────────────────────┘
        ⌄ 펼치면
        ├ GPU 0,1  Restormer · SR/SSL2SL · 03:14:22
        │          CUDA_VISIBLE_DEVICES=0,1 python train.py -opt …   [복사] [📁]
        └ GPU 2    SwinIR    · SR/BSR-x4 · 00:41:10
```

- 접힌 상태는 지금처럼 한 줄, 펼치면 실행 중인 코드까지 — 상시로는 자리를 안 먹게.
- 같은 run이 쓰는 GPU는 같은 색(시리즈 팔레트 순환).
- 슬롯 호버 툴팁: 모델 · Work · 경과 시간 · 실행 명령어 전문.
- GPU 충돌 경고: 서로 다른 running run이 같은 GPU 인덱스를 잡고 있으면 그 슬롯을 `failed` 색 테두리로 표시.
  (실수로 같은 GPU에 두 개 띄운 상황을 잡아 줍니다.)

### 5.2 입력 폼

`Device` 자유 입력 대신 **GPU 체크박스**를 씁니다.
Server를 고르면 그 서버의 GPU 목록이 `[0][1][2][3]…` 토글로 나오고,
선택 결과가 `gpu_indices` 에 저장되며 `CUDA_VISIBLE_DEVICES=0,1` 문자열을 자동 생성합니다.

---

## 6. R7 — 다크 테마

`docs/STYLE_GUIDE.md` 에 토큰과 컴포넌트 규격을 정리했습니다. 구현은:

```
dl_exp_manager/theme/tokens.py     # 색·크기·간격 단일 출처
dl_exp_manager/theme/dark.qss.tpl  # {{token}} 치환 템플릿
dl_exp_manager/theme/fonts.py      # 폰트 스택 해석 / 번들 폰트 로드
assets/fonts/                      # (선택) Pretendard, JetBrains Mono — 둘 다 OFL
```

- 현재 코드에 흩어진 하드코딩 색(`#1a73e8`, `#666`, `STATUS_COLORS`)을 전부 토큰 참조로 교체합니다.
- `QPalette` + QSS 를 **함께** 설정합니다. QSS만으로는 툴팁·컨텍스트 메뉴 일부가 OS 기본 밝은 색으로 남습니다.
- 폰트 동봉 여부는 결정이 필요합니다 (§8-3).

---

## 7. 추가 제안 (승인 필요)

실험 아카이빙 도구로 쓰면서 실제로 아쉬울 지점 순으로 정리했습니다.
**번호만 알려주시면 그 항목만 범위에 넣겠습니다.**

| # | 제안 | 왜 유용한가 | 비용 | 추천 |
|---|---|---|---|---|
| 1 | **최고 성능 자동 하이라이트** | 같은 Work 안에서 최고 PSNR 행/셀을 강조. `higher_is_better` 를 이미 정의하므로 거의 공짜. 표를 눈으로 훑는 시간이 확 준다 | XS | ★★★ |
| 2 | **config.yml 붙여넣기 → 폼 자동 채우기** ✅ | YAML을 파싱해 model/dataset/batch/lr/optimizer/경로를 자동 추출. 등록 노동이 대부분 사라진다. → `log_parser.py` + 학습 로그의 최근 검증 지표·소요 시간까지 확장 구현(§11) | S | ★★★ |
| 3 | **Run 비교 뷰 (2~3개 선택)** ✅ | 지표 나란히 + config diff. 실험 관리에서 가장 자주 하는 행동인데 지금은 불가능. → `widgets/compare_dialog.py` (§11) | M | ★★★ |
| 4 | **경로 존재 여부 배지** | 결과 폴더가 아직 마운트 안 됐는지 / 서버에서 지워졌는지 표시. 오래된 아카이브일수록 값어치가 커진다 | S | ★★☆ |
| 5 | **DB 자동 백업 (종료 시 N개 순환)** | SQLite 단일 파일이라 실수 한 번에 전부 날아간다. 보험 | XS | ★★★ |
| 6 | **전역 검색 `Ctrl+K`** | Task/Work/run을 가로질러 모델명·경로로 즉시 점프 | S | ★★☆ |
| 7 | **★ 즐겨찾기 + 태그 + 실패 사유 필드** | 실패 실험도 자산이다. "OOM이었나?"를 6개월 뒤에 답할 수 있게 | S | ★★☆ |
| 8 | **컬럼 프리셋 (간단히/전체/논문용)** | Task별 컬럼이 늘어나면 반드시 필요해진다 | S | ★★☆ |
| 9 | **결과 폴더 드래그&드롭 등록** | 탐색기에서 폴더를 창에 떨구면 경로 자동 입력 + 로그 탐지 | S | ★☆☆ |
| 10 | **마크다운/HTML 리포트 내보내기** | 선택한 run들을 표로 뽑아 슬랙·문서에 바로 붙여넣기 | S | ★☆☆ |
| 11 | **로그 tail 뷰어** | `result_path` 안의 `train.log` 마지막 N줄 표시 | M | ★☆☆ |
| 12 | **SSH `nvidia-smi` 실시간 폴링** | 서버 GPU 실사용을 자동 반영해 수동 입력이 사라진다. 다만 자격증명·네트워크 의존이 생겨 앱 성격이 "아카이버"에서 "모니터"로 바뀐다 | L | 보류 권장 |

**제가 고른다면 1, 2, 5** 를 먼저 넣겠습니다. 비용 대비 체감이 가장 크고 다른 작업과 충돌하지 않습니다.
3번은 시각화 패널(R8)과 설계를 같이 하는 편이 낫습니다.

---

## 8. 결정 사항 (기본값으로 진행, 바꾸고 싶으면 알려 주세요)

1. **기능별 분할 + UI 쓰기**로 갔습니다. 처음엔 단일 파일이었으나 너무 길어져
   `servers.yaml` / `defaults.yaml` / `tasks/<이름>.yaml` 로 나눴습니다.
   구버전 단일 파일은 첫 실행 때 자동으로 분할됩니다.
   (대안이던 "기본값 파일 + 사용자 오버레이" 2단 구조는 "내 설정이 어디 있지?"를 만들어서 채택하지 않았습니다.)
2. **ruamel.yaml 은 선택 의존성**으로 두었습니다. 설치돼 있으면 주석을 보존하며 저장하고,
   없으면 PyYAML 로 동작합니다(이때 UI 저장 시 주석 소실). 하드 의존성은 만들지 않았습니다.
3. **폰트는 동봉하지 않았습니다.** 시스템 설치분을 스택 순서로 찾아 씁니다.
   `assets/fonts/` 에 ttf/otf 를 넣으면 자동 등록되므로, 원하시면 Pretendard 를 넣기만 하면 됩니다.
4. **다크가 기본이고 라이트도 유지**합니다 (`--theme light`). 토큰만 갈아끼우므로 위젯 코드는 동일합니다.
5. **§7 제안은 아직 하나도 구현하지 않았습니다.** 채택할 번호를 알려 주세요.

---

## 9. 작업 순서

의존성을 고려한 순서입니다. 각 단계 끝에서 앱이 항상 동작하도록 나눴습니다.

| Phase | 내용 | 상태 |
|---|---|---|
| **P0** | DB 마이그레이션 프레임 + `config_store.py` + 테마 토큰 골격 | ✅ |
| **P1** | 다크 테마 + 폰트 적용, 하드코딩 색 제거 | ✅ |
| **P2** | `options.yaml` 연동 콤보박스 + 드롭다운 `＋` / 우클릭 / F2 | ✅ |
| **P3** | Task별 컬럼 + 커스텀 필드(`extra_json`) + 헤더 우클릭 편집 | ✅ |
| **P4** | GPU 인지형 서버 상태 패널 + GPU 선택 폼 | ✅ |
| **P5** | 우클릭/F2 편집을 전 표면에 일관 적용 (`editing.py`) | ✅ |
| **P6** | 채택된 §7 항목 | 대기 |

### 구현하며 바뀐 것

- **`_is_task_scoped`** — 옵션 이름 변경/삭제가 Task 전용 목록과 `defaults` 중 어디에 적용될지
  판단하는 규칙이 필요했습니다. Task 가 그 필드를 직접 정의하고 있으면 Task 범위, 아니면 defaults 범위입니다.
- **Task 전용 목록의 첫 편집** — `defaults` 를 상속하던 필드를 Task 에서 처음 고치면,
  보이던 값들을 그대로 복사한 뒤 편집합니다. 그러지 않으면 항목 하나 추가했다가 나머지가 사라집니다.
- **설정에 없는 지표도 숨기지 않음** — 기록에 값이 있는데 `metrics` 정의에 없는 지표는
  컬럼 끝에 덧붙입니다. 설정을 고쳤다고 이미 쌓인 데이터가 안 보이면 안 됩니다.
- **`distinct_values` 에 Task 범위** — 기록에 남은 옛 값을 콤보에 합칠 때 범위를 걸지 않으면
  Classification 목록에 SR 모델이 섞여 들어옵니다.
- **QSS 토큰 검증** — 템플릿에 정의되지 않은 토큰이 있으면 `render_qss` 가 예외를 냅니다.
  색 오타가 조용히 무시되면 나중에 찾기 어렵습니다.
- **설정 파일 기능별 분할** — 단일 파일이 286줄이 되어 못 보겠다는 지적을 받고 나눴습니다.
  값의 출처 파일을 추적해 쓰기를 되돌리는 구조가 필요했습니다.
- **깨진 정의를 내장 정의로 덮지 않음** — 정의가 있었는데 전부 깨진 경우까지 기본값을 채우면
  사용자의 SR 을 내장 SR 이 가리고, 다음 UI 편집이 원본 파일을 덮어씁니다.
  정의가 **아예 없을 때만** 기본값을 넣습니다.

각 Phase는 별도 커밋으로 쌓고, PR #1 에 이어집니다.

---

## 10. R8 — 시각화 패널 (일부 구현)

### 10.1 넣을 뷰

1. **지표 비교 막대/산점도** — 선택한 run들의 PSNR/SSIM 비교. Task의 `metrics` 정의를 그대로 축으로. (계획만)
2. **성능-속도 산점도 (Pareto)** — x=Latency, y=PSNR. 배포 모델 고를 때 쓰는 그림. Inference 탭 데이터로 바로 그려집니다. (계획만)
3. **학습 곡선** ✅(축소 구현, §11) — `result_path` 의 로그를 파싱해 iteration별 지표를 하나 골라 라인 차트로.
   `log_parser.py` 가 BasicSR류 로그 포맷을 관대하게 파싱하고, `widgets/curve_chart.py` 가 pyqtgraph 없이
   QPainter로 직접 그립니다. **정규식 플러그인화는 아직입니다** — 지금은 흔한 로그 관례 하나만 지원합니다.
4. **이미지 결과 뷰어** ✅(대폭 축소, §11) — GT/Input/여러 모델 출력을 동기화 비교하는 화면 대신,
   결과 폴더에서 **대표 이미지 한 장만** 자동으로 찾아 보여주는 `widgets/image_viewer.py` 로 범위를 줄여 구현했습니다.
   여러 장 동기화 비교·크롭 확대는 아직 계획만입니다.
5. **Work 요약 대시보드** — 실험 수, 성공/실패 비율, 서버별 GPU 시간 누적. (계획만)

### 10.2 기술 선택

- **pyqtgraph** 권장: Qt 네이티브라 다크 테마가 그대로 먹고, 수천 점도 즉시 반응. 이미지 뷰어(`ImageItem` + ROI)도 같은 라이브러리로 해결됩니다.
- matplotlib은 **내보내기(논문용 PDF/PNG)** 경로에만 씁니다. 두 라이브러리의 역할을 섞지 않습니다.
- 색은 `STYLE_GUIDE` §2.6 시리즈 팔레트를 공유합니다.

### 10.3 지금 설계가 준비해 둘 것

- `metrics` 정의에 `higher_is_better`, `unit`, `digits` 가 이미 있어 축 라벨·정렬 방향을 그대로 씁니다.
- 이미지 비교를 하려면 run마다 **출력 이미지 폴더**를 알아야 합니다 → `result_path` 외에
  `image_dir` 같은 커스텀 필드를 `extra_json` 으로 붙일 수 있게 P3에서 열어 둡니다.
- 학습 곡선용 로그 경로도 같은 방식으로 확장합니다.

---

## 11. 2026-09 세션 — 등록 폼 개편 + 실험 관리 기능 확장

사용자 피드백을 받아 진행한 작업 묶음입니다. 각 항목은 별도 커밋으로 쌓았습니다.

### 11.1 New Run 폼 개편

- **Server** — 자유 입력/추가가 가능한 콤보 대신, 상단 서버 바(`config.servers`)에 등록된 목록에서만
  고르는 `ServerCombo` 로 교체했습니다. 서버 추가/삭제는 상단 서버 바에서만 합니다.
- **GPU** — 슬롯 체크박스 대신 **개수** 스핀박스로 단순화했습니다. `gpu_indices` 컬럼은 이제 개수
  문자열(`"2"`)을 저장하며, 예전 콤마 인덱스 목록(`"0,1"`)도 개수로 읽어 하위 호환됩니다
  (`utils.parse_gpu_count`). 서버 상태 바의 충돌 표시도 "정확한 슬롯 충돌" 대신 "개수 초과"로 바뀌었습니다.
- **Status** 기본값을 `queued` 로 바꿨습니다.
- 콤보박스 드롭다운 화살표가 플랫폼에 따라 안 보이던 문제를 QSS 에서 강조색 삼각형으로 직접 그려 고쳤습니다.
- **평가 지표를 Task 안에서 공유**합니다 — 어느 Run 에서든 새 지표 값을 입력해 저장하면 그 Task 의 지표로
  자동 등록되고, 다음 New Run 부터 값 빈 상태로 미리 채워집니다(`MetricsEditor.set_value`).
- **Inference 폼**은 Server/GPU 를 없애고, 같은 Work 의 Train 실행 중 하나(**Source Train Run**)와
  **Model Epoch / Checkpoint** 를 고르는 형태로 바꿨습니다. Source Train Run 을 고르면 Model 이 자동
  채워집니다. `inference_runs` 에 `source_train_run_id`, `checkpoint_epoch` 컬럼을 추가했습니다(스키마 v4).

### 11.2 Run 히스토리

`run_history` 테이블(run_kind, run_id, action, detail, created_at)에 생성/수정(필드별 변경 diff)/복제를
기록합니다. 상세 패널의 새 **History** 탭에서 시각·내용을 확인합니다. Paths 와 Command/config.yml/Metrics/
History 탭은 이제 **행을 선택했을 때만** 나타납니다(선택 전엔 보여줄 내용이 없으므로).

### 11.3 자동 로깅 — train.py 결과에서 폼 채우기

`log_parser.py` 가 두 가지를 관대하게 파싱합니다.

- `parse_train_config(path)` — config.yaml(BasicSR류 스키마를 우선 시도: `network_g.type`,
  `datasets.train.name/dataroot_gt/batch_size_per_gpu`, `train.total_iter`, `train.optim_g.type/lr`, `scale`)
  에서 Model/Dataset/Dataset Path/Batch/LR/Optimizer/Epoch 를 추출합니다.
- `parse_loss_log(path)` — `iter: N` 학습 로그 줄과 `# key: value` 검증 로그 줄(BasicSR 관례)에서
  iteration별 지표(학습 곡선용)와 가장 최근 검증 지표, 첫/마지막 타임스탬프로 추정한 소요 시간을 뽑습니다.

형식을 못 알아봐도 예외 없이 빈 결과만 돌려주며, 채운 값은 저장 전에 폼에서 그대로 확인·수정할 수 있습니다.
폼의 **⇪ Parse** 버튼, 또는 결과 폴더를 Result Folder Path 에 드래그&드롭하면 자동으로 실행됩니다.

### 11.4 대표 이미지 뷰어 / Run 비교 / 학습 곡선

§7-#3, §10.1-#3/#4 참고 — `widgets/image_viewer.py`(대표 이미지 한 장), `widgets/compare_dialog.py`
(2~3개 Run 비교), `widgets/curve_chart.py`(pyqtgraph 없이 QPainter 로 직접 그리는 라인 차트)로 구현했습니다.

### 11.5 Work 별 데이터셋 레지스트리

데이터셋을 **Work 단위**로 등록합니다(Task 단위 옵션 목록과는 별개). `datasets` 테이블
(work_id, name, variant, path, sample_count, notes)에 이름 + 선택적 Variant + 경로 + 총 데이터
개수를 저장합니다. 같은 이름이라도 Variant 를 다르게 두면 "전체 페어"와 "특정 서브셋"을 별개
항목으로 등록할 수 있습니다(예: `DIV2K · Full Pair` / `DIV2K · Subset A`). 좌측 네비게이션(§11.6)에서
인라인으로 관리하고, 등록/수정 폼의 **Dataset 콤보 자체가 이 레지스트리와 바로 연동**됩니다(§11.7).

처음 구현할 때는 기존 Dataset 필드(Task 옵션 자유 입력)를 그대로 두고 그 옆에 별도 "Registered
Dataset" 로더 콤보를 추가하는 additive 방식으로 갔습니다. 그런데 사용자가 "New Run 에서 Dataset
선택이 Work 데이터셋과 연동이 안 된다"고 지적했습니다 — 메인 Dataset 필드를 보고 있었는데 그건
여전히 옛날 방식(Task 옵션)이었고, 새로 만든 로더는 Paths 섹션 아래 별도 행이라 눈에 안 띄었던
겁니다. §11.7 에서 두 콤보를 하나로 합쳤습니다.

### 11.6 좌측 네비게이션 드릴다운 개편 (Option A)

`QTreeWidget` 기반 Task ▸ Work 트리를, 한 번에 한 단계만 보여주는 **드릴다운 + 브레드크럼**으로
바꿨습니다(`nav_panel.py` 전면 재작성). `All Tasks ▸ Task ▸ Work` 브레드크럼으로 언제든 위 단계로
돌아가고, Work 까지 들어가면 그 Work 의 Dataset(§11.5)이 그 자리에 바로 나와 추가/수정/삭제할 수
있습니다. 좌클릭 = 드릴다운, 우클릭 = 이름변경/삭제 메뉴, F2/Del/Ins 는 현재 단계의 "활성 대상"에
적용됩니다. 세부 구현 노트:

- **공개 API 는 그대로 유지**했습니다(`selectionChanged`, `current_task_id`/`current_work_id`,
  `refresh(select_work_id=, select_task_id=)`, `add_task`/`add_work`) — `main_window.py` 는 무변경입니다.
- **`refresh()` 인자 없이 호출될 때(매 저장마다)는 현재 드릴다운 위치를 그대로 유지**합니다. 트리였을 때는
  선택이 항상 남아 있어 문제가 안 됐지만, 드릴다운 구조에서는 "저장할 때마다 처음 화면으로 튕겨나가는" 버그가
  되기 쉬워 `_initialized` 플래그로 "최초 진입"과 "사용자가 위 단계로 나간 상태"를 구분했습니다.
- **최초 진입 시엔 실제 Run 이 있는 Task 로 자동으로 들어갑니다**(예전 트리의 `_first_selectable` 과 동일한
  규칙). 그냥 알파벳순 첫 Task 로 들어가면 신규 설치 시 항상 비어 있는 Task(예: Classification)에서
  시작하는 문제가 있었습니다.

레이아웃 시안은 사용자에게 A(드릴다운) / B(트리 유지 + 요약 줄) / C(평평한 아코디언) 3가지를 스크린샷으로
제시했고, A 로 확정했습니다. B/C 는 채택하지 않았지만 나중에 다시 참고할 수 있도록 시안만 남겨 둡니다.

### 11.7 후속 수정 — Dataset 콤보를 레지스트리와 직접 연동, 총 데이터 개수 추가

§11.5 의 "Registered Dataset" 로더 콤보를 없애고, **New Run 폼의 메인 Dataset 필드 자체**를
`widgets/dataset_dialog.py::DatasetCombo` 로 교체했습니다. `ManagedCombo` 와 같은 조작감(드롭다운
맨 아래 `＋ 새 데이터셋 추가…`, 항목 우클릭/F2/Del 로 수정·삭제)을 그대로 이 Work 의 데이터셋
레지스트리에 대해 제공합니다. 옆의 📦 버튼은 여러 건을 한눈에 보고 고치는 `DatasetManagerDialog`
(표 형태)를 그대로 열어 줍니다 - 인라인 콤보와 표 다이얼로그를 상호 보완적으로 남겨 뒀습니다.

또한 `datasets` 테이블에 **`sample_count`(총 데이터 개수)** 컬럼을 추가했습니다(스키마 v5). 값이
없으면 "0개"와 구분하기 위해 `NULL` 로 둡니다(`DatasetEditDialog` 의 스핀박스는 0 을
"(unspecified)" 로 표시). 좌측 네비게이션의 인라인 Dataset 행과 `DatasetManagerDialog` 의 표에도
개수를 함께 보여줍니다.

---

## 12. 2026-09 세션 (이어서) — Dataset 이미지 정보, New Run 폼 좌우 분할, PySide6 크래시 수정

### 12.1 Dataset 에 이미지 크기 / 확장자

`datasets` 테이블에 `image_size`(예: `256x256`), `extension`(예: `tiff`) 컬럼을 추가했습니다
(스키마 v6). `DatasetEditDialog`/`DatasetManagerDialog`/네비게이션 인라인 행에 모두 반영했고,
Inference 폼에서 Dataset 을 고르면 **Input size 가 이 값으로 자동 채워집니다**(직접 수정도 가능).

### 12.2 New Run 폼을 좌(실행 설정) / 우(경로 + 실행 코드) 로 분할

`BaseRunPanel._build_form_area()` 를 `QSplitter` 기반 2단 구성으로 다시 짰습니다. 각 사이드가
독립된 스크롤 영역(`QFormLayout` 두 개, `self.form_layout`/`self.right_form_layout`)이고,
Save/Clear/Cancel 버튼은 스크롤 밖으로 빼서 폼을 아무리 내려도 항상 보이게 했습니다.

Train 은 요청받은 순서가 기존 순서와 그대로 맞아떨어져 필드 재배치가 필요 없었습니다. Inference 는
순서가 크게 달라(같은 Work 의 Train Run 을 가장 먼저 고르는 흐름) `_build_left_fields` 를 통째로
오버라이드했습니다 — 공용 위젯 생성 로직(`_make_work_combo`/`_make_server_combo`/
`_make_dataset_row`/`_make_status_combo`/`_make_started_row`/`_make_duration_edit`)을 따로
빼 둬서, 순서를 재배치하는 서브클래스도 위젯 생성 코드를 중복하지 않게 했습니다. 요청받은 9개
필드(Work ID/Train Run/Epoch·Iter/Dataset/Status/Server/Latency/Input size/Started At) 사이에,
명시되진 않았지만 여전히 필요한 필드(Checkpoint Path·Model 은 Train Run 근처에, Device·Throughput
은 Latency 근처에)를 의미상 가장 가까운 이웃 옆에 끼워 넣었습니다 — 명시된 필드들의 상대 순서는
그대로 두고요.

`SHOW_SERVER_GPU` 하나였던 클래스 플래그를 `SHOW_SERVER`/`SHOW_GPU` 로 나눴습니다. Inference 는
Server 는 다시 보여주되(“어느 서버에서 돌렸는지”는 여전히 유용) GPU 는 계속 뺍니다(Train Run +
Epoch/Iter 가 이미 실행을 특정하므로).

### 12.3 PySide6 에서 New Run 팝업이 바로 죽던 문제

지난 세션에서 넣은 `editing.disable_wheel_scrolling()` 이 `root.findChildren((QComboBox,
QAbstractSpinBox))` 처럼 타입을 튜플로 넘겼는데, **PyQt6 는 튜플을 받지만 PySide6 는 한 번에 타입
하나만 받습니다** — PySide6 사용자는 New Run 팝업을 여는 순간 `TypeError` 로 죽었습니다. 타입별로
`findChildren` 을 따로 호출하도록 고쳤습니다. 실제로 PySide6 를 설치해 재현 후 수정을 검증했습니다.

**교훈**: `tests/test_widgets.py` 와 `tests/test_columns_and_migration.py` 는
`pytest.importorskip("PyQt6.QtWidgets", ...)` 로 PyQt6 를 하드코딩하고 있어서, PySide6 전용
환경에서는 이 두 파일의 테스트가 전부 **건너뛰어질 뿐 실패하지 않습니다** — 이번 버그가 테스트를
통과하고도 사용자 환경에서 터진 이유입니다. `dl_exp_manager.qt` 를 통해 바인딩을 알아내 그걸로
`importorskip` 하도록 바꾸면 두 바인딩 모두에서 실제로 돌려볼 수 있는데, 아직 안 했습니다(§7 후속
후보).

### 12.4 `config/servers.yaml` 을 git 에서 뺌 — 템플릿만 추적

`servers.yaml` 에는 실서버 IP/구성이 들어가는데 이제까지 저장소에 그대로 커밋돼 있었습니다.
`config/servers.template.yaml` (BUILTIN 과 동일한 placeholder 서버 4개)을 새로 만들어 git 에
추적시키고, 실제 `config/servers.yaml` 은 `.gitignore` 에 추가한 뒤 `git rm --cached` 로 추적만
해제했습니다(로컬 파일은 그대로 둠 — 이 세션 중 사용자가 UI 로 등록해 둔 "DGX" 서버 항목까지
같이 지우면 작업 내용을 잃으므로).

`config_store.py` 의 런타임 동작은 사실 거의 안 건드려도 됐습니다 — `servers.yaml` 이 없을 때
BUILTIN 값을 메모리에만 채우고 **디스크에 다시 쓰지는 않는** 경로가 이미 있었습니다
(`load()` 의 "정상 로드" 분기, `options.yaml` 자체는 있고 `servers.yaml` 만 없는 경우). 이 저장소는
`config/options.yaml` 을 계속 추적하므로 fresh clone 은 항상 이 분기를 타 — 클론 직후 실행해도
`servers.yaml` 이 자동 생성되지 않고, placeholder 서버 4개로 뜨면서 상태바에 템플릿을 복사하라는
안내만 표시됩니다. (반대로 `options.yaml` 자체가 아예 없는 완전 백지 상태의 "첫 실행" 경로는
그대로 뒀습니다 — 이건 저장소 관리가 아니라 `config/` 폴더 자체를 통째로 잃어버렸을 때의 복구
경로라 성격이 다르고, `tests/test_config_store.py::make_config()` 를 비롯한 다수 테스트가 이
경로에서 바로 쓸 수 있는 `servers.yaml` 이 생긴다고 가정하고 있어 건드리면 그 테스트들이 전부
깨집니다.)

README 설치 안내에 `cp config/servers.template.yaml config/servers.yaml` 단계를 추가했습니다.

### 12.5 Train New Run 에 Crop size 추가

`train_runs` 에 `crop_size` 컬럼을 추가했습니다(스키마 v7). Training Hyperparameters 섹션에서
Batch size 바로 아래에 넣었습니다(둘 다 같은 데이터 파이프라인 설정이라 나란히 두는 게 자연스러움).
`log_parser.parse_train_config` 도 BasicSR 류 config.yaml 의 `datasets.train.gt_size` /
`crop_size` / `patch_size` 를 관대하게 시도해 자동으로 채웁니다. 값 형식은 자유 텍스트라
`256x256` 처럼 정사각형이든 `256`처럼 한 변만 적든 그대로 저장됩니다(다른 하이퍼파라미터
필드들과 동일한 관례).

### 12.6 Dataset 에 등록일 표시 + 폴더 아이콘으로 경로 바로 열기

`datasets` 테이블에는 이미 `created_at` 이 있었지만 화면 어디에도 보여주지 않고 있었습니다 -
스키마 변경 없이 `DatasetManagerDialog` 표에 "Registered" 컬럼(날짜만, `YYYY-MM-DD`)을,
좌측 네비게이션의 인라인 Dataset 행에는 `image_size · extension · registered 2026-09-04`
형태로 메타 줄에 붙였습니다.

인라인 Dataset 행의 ✎(수정)/🗑(삭제) 아이콘 왼쪽에 📁 버튼을 추가해 그 데이터셋의 `path` 를
바로 OS 파일 탐색기로 열 수 있게 했습니다 - 이미 있던 `utils.open_in_file_manager` +
`common.toast` 조합을 그대로 재사용했습니다(로그 뷰어/이미지 뷰어의 "Open Folder" 버튼과
같은 패턴). 경로가 비어 있으면 `open_in_file_manager` 가 "Path is empty." 를 그대로 돌려주고
toast 로 보여줘서 별도 예외 처리가 필요 없었습니다.

**교훈**: `toast()` 는 부모 윈도우에 `statusBar()` 가 없으면 `QMessageBox` 로 떨어지는데,
테스트에서 `NavigationPanel` 을 `QMainWindow` 없이 단독으로 띄운 채 이 경로를 그대로 타면
offscreen 환경에서 모달이 뜬 채 `exec()` 가 영원히 안 끝나 테스트 프로세스 전체가 멈춥니다
(실제로 처음 테스트를 그렇게 짰다가 pytest 가 걸려서 중간에 죽여야 했음). 이런 위젯을
단독으로 테스트할 땐 `toast` 자체를 monkeypatch 해서 우회해야 합니다.

### 12.7 후속 수정 — 등록일을 Edit Dataset 에서 직접 고칠 수 있게, 표시에서 "registered" 라벨 제거

§12.6 에서는 `created_at` 을 읽기 전용으로만 보여줬는데, 다른 항목들처럼 고칠 수 있어야
한다는 피드백을 받아 `DatasetEditDialog` 에 `QDateEdit`("Registered:") 을 추가했습니다.
새로 등록할 땐 오늘 날짜가 기본값이고, 기존 데이터셋을 열면 그 `created_at` 날짜로 채워집니다.
`result_values()` 가 8-튜플로 늘어나(`registered_at` 추가), 이걸 쓰는 4곳
(`DatasetManagerDialog._add/_edit_selected`, `DatasetCombo.add_item/_edit_row`,
`NavigationPanel._add_dataset/_edit_dataset`)을 전부 맞춰 고쳤습니다.

`db.add_dataset`/`update_dataset` 에 `created_at: str = ""` 파라미터를 추가했습니다 - 값을
주면 그 날짜로, 안 주면(빈 문자열) add 는 지금 시각을, update 는 **기존 값을 그대로 유지**합니다
(날짜 필드를 안 건드리고 다른 필드만 고치는 기존 호출부가 있어도 등록일이 저절로 튀지 않게).

표시 쪽은 "registered 2026-09-04" 처럼 라벨을 붙였던 걸 날짜만("2026-09-04") 보이도록
줄였습니다 - `DatasetManagerDialog` 표의 "Registered" 컬럼 헤더가 이미 라벨 역할을 하고,
네비게이션 인라인 행은 옆에 image_size/extension 과 나란히 붙어 있어 굳이 문구가 없어도
날짜라는 게 맥락상 읽힙니다.

### 12.8 UI 폰트 우선순위 조정

"UI 폰트를 Noto Sans KR 로 바꿔줘" 요청으로 시작했는데, 이 개발 환경엔 Noto Sans KR 이
설치돼 있지 않아 그대로 1순위로 올려도 다음 순위(Apple SD Gothic Neo)로 그대로 폴백돼
화면상 변화가 없었습니다. 이를 설명하고 "폰트 파일을 assets/fonts/ 에 번들할지 / 시스템에
직접 설치할지" 물었더니 다른 폰트를 추천해 달라고 해서, Pretendard(기존 1순위)·Noto Sans
KR·SUIT 중 고르게 했고 최종적으로 **"Apple SD Gothic Neo 를 1순위로"** 로 정리됐습니다.

`UI_FONT_STACK`(`theme/fonts.py`) 을 `Apple SD Gothic Neo → Malgun Gothic → Noto Sans KR →
Noto Sans CJK KR → Pretendard → Pretendard Variable → SUIT → Inter → Segoe UI Variable →
Segoe UI` 순으로 재정렬했습니다. macOS/Windows 기본 탑재 고딕 폰트를 최우선으로 둬서 별도
설치 없이 항상 의도한 폰트로 뜨고, 그 외 환경(Linux 등)에서는 한글+라틴을 한 벌로 커버하는
폰트들로 순서대로 폴백합니다. 폰트 파일 자체는 (기존 설계 그대로) 번들하지 않았습니다 -
`assets/fonts/` 에 직접 넣으면 그 폰트가 최우선으로 등록되는 기존 메커니즘은 그대로 살아있어,
특정 폰트를 무조건 보장하고 싶으면 그 경로를 쓰면 됩니다.

바로 다음 요청("Pretendard를 디폴트로 바꿔줘")으로 다시 **Pretendard 1순위**로 되돌렸습니다 -
결과적으로 최초 스택과 같아졌지만, macOS/Windows 기본 고딕을 Noto Sans KR 보다 앞에 두는
폴백 순서(§12.8 에서 정리한 순서)는 그대로 유지했습니다.

### 12.9 네비게이션 Dataset 목록에서 경로 텍스트 제거

좌측 네비게이션의 인라인 Dataset 행에 있던 경로 표시 줄(`path_label`, 모노폰트 소문자)을
없앴습니다. §12.6 에서 추가한 📁 버튼이 이미 그 경로를 바로 열어 주므로, 길고 잘리기 쉬운
절대경로 텍스트를 사이드바에 그대로 노출할 필요가 줄었고, 애초에 이 드릴다운 자체가
"심플하게 보여주는" 것이 목표였던 UI 라(§11.6) 정보를 덜어내는 방향과도 맞습니다.
`DatasetManagerDialog` 표의 Path 컬럼은 그대로 뒀습니다 - 그쪽은 여러 데이터셋의 경로를 한눈에
비교/관리하는 용도라 텍스트로 보이는 게 여전히 유용합니다.
