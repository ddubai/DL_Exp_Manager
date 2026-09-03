# 개발 계획 (v2)

이 문서는 v2 의 설계안이자 진행 기록입니다.
**R1~R7 은 구현 완료**(§9 참고), **R8(시각화 패널)은 계획만** 있습니다.
**§7 의 추가 제안은 아직 구현하지 않았습니다** — 확인 후 채택할 항목만 진행합니다.

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

`config/options.yaml` (저장소 추적). 실험용 `config.yml`(입력 폼의 그 필드)과
헷갈리지 않도록 이름을 **`options.yaml`** 로 구분합니다.

**단일 파일 + UI 쓰기** 방식을 제안합니다. 손으로 편집한 내용과 UI로 추가한 내용이
한 파일에 모이고, git으로 변경 이력이 남습니다.
→ 주석이 날아가지 않도록 `ruamel.yaml`(round-trip 로더)을 씁니다. PyYAML은 주석을 버립니다.

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
| 좌측 트리 — DL Task | ✅ | ✅ | ✅ | DB `tasks` |
| 좌측 트리 — Work ID | ✅ | ✅ | ✅ | DB `works` |
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
| 2 | **config.yml 붙여넣기 → 폼 자동 채우기** | YAML을 파싱해 model/dataset/batch/lr/optimizer/경로를 자동 추출. 등록 노동이 대부분 사라진다 | S | ★★★ |
| 3 | **Run 비교 뷰 (2~3개 선택)** | 지표 나란히 + config diff. 실험 관리에서 가장 자주 하는 행동인데 지금은 불가능 | M | ★★★ |
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

1. **단일 파일 + UI 쓰기**로 갔습니다. `config/options.yaml` 하나를 손으로도 UI로도 편집합니다.
   (대안이던 2단 오버레이는 "내 설정이 어디 있지?"를 만들어서 채택하지 않았습니다.)
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

각 Phase는 별도 커밋으로 쌓고, PR #1 에 이어집니다.

---

## 10. R8 — 시각화 패널 (계획만)

이번 구현 범위 밖이지만, 위 설계가 이걸 막지 않도록 미리 적어 둡니다.

### 10.1 넣을 뷰

1. **지표 비교 막대/산점도** — 선택한 run들의 PSNR/SSIM 비교. Task의 `metrics` 정의를 그대로 축으로.
2. **성능-속도 산점도 (Pareto)** — x=Latency, y=PSNR. 배포 모델 고를 때 쓰는 그림. Inference 탭 데이터로 바로 그려집니다.
3. **학습 곡선** — `result_path` 의 로그/CSV를 파싱해 iteration별 지표. 로그 포맷이 프로젝트마다 달라 **파서를 플러그인화**해야 합니다(`options.yaml` 에 정규식 지정).
4. **이미지 결과 뷰어** — SR/DN이므로 이게 실제로 제일 자주 볼 화면입니다.
   GT / Input / 여러 모델 출력을 **동기화된 확대·이동 + 좌우 슬라이더**로 비교, 크롭 확대(zoom-in box) 지원.
5. **Work 요약 대시보드** — 실험 수, 성공/실패 비율, 서버별 GPU 시간 누적.

### 10.2 기술 선택

- **pyqtgraph** 권장: Qt 네이티브라 다크 테마가 그대로 먹고, 수천 점도 즉시 반응. 이미지 뷰어(`ImageItem` + ROI)도 같은 라이브러리로 해결됩니다.
- matplotlib은 **내보내기(논문용 PDF/PNG)** 경로에만 씁니다. 두 라이브러리의 역할을 섞지 않습니다.
- 색은 `STYLE_GUIDE` §2.6 시리즈 팔레트를 공유합니다.

### 10.3 지금 설계가 준비해 둘 것

- `metrics` 정의에 `higher_is_better`, `unit`, `digits` 가 이미 있어 축 라벨·정렬 방향을 그대로 씁니다.
- 이미지 비교를 하려면 run마다 **출력 이미지 폴더**를 알아야 합니다 → `result_path` 외에
  `image_dir` 같은 커스텀 필드를 `extra_json` 으로 붙일 수 있게 P3에서 열어 둡니다.
- 학습 곡선용 로그 경로도 같은 방식으로 확장합니다.
