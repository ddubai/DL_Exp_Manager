# 사용자 가이드

이 문서는 **DL Experiment Manager 를 처음 써 보는 사람**을 위한 안내입니다.
"이게 왜 이렇게 동작하나" 같은 설계 이유가 궁금하면 [`README.md`](../README.md)와
[`ROADMAP.md`](ROADMAP.md)를, 색·폰트·간격 같은 디자인 규격은
[`STYLE_GUIDE.md`](STYLE_GUIDE.md)를 보세요. 여기서는 **화면을 어떻게 누르면
되는지**만 다룹니다.

---

## 1. 이 앱이 하는 일

4대의 학습 서버에서 따로따로 돌아가는 실험을, 여러분 컴퓨터의 SQLite 파일
하나에 모아 관리합니다. 계층은 이렇게 세 단계입니다.

```
DL Task            예: Denoising, Super-Resolution
  └─ Work           예: N2N-Base (그 Task 안의 하나의 실험 묶음)
       ├─ Dataset   그 Work 에서 쓰는 데이터셋들(경로 포함)
       ├─ Train     학습 실행 기록
       └─ Evaluation 평가 실행 기록
```

왼쪽 화면에서 Task → Work 순서로 들어가고, 그 안에서 Train/Evaluation 표를 봅니다.

---

## 2. 설치하고 처음 실행하기

```bash
pip install -r requirements.txt

# 실서버 주소를 직접 채워 넣습니다 (이것만 손으로 복사해야 합니다)
cp config/servers.template.yaml config/servers.yaml
# 복사한 파일을 열어 Server 1~4 의 host 주소를 실제 값으로 바꿔 주세요.
# 이 단계를 건너뛰어도 앱은 안 죽습니다 - placeholder 서버 4개로 뜹니다.

python main.py --sample   # 예시 데이터가 채워진 채로 실행 (기능을 눌러보기 좋습니다)
python main.py            # 평소에는 --sample 없이 실행합니다
```

처음 실행하면 `config/` 아래에 필요한 설정 파일들이 자동으로 만들어집니다
(`servers.yaml` 만 예외 - 위에서 직접 복사한 파일을 씁니다). 아무것도 더 안
하고 바로 서버·Task·Work·Dataset 을 등록하면 됩니다.

**둘러볼 데이터가 필요하면** 이미 만들어 둔 예시 DB 를 여세요 (57건의 실험
기록 + 실제로 열리는 로그/이미지 몇 개가 들어 있습니다):

```bash
python main.py --db sample_experiments.db
```

---

## 3. 화면 구성 한눈에 보기

```
┌─────────────────────────────────────────────────────────────────┐
│ Servers: ● Server 1 (2/4)  ○ Server 2 (0/4)  ...            [+] │ ← 서버 상태 바
├───────────────┬─────────────────────────────────────────────────┤
│ All Tasks     │  [Train] [Evaluation]              Task ▸ Work  │ ← 탭 + 지금 위치
│ Denoising     │  ─────────────────────────────────────────────  │
│ · N2N-Base    │  [+ New Run] [검색...] [필터] [★][↻][⤓][⧉]...  │ ← 툴바
│ · RealNoise   │  ┌───┬────────┬────────┬───────┬─────┬──────┐  │
│               │  │ID │ Status │ Server │ Model │ ... │ PSNR │  │ ← 실행 목록
│ [DATASET]     │  ├───┼────────┼────────┼───────┼─────┼──────┤  │
│ DIV2K  📁✏️🗑  │  │...                                        │  │
│ + Add dataset │  └───┴────────┴────────┴───────┴─────┴──────┘  │
│               │  ─────────────────────────────────────────────  │
│               │  선택한 행의 상세 (경로 / 실행 명령어 / config)  │ ← 상세 패널
└───────────────┴─────────────────────────────────────────────────┘
```

- **왼쪽**: Task ▸ Work 드릴다운. Work 안에 들어가면 그 Work 의 **Dataset 목록**이
  바로 그 자리에 나옵니다.
- **가운데 위**: 지금 보고 있는 Work 의 Train 또는 Evaluation 실행 목록(표).
- **가운데 아래**: 표에서 행을 하나 고르면 나타나는 상세 정보.

---

## 4. 5분 튜토리얼 - 처음부터 끝까지 한 번 해보기

### 4-1. 서버 등록

상단 서버 바 오른쪽 끝의 **`+`** 버튼 → 이름/주소/GPU 목록 입력 → 저장.
(`config/servers.yaml` 을 미리 복사해 뒀다면 서버 4개가 이미 보일 겁니다.)

### 4-2. Task 만들기

왼쪽 위 **`+`**(또는 메뉴 `Edit ▸ Add DL Task`, 단축키 `Ctrl+Shift+T`) →
이름 입력 (예: `Denoising`). 이미 4개(Denoising, Super-Resolution, Clustering,
Classification)가 기본으로 들어 있으니, 새 연구 주제가 아니면 이 단계는
건너뛰어도 됩니다.

### 4-3. Work 만들기

Task 를 선택한 상태에서 `+`(또는 `Ctrl+Shift+W`) → Work 이름 입력
(예: `N2N-Base`). Work 는 "이 주제 안에서 지금 진행 중인 하나의 실험 묶음"
정도로 생각하면 됩니다.

### 4-4. Dataset 등록

Work 를 선택하면 왼쪽 화면 아래에 그 Work 의 Dataset 목록이 보입니다.
**`+ Add dataset`** 을 눌러 다음을 채웁니다.

| 필드 | 뜻 | 예시 |
|---|---|---|
| Name | 데이터셋 이름 | `DIV2K` |
| Variant | (선택) 같은 이름이라도 판이 다를 때 | `Full Pair` / `Subset A` |
| Path | 실제 경로 | `/mnt/data/DIV2K/train` |
| Total samples / Image size / Extension | (선택) 참고용 메타정보 | `800` / `256x256` / `png` |
| **Device** | 명령어의 `data=` 자리 **앞부분** | `server1` |
| **Abbreviation** | 명령어의 `data=` 자리 **뒷부분** | `div2k` |

Device/Abbreviation 을 채워 두면, 아래 4-6 에서 만드는 실행 명령어에
`data=server1/div2k` 처럼 자동으로 들어갑니다. **비워 두면 그 자리는
명령어에서 통째로 빠집니다** — 깨진 인자가 남는 대신 조용히 사라집니다.

> 데이터셋은 왼쪽 화면 말고 New Run 등록 폼 안의 **Dataset** 콤보박스에서도
> 바로 추가할 수 있습니다(맨 아래 `+ 새 데이터셋 추가…`).

### 4-5. Train Run 등록

가운데 위 툴바의 **`+ New Run`** 클릭. 왼쪽엔 실행 설정, 오른쪽엔 경로와
실행 명령어를 입력하는 2단 화면이 뜹니다.

1. Server / GPU 개수 / Model / Dataset 을 고릅니다. Dataset 을 고르면 등록해 둔
   경로가 자동으로 채워집니다.
2. Status 는 기본이 `queued` 입니다. 지금 바로 돌릴 거면 `running` 으로
   바꿔도 됩니다.
3. Batch size / Crop size / LR / Optimizer / Epochs 같은 하이퍼파라미터를
   채웁니다(다 채울 필요 없습니다 - 안 채운 항목은 명령어에서 빠집니다).

### 4-6. 실행 명령어 자동 생성

오른쪽 아래 **Execution Command** 칸 위의 **`⚙ Generate`** 버튼을 누르면,
지금까지 채운 폼 값으로 실행 명령어를 만들어 줍니다. 예:

```
python train.py algo=dn/noise2noise data=server1/div2k model=dn/NAFNet +batch_size=16 +max_epoch=200
```

이 형식은 `config/task-defs/<Task>.yaml` 의 `commands:` 에서 정합니다(6절에서
바꾸는 법을 다룹니다). 폼 값을 바꾸면 명령어도 따라 바뀌지만, **명령어 칸을
직접 고치는 순간부터는 더 이상 자동으로 안 바뀝니다** - 손으로 조정한 걸
존중합니다.

**`+ Register Run`** 을 누르면 저장됩니다.

### 4-7. 학습이 끝나면

1. 표에서 그 행을 다시 열어 Status 를 `done` 으로, 지표(PSNR/SSIM 등)를
   입력합니다.
2. 결과 폴더에 `config.yaml`(BasicSR 류)과 로그 파일이 있다면, 등록 폼의
   **`⇪ Parse`** 버튼이 Model/Dataset/하이퍼파라미터/지표/소요시간을 자동으로
   읽어 채워 줍니다(값은 저장 전에 눈으로 확인·수정할 수 있습니다).
3. 상세 패널의 **`📈 Training Curve`** 로 학습 로그를 그래프로, **`🖼 View Image`**
   로 결과 폴더의 대표 이미지를, **`📄 View Log`** 로 로그 파일을 바로 볼 수
   있습니다.

### 4-8. Evaluation 으로 이어가기

Train 표에서 그 실행을 **우클릭 → `▷ Create Evaluation Run from This`**.
Evaluation 탭으로 넘어가면서 같은 모델·데이터셋·서버·config group(예: `algo`)
을 그대로 물려받은 폼이 열리고, 평가 명령어까지 미리 만들어져 있습니다.
Checkpoint 경로/Epoch 만 채우면 됩니다.

### 4-9. 여러 실행 비교하기

표에서 Ctrl(또는 Cmd) 을 누른 채 2~8개 행을 고르고 **`⇄ Compare`**.

- **Metrics / Params** 탭: 값이 다른 칸만 하이라이트된 표.
- **📊 Chart** 탭: 지표별 막대 그래프(지표마다 자기 스케일 안에서 정규화 - PSNR
  과 SSIM 을 같은 축에 억지로 안 넣습니다). 제일 좋은 값의 막대는 테두리로
  강조됩니다.
- **config.yaml Diff** 탭: 2개를 골랐으면 한 장의 unified diff, 3개 이상이면
  Run 별 config 를 각각 탭으로.

---

## 5. 그 외에 알아 두면 좋은 기능들

| 기능 | 위치 | 설명 |
|---|---|---|
| 전역 검색 | `Ctrl+K` | Task/Work/Run 을 이름·경로·메모까지 가로질러 검색 |
| 즐겨찾기 | 상세 패널의 `☆ Favorite` / 표의 별 컬럼 | 툴바의 `★ Favorites` 로 즐겨찾기만 필터 |
| 태그 / 실패 사유 | New Run 폼 | Status 를 `failed` 로 하면 실패 사유 입력란이 나타남 |
| 컬럼 구성 바꾸기 | 표 헤더 우클릭 | 컬럼 추가/제거/이름변경. 위치·폭은 재시작해도 유지됨 |
| 정렬 | 헤더 클릭 | 숫자 컬럼은 문자열이 아니라 실제 크기 순 |
| 복제 | `⎘ Duplicate` | 같은 설정으로 복제(상태는 `queued` 로 리셋) |
| 내보내기 | `⤓ Export` / `File ▸ Export Current Tab to CSV`(`Ctrl+E`) | CSV(엑셀 한글 안전), 선택 행/표 전체 TSV 클립보드 복사, Markdown/HTML 리포트 |
| History | 상세 패널의 History 탭 | 그 실행이 언제 생성/수정/복제됐는지, 뭐가 바뀌었는지 |
| 다크/라이트 | `View ▸ Theme` | 재시작 없이 즉시 전환, 다음 실행에도 유지 |
| DB 백업 | 자동 | 앱을 끌 때마다 `backups/` 에 스냅샷 5개까지 보관 |
| 다른 DB 열기 | `File ▸ Open DB…`(`Ctrl+O`) | 프로젝트마다 다른 `.db` 파일을 쓰고 싶을 때 |
| 예시 데이터 넣기 | `Tools ▸ Insert Sample Data` | 지금 DB 에 예시 실행 기록을 더 채워 넣습니다(이미 데이터가 있어도 동작) |

---

## 6. 설정 파일 살짝 맛보기

콤보박스 선택지, 표 컬럼, 지표, 실행 명령어 템플릿은 전부 `config/` 아래
YAML 파일로 관리됩니다. **UI 에서 바꿔도 되고 파일을 직접 열어 고쳐도
됩니다** - 앱이 즉시 반영합니다.

```
config/
  defaults.yaml          모든 Task 가 같이 쓰는 기본 선택지 (model/dataset/optimizer)
  params.yaml             +batch_size=16 처럼 인자를 어떻게 쓸지 (아래 참고)
  task-defs/
    Denoising.yaml         이 Task 만의 선택지 · 지표 · 표 컬럼 · 명령어 템플릿
    Super-Resolution.yaml
    ...
```

지금 보고 있는 Task 의 설정 파일을 바로 열려면 `Ctrl+Shift+O`
(또는 `Tools ▸ Open Current Task Config File`).

### `task-defs/<Task>.yaml` 의 `commands:`

```yaml
commands:
  train: python train.py algo={task_short}/{algo} data={dataset_device}/{dataset_abbr}
    model={task_short}/{model} <batch_size> <crop_size> <lr> <epochs>
```

- `{이름}` 은 **값만** 채웁니다(예: `{model}` → `NAFNet`).
- `<이름>` 은 **인자 전체**를 채웁니다(예: `<batch_size>` → `+batch_size=16`).
  값이 비어 있으면 그 인자가 통째로 사라집니다.
- `algo` 처럼 Task 마다 다른 선택지를 쓰려면 그 Task 파일의 `options:` 에
  이름만 추가하면 폼에 콤보박스가 자동으로 생깁니다.

### `config/params.yaml` - `<batch_size>` 가 정확히 어떻게 펼쳐지는지

```yaml
style: {prefix: '+', separator: '='}   # 기본값: +batch_size=16
params:
  epochs: {name: max_epoch}             # → +max_epoch=200
  batch_size: {prefix: '--', separator: ' ', name: batch-size}  # → --batch-size 16
```

학습 코드가 `+batch_size` 대신 `+batchsize` 를 쓰거나, Hydra 대신 argparse
스타일(`--batch-size 16`)을 쓴다면 **이 파일 한 곳만** 고치면 모든 Task 의
명령어가 한 번에 바뀝니다.

---

## 7. 문제 해결

**Q. 서버 바에 서버가 4개(Server 1~4, 가짜 주소)만 보여요.**
`config/servers.yaml` 을 아직 안 만드신 겁니다. 2절의 `cp` 명령을 실행하고
실제 주소로 채워 넣으세요.

**Q. `⚙ Generate` 를 눌렀더니 "unknown placeholder" 경고가 떠요.**
그 Task 의 명령어 템플릿이 쓰는 이름(`{algo}` 등)이 `options:` 에 정의돼
있지 않다는 뜻입니다. `Ctrl+Shift+O` 로 그 Task 파일을 열어 `options:` 에
이름을 추가하거나, `commands:` 에서 그 부분을 지우세요.

**Q. VSCode 에서 `config/task-defs/*.yaml` 이 빨간줄 투성이예요.**
VSCode 의 Ansible 확장이 폴더 이름만 보고 오해하는 문제였는데, 폴더 이름을
`task-defs` 로 바꾸고 파일에 `$schema=none` 안전장치도 넣어서 이제는
발생하지 않을 겁니다. 그래도 남아 있으면 창을 한 번 새로고침
(Cmd/Ctrl+Shift+P → "Reload Window")해 보세요.

**Q. 실수로 실행을 지웠어요 / 잘못 편집했어요.**
`backups/` 폴더에 앱을 끌 때마다 만들어 둔 스냅샷이 최대 5개 있습니다.
`File ▸ Open DB…` 로 원하는 백업 파일을 열어 필요한 값을 확인한 뒤,
원래 DB 로 옮기면 됩니다.

**Q. Task 이름을 실수로 다르게 등록해서 설정을 못 찾아요.**
Task 조회는 **이름을 정확히 일치**시켜서 합니다(예: `Super-Resolution` 과
`SuperResolution` 은 다른 이름입니다). 왼쪽에서 Task 를 우클릭(또는 `F2`)해
이름을 `config/task-defs/` 안의 파일 이름과 똑같이 맞추면 바로 연결됩니다.
