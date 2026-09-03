"""플랫폼 유틸리티 - OS 파일 탐색기 열기, 시간/메트릭 포맷, CSV 내보내기."""
from __future__ import annotations

import csv
import html
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# OS 파일 탐색기 연동
# ---------------------------------------------------------------------------
def open_in_file_manager(path: str, reveal: bool = False) -> tuple[bool, str]:
    """OS 기본 파일 탐색기에서 `path` 를 연다.

    macOS -> Finder(`open`), Windows -> 탐색기(`os.startfile`/`explorer`),
    Linux -> `xdg-open`.

    Args:
        path: 열려는 폴더 또는 파일 경로. (로컬/마운트된 네트워크 경로 모두 가능)
        reveal: True 이면 파일을 '선택된 상태'로 상위 폴더를 연다.
                파일 경로가 아닌 폴더라면 무시된다.

    Returns:
        (성공 여부, 사용자에게 보여줄 메시지)
    """
    if not path or not path.strip():
        return False, "Path is empty."

    target = os.path.expanduser(os.path.expandvars(path.strip()))

    if not os.path.exists(target):
        # 존재하지 않으면 가장 가까운 상위 폴더라도 열어 준다.
        parent = _nearest_existing_parent(target)
        if parent is None:
            return False, f"Path not found:\n{target}"
        ok, msg = _launch_file_manager(parent, reveal=False)
        if ok:
            return False, (
                f"The path does not exist, so its parent folder was opened instead.\n"
                f"Requested: {target}\nOpened: {parent}"
            )
        return False, msg

    if reveal and os.path.isfile(target):
        return _launch_file_manager(target, reveal=True)

    if os.path.isfile(target):
        target = os.path.dirname(target) or target

    return _launch_file_manager(target, reveal=False)


def _nearest_existing_parent(path: str) -> str | None:
    current = os.path.abspath(path)
    while True:
        parent = os.path.dirname(current)
        if parent == current:
            return None
        if os.path.isdir(parent):
            return parent
        current = parent


def _launch_file_manager(target: str, reveal: bool) -> tuple[bool, str]:
    system = platform.system()
    try:
        if system == "Darwin":  # macOS / Finder
            args = ["open", "-R", target] if reveal else ["open", target]
            subprocess.Popen(args)
        elif system == "Windows":  # Windows / 탐색기
            if reveal:
                # explorer 는 select 성공 시에도 exit code 1 을 반환하므로 check 하지 않는다.
                subprocess.Popen(["explorer", f"/select,{os.path.normpath(target)}"])
            else:
                os.startfile(os.path.normpath(target))  # type: ignore[attr-defined]
        else:  # Linux 등 freedesktop
            subprocess.Popen(
                ["xdg-open", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except FileNotFoundError:
        return False, f"Could not find a file manager command on this platform ({system})."
    except OSError as exc:
        return False, f"Could not open folder:\n{target}\n\n{exc}"
    return True, f"Opened: {target}"


def open_terminal_here(path: str) -> tuple[bool, str]:
    """해당 경로에서 터미널을 연다(가능한 플랫폼에 한해)."""
    target = os.path.expanduser(path.strip()) if path else ""
    if not os.path.isdir(target):
        return False, "Not a folder path."
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-a", "Terminal", target])
        elif system == "Windows":
            subprocess.Popen(["cmd", "/c", "start", "cmd", "/K", f"cd /d {target}"], shell=False)
        else:
            for term in ("x-terminal-emulator", "gnome-terminal", "konsole", "xterm"):
                try:
                    subprocess.Popen([term], cwd=target)
                    break
                except FileNotFoundError:
                    continue
            else:
                return False, "Could not find a terminal emulator."
    except OSError as exc:
        return False, str(exc)
    return True, f"Opened terminal: {target}"


# ---------------------------------------------------------------------------
# 결과 폴더 탐지 (드래그&드롭 등록, 로그 tail 뷰어)
# ---------------------------------------------------------------------------
_CONFIG_EXTS = (".yml", ".yaml")


def scan_result_folder(folder: str) -> dict[str, str | None]:
    """폴더 바로 아래(하위 폴더 재귀 없이)에서 config 파일과 로그 파일을 추정해 찾는다.

    학습 결과 폴더는 보통 config.yml / train.log 를 최상위에 두는 관례를 따른다고 가정한다.
    """
    result: dict[str, str | None] = {"config": None, "log": None}
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        return result
    for name in entries:
        full = os.path.join(folder, name)
        if not os.path.isfile(full):
            continue
        lower = name.lower()
        if result["config"] is None and lower.endswith(_CONFIG_EXTS):
            result["config"] = full
        if result["log"] is None and (
            lower.endswith(".log") or ("log" in lower and lower.endswith(".txt"))
        ):
            result["log"] = full
    return result


def tail_file(path: str, max_lines: int = 400, max_bytes: int = 512_000) -> str:
    """파일 끝부분 최대 `max_lines` 줄을 돌려준다.

    큰 로그 파일 전체를 읽지 않도록, 파일 끝에서 `max_bytes` 만 읽어 그 안에서 줄을 센다.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fp:
            if size > max_bytes:
                fp.seek(size - max_bytes)
            data = fp.read()
    except OSError as exc:
        return f"(could not read file: {exc})"
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


# ---------------------------------------------------------------------------
# 시간 / 숫자 포맷
# ---------------------------------------------------------------------------
_DURATION_RE = re.compile(
    r"^\s*(?:(?P<d>\d+(?:\.\d+)?)\s*d)?\s*"
    r"(?:(?P<h>\d+(?:\.\d+)?)\s*h)?\s*"
    r"(?:(?P<m>\d+(?:\.\d+)?)\s*m)?\s*"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s)?\s*$",
    re.IGNORECASE,
)


def parse_duration(text: str) -> float | None:
    """'3h 20m', '1d2h', '01:30:00', '5400' 등을 초(second)로 변환한다."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    # HH:MM:SS / MM:SS
    if ":" in raw:
        parts = raw.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        nums = nums[-3:]
        while len(nums) < 3:
            nums.insert(0, 0.0)
        return nums[0] * 3600 + nums[1] * 60 + nums[2]

    # 순수 숫자 -> 초
    try:
        return float(raw)
    except ValueError:
        pass

    match = _DURATION_RE.match(raw)
    if not match or not any(match.groupdict().values()):
        return None
    g = match.groupdict()
    total = 0.0
    total += float(g["d"] or 0) * 86400
    total += float(g["h"] or 0) * 3600
    total += float(g["m"] or 0) * 60
    total += float(g["s"] or 0)
    return total


def format_duration(seconds: float | int | None) -> str:
    """초를 'Xd HH:MM:SS' 형태의 사람이 읽기 쉬운 문자열로."""
    if seconds is None:
        return ""
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return ""
    if total < 0:
        return ""
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def parse_iso(text: str | None) -> datetime | None:
    if not text:
        return None
    raw = str(text).strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def elapsed_since(started_at: str | None) -> float | None:
    start = parse_iso(started_at)
    if start is None:
        return None
    return max(0.0, (datetime.now() - start).total_seconds())


def format_number(value: Any, digits: int = 4) -> str:
    """메트릭 값을 표시용 문자열로. 숫자가 아니면 원본 문자열."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num != num:  # NaN
        return ""
    if abs(num) >= 1e6 or (num != 0 and abs(num) < 1e-4):
        return f"{num:.3e}"
    text = f"{num:.{digits}f}".rstrip("0").rstrip(".")
    return text or "0"


def coerce_number(value: Any) -> Any:
    """가능하면 float 로, 아니면 원본 그대로."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text:
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def parse_gpu_count(text: Any) -> int:
    """GPU 개수를 해석한다.

    새 형식은 단순 개수("2")이고, 옛 형식은 콤마로 구분된 인덱스 목록("0,1")이다.
    콤마가 있으면 인덱스 개수를, 아니면 숫자 그대로를 개수로 본다.
    """
    text = str(text or "").strip()
    if not text:
        return 0
    if "," in text:
        return len([part for part in text.split(",") if part.strip() != ""])
    try:
        return int(float(text))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# 메트릭 JSON
# ---------------------------------------------------------------------------
def loads_metrics(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items()}


def dumps_metrics(metrics: dict[str, Any] | None) -> str:
    clean = {
        str(k).strip(): coerce_number(v)
        for k, v in (metrics or {}).items()
        if str(k).strip() != ""
    }
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def metrics_to_text(metrics: dict[str, Any] | None, sep: str = " · ") -> str:
    if not metrics:
        return ""
    return sep.join(f"{k}={format_number(v)}" for k, v in sorted(metrics.items()))


# ---------------------------------------------------------------------------
# CSV / 클립보드용 직렬화
# ---------------------------------------------------------------------------
def write_csv(path: str, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    """CSV 파일로 저장. Excel 한글 깨짐 방지를 위해 utf-8-sig 사용. 반환값은 기록한 행 수."""
    count = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(list(headers))
        for row in rows:
            writer.writerow(["" if c is None else c for c in row])
            count += 1
    return count


def rows_to_tsv(headers: Sequence[str] | None, rows: Iterable[Sequence[Any]]) -> str:
    """클립보드 붙여넣기(Excel/Sheets)용 TSV 문자열."""
    lines: list[str] = []
    if headers:
        lines.append("\t".join(str(h) for h in headers))
    for row in rows:
        lines.append("\t".join("" if c is None else str(c).replace("\t", " ") for c in row))
    return "\n".join(lines)


def platform_label() -> str:
    return f"{platform.system()} {platform.release()} · Python {sys.version.split()[0]}"


# ---------------------------------------------------------------------------
# 리포트 내보내기 (#10) - Markdown / HTML
# ---------------------------------------------------------------------------
def render_markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def render_markdown_report(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], generated_at: str | None = None
) -> str:
    generated_at = generated_at or now_iso()
    parts = [
        f"# {title}",
        "",
        f"_Generated {generated_at} · {len(rows)} row(s)_",
        "",
        render_markdown_table(headers, rows) if rows else "_(no rows)_",
        "",
    ]
    return "\n".join(parts)


def render_html_report(
    title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]], generated_at: str | None = None
) -> str:
    generated_at = generated_at or now_iso()

    def esc(value: Any) -> str:
        return html.escape("" if value is None else str(value))

    thead = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows
    ) or f'<tr><td colspan="{len(headers)}">(no rows)</td></tr>'

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{esc(title)}</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; margin: 24px; color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; white-space: nowrap; }}
  th {{ background: #f2f2f2; position: sticky; top: 0; }}
  tr:nth-child(even) {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>{esc(title)}</h1>
<div class="meta">Generated {esc(generated_at)} &middot; {len(rows)} row(s)</div>
<table>
<thead><tr>{thead}</tr></thead>
<tbody>{body}</tbody>
</table>
</body>
</html>
"""
