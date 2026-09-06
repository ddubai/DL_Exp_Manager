"""실행 명령어 생성 - Task 별 템플릿 + 폼 값 -> 실제로 붙여넣을 수 있는 한 줄.

Hydra 처럼 `key=value` 를 나열하는 CLI 를 염두에 둔다::

    python train.py algo=dn/noise2noise data=dn/dataset1 model=dn/UNet +batch_size=16

핵심은 **값이 비면 그 토큰을 통째로 지우는 것**이다. 단순 문자열 치환으로는
배치 크기를 안 적었을 때 `+batch_size=` 같은 깨진 인자가 남아 명령어가 실행되지
않는다. 그래서 템플릿을 공백으로 쪼갠 뒤, 토큰 안의 자리표시자가 하나라도 비어
있으면 그 토큰을 버린다.

템플릿은 `config/tasks/<Task>.yaml` 의 `commands:` 에 있고, 자리표시자 이름은
폼 필드 이름을 그대로 쓴다(`{model}`, `{dataset}`, `{batch_size}`, 사용자 정의
옵션이면 `{algo}` 처럼). 앱이 모르는 이름도 예외 없이 "빈 값"으로 보고 넘어가되,
어떤 이름이었는지는 돌려줘서 호출부가 알려 줄 수 있게 한다.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass
class RenderedCommand:
    text: str = ""
    dropped: list[str] = field(default_factory=list)  # 값이 비어서 빠진 자리표시자 이름
    unknown: list[str] = field(default_factory=list)  # 템플릿에는 있지만 앱이 모르는 이름

    def __bool__(self) -> bool:
        return bool(self.text)


def placeholders_in(template: str) -> list[str]:
    """템플릿이 쓰는 자리표시자 이름을 등장 순서대로(중복 없이) 돌려준다."""
    seen: list[str] = []
    for name in _PLACEHOLDER_RE.findall(template or ""):
        if name not in seen:
            seen.append(name)
    return seen


def render_command(template: str, values: dict[str, object]) -> RenderedCommand:
    """`template` 의 자리표시자를 `values` 로 채운다.

    - 값이 비었거나(`None`/`""`) 앱이 모르는 이름이면 **그 토큰을 통째로 버린다**.
      (`+batch_size={batch_size}` 는 배치를 안 적으면 아예 사라진다)
    - 공백이 든 값은 셸에서 안전하도록 따옴표를 씌운다.
    - 템플릿이 여러 줄이어도(YAML 접힌 블록) 공백 기준으로 한 줄로 합친다.
    """
    result = RenderedCommand()
    if not template or not template.strip():
        return result

    out_tokens: list[str] = []
    for token in template.split():
        names = _PLACEHOLDER_RE.findall(token)
        if not names:
            out_tokens.append(token)
            continue

        rendered = token
        keep = True
        for name in names:
            if name not in values:
                if name not in result.unknown:
                    result.unknown.append(name)
                keep = False
                continue
            text = "" if values[name] is None else str(values[name]).strip()
            if not text:
                if name not in result.dropped:
                    result.dropped.append(name)
                keep = False
                continue
            rendered = rendered.replace("{" + name + "}", shlex.quote(text))
        if keep:
            out_tokens.append(rendered)

    result.text = " ".join(out_tokens)
    return result
