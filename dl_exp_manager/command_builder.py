"""실행 명령어 생성 - Task 별 템플릿 + 폼 값 -> 실제로 붙여넣을 수 있는 한 줄.

Hydra 처럼 `key=value` 를 나열하는 CLI 를 염두에 둔다::

    python train.py algo=dn/noise2noise data=dn/dataset1 model=dn/UNet +batch_size=16

템플릿에는 자리표시자가 두 종류 있다.

    {batch_size}   값만            ->  16
    <batch_size>   인자 전체        ->  +batch_size=16

`<...>` 의 모양(`+` 를 붙일지, 이름을 `batch_size` 로 쓸지 `batchsize` 로 쓸지,
`=` 로 붙일지 공백으로 띄울지)은 Task 파일이 아니라 `config/params.yaml` 한 곳에서
정한다 - `ParamStyle` 이 그 파일을 담는 그릇이다.

핵심은 **값이 비면 그 토큰을 통째로 지우는 것**이다. 단순 문자열 치환으로는
배치 크기를 안 적었을 때 `+batch_size=` 같은 깨진 인자가 남아 명령어가 실행되지
않는다. 그래서 템플릿을 공백으로 쪼갠 뒤, 토큰 안의 자리표시자가 하나라도 비어
있으면 그 토큰을 버린다.

템플릿은 `config/tasks/<Task>.yaml` 의 `commands:` 에 있고, 자리표시자 이름은
폼 필드 이름을 그대로 쓴다(`{model}`, `{dataset}`, `<batch_size>`, 사용자 정의
옵션이면 `<algo>` 처럼). 앱이 모르는 이름도 예외 없이 "빈 값"으로 보고 넘어가되,
어떤 이름이었는지는 돌려줘서 호출부가 알려 줄 수 있게 한다.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
# `<name>` - 인자 하나를 통째로. 셸 리다이렉션(`< file`, `2>&1`, `<<EOF`)은
# 여는 꺾쇠 바로 뒤에 식별자 + 닫는 꺾쇠가 오는 이 모양과 겹치지 않는다.
_ARG_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>")


@dataclass
class RenderedCommand:
    text: str = ""
    dropped: list[str] = field(default_factory=list)  # 값이 비어서 빠진 자리표시자 이름
    unknown: list[str] = field(default_factory=list)  # 템플릿에는 있지만 앱이 모르는 이름

    def __bool__(self) -> bool:
        return bool(self.text)


@dataclass
class ParamSpec:
    """파라미터 하나를 명령줄에 어떻게 쓸지."""

    name: str = ""                  # CLI 에서 쓰는 이름 (비면 필드 이름 그대로)
    prefix: str | None = None       # None 이면 ParamStyle 의 공통값
    separator: str | None = None    # None 이면 ParamStyle 의 공통값
    template: str | None = None     # 완전 수동. {name} / {value} 를 쓴다


@dataclass
class ParamStyle:
    """`config/params.yaml` 의 내용. `<name>` 을 실제 인자 문자열로 만든다."""

    prefix: str = "+"
    separator: str = "="
    params: dict[str, ParamSpec] = field(default_factory=dict)

    def spec(self, field_name: str) -> ParamSpec:
        """등록되지 않은 파라미터도 공통 스타일 + 자기 이름으로 동작한다."""
        return self.params.get(field_name) or ParamSpec()

    def cli_name(self, field_name: str) -> str:
        return self.spec(field_name).name or field_name

    def render_arg(self, field_name: str, value: str) -> str:
        """`("epochs", "200")` -> `+max_epoch=200`. 값 인용까지 끝낸 결과를 준다."""
        spec = self.spec(field_name)
        name = spec.name or field_name
        quoted = shlex.quote(value)
        if spec.template:
            return spec.template.replace("{name}", name).replace("{value}", quoted)
        prefix = self.prefix if spec.prefix is None else spec.prefix
        separator = self.separator if spec.separator is None else spec.separator
        return f"{prefix}{name}{separator}{quoted}"

    @classmethod
    def from_dict(cls, raw: Any) -> "ParamStyle":
        """YAML 로 읽은 값에서 만든다. 이상한 값은 조용히 기본값으로 되돌린다."""
        style = cls()
        if not isinstance(raw, dict):
            return style
        common = raw.get("style")
        if isinstance(common, dict):
            if isinstance(common.get("prefix"), str):
                style.prefix = common["prefix"]
            if isinstance(common.get("separator"), str):
                style.separator = common["separator"]
        params = raw.get("params")
        if isinstance(params, dict):
            for key, body in params.items():
                style.params[str(key)] = _spec_from(body)
        return style


def _spec_from(body: Any) -> ParamSpec:
    if body is None:
        return ParamSpec()
    if isinstance(body, str):
        # `epochs: max_epoch` - 이름만 바꾸는 가장 흔한 경우의 지름길
        return ParamSpec(name=body)
    if not isinstance(body, dict):
        return ParamSpec()
    spec = ParamSpec()
    for attr in ("name", "prefix", "separator", "template"):
        value = body.get(attr)
        if isinstance(value, str):
            setattr(spec, attr, value)
    return spec


def placeholders_in(template: str) -> list[str]:
    """템플릿이 쓰는 자리표시자 이름을 등장 순서대로(중복 없이) 돌려준다.

    `{name}` 과 `<name>` 을 모두 센다.
    """
    seen: list[str] = []
    for match in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}|<([A-Za-z_][A-Za-z0-9_]*)>", template or ""):
        name = match.group(1) or match.group(2)
        if name not in seen:
            seen.append(name)
    return seen


def render_command(
    template: str, values: dict[str, object], style: ParamStyle | None = None
) -> RenderedCommand:
    """`template` 의 자리표시자를 `values` 로 채운다.

    - `{name}` 은 값만, `<name>` 은 `params.yaml` 이 정한 인자 전체로 바뀐다.
    - 값이 비었거나(`None`/`""`) 앱이 모르는 이름이면 **그 토큰을 통째로 버린다**.
      (`<batch_size>` 는 배치를 안 적으면 아예 사라진다)
    - 공백이 든 값은 셸에서 안전하도록 따옴표를 씌운다.
    - 템플릿이 여러 줄이어도(YAML 접힌 블록) 공백 기준으로 한 줄로 합친다.
    """
    result = RenderedCommand()
    if not template or not template.strip():
        return result
    style = style or ParamStyle()

    out_tokens: list[str] = []
    for token in template.split():
        names = _PLACEHOLDER_RE.findall(token) + _ARG_RE.findall(token)
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
            rendered = rendered.replace("<" + name + ">", style.render_arg(name, text))
            rendered = rendered.replace("{" + name + "}", shlex.quote(text))
        if keep:
            out_tokens.append(rendered)

    result.text = " ".join(out_tokens)
    return result
