"""자습서 그림(SVG) 공통 도구.

규칙
- 색은 클래스로만 지정 (s-* = 선, f-* = 채움). 앱에서는 테마 색으로 바뀌고,
  앱 밖(GitHub 등)에서는 아래 STYLE의 기본색(밝은 배경용 + 다크 모드)으로 보임
- 글자는 <text class="f-fg"> 처럼 채움 클래스를 씀. 글꼴은 지정하지 않음(앱 글꼴을 따름)
- id 속성은 쓰지 않음 (한 화면에 그림 여러 개가 들어갈 때 충돌)
- 크기는 viewBox로만 정함
"""
from html import escape

STYLE = """<style>
.vdfig{font-family:system-ui,-apple-system,'Apple SD Gothic Neo','Malgun Gothic','Noto Sans KR',sans-serif}
.vdfig .s-fg{stroke:#1c2330}.vdfig .s-mu{stroke:#8a93a3}.vdfig .s-ac{stroke:#2563eb}.vdfig .s-bd{stroke:#c2410c}.vdfig .s-ok{stroke:#15803d}
.vdfig .f-fg{fill:#1c2330}.vdfig .f-mu{fill:#5a6475}.vdfig .f-ac{fill:#2563eb}.vdfig .f-acs{fill:#e6eefe}.vdfig .f-bd{fill:#c2410c}
.vdfig .f-bds{fill:#fdeee6}.vdfig .f-ok{fill:#15803d}.vdfig .f-sf{fill:#f0f2f5}.vdfig .f-bg{fill:#ffffff}
@media (prefers-color-scheme:dark){
.vdfig .s-fg{stroke:#e6e9ef}.vdfig .s-mu{stroke:#737c8c}.vdfig .s-ac{stroke:#6d9bff}.vdfig .s-bd{stroke:#fb8b5b}.vdfig .s-ok{stroke:#4ade80}
.vdfig .f-fg{fill:#e6e9ef}.vdfig .f-mu{fill:#a7afbd}.vdfig .f-ac{fill:#6d9bff}.vdfig .f-acs{fill:#1f2b45}.vdfig .f-bd{fill:#fb8b5b}
.vdfig .f-bds{fill:#3a2218}.vdfig .f-ok{fill:#4ade80}.vdfig .f-sf{fill:#222733}.vdfig .f-bg{fill:#1a1e26}}
</style>"""


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.label, self.parts = w, h, label, []

    def add(self, s):
        self.parts.append(s)
        return self

    def line(self, x1, y1, x2, y2, cls="s-fg", width=1.5, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        return self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, cls="s-fg f-acs", width=1.5, rx=0):
        return self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" stroke-width="{width}"/>')

    def circle(self, cx, cy, r, cls="f-bd"):
        return self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" class="{cls}"/>')

    def path(self, d, cls="s-fg", width=1.5, fill="none"):
        f = "" if "f-" in cls else f' fill="{fill}"'
        return self.add(f'<path d="{d}" class="{cls}" stroke-width="{width}"{f}/>')

    def text(self, x, y, s, cls="f-fg", size=13, anchor="middle", weight=None):
        w = f' font-weight="{weight}"' if weight else ""
        return self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}"{w}>{escape(s)}</text>')

    def save(self, path):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" class="vdfig" viewBox="0 0 {self.w} {self.h}" '
               f'role="img" aria-label="{escape(self.label)}">\n{STYLE}\n{body}\n</svg>\n')
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)
