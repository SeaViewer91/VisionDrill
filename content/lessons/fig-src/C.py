"""부록 C 보조: 1~61강 원고의 수식($...$, $$...$$)을 모두 뽑아 강별 목록과 기호의 첫 등장 강을 출력함.

C.md는 이 출력을 보고 사람이 손으로 정리함(B.py처럼 통째로 생성하지 않음). 다만 C.md의 "첫 등장 강"은
이 스크립트가 낸 값을 씀. 강 원고를 고친 뒤에는 다시 돌려 첫 등장 강이 바뀌었는지 확인함.
    python3 content/lessons/fig-src/C.py            # 강별 수식 목록 + 기호 첫 등장 표
    python3 content/lessons/fig-src/C.py --list     # 강별 수식 목록만
    python3 content/lessons/fig-src/C.py --find '\\\\sigma\\(' 'IoU'   # 정규식이 수식에 처음 나오는 강
"""
import os, re, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.dirname(HERE)


def lesson_math(path):
    """원고 한 편의 수식을 등장 순서대로 돌려줌. 코드 블록·인라인 코드 안의 것은 뺌."""
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"^---\n.*?\n---\n", "", s, flags=re.S)        # 머리말
    s = re.sub(r"```.*?```", "", s, flags=re.S)                 # 코드 블록
    s = re.sub(r"`[^`\n]*`", "", s)                             # 인라인 코드
    out = []
    for m in re.finditer(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", s, flags=re.S):
        tex = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        out.append(("B" if m.group(1) is not None else "I", " ".join(tex.split())))
    return out


def all_lessons():
    res = {}
    for p in sorted(glob.glob(os.path.join(L, "[0-9][0-9].md"))):
        n = int(os.path.basename(p)[:2])
        if 1 <= n <= 61:
            res[n] = lesson_math(p)
    return res


# 기호 관례 표에 올린 기호: 이름 → 수식 안에서 찾는 정규식
SYMBOLS = {
    "n (표본 수)": r"(?<![a-zA-Z\\])n(?![a-zA-Z])",
    "N": r"(?<![a-zA-Z\\])N(?![a-zA-Z])",
    "x_i": r"x_i|x_\{i\}",
    "\\bar{x}": r"\\bar\{?x",
    "\\hat{y}": r"\\hat\{?y",
    "\\hat{\\beta}": r"\\hat\{?\\beta",
    "\\mu": r"\\mu",
    "\\sigma (기호)": r"\\sigma(?![a-zA-Z])",
    "\\sigma( (함수)": r"\\sigma\s*\(",
    "\\sigma^2": r"\\sigma\^",
    "s (표본 표준편차)": r"(?<![a-zA-Z\\])s(?![a-zA-Z_])",
    "p": r"(?<![a-zA-Z\\])p(?![a-zA-Z])",
    "\\alpha": r"\\alpha",
    "\\beta": r"\\beta",
    "\\lambda": r"\\lambda",
    "\\theta": r"\\theta",
    "\\eta": r"\\eta(?![a-zA-Z])",
    "\\epsilon/\\varepsilon": r"\\(var)?epsilon",
    "\\gamma": r"\\gamma",
    "\\tau": r"\\tau",
    "\\rho": r"\\rho",
    "\\kappa": r"\\kappa",
    "\\log": r"\\log",
    "\\ln": r"\\ln",
    "\\exp/e^": r"\\exp|(?<![a-zA-Z\\])e\^",
    "\\mathbf (굵은 벡터·행렬)": r"\\mathbf|\\boldsymbol",
    "^\\top (전치)": r"\^\{?\\top|\^T(?![a-zA-Z])|\^\{T\}",
    "\\sum": r"\\sum",
    "\\nabla": r"\\nabla",
    "\\partial": r"\\partial",
    "\\lvert/\\vert (절댓값·넓이)": r"\\lvert|\\vert|\|",
    "\\cap/\\cup": r"\\cap|\\cup",
    "IoU": r"IoU|\\text\{IoU\}|\\mathrm\{IoU\}",
    "B, G (박스)": r"(?<![a-zA-Z\\])B(?![a-zA-Z])|(?<![a-zA-Z\\])G(?![a-zA-Z])",
    "TP/FP/FN": r"TP|FP|FN",
    "\\mathrm{softmax}": r"softmax",
    "\\operatorname{E}/\\mathbb{E}": r"\\mathbb\{E\}|\\operatorname\{E\}|E\[|E\(",
    "\\mathrm{Var}": r"Var",
    "R^2": r"R\^2|R\^\{2\}",
    "z": r"(?<![a-zA-Z\\])z(?![a-zA-Z])",
    "t": r"(?<![a-zA-Z\\])t(?![a-zA-Z])",
    "k (커널·겹 수 등)": r"(?<![a-zA-Z\\])k(?![a-zA-Z])",
    "K": r"(?<![a-zA-Z\\])K(?![a-zA-Z])",
    "W (가중치)": r"(?<![a-zA-Z\\])W(?![a-zA-Z])",
    "b (편향)": r"(?<![a-zA-Z\\])b(?![a-zA-Z])",
    "\\odot": r"\\odot",
    "\\otimes": r"\\otimes",
    "do(": r"do\(|\\mathrm\{do\}|\\operatorname\{do\}",
    "\\operatorname*{arg": r"arg\s*\\?m",
}


def first_lesson(regex, data):
    rx = re.compile(regex)
    for n in sorted(data):
        for _, tex in data[n]:
            if rx.search(tex):
                return n, tex
    return None, None


def lessons_with(regex, data):
    rx = re.compile(regex)
    return [n for n in sorted(data) if any(rx.search(t) for _, t in data[n])]


def main():
    data = all_lessons()
    args = sys.argv[1:]
    if args and args[0] == "--find":
        for r in args[1:]:
            n, tex = first_lesson(r, data)
            print(f"{r}\t첫 등장 {n}강\t{tex}\t전체 {lessons_with(r, data)}")
        return
    total = sum(len(v) for v in data.values())
    print(f"# 1~61강 수식 {total}개\n")
    for n in sorted(data):
        print(f"## {n}강 ({len(data[n])}개)")
        seen = set()
        for kind, tex in data[n]:
            if tex in seen:
                continue
            seen.add(tex)
            print(f"  [{kind}] {tex}")
    if args and args[0] == "--list":
        return
    print("\n# 기호 첫 등장 강\n")
    for name, r in SYMBOLS.items():
        n, tex = first_lesson(r, data)
        print(f"{name}\t{n}강\t{tex}\t전체 {lessons_with(r, data)}")


if __name__ == "__main__":
    main()
