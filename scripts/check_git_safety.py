#!/usr/bin/env python3
"""
檢查 repo 裡有沒有不該進 git 的東西 —— 包含已經刪掉但還留在歷史裡的。

用法（在 repo 根目錄）：
    py scripts/check_git_safety.py

★ 為什麼需要這支：

  .gitignore 只擋「還沒被追蹤」的檔案。已經 commit 過的檔案不受影響，
  而且 **git rm 之後它還留在歷史裡** —— push 過的話別人 clone 下來
  照樣拿得到。

  Stage II 評審會抽驗程式碼，他們看得到 commit 歷史。
  我們的核心主張是「影像不出床邊」，repo 歷史裡躺著病房畫面的話，
  那句話當場破功。

這支腳本分三層查：

    1. 現在被追蹤的檔案      → git rm --cached 就能處理
    2. 歷史裡出現過的檔案 ★   → 要重寫歷史才清得掉
    3. 有沒有推上遠端 ★★      → 推過的話，重寫歷史還要 force push
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict

# ★ 永遠不算問題的檔名。
#
#   .gitkeep 是刻意放的空檔案，用途是讓 data/clips/ 與 results/
#   這種空目錄能進 git —— 新 clone 下來沒有這些目錄的話，
#   dump_keypoints.py 和 run_demo.py 會因為路徑不存在而失敗。
#
#   第一版把它判成「影像與素材」，那個誤報比漏報更糟：
#   在最高嚴重度上喊狼來了，下次真的有素材進 repo 時沒人會當一回事。
ALWAYS_OK = {".gitkeep", ".gitignore", ".gitattributes", ".keep"}

# 分級：privacy 是會讓提案破功的，其他只是檔案大小/授權問題
RULES = [
    # ★ 順序有意義：這條要排在「影像與素材」之前，
    #   否則 data/clips/* 會先把 .json 吃掉，分級又變回 [影像]。
    # ★ 關鍵點 dump 不等同影像。
    #
    #   第一版把它跟影片放同一級，結果對一個 50 KB、只有座標沒有畫面、
    #   內容還是自己做開合跳的 test.json，建議「重寫歷史 + force push」。
    #   那個代價（四個人重新 clone、繳件前一週動歷史）遠高於收益。
    #
    #   影像進歷史才值得重寫；座標進歷史停止追蹤就夠了 ——
    #   除非它是從真實病人影片 dump 出來的，那要另外判斷。
    ("derived", "關鍵點 dump（有座標，無畫面）", [
        "data/*.json", "data/**/*.json",
    ]),
    ("privacy", "影像與素材", [
        "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm",
        "data/clips/*", "data/raw/*",
        "data/*.png", "data/*.jpg", "data/*.jpeg",
    ]),
    ("size", "模型權重", [
        "*.pt", "*.pth", "*.onnx", "*.hef", "*.engine",
    ]),
    ("noise", "執行產物", [
        "results/*", "*.jsonl",
    ]),
    ("noise", "快取與系統檔", [
        "__pycache__/*", "*.pyc", "Thumbs.db", "desktop.ini", ".DS_Store",
    ]),
]

# ★ 放在 repo 根目錄的資料檔。
#
#   第一版漏掉了這一類：規則只查 data/ 與 results/ 底下，
#   但 dump_keypoints.py 的輸出如果直接寫在根目錄（test.json、diag.png），
#   就繞過了 .gitignore 的所有規則。
#
#   這些沒有畫面所以不是隱私問題，但它們是從素材衍生的，而且放錯位置。
MISPLACED_EXT = {".json", ".png", ".jpg", ".jpeg", ".csv",
                 ".mp4", ".mov", ".avi", ".jsonl"}

# 用文字不用 emoji —— Windows 終端機字型不一定有 emoji，會變成方框
MARK = {"privacy": "[影像]", "derived": "[衍生]", "size": "[大小]",
        "noise": "[雜訊]", "misplaced": "[位置]"}


def git(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "").strip()
    except FileNotFoundError:
        print("找不到 git 指令。確認有裝 git 並且在 PATH 裡。")
        sys.exit(2)


def matches(path: str, patterns: list[str]) -> bool:
    import fnmatch
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        # data/**/*.json 這種要額外處理
        if "**" in pat:
            head = pat.split("**")[0]
            tail = pat.split("**")[-1].lstrip("/")
            if path.startswith(head) and fnmatch.fnmatch(path.rsplit("/", 1)[-1], tail):
                return True
    return False


def classify(paths: set[str]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in sorted(paths):
        name = p.rsplit("/", 1)[-1]
        if name in ALWAYS_OK:
            continue
        for level, label, pats in RULES:
            if matches(p, pats):
                out[(level, label)].append(p)
                break
        else:
            # 根目錄的資料檔：沒有 "/" 代表在 repo 根
            if "/" not in p:
                ext = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
                if ext in MISPLACED_EXT:
                    out[("misplaced", "放在根目錄的資料檔")].append(p)
    return out


def show(title: str, grouped: dict, limit: int = 12) -> bool:
    if not grouped:
        print(f"  ✓ {title}：沒有")
        return False
    print(f"  ✗ {title}：")
    for (level, label), paths in sorted(grouped.items()):
        print(f"      {MARK[level]} {label}（{len(paths)} 個）")
        for p in paths[:limit]:
            print(f"         {p}")
        if len(paths) > limit:
            print(f"         … 另外 {len(paths) - limit} 個")
    return True


def main() -> None:
    print("=" * 64)
    print("  git 安全檢查 — 有沒有不該進 repo 的東西")
    print("=" * 64)

    code, _ = git("rev-parse", "--is-inside-work-tree")
    if code != 0:
        print("\n這個資料夾還不是 git repo（沒有 .git）。")
        print("代表什麼都還沒 commit、也還沒 push —— 現在加 .gitignore 剛剛好。")
        sys.exit(0)

    # ── 基本狀態 ──────────────────────────────────────────
    print("\n1. 基本狀態")
    print("-" * 64)
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    code, count = git("rev-list", "--count", "HEAD")
    n_commits = int(count) if code == 0 and count.isdigit() else 0
    print(f"  分支        {branch}")
    print(f"  commit 數   {n_commits}")

    if n_commits == 0:
        print("\n  還沒有任何 commit。加上 .gitignore 再 commit 就不會有問題。")
        sys.exit(0)

    _, remotes = git("remote", "-v")
    has_remote = bool(remotes)
    print(f"  遠端        {remotes.splitlines()[0] if has_remote else '（沒有設定）'}")

    pushed = False
    if has_remote:
        git("fetch", "--quiet")
        code, upstream = git("rev-parse", "--abbrev-ref", "@{upstream}")
        if code == 0:
            _, ahead = git("rev-list", "--count", f"{upstream}..HEAD")
            _, behind = git("rev-list", "--count", f"HEAD..{upstream}")
            print(f"  追蹤        {upstream}（本地多 {ahead}、遠端多 {behind}）")
            _, remote_commits = git("rev-list", "--count", upstream)
            pushed = remote_commits.isdigit() and int(remote_commits) > 0
        else:
            print("  追蹤        這個分支還沒推上去過")

    print(f"  推送狀態    {'★ 已經推上遠端' if pushed else '尚未推送'}")

    # ── 現在被追蹤的 ──────────────────────────────────────
    print("\n2. 現在被 git 追蹤的檔案")
    print("-" * 64)
    _, tracked = git("ls-files")
    g_tracked = classify(set(tracked.splitlines()))
    bad_tracked = show("不該追蹤的檔案", g_tracked)

    # ── 歷史裡出現過的 ★ ──────────────────────────────────
    print("\n3. 歷史裡出現過的檔案（★ 包含已經刪掉的）")
    print("-" * 64)
    _, hist = git("log", "--all", "--pretty=format:", "--name-only",
                  "--diff-filter=A")
    ever = {l.strip() for l in hist.splitlines() if l.strip()}
    g_hist = classify(ever)
    # 只報「現在沒追蹤、但歷史有」的，現在還在的上面已經列過
    g_hist_only = {}
    for k, paths in g_hist.items():
        only = [p for p in paths if p not in set(tracked.splitlines())]
        if only:
            g_hist_only[k] = only
    bad_hist = show("歷史殘留（git rm 清不掉）", g_hist_only)

    # ── 最大的物件 ────────────────────────────────────────
    print("\n4. 歷史裡最大的幾個檔案")
    print("-" * 64)
    code, out = git("rev-list", "--objects", "--all")
    if code == 0 and out:
        _, catf = subprocess.run(
            ["git", "cat-file", "--batch-check=%(objecttype) %(objectname) "
             "%(objectsize) %(rest)"],
            input=out, capture_output=True, text=True,
            encoding="utf-8", errors="replace").stdout, None
        rows = []
        for line in _.splitlines():
            parts = line.split(maxsplit=3)
            if len(parts) == 4 and parts[0] == "blob":
                rows.append((int(parts[2]), parts[3]))
        rows.sort(reverse=True)
        if rows:
            for size, path in rows[:8]:
                mb = size / 1048576
                flag = "  ←" if mb > 5 else ""
                print(f"  {mb:7.2f} MB  {path}{flag}")
        else:
            print("  （沒有 blob）")

    # ── 結論 ──────────────────────────────────────────────
    print("\n" + "=" * 64)
    if not bad_tracked and not bad_hist:
        print("  ✓ 乾淨。沒有影像、權重或產物進到 repo 或歷史裡。")
        print("=" * 64)
        sys.exit(0)

    print("  需要處理")
    print("=" * 64)

    if bad_tracked:
        # ★ 列出實際找到的路徑，不要寫死檔名 ——
        #   git rm --cached 遇到沒被追蹤的檔名會整條指令失敗。
        paths = [p for ps in g_tracked.values() for p in ps]
        quoted = " ".join(f'"{p}"' if " " in p else p for p in paths)
        print("\n【現在還被追蹤的】移出追蹤，檔案會留在硬碟上：\n")
        print(f"    git rm --cached {quoted}")
        print('    git commit -m "Stop tracking footage, results and weights"\n')

    if bad_hist or bad_tracked:
        privacy_hit = any(k[0] == "privacy"
                          for k in {**g_tracked, **g_hist_only})
        if privacy_hit:
            # filter-repo 用「路徑前綴」比逐檔列乾淨
            # results/ 整個丟掉；data/ 只丟子目錄，data/annotations.md 要留
            prefixes = set()
            for ps in list(g_tracked.values()) + list(g_hist_only.values()):
                for p in ps:
                    if p.startswith("results/"):
                        prefixes.add("results")
                    elif p.startswith("data/") and p.count("/") >= 2:
                        prefixes.add("/".join(p.split("/")[:2]))
                    elif p.startswith("data/"):
                        prefixes.add(p)
            prefixes = sorted(prefixes)
            args = " ".join(f"--path {x}" for x in prefixes) or "--path data/clips"
            print("【[影像] 歷史裡有影像或素材】這是唯一值得重寫歷史的情況。\n")
            print("  git rm 之後檔案還在歷史裡，clone 下來照樣拿得到。要真的清掉：\n")
            print("    pip install git-filter-repo")
            print(f"    git filter-repo {args} --invert-paths\n")
            print("  ⚠️ 這會重寫所有 commit 的 hash。隊友手上的 clone 會對不上，")
            print("     要請他們重新 clone，不要用 git pull。\n")
            if pushed:
                print("""  ⚠️⚠️ 而且這個 repo 已經推上遠端了，重寫歷史之後要：

    git push --force --all

     在那之前，GitHub 上那份還是舊的。如果 repo 是 public，
     先改成 private 再處理。
""")
        elif any(k[0] in ("derived", "size", "noise")
                 for k in {**g_tracked, **g_hist_only}):
            print("""【沒有影像進到 repo】停止追蹤就夠了，不要重寫歷史。

  關鍵點 dump、權重、產物進歷史只是讓 repo 變大，沒有隱私問題。
  force push 要所有人重新 clone，繳件前一週動歷史，代價高於收益。

  ⚠️ 例外：如果某個 dump 是從**真實病人**影片做出來的，
     那就算只有座標也要當影像處理。目前的素材都是模擬場景。
""")

    misplaced = [p for (lvl, _), ps in {**g_tracked, **g_hist_only}.items()
                 if lvl == "misplaced" for p in ps]
    if misplaced:
        print("""【[位置] 根目錄的資料檔】不是隱私問題，但放錯地方。

  放在根目錄會繞過 .gitignore 的所有規則。搬到該在的位置：

    關鍵點 dump（*.json）→ data/clips/
    診斷圖、產出（*.png）→ results/

  搬完之後 .gitignore 就會自動擋住它們。
""")
        for p in misplaced:
            print(f"    {p}")
        print()

    print("分級：[影像] 唯一值得重寫歷史的　[衍生] 有座標無畫面　"
          "[大小] 檔案大小／授權　[位置] 放錯地方　[雜訊] 雜訊")
    sys.exit(1)


if __name__ == "__main__":
    main()