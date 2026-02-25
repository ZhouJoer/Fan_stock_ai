"""Matplotlib 中文字体配置工具。

解决 Windows / Linux / macOS / Docker 下 matplotlib 图表中文显示为方块的问题。
在生成任何含中文的图表前调用 setup_chinese_font() 即可。

跨平台支持：
  - Windows: 从 C:\\Windows\\Fonts 显式加载
  - Linux/Docker: 从 /usr/share/fonts 加载（需在镜像中安装 fonts-wqy-microhei 等）
  - macOS: 使用系统字体或 matplotlib 已缓存字体

若仍乱码，可删除 matplotlib 字体缓存后重试：
  - Windows: %USERPROFILE%\\.matplotlib\\fontlist-*.json
  - Linux: ~/.cache/matplotlib/fontlist-*.json
"""
from __future__ import annotations

import platform
from pathlib import Path

# 各平台显式加载的字体路径：(路径, 字体族名)
# 按优先级排列，找到第一个可用即使用
_FONT_PATHS: list[tuple[Path, str]] = []


def _init_font_paths() -> None:
    """按当前平台初始化可用的字体路径列表。"""
    global _FONT_PATHS
    if _FONT_PATHS:
        return

    if platform.system() == "Windows":
        base = Path("C:/Windows/Fonts")
        _FONT_PATHS = [
            (base / "msyh.ttc", "Microsoft YaHei"),
            (base / "simhei.ttf", "SimHei"),
            (base / "simsun.ttc", "SimSun"),
            (base / "simkai.ttf", "KaiTi"),
        ]
    else:
        # Linux / Docker / macOS：常见字体安装路径（Debian: /usr/share/fonts/truetype/wqy/）
        candidates = [
            # Debian/Ubuntu: fonts-wqy-microhei（主路径）
            (Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"), "WenQuanYi Micro Hei"),
            (Path("/usr/share/fonts/truetype/wqy-microhei/wqy-microhei.ttc"), "WenQuanYi Micro Hei"),
            # Debian/Ubuntu: fonts-noto-cjk
            (Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), "Noto Sans CJK SC"),
            (Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"), "Noto Sans CJK SC"),
            # Fedora/CentOS
            (Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"), "Noto Sans CJK SC"),
            # macOS 系统字体（若通过 X11/Docker 等运行）
            (Path("/System/Library/Fonts/PingFang.ttc"), "PingFang SC"),
            (Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), "Arial Unicode MS"),
        ]
        _FONT_PATHS = [(p, name) for p, name in candidates if p.exists()]


def setup_chinese_font() -> None:
    """配置 matplotlib 使用支持中文的字体。

    跨平台：Windows / Linux / Docker / macOS 均可工作。
    Docker 镜像需安装中文字体，例如：apt-get install fonts-wqy-microhei
    """
    import matplotlib
    matplotlib.use("Agg")  # 确保非交互后端
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # 常见中文字体名（用于回退和优先级）
    font_names = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "KaiTi",
        "FangSong",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "STSong",
        "STKaiti",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    _init_font_paths()
    chosen = None

    # 1. 从显式路径加载（跨平台，不依赖 matplotlib 字体缓存）
    for path, ffamily in _FONT_PATHS:
        if path.exists():
            try:
                path_str = str(path.resolve())
                font_manager.fontManager.addfont(path_str)
                # 使用 addfont 后 FontEntry 中的实际 name（来自字体元数据）
                for fe in font_manager.fontManager.ttflist:
                    if getattr(fe, "fname", "").replace("\\", "/") == path_str.replace("\\", "/"):
                        chosen = fe.name
                        break
                if chosen:
                    break
                chosen = ffamily  # 回退到预设名
                break
            except Exception:
                continue

    # 2. 从 matplotlib 已缓存字体中查找（如本地开发时系统已安装）
    if not chosen:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in font_names:
            if name in available:
                chosen = name
                break

    # 3. Linux: 用 glob 查找 wqy-microhei（不同 distro 路径可能不同）
    if not chosen:
        for fonts_root in [Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]:
            if not fonts_root.exists():
                continue
            try:
                for p in fonts_root.rglob("wqy-microhei*.ttc"):
                    path_str = str(p.resolve())
                    try:
                        font_manager.fontManager.addfont(path_str)
                        for fe in font_manager.fontManager.ttflist:
                            if getattr(fe, "fname", "").replace("\\", "/") == path_str.replace("\\", "/"):
                                chosen = fe.name
                                break
                        if not chosen:
                            chosen = "WenQuanYi Micro Hei"
                    except Exception:
                        continue
                    if chosen:
                        break
                if chosen:
                    break
            except Exception:
                continue

    # 4. 尝试项目内捆绑字体（可选：将字体放在 utils/fonts/ 下）
    if not chosen:
        _this_dir = Path(__file__).resolve().parent
        bundled = [
            (_this_dir / "fonts" / "WenQuanYiMicroHei.ttf", "WenQuanYi Micro Hei"),
            (_this_dir / "fonts" / "wqy-microhei.ttc", "WenQuanYi Micro Hei"),
        ]
        for path, ffamily in bundled:
            if path.exists():
                try:
                    path_str = str(path.resolve())
                    font_manager.fontManager.addfont(path_str)
                    for fe in font_manager.fontManager.ttflist:
                        if getattr(fe, "fname", "").replace("\\", "/") == path_str.replace("\\", "/"):
                            chosen = fe.name
                            break
                    if not chosen:
                        chosen = ffamily
                    break
                except Exception:
                    continue

    if chosen:
        others = [f for f in font_names if f != chosen]
        plt.rcParams["font.sans-serif"] = [chosen] + others
    else:
        plt.rcParams["font.sans-serif"] = font_names

    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号
