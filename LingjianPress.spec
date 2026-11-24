from pathlib import Path

root = Path(SPECPATH)
samples = [root / "assets" / name for name in ("reference.png", "line-replay.mp4")]
analysis = Analysis(
    [str(root / "main.py")],
    datas=[(str(path), "assets") for path in samples if path.is_file()],
    excludes=["tkinter"],
)

# Qt uses the Windows ICU API. The bundled Poppler tools can put their
# incompatible ICU build on the packaging machine's PATH.
analysis.binaries = [
    item for item in analysis.binaries
    if not (Path(item[0]).name.lower().startswith("icu") and "poppler" in item[1].lower())
]

archive = PYZ(analysis.pure)
program = EXE(
    archive, analysis.scripts, [],
    exclude_binaries=True,
    name="LingjianPress",
    console=False,
    upx=False,
)
app = COLLECT(program, analysis.binaries, analysis.datas, name="LingjianPress", upx=False)
