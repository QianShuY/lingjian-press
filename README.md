# 灵鉴 · 冲压检测

级进模检测的 Windows 桌面原型，使用 OpenCV 和 PySide6。可在模拟产线中验证自动触发、位置跟随、孔位检查、疑似杂物标记和异常留图。现场回放可以用开模画面验证送料前沿和手动设置的孔位。

## 运行

在 Windows 上使用 64 位 Python 3.11。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

在「运行」页点「启动模拟产线」。在「模具配方」里可切换内置配方，或用自己的良品图设置孔位和检查区。配方和异常记录保存在本机的 `%LOCALAPPDATA%\LingjianPress`。

## 现场回放

已有现场原型可在项目的 `assets/line-replay.mp4` 放回原回放视频，再打开「现场回放」。当前开模条件、下模基准和默认送料线按原视频取景设置。可点「设置检测位置」重画送料线或选择待检孔。缺少视频时，程序会进入模拟产线。

现场素材和检测记录存放在本地。叠料、跳料等待异常样本，工业相机与 PLC 联机和节拍仍待现场验证。

## 打包

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller==6.21.0
.\.venv\Scripts\python.exe -m PyInstaller LingjianPress.spec
```

程序位于 `dist/LingjianPress`。分发时复制整个文件夹。打包会带上本地已有的回放视频和参考图。需要分发模拟版时，从干净的仓库副本打包。
