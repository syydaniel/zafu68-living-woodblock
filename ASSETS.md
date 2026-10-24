# 第三方软件、模型、素材与内容来源

这部片子用到的外部软件、模型、素材库和内容来源。它们都不属于本仓库，各自遵循原作者的许可；本仓库不分发其中任何文件。标为“仅链接”的是付费或不允许再分发的内容。

## 软件与在线服务

| 名称 | 链接 | 许可 | 在片中的用途 |
|---|---|---|---|
| Claude Code（Anthropic） | <https://www.anthropic.com/claude-code> | 商业服务 | AI 编程助手：编写并运行全部脚本，调用下面的工具，逐帧检查 |
| Codex CLI（OpenAI） | <https://github.com/openai/codex> | Apache-2.0；其中的图像生成是 ChatGPT 付费账号的服务 | 批量生成木刻版画关键帧和片尾贺卡素材（第 02、06 章） |
| LibTV | <https://www.liblib.tv/> | 商业服务，按积分计费（仅链接） | 通过它的 MCP 接口调用 Seedance 2.5（第 03 章） |
| Seedance 2.5（字节跳动） | <https://seed.bytedance.com/> | 商业模型（仅链接） | 16 段首帧 / 首尾帧图生视频，720p，24 fps，无声 |
| DaVinci Resolve Studio 21.1 | <https://www.blackmagicdesign.com/products/davinciresolve> | 商业软件（仅链接） | 用脚本接口装配时间线、合成、输出 1080p（第 08 章） |
| FFmpeg 8.0.1（含 libx264） | <https://ffmpeg.org/> | LGPL-2.1+，含 x264 的构建为 GPL-2.0+ | 抽帧、转码、ProRes 编码、8D 版与手机版、响度和画面检查 |
| Node.js 25 | <https://nodejs.org/> | MIT | 运行 three.js 离线渲染脚本 |
| three.js r186 | <https://threejs.org/> | MIT | 树苗长成大树那段的落叶和翅果叠加层（第 04 章） |
| playwright-core 1.64.1 | <https://playwright.dev/> | Apache-2.0 | 驱动无头 Chrome 逐帧渲染 three.js 页面 |
| Google Chrome | <https://www.google.com/chrome/> | 专有免费软件（仅链接） | three.js 页面的 WebGL 渲染 |
| Python 3.12 / 3.11 | <https://www.python.org/> | PSF-2.0 | 画面环境与声音环境的全部脚本 |
| uv | <https://github.com/astral-sh/uv> | Apache-2.0 或 MIT | 管理 Python 环境 |
| Swift 编译器 | <https://www.swift.org/> | Apache-2.0 | 编译空间声渲染程序 spat.swift |

## 本地模型

| 名称 | 链接 | 许可 | 在片中的用途 |
|---|---|---|---|
| Depth Anything V2 Large | <https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf> | CC BY-NC 4.0 | 粒子转场里给画面估计深度（第 05 章） |
| LaMa（经 simple-lama-inpainting） | <https://github.com/advimman/lama>、<https://github.com/enesmsahin/simple-lama-inpainting> | Apache-2.0 | 清理风格参考图里的远处高楼（第 02 章） |
| whisper-large-v3-turbo（MLX 版，mlx-whisper） | <https://huggingface.co/mlx-community/whisper-large-v3-turbo> | MIT | 配音转写与逐句切分（第 01 章） |
| CLAP（laion/larger_clap_general） | <https://huggingface.co/laion/larger_clap_general> | Apache-2.0 | 按文字描述检索和筛选音效库（第 07 章） |

## Python 库

| 名称 | 链接 | 许可 | 用途 |
|---|---|---|---|
| NumPy | <https://numpy.org/> | BSD-3-Clause | 全部画面与声音计算 |
| SciPy | <https://scipy.org/> | BSD-3-Clause | 插值、滤波、信号处理 |
| OpenCV | <https://opencv.org/> | Apache-2.0 | 光流（DIS、LK）、缩放、图像处理 |
| Pillow | <https://python-pillow.org/> | MIT-CMU | 读写图片、排版文字 |
| scikit-image | <https://scikit-image.org/> | BSD-3-Clause | 图像度量与处理 |
| PyTorch | <https://pytorch.org/> | BSD-3-Clause | 运行深度、修补和 CLAP 模型 |
| Hugging Face Transformers | <https://github.com/huggingface/transformers> | Apache-2.0 | 加载 Depth Anything 与 CLAP |
| MLX / mlx-whisper | <https://github.com/ml-explore/mlx> | MIT | 在 Apple 芯片上运行 Whisper |
| moderngl | <https://github.com/moderngl/moderngl> | MIT | 竹林 GLSL 引擎的离线渲染 |
| librosa | <https://librosa.org/> | ISC | 音频分析（基频、MFCC） |
| python-soundfile | <https://github.com/bastibe/python-soundfile> | BSD-3-Clause | 读写音频 |
| pyloudnorm | <https://github.com/csteinmetz1/pyloudnorm> | MIT | 响度测量与归一 |
| pedalboard（Spotify） | <https://github.com/spotify/pedalboard> | GPL-3.0 | 混响、均衡等音频效果 |
| FluidSynth | <https://www.fluidsynth.org/> | LGPL-2.1 | 渲染钢琴声部 |
| requests | <https://requests.readthedocs.io/> | Apache-2.0 | 收集公开网页上的参考照片 |

## 声音素材

| 名称 | 链接 | 许可 | 在片中的用途 |
|---|---|---|---|
| Salamander Grand Piano V3（FreePats SF2 版，原始采样 Alexander Holm） | <https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html> | CC BY 3.0 | 配乐的钢琴 |
| Sonatina Symphonic Orchestra（Mattias Westlund） | <https://github.com/peastman/sso> | Creative Commons Sampling Plus 1.0 | 配乐的管弦乐 |
| Final Cut Pro Sound Effects（Apple） | <https://www.apple.com/final-cut-pro/> | Apple 软件许可（仅链接） | 风、水、鸟鸣、纸、落叶等音效 |
| Fairlight Sound Library（Blackmagic Design） | <https://www.blackmagicdesign.com/support/family/davinci-resolve-and-fusion> | Blackmagic Design 许可（仅链接） | 环境声与音效 |
| Sonniss #GameAudioGDC Bundle 2026 | <https://sonniss.com/gameaudiogdc> | Sonniss 许可：可用于作品，不可原样再分发（仅链接） | 自然环境声、纸张拟音 |
| Apple 脉冲响应（Space Designer） | <https://www.apple.com/logic-pro/> | Apple 软件许可（仅链接） | 配乐与音效的厅堂混响 |
| Apple AVAudioEnvironmentNode（HRTF） | <https://developer.apple.com/documentation/avfaudio/avaudioenvironmentnode> | macOS 系统框架 | 8D 双耳版的空间化 |

## 字体

| 名称 | 链接 | 许可 | 用途 |
|---|---|---|---|
| 思源宋体 Noto Serif SC | <https://fonts.google.com/noto/specimen/Noto+Serif+SC> | SIL OFL 1.1 | 字幕、年份、日期 |

## 画面与文字内容的来源（不随仓库分发）

| 来源 | 链接 | 版权 | 用途 |
|---|---|---|---|
| 浙江农林大学校友总会“母校印记”栏目与“忆母校的门”回忆文章 | <https://xyzh.zafu.edu.cn/> | 学校及原作者 | 历代校门和老楼的老照片，重绘成版画关键帧 |
| 浙江农林大学官网、官方公众号、图书馆网站 | <https://www.zafu.edu.cn/> | 学校 | 东湖校区原址与建设中的照片、校园掠影、图书馆照片 |
| 指导老师陈胜伟提供的照片 | 不公开 | 提供者 | 今天的东湖校门、校园森林、木化石 |
| 校园实拍视频 | 不公开 | 拍摄者 | 只作构图参考，不出现在片中 |
| 浙江农林大学标志与标准字 | <https://www.zafu.edu.cn/> | 学校 | 片尾校名 |
| 宣传片旁白文案 | 见 [第 01 章](docs/01_文案配音与时间表.md) | 学校 | 旁白与字幕 |
| 真人配音录音（配音：周文韬） | 不公开 | 配音员与学校 | 旁白 |
