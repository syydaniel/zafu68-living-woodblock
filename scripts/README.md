# 脚本说明

这里是成片最终流程实际用到的脚本和数据，按文档章节分文件夹。每个文件夹里保留原工程的相对路径：原工程根目录是 `~/Movies/ZAFU2`；竹林引擎来自更早的工程 `~/Movies/ZAFU`，放在 `04_code_shots/zafu_v1/`；共用的声音引擎来自短片《奶蛙·彼岸》的工程，放在 `07_sound/naiwa_engine/`（它也公开在 <https://github.com/syydaniel/naiwa-beyond>）。

## 路径与环境

- 脚本里的个人目录统一改写成 `~`。有些路径写在普通字符串里（例如 `sys.path.insert` 和默认目录），Python 不会自动展开 `~`，自己运行时请换成本机的实际目录。
- 声音环境的解释器写成 `~/venvs/audio/bin/python`（原工程里是一个 Python 3.11 虚拟环境）；画面环境是原工程里的 `.venv_ai`（Python 3.12）。两个环境装了哪些库，见 [第 00 章](../docs/00_制作环境与工具链.md)。
- 校园实拍视频的目录写成 `~/footage/campus`，它们只作构图参考，不随仓库提供。
- 视频、照片、配音、模型权重和渲染结果都不在仓库里，所以脚本不能直接跑出成片；它们完整记录了每一步怎样做、用什么参数，可以对照文档阅读或改造成自己的流程。
- 只改过注释和文档字符串，代码本身与制作时完全一致。

## 00 制作环境与工具链（`00_tools/`）

| 文件 | 用途 |
|---|---|
| `package-lock.json` | Node 依赖的锁定版本 |
| `package.json` | Node 依赖清单（playwright-core 驱动无头 Chrome 渲染 three.js 落叶叠加层） |

## 01 文案、配音与时间表（`01_text_voice/`）

| 文件 | 用途 |
|---|---|
| `vo/audition/zwt2_timing.json` | 成片用的每句切点与长度（u18 为补录第 4 段），mix_final.py 与 edit18 读取 |
| `vo/audition/zwt_timing.json` | 按能量切句后每句在原始录音里的切点与长度 |
| `vo/bday/asr.json` | 6 段结尾补录的 whisper 逐字转写（内容核对） |
| `vo/zwt/zwt11_asr.json` | whisper-large-v3-turbo（MLX）对配音原始录音的转写，含逐字时间 |
| `vo/zwt/zwt_words.json` | 每句的逐字时间（相对切点），edit18 读取 |
| `work/v18/align_chunks.py` | 按配音里的真实停顿测每个短语的开口时间，写 chunks.json |
| `work/v18/carve_time.json` | 校训刻字的开始时间（74.57 秒），供音效设计 |
| `work/v18/chunks.json` | 每句的短语与开口时间（字幕、竖排字、刻字时间的来源） |
| `work/v18/cues.json` | 每句切点在片中的秒数，供 audio/v3/mix_final.py 放置旁白 |

## 02 照片变成版画关键帧（`02_keyframes/`）

| 文件 | 用途 |
|---|---|
| `codex/jobs/v15_test.json` | Codex 作业：东湖图书馆航拍重印成版画，裁切去高楼后作风格锚点 1 |
| `codex/jobs/v16_KA.json` | Codex 作业：图书馆设定图、AI-7 尾帧、AI-9 尾帧、AI-10 首帧及槭树、林荫路画法参考 |
| `codex/jobs/v16_KB.json` | Codex 作业：AI-4 首帧（俯瞰图书馆）、AI-5 首帧（晨雾图书馆） |
| `codex/jobs/v16_KC.json` | Codex 作业：四张植物卡片、鹿鸣岛夏日、校训石空白石面、彩色天目山等静帧关键帧 |
| `codex/jobs/v16_KD.json` | Codex 作业：AI-9 雪景首帧、AI-10 金云尾帧、AI-7 银杏近景首帧、幼苗画法参考 |
| `codex/jobs/v16_KF.json` | Codex 作业：AI-G 首帧：翅果躺在湿土上 |
| `codex/jobs/v16_KG.json` | Codex 作业：AI-G 尾帧：同机位翅果发芽 |
| `codex/jobs/v16_KH.json` | Codex 作业：研究图版：翅果与种端剖面，不带文字 |
| `codex/jobs/v16_T1.json` | Codex 作业：T1 三张：墨版天目山（AI-1 首帧、小星球）、院士林草地（风格锚点 2）、黑天鹅（AI-8 首帧、风格锚点 3） |
| `codex/jobs/v16_T1L.json` | Codex 作业：T1b 尾帧：草坪上挂满翅果的大槭树，不带建筑 |
| `codex/jobs/v17_E.json` | Codex 作业：东湖原址 2000 年与建设中 2001 年两张老照片的忠实重印（E2 首尾帧） |
| `codex/jobs/v17_motto.json` | Codex 作业：校训石“求真敬业”真实书法的描摹 |
| `codex/jobs/v18_K.json` | Codex 作业：指导老师照片的三张重印：今天的东湖校门（保留六个红字）、校园森林、木化石 |
| `codex/run_batch.py` | Codex CLI 批处理：按任务 JSON 调用内置图像生成，输出版画关键帧到 codex/out |
| `libtv_jobs/kf/jobs_v17.json` | 年代关键帧作业：老校门、林化大楼、2004 年图书馆与今天等真实老照片的忠实重印，按年代套色 |
| `libtv_jobs/kf/jobs_v17b.json` | K_S2_last 作业：在 K_S1_last 版画上拆掉竹脚手架成为落成大楼，其余像素不动 |
| `libtv_jobs/kf/run_kf.py` | 同一批处理脚本的副本，用于年代关键帧（读 libtv_jobs/kf/*.json，输出 libtv_jobs/kf/out/） |
| `tools/bing_images.py` | 按关键词检索学校官网、图书馆网站等参考照，保留原文件与 EXIF 拍摄日期，来源写入 index.json |
| `tools/crawl_xyzh.py` | 抓取校友总会网站“母校印记”等四个栏目的老照片，记录标题、日期、来源页、SHA1 和图注到 index.json |
| `tools/depth.py` | Depth Anything V2 Large 相对深度，输出 16 位视差 PNG，作为代码版式的聚类特征和远山留白依据 |
| `tools/extract_footage.sh` | 把作者素材库里用到的航拍、微距视频抽成 JPEG 帧序列（footage/<名字>/），供挑选关键帧依据 |
| `tools/woodblock.py` | 从真实照片生成四色木刻版式（交给 Codex 的 Image 1），并输出印版、三联对比图和边缘重合率 |
| `tools/woodblock2.py` | 全片色板（宣纸、墨、石青、石绿、赭石）与木刻版式工具，被 scenes.py 和 snap.py 导入 |
| `work/v16k/clean_h3.py` | 用 LaMa 局部去掉风格锚点 1（东湖图书馆版画）远处的高层住宅，只贴回遮罩内像素 |
| `work/v16k/jobs.py` | 生成 v16_KA 到 v16_KD 四批 Codex 作业：统一的风格、光向、对位、清理四段提示词和三张风格锚点 |
| `work/v16k/prep.py` | 关键帧依据的准备：按 SRC 表裁成 16:9 的 1920×1080，算深度，再调用 woodblock.py 生成代码版式 |

## 03 AI 视频（`03_ai_video/`）

| 文件 | 用途 |
|---|---|
| `libtv_jobs/clips.json` | 成片用到的 16 段的参数、关键帧、提示词、片中位置与质检结论（由 queue.json 整理，去掉了平台编号） |
| `libtv_jobs/qc.sh` | 单段质检：规格、联系表、首尾帧抽取、首帧 SSIM |
| `libtv_jobs/qc_pair.sh` | 下载一段并对首尾两张关键帧做质检，检查有无音轨 |

## 04 代码镜头、竹林引擎与 three.js（`04_code_shots/`）

| 文件 | 用途 |
|---|---|
| `libtv_jobs/fx/comp.py` | 叠加层质检：把落叶帧合成到 T1b 背景帧上拼成联络表 |
| `libtv_jobs/fx/page/index.html` | three.js 离线渲染页面（透明背景，import map 指向本地 three.module.js） |
| `libtv_jobs/fx/page/main.js` | three.js 离线渲染外壳：按 URL 参数建渲染器，每帧 K 个子帧平均做运动模糊，首尾帧单样本 |
| `libtv_jobs/fx/page/t1b_leaves.js` | 两阵槭叶和翅果掠过镜头的实例化效果（盖住 T1b 的两次季节跳色） |
| `libtv_jobs/fx/post.py` | 原始 RGBA 转直通 alpha 的 PNG，再编码 ProRes 4444（FX_T1b_leaves.mov） |
| `libtv_jobs/fx/render.mjs` | 用 playwright-core 驱动无头 Chrome 逐帧渲染，保存原始 RGBA |
| `work/v16k/planet.py` | 开场小星球基类 Planet（edit18 的 Planet18 继承它，负责 5.6 秒与 AI-1 的直切衔接）；被 edit16 导入 |
| `work/v18/bamboo_look.py` | 竹子镜头的版画调色（渐变映射加墨线边缘） |
| `zafu_v1/camera.py` | Catmull-Rom 相机路径与 uniforms |
| `zafu_v1/edl.py` | v1 镜头表：竹林镜头时段、子帧数、转场、调色关键帧、相机路径 BAMBOO_CAM、竹子生长 uniforms |
| `zafu_v1/edl_v3b.py` | 在 edl.py 上改为 16:9 无黑边、无颗粒，渲染竹林时用 --edl edl_v3b |
| `zafu_v1/engine.py` | v1 GLSL 渲染器（moderngl）：子帧运动模糊与景深、辉光、ACES 调色、16 位读回、ProRes 输出；渲染竹林原片 |
| `zafu_v1/post/post.glsl` | v1 后期通道：子帧累加、转场、辉光、光晕、最终调色（ACES、白平衡、暗角、颗粒、画幅） |
| `zafu_v1/shot_light.py` | v1 光之树镜头；edl.py 初始化时加载（竹林画面不用） |
| `zafu_v1/shot_roots.py` | v1 根系镜头；edl.py 初始化时加载，竹林原片开头 0.48 秒的白光过渡来自它 |
| `zafu_v1/shots.py` | 全屏片元着色器镜头类 FragShot |
| `zafu_v1/shots/bamboo.glsl` | 竹林场景着色器：毛竹林、破土竹笋、竹冠体积光、尘埃、竹叶掠过、薄透镜景深 |
| `zafu_v1/shots/common.glsl` | 着色器共用的噪声、哈希与相机函数 |
| `zafu_v1/shots/herb.glsl` | v1 标本镜头着色器（初始化时编译，竹林画面不用） |
| `zafu_v1/shots/land.glsl` | v1 山水镜头着色器（初始化时编译，竹林画面不用） |
| `zafu_v1/shots/light_bg.glsl` | v1 光之树背景着色器（初始化时编译，竹林画面不用） |
| `zafu_v1/shots/micro.glsl` | v1 显微镜头着色器（初始化时编译，竹林画面不用） |
| `zafu_v1/shots/roots_bg.glsl` | v1 根系镜头背景着色器（初始化时编译） |
| `zafu_v1/textgen.py` | v1 文字纹理生成；edl.py 初始化时导入（渲染时 --notext 关闭文字，画面不含文字） |
| `zafu_v1/timeline.py` | v1 时间表（90 BPM），给出竹林镜头起点 T[bamboo] = 21.333 秒 |

## 05 剪辑与镜头处理（`05_edit/`）

| 文件 | 用途 |
|---|---|
| `work/v14/comp.py` | 文字与合成底层：字体加载、字形栅格化、RiseText 浮现文字、char_times、over() 合成、水墨晕开 T_ink、转场注册表 TRANS |
| `work/v14/edit14.py` | 被 edit16 导入：旧版 film() 提供开场问句的文字对象与 INKC 墨色（edit18.film() 调用链的一部分） |
| `work/v14/paper.py` | 宣纸与墨线工具（Ink 墨线，研究图版的细线标注）；被 comp.py 和 scenes.py 导入 |
| `work/v14/word_times.json` | comp.py 导入时读取的旧逐字时间表（随后被 edit18 用新配音的时间整体替换），保留以便脚本能运行 |
| `work/v16k/edit16.py` | 被 edit17 导入：关键帧查找 kf()、AI 镜头基类 AIShot、逐帧渲染循环 render()，并定义开场竖排问句 |
| `work/v16k/edit17.py` | 字幕原文表 LINES 与字幕样式 InkCaption、长镜头类 Chain（被 edit18 导入，与第 05 章共用） |
| `work/v16k/edit18.py` | 全片镜头表：每句开口时间、镜头起止与取段、转场表、全部屏幕文字（与第 05 章共用，仓库里保留一份即可） |
| `work/v16k/particles3d.py` | 粒子汇聚转场：Depth Anything V2 深度、三维颜料颗粒、汇聚点、宣纸墨晕 Bleed（16.6 秒和 37.8 秒） |
| `work/v16k/scenes.py` | 读图 load()、宣纸 paper()、推拉镜头基类 Parallax（Parallax18 的父类，研究图版的推镜） |
| `work/v18/cadence.json` | 17 段 AI 视频的逐帧位移与锐度，AIShotMap 把每帧放回真实时间的输入 |
| `work/v18/cadence.py` | 测量 AI 视频的 4 帧节奏：逐帧真实位移 steps、锐度 sharp、柔化量 sigma，写入 cadence.json |
| `work/v18/motion.json` | 每段 AI 视频和竹林渲染的逐帧运动量，按运动量变速的输入 |
| `work/v18/motion_profiles.py` | 测量每段 AI 视频每帧的运动量（DIS 光流第 90 百分位）写入 motion.json |
| `work/v18/multiplane.py` | 被 edit18 在导入时加载（类 Multiplane 在成片中没有调用），保留以便 edit18 能原样运行 |

## 06 片尾：校庆生日贺卡（`06_ending/`）

| 文件 | 用途 |
|---|---|
| `codex/jobs/v19_title.json` | Codex 作业：片尾生日贺卡素材：年轮木片、两枝银杏、银杏叶精灵图、金色小枝 |

## 07 声音（`07_sound/`）

| 文件 | 用途 |
|---|---|
| `audio/v3/LIB_SCAN_v19.md` | 音效库 CLAP 加频谱扫描报告：乌鸦、海鸥、狗叫的来源与排除，床、鸟鸣、钟铃、纸、翅膀、whoosh 的干净片段与 D 调变速比 |
| `audio/v3/design_helpers.py` | 音效设计辅助函数：one()、bed()、合成光音 bell()、D 调音高表 NOTE |
| `audio/v3/design_zafu_v21.py` | 音效设计：连续风底、场景环境床、鸟鸣、转场声、片尾、13 处倒放预冲、三处低频托底、8D 三维运动，输出 pre21 的立体声与双耳 FX/AMB |
| `audio/v3/mix_final.py` | 最终混音：人声链（去口水音、去齿音、平衡器、每句 -22 LUFS）、避让、频段让位、人声与背景间距、慢推子、-16 LUFS，输出立体声与双耳母带及 DX/MX/FX/AMB 分轨（成片用 out/v23） |
| `music/mix_zafu.py` | 配乐厅堂混音：声部座位与电平、Opulent Ballroom 音乐厅、指挥式慢推子 ride，输出 mx_stereo.wav |
| `music/mix_zafu_bin.py` | 配乐 3D 双耳混音：听者坐在乐团中间，几处声部绕头转动，输出 mx_bin.wav |
| `music/render_v10.sh` | 配乐一键流程：作曲、采样渲染、厅堂立体声混音、3D 双耳混音 |
| `music/score_lib_v13_head.py` | 作曲库：主题 P1 到 P4 与和声、Cue 速度图、mel/prog/lh/pad/tutti/held/swell 等写法函数（钢琴柱式和弦不滚奏） |
| `music/score_lib_v13_writer.py` | 写 MIDI、notes.json、score.json，并输出空白、速度、低表情进入的检查报告 |
| `music/score_zafu_v10.py` | 程序作曲：D 大调、钉在画面上的速度图、五段配器、三个重音 76.9/77.8/78.85 s、106.0 s 全奏和弦，输出各声部 MIDI |
| `naiwa_engine/audio/v8/clap_tool.py` | CLAP“耳朵”：音频与文字嵌入、按标签打分，用于音效库扫描 |
| `naiwa_engine/audio/v8/eng.py` | 对象式声音引擎：立体声、5.1、双耳布局，真实脉冲响应混响，HRTF 方向增强 BIN_EQ、早期反射、orbit() 绕头路径 |
| `naiwa_engine/audio/v8/lib_index.py` | 给三个本地音效库建 CLAP 索引（build / add），按文字描述检索 |
| `naiwa_engine/audio/v8/mix_v8.py` | 提供 mix_final.py 使用的慢推子 leveler() |
| `naiwa_engine/audio/v8/snd.py` | 按名字取音效库文件、截取、变速移调、循环、前视限幅器 |
| `naiwa_engine/audio/v8/spat/spat.swift` | 离线双耳渲染器源码：AVAudioEnvironmentNode + Apple HRTFHQ，按关键帧移动声源 |
| `naiwa_engine/music/v8/render_score.py` | 逐声部渲染配乐干声：钢琴用 FluidSynth + Salamander，其余用 Sonatina 采样 |
| `naiwa_engine/music/v8/sso.py` | Sonatina Symphonic Orchestra 的小型 SFZ 采样播放器 |

## 08 达芬奇装配、输出与检查（`08_resolve_output/`）

| 文件 | 用途 |
|---|---|
| `work/v16k/build_resolve.py` | 在 DaVinci Resolve 脚本环境中按 plan.json 自动建时间线：V1 画面与交叉叠化、V2/V3 特殊转场与落叶叠加层、V4 文字层、V5 宣纸纹理（正片叠底 35%）、A1 到 A4 四条声音分轨 |
| `work/v16k/export18.py` | 导出代码镜头、重排时间后的 AI 镜头、预渲染转场、透明字幕层和 plan.json，供达芬奇装配 |
| `work/v16k/scan_jumps.py` | 成片跳变扫描：逐帧画面差和颜色跳变超过局部水平的帧 |
| `work/v18/join_sheet.py` | 接缝联络表：每个接点前后 ±0.6 秒 7 帧拼成一行 |
