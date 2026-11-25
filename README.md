# 浏览器端 AI 工具集 · Transformers.js / WebLLM / 本地 RAG

在浏览器里本地运行 AI 模型，没有后端服务、没有 API Key，输入内容不会离开设备。仓库包含四个可独立打开的页面：

| 页面 | 内容 |
|---|---|
| `index.html` | 文本分类 / 语义相似度 / 零样本分类 / 本地流式对话（Transformers.js + ONNX Runtime Web，WebGPU/WASM） |
| `webllm.html` | WebLLM（MLC）版本的本地流式对话，仅 WebGPU，用于两种引擎的对比 |
| `rag.html` | **本地知识库问答（RAG）**：文档分块 → 本地向量化 → IndexedDB 检索 → Top-K → 流式回答 + 引用来源 |


**技术栈**：Transformers.js v3.8.1 · ONNX Runtime Web · WebGPU / WASM · 原生 HTML/CSS/JS（零构建、单文件页面）

## 功能

| 功能 | 模型 | 说明 |
|---|---|---|
| 情感分析 | `Xenova/bert-base-multilingual-uncased-sentiment` | 中英文多语言，5 档（极差→极好） |
| 语义相似度 | `Xenova/all-MiniLM-L6-v2` | 句向量 + 余弦相似度，支持连续 5 次基准测速 |
| 零样本分类 | `Xenova/nli-deberta-v3-small` | 给候选标签直接分类（英文效果最佳） |
| 本地流式对话 | `onnx-community/Qwen2.5-0.5B-Instruct` | Chat Template + TextStreamer 逐 token 输出，支持停止、首 token 延迟与 token/s 统计 |
| WebLLM 对照实现 | `Qwen2.5-0.5B/1.5B-Instruct`（MLC 格式） | 独立页面 `webllm.html`：WebGPU + WebLLM 引擎的流式对话，用于对比两种前端推理方案 |
| 本地知识库问答（RAG） | `all-MiniLM-L6-v2` 向量 + `Qwen2.5-0.5B-Instruct` 生成 | 独立页面 `rag.html`：文档上传/粘贴 → 分块 → 向量化 → IndexedDB → Top-K 检索 → 流式回答并输出引用 |

## 架构

~~~text
用户输入
  → Tokenizer（WASM，Transformers.js）
  → ONNX Runtime Web（后端选择：WebGPU / WASM）
  → 模型推理（按 dtype 量化：fp16 / q8 / q4 / q4f16）
  → 后处理 / TextStreamer（对话场景逐 token 回调）
  → 页面渲染
~~~

实现要点：

- **三级后端降级**：`WebGPU(fp16/q4f16) → WebGPU(q8/q4) → WASM(q8/q4)`，任一步失败自动回退并在界面提示；
- **模型按需加载**：首次运行对应任务时才下载，之后使用浏览器 Cache Storage（不可用时自动降级为不缓存）；
- **网络容错**：jsDelivr → esm.sh → unpkg 三个 CDN 依次尝试；模型源支持 hf-mirror.com / huggingface.co 切换；
- **可观测性**：模型下载进度、首载耗时、单次推理耗时、连续 5 次均值、对话首 token 延迟与 token/s；
- **流式对话**：基于 Transformers.js `TextStreamer` 的 token 回调渲染，支持中途停止输出与多轮上下文（保留最近 6 条消息）。

## 两个实现与选型对比

本仓库同时放了两套浏览器推理实现，点顶部徽章可在两页之间切换：

| 维度 | Transformers.js 版（index.html） | WebLLM 版（webllm.html） |
|---|---|---|
| 引擎 | Transformers.js v3 + ONNX Runtime Web | WebLLM（MLC LLM） |
| 模型格式 | ONNX（fp16 / q8 / q4） | MLC 编译格式（q4f16 等） |
| 后端 | WebGPU 优先，WASM 回退 | 仅 WebGPU |
| 任务范围 | 分类 / 向量 / 生成，可多任务共存 | 以对话生成为主，WebGPU 生成更快 |
| 首次下载 | 20MB–450MB（按任务） | 350MB–1GB（按模型） |
| 国内网络 | 支持 hf-mirror 镜像 | 依赖 HF / GitHub raw，通常需代理 |
| 典型场景 | 轻量多任务、隐私文本处理 | 追求 WebGPU 生成速度的本地对话 |

结论：Transformers.js 覆盖任务广、有 WASM 兜底、国内网络友好；WebLLM 在 WebGPU 上生成性能更好但只支持 WebGPU、依赖外网下载。实际选型按「设备覆盖率 vs 生成性能 vs 下载成本」权衡。

## 本地知识库问答（RAG）实现

*（页面：`rag.html`）*

1. **文档入库**：支持 .txt / .md / .json / .csv 上传或直接粘贴；按可配置的分块大小与重叠切分；
2. **向量化**：`all-MiniLM-L6-v2`（q8 量化）逐块编码为归一化向量；
3. **持久化**：文档与向量写入浏览器 IndexedDB，刷新/重开页面不丢失；
4. **检索**：问题向量与全部块向量计算余弦相似度，取 Top-K 并按阈值过滤；
5. **生成**：把命中片段编号后拼进提示词，Qwen2.5-0.5B 流式生成，回答中标注 `[1][2]` 并展示可点击的来源片段；
6. **观测**：索引进度、向量总数、检索耗时、生成首 token 延迟与总耗时。

设计取舍：文档与向量不离开本机；不依赖后端分块/embedding 服务；检索用暴力余弦（演示规模足够，后续可换 HNSW/IndexedDB 索引优化）。

## 运行

### 方式 1：本地服务器（推荐）

~~~bash
python -m http.server 8123
# 或：npx http-server . -p 8123 -c-1
~~~

打开 <http://127.0.0.1:8123>（Windows 可双击 `本地预览.cmd`）。

### 方式 2：直接打开 index.html

Chrome / Edge 一般可用；若被 `file://` 安全策略拦住，请改用方式 1。

### 使用建议

1. 浏览器用较新的 Chrome / Edge（WebGPU 需 113+）；
2. 模型源国内优先 `hf-mirror.com`，下载卡住时切换 `huggingface.co`（通常需要代理）；
3. 首次运行会下载模型：相似度约 20–30MB、情感约 100–200MB、零样本约 40–60MB、对话约 300–450MB；
4. 对话模型较大，WebGPU 体验更好；内存紧张时改选 `WASM CPU`。

## 验证状态

| 检查项 | 结果 |
|---|---|
| 内联模块脚本语法（node --check） | ✅ 通过 |
| DOM id 与 JS 引用一致性 | ✅ 全部匹配 |
| CDN 入口可达（3.8.1 × 3） | ✅ HTTP 200 |
| 模型仓库与 ONNX 文件可达 | ✅ HTTP 200（hf-mirror） |
| 浏览器实跑（分类 / 相似度） | ✅ 已在本机 Chrome 跑通 |
| 浏览器实跑（流式对话 / 基准） | ⏳ 待验证 |
| WebLLM 版（webllm.html） | ⏳ 待验证：需 WebGPU + 代理（模型来自 HF / GitHub raw） |
| RAG 版（rag.html） | ⏳ 待验证：需下载向量模型与对话模型 |

## 目录结构

~~~text
transformers-js-demo/
├── index.html   # Transformers.js 版（分类 / 相似度 / 零样本 / 流式对话）
├── webllm.html  # WebLLM 版（WebGPU 流式对话，用于方案对比）
├── rag.html     # 本地知识库问答（RAG：向量检索 + 引用）
├── README.md
├── LICENSE
└── .gitignore
~~~

## 已知限制与后续方向

- 模型下载体积受量化方式影响；对话模型首次加载较慢；
- 移动端内存与算力有限，长上下文容易 OOM；
- WebGPU 在部分设备/浏览器不可用，依赖 WASM 回退；
- 后续可做：Service Worker 预缓存、Web Worker 隔离推理、多模型并发调度、RAG 检索增强示例。

## License

MIT。模型版权归 HuggingFace 上对应模型作者所有，本仓库仅做调用演示。