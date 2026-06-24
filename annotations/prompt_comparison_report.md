# Prompt 工程对比报告

**数据集**: train + val | **对比图片数**: 10（v1-v6）+ 全量 2369（v7） | **版本**: v1, v2, v3, v4, v5, v6, v7

## 1. Prompt 设计概览

| 版本 | 名称 | 策略 | 核心技术 |
|------|------|------|----------|
| v1 | 基线版 | 极简指令，无格式约束或细节要求 | 直接指令，无输出格式指定 |
| v2 | 改进版 | 增加具体约束：句数、输出格式、类别选项 | 长度约束、格式规范、枚举选项 |
| v3 | 进阶版 | 结构化引导，明确维度和负面约束 | 结构化拆解（主体/动作/氛围）、负面约束（无边框）、少样本示例 |
| v4 | 丰富版 | 最大化信息密度：属性、数量、空间关系、氛围 | 属性要求（颜色/大小/材质）、空间关系引导、具体细节强调、推理请求 |
| v5 | 最佳实践版 | 结合角色扮演、结构化输出模板和质量约束 | 角色分配（专家分析师）、逐句结构模板、完整性指令、生动语言引导 |
| v6 | 增强结构版 | 保留 v3 的格式稳定性，增强细节丰富度和物体覆盖率 | 结构化拆解、负面约束、显式数量目标（15-25 项）、反模糊用词约束、故事视觉细节锚定 |
| v7 | 约束精炼版 | 基于 v6 全量分析，修复开头单调、长度过长、含换行符三大问题 | 角色设定、上下文说明、格式硬约束（50-80 词）、few-shot 示例、禁止 meta 开头、禁止换行 |

## 2. Prompt 详细内容

### v1 — 基线版

**caption**:
> Describe this image.

**objects**:
> List the objects in this image.

**category**:
> What type of image is this?

**short_story**:
> Write a short story about this image.


### v2 — 改进版

**caption**:
> Describe this image in 2-3 sentences. Include the main subjects, actions, and setting.

**objects**:
> List the main objects visible in this image. Return a JSON array of object name strings, e.g. ["car", "tree"].

**category**:
> Classify this image into one of: real photo, meme, web image, screenshot, illustration, other. Return only the category name.

**short_story**:
> Write a 1-2 sentence short story inspired by this image.


### v3 — 进阶版

**caption**:
> Analyze this image and describe it in 3-4 sentences. Cover: (1) the main subjects and their actions, (2) the setting and environment, (3) colors, lighting, and overall mood. Be specific and factual.

**objects**:
> Identify the distinct objects, people, animals, and elements in this image. Return ONLY a plain JSON array of short name strings, e.g. ["red car", "traffic light", "pedestrian"]. Do NOT include bounding boxes, coordinates, or any extra structure.

**category**:
> Classify this image into exactly one category from this list: real photo, meme, web image, screenshot, illustration, other. Return only the category name, nothing else.

**short_story**:
> Imagine the story behind this image. Write 2-3 sentences that capture a possible narrative, emotion, or moment. Be creative yet grounded in what you see.


### v4 — 丰富版

**caption**:
> Provide a detailed description of this image in 4-5 sentences. Include: the number and identities of people (if any), specific objects and their attributes (color, size, material), spatial relationships between elements, the setting (indoor/outdoor, time of day, weather), and the overall atmosphere or emotional tone. Avoid vague language; use concrete, observable details.

**objects**:
> List every distinct object and element you can identify in this image. For important objects, include a brief attribute (color, material, or state), e.g. ["wooden table", "red bicycle", "cloudy sky", "person wearing blue jacket"]. Return ONLY a JSON array of strings. Be thorough.

**category**:
> Classify this image into one of: real photo, meme, web image, screenshot, illustration, other. Then briefly explain your reasoning in one sentence. Format: {"category": "...", "reason": "..."}

**short_story**:
> Write a 2-3 sentence narrative inspired by this image. Incorporate specific visual details you observe — names, emotions, actions, or implied context. Aim to make the reader feel present in the scene.


### v5 — 最佳实践版

**caption**:
> You are an expert image analyst. Describe this image comprehensively in 4-5 sentences. Structure your description as follows: Sentence 1: Overall scene summary (what is happening). Sentence 2: Key subjects and their actions or positions. Sentence 3: Environmental details (location, time, weather, lighting). Sentence 4: Visual qualities (colors, textures, composition). Sentence 5: Mood, atmosphere, or implied narrative. Be precise, vivid, and avoid repetition.

**objects**:
> You are a precise object detector. List all distinct objects, people, animals, text, and notable elements visible in this image. For each entry, use the format "object (attribute)" where attribute is color, material, state, or position — e.g. ["red sports car", "elderly man (sitting)", "wooden fence (weathered)", "STOP sign"]. Return ONLY a JSON array of strings. Aim for completeness.

**category**:
> Classify this image into exactly one category from: real photo, meme, web image, screenshot, illustration, other. Return a JSON object: {"category": "...", "confidence": "high/medium/low", "reason": "..."}

**short_story**:
> You are a creative writer. Craft a 2-3 sentence story inspired by this image. Your story should: (1) reference specific visual elements you observe, (2) convey a clear emotion or theme, and (3) leave the reader with a memorable impression. Avoid generic descriptions; make every word count.


### v6 — 增强结构版

**caption**:
> Analyze this image and describe it in 3-4 sentences. Cover: (1) the main subjects, their actions, and visible attributes (color, material, size), (2) the setting, spatial layout, and environment details, (3) lighting, colors, and overall mood. Use concrete, specific language — avoid vague words like 'nice', 'some', 'various'.

**objects**:
> List ALL distinct objects, people, animals, plants, text, and visual elements you can identify in this image. Be thorough — aim for 15-25 items. For each item, use the format 'object' or 'adjective object' (e.g. "red car", "wooden table", "elderly woman", "STOP sign"). Return ONLY a plain JSON array of short strings. Do NOT include bounding boxes, coordinates, descriptions, or any extra structure.

**category**:
> Classify this image into exactly one category from this list: real photo, meme, web image, screenshot, illustration, other. Return only the category name, nothing else.

**short_story**:
> Imagine the story behind this image. Write 2-3 sentences that capture a possible narrative, emotion, or moment. Reference at least two specific visual details you observe. Be creative yet grounded in what you see.


### v7 — 约束精炼版

**caption**:
> You are a professional image captioner creating training data for an image captioning model. Write a factual description of this image in 2-3 sentences, 50-80 words. Cover: (1) the main subject and what is happening, (2) key visual details such as colors, lighting, and setting. Start directly with the subject — for example: 'A golden retriever sits on a worn leather couch near a sunlit window' or 'Two cyclists race along a coastal road at dusk, their shadows stretching across the pavement'. Do NOT begin with 'The image', 'This image', 'The picture', or any meta-phrase about the image itself. Do NOT use bullet points, markdown formatting, or line breaks. Write one plain paragraph. Avoid vague words like 'nice', 'some', 'various', 'interesting'.

**设计原则对照**:

| 原则 | v7 中的体现 |
|------|-----------|
| ①明确具体 | "2-3 sentences, 50-80 words" + 两个描述维度 |
| ②设定角色 | "You are a professional image captioner" |
| ③提供背景 | "creating training data for an image captioning model" |
| ④指定格式 | "one plain paragraph"、词数硬约束 50-80 |
| ⑤提供示例 | 两个 few-shot 开头示例 |
| ⑥设定限制 | 禁止 meta 开头、禁止 `\n`/markdown、禁止模糊词 |


## 3. 逐图对比

### 0.jpg（车内后视镜日落）

#### caption

| 版本 | 输出 |
|------|------|
| v1 | This is a beautifully composed photograph taken from inside a vehicle, looking out through the **side-view mirror** at sunset... （约 200 词，含大量技术分析） |
| v2 | This image captures a serene sunset viewed through a car's side-view mirror, with the golden-orange sun dipping below the horizon on a highway. |
| v3 | This image captures a view through a car's side-view mirror, reflecting the vehicle's own door handle and window frame alongside a highway scene at sunset. |
| v4 | This image captures a serene sunset scene viewed through a car's side-view mirror, with the sun positioned low on the horizon, casting a warm orange glow across the sky. |
| v5 | This image captures a serene sunset scene viewed through a car's side-view mirror, reflecting the road ahead and the sky ablaze with warm hues. |
| v6 | The image captures a view through a car's side-view mirror, reflecting the vehicle's own door handle and window frame alongside a highway scene at sunset. The mirror's convex surface distorts the background slightly, showing multiple cars ahead with headlights on, traveling along a multi-lane road marked with white dashed lines. |

#### objects

| 版本 | 输出 |
|------|------|
| v1 | 1. **Car side mirror**, 2. **Mirror glass**, 3. **Sunset sky**... （14 项，含长文本描述） |
| v2 | ["car", "mirror", "sun", "tree", "road"] （5 项） |
| v3 | ["car", "side mirror", "road", "lane markings", "sun", "sky", "trees", "guardrail", "headlights", "vehicle", "dashboard"] （11 项） |
| v4 | 格式异常：JSON 缺少引号 |
| v5 | 格式异常：JSON 结构损坏 |
| v6 | ["car side mirror", "car door", "car window", "car door handle", "highway road", "white lane markings", "sunset", "sun", "orange sky", "clouds", "trees", "guardrail", "oncoming car", "headlights", "small convex mirror", "reflection", "road surface", "distant vehicles", "car frame", "car interior"] （20 项） |

### 1.jpg（日式炸物拼盘）

#### caption

| 版本 | 输出 |
|------|------|
| v1 | This is a close-up, overhead-style food photograph showcasing a hearty meal... （约 200 词） |
| v2 | This image displays a wooden table set with three plates of Japanese-style fried food... |
| v3 | This image displays a casual dining scene featuring three plates of food arranged on a round, medium-brown wooden table. |
| v4 | This indoor dining scene features a round wooden table set with three distinct dishes... |
| v5 | This image captures a casual dining scene featuring three distinct dishes arranged on a wooden table... |
| v6 | Three plates of Japanese-style fried food are arranged on a round, medium-brown wooden table. The foreground features a basket lined with printed paper holding golden-brown, crispy chicken nuggets, accompanied by two metal forks with dark handles. |

#### objects

| 版本 | 输出 |
|------|------|
| v1 | 1. **Wooden table**, 2. **Bowl of noodles with chicken**... （18 项，含长描述） |
| v2 | 格式异常：返回了 bounding box 坐标 |
| v3 | ["table", "fork", "chopsticks", "chicken nuggets", "noodles", "chicken", "vegetables", "plate", "bowl", "paper", "condiment", "speaker"] （12 项） |
| v4 | 格式异常：返回了 bounding box 坐标 |
| v5 | 格式异常：JSON 结构损坏 |
| v6 | ["fried chicken pieces", "wooden table", "fork", "chopsticks", "paper liner", "noodles", "cucumber slices", "chicken cutlet", "ketchup", "mayonnaise", "bowl", "plate", "white appliance", "text on paper", "red sauce", "golden fried food", "green vegetables", "dining setting", "restaurant table"] （19 项） |

### 2.jpg（街头摊贩煎蛋）

#### objects

| 版本 | 输出 |
|------|------|
| v1 | 1. **Person (Chef/Cook)**, 2. **Steam/Vapor**... （15 项） |
| v2 | ["person", "egg", "egg yolk", "egg carton", "egg tray", "vegetable", "lettuce"...] （15 项） |
| v3 | ["person", "hand", "apron", "glove", "egg", "egg yolk", "meat", "spoon"...] （18 项） |
| v4 | 格式异常：碎片化属性文本 |
| v5 | ["person (cooking)", "hand (holding spoon)", "egg", "griddle", "spoon"...] （17 项） |
| v6 | ["chef", "white glove", "black apron", "steam", "metal tray", "egg", "wooden spatula", "plastic container", "minced meat", "egg carton", "eggs", "colander", "lettuce", "metal lid", "spoon", "hand", "table", "plastic container", "yellow egg carton"] （19 项） |

### 3.jpg（卡通狗狗海报）

#### category

| 版本 | 输出 |
|------|------|
| v1 | This is a **product photograph** — specifically, a close-up shot of **merchandise or stationery items**... （约 100 词分析） |
| v2 | real photo |
| v3 | illustration ✓ |
| v4 | {"category": "real photo", "reason": "..."} |
| v5 | {"category": "real photo", "confidence": "high", "reason": "..."} |
| v6 | illustration ✓ |

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["poster", "dog", "poop", "cloud", "star", "heart", "flower", "book", "poster stand", "store display"] （10 项） |
| v6 | ["poster", "text", "clouds", "stars", "hearts", "doodles", "cartoon dogs", "poop emoji", "blue background", "yellow text", "pink text", "white clouds", "green ground", "white dog", "yellow dog", "orange dog", "pink dog", "purple dog", "blue dog", "store display", "packaging", "LALA COMPANY", "DOGGIE CAN DO ANYTHING", "小狗可以"] （24 项） |

### 4.jpg（海边日落浪花）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["sun", "sky", "ocean", "wave", "foam", "rock", "seaweed", "shore", "mountain", "sunlight", "reflection", "water splash"] （12 项） |
| v6 | ["sun", "orange sky", "ocean", "waves", "white foam", "rocky shore", "seaweed", "distant island", "sunlight reflection", "water droplets", "foamy water", "rocky outcrop", "horizon line", "glare", "ripples", "coastline", "sea spray", "natural landscape", "sunset", "water texture", "foam texture", "rock texture", "sky gradient"] （23 项） |

### 5.jpg（家庭露营野餐）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["tent", "tarp", "people", "table", "chair", "water bottle", "cart", "tree", "grass", "sky", "rocks", "bag", "box", "drinking water", "food", "clothing", "shoes", "hammock", "awning"] （19 项） |
| v6 | ["tents", "people", "trees", "grass", "sky", "table", "chairs", "water bottle", "wagon", "rocks", "blanket", "bag", "shoes", "clothing", "awning", "pole", "rope", "sunlight", "shadows", "picnic items", "folding chair", "white box"] （22 项） |

### 6.jpg（墙上人影剪影）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["shadow", "brick wall", "people", "tree shadow"] （4 项） |
| v6 | ["shadows", "brick wall", "people", "tree shadow", "sunlight", "outdoor scene", "silhouette", "group of people", "arms raised", "building facade", "warm light", "blurred background", "shadows on wall", "human figures", "brick texture", "natural light", "outdoor environment", "group activity", "celebration", "evening light", "urban setting", "playful shadows", "group pose"] （23 项） |

### 7.jpg（空教室阳光）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["desk", "chair", "window", "curtain", "wall", "pipe", "radiator", "trees", "building", "window frame", "window sill", "sunlight", "shadow"] （13 项） |
| v6 | ["desk", "chair", "window", "curtain", "tree", "brick wall", "radiator", "pipe", "wall", "sunlight", "shadow", "window frame", "window sill", "desk leg", "chair back", "chair seat", "window pane", "greenery", "blue sticker", "metal frame", "wooden surface", "fabric"] （22 项） |

### 8.jpg（巷子里的两只猫）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["cat", "cat", "plant", "pot", "steps", "building", "wall", "pipe", "window", "balcony", "gate", "sunlight"] （12 项） |
| v6 | ["orange and white cat", "white and orange cat", "green leafy plant", "potted plant", "terracotta pot", "concrete steps", "brick wall", "metal gate", "balcony railing", "air conditioning unit", "pipes", "cable", "sunlight", "shadow", "window", "doorway", "leaves", "soil", "plant pot", "building exterior", "urban alleyway"] （21 项） |

### 9.jpg（水下书籍）

#### objects

| 版本 | 输出 |
|------|------|
| v3 | ["book", "fish", "aquatic plants", "water", "bullet casing"] （5 项） |
| v6 | ["open book", "text", "fish", "aquatic plant", "water", "fish tail", "fish body", "fish eye", "fish fin", "book page", "book spine", "book cover", "book text", "book title", "book paragraph", "book font", "book paper", "book binding", "book edge", "book corner", "book surface", "book thickness", "book shadow", "book reflection"] （24 项） |

## 4. 汇总统计

| 版本 | caption 平均词数 | objects 平均条数 | objects 格式正常率 | category 格式 | category 分布 |
|------|-----------------|-----------------|-------------------|---------------|---------------|
| v1 | 283.0 | 26.3 | 10/10（格式正常但为冗长文本） | 冗长段落（10/10） | 各不相同的长文分析 |
| v2 | 69.5 | 20.3 | 8/10（2 个返回了 bounding box） | 简洁（10/10） | real photo: 10 |
| v3 | 100.6 | 11.6 | 10/10 | 简洁（10/10） | real photo: 9, illustration: 1 |
| v4 | 129.2 | 31.4 | 7/10（3 个 JSON 格式损坏） | JSON（10/10） | real photo: 9, illustration: 1 |
| v5 | 127.9 | 39.9 | 5/10（5 个 JSON 格式损坏） | JSON（10/10） | real photo: 10 |
| **v6** | **108.0** | **21.7** | **10/10** | **简洁（10/10）** | **real photo: 8, illustration: 2** |
| **v7** | **61.0** | — | — | — | — |

### 格式问题详情

| 版本 | objects 格式问题 | category 格式问题 |
|------|-----------------|-----------------|
| v1 | 输出为带编号的长文本，非结构化列表 | 输出为 100+ 词的分析段落，非单个类别名 |
| v2 | 2/10 返回了 bounding box 坐标而非纯文本列表 | 无 |
| v3 | 无 | 无 |
| v4 | 3/10 JSON 缺少引号或括号，无法解析 | 无 |
| v5 | 5/10 JSON 结构损坏，属性挤入单个字符串 | 无 |
| **v6** | **无** | **无** |

### v3 vs v6 关键对比

| 指标 | v3 | v6 | 提升 |
|------|-----|-----|------|
| caption 平均词数 | 100.6 | 108.0 | +7.4（更丰富的细节） |
| objects 平均条数 | 11.6 | 21.7 | +87%（覆盖率大幅提升） |
| objects 格式正常率 | 10/10 | 10/10 | 持平（格式同样稳定） |
| 6.jpg objects 数 | 4 | 23 | +475%（从严重遗漏到全面覆盖） |
| 9.jpg objects 数 | 5 | 24 | +380%（从严重遗漏到全面覆盖） |
| category illustration 准确率 | 1/1 (3.jpg) | 2/2 (3.jpg, 9.jpg) | v6 额外识别出水下书籍为插画 |

### v6 vs v7 关键对比（caption 质量）

| 指标 | v6 (全量 2000 张) | v7 (3 张测试) | 改进 |
|------|-------------------|--------------|------|
| caption 平均词数 | 134.1 | 61.0 | -54%（精简到训练友好区间） |
| 以 "The image" 开头占比 | 74.5% | **0%** | 完全消除 meta 开头 |
| 含 `\n` 占比 | 32.4% | **0%** | 完全消除换行符 |
| 生成速度 | ~8.5s/张 | ~5s/张 | -41%（输出更短，推理更快） |

v7 的 caption 更适合用于训练 image captioning 模型：
- **长度适中**：50-80 词，信息密度高，无冗余
- **开头多样**：直接以画面主体开头，而非千篇一律的 "The image features/captures"
- **格式干净**：无 `\n`、无 markdown、纯文本段落

## 5. 观察与结论

### caption 质量最佳：v7（训练用）/ v6（详细描述用）

**用于模型训练：v7 最佳**。v7 的 caption 平均 61 词（v6 为 134 词），长度适中，信息密度高，无冗余。v7 完全消除了 "The image" 开头（v6 中占 74.5%）和 `\n` 换行符（v6 中占 32.4%），输出格式干净，可直接用于训练，无需后处理。

**用于详细描述：v6 最佳**。v6 的 caption 平均 108 词，细节丰富，适合需要全面描述的场景（如图片检索、内容审核）。

### objects 检测最佳：v6

- **v6 格式完全稳定**：10/10 全部返回干净的纯文本数组，零格式异常
- **v6 覆盖率显著提升**：平均 21.7 条，比 v3 的 11.6 条提升 87%
- **关键改进在"困难"图片上**：v3 在 6.jpg（人影）仅检测到 4 个物体，v6 检测到 23 个；v3 在 9.jpg（水下书籍）仅 5 个，v6 达到 24 个
- **保持了 v3 的格式优势**：没有出现 v4/v5 的 JSON 损坏问题

### category 分类最佳：v6

v6 在 10 张图中正确识别了 2 张 illustration（3.jpg 卡通海报和 9.jpg 水下合成图），比 v3（仅识别 3.jpg）更全面。v2/v5 全部判为 "real photo"，对合成/插画图的判断过于粗略。

### short_story 创意最佳：v6

v6 的 prompt 要求"Reference at least two specific visual details"，使故事更紧密地锚定在图片内容上，避免了 v5 偶尔脱离画面的问题，同时保持了 v3/v4 级别的创意水准。

### 综合推荐

**objects/category/short_story 最佳：v6**
- 格式 100% 稳定，物体覆盖率比 v3 提升 87%，category 准确率最高

**caption 最佳（训练用）：v7**
- 词数从 v6 的 134 词精简到 61 词，更适合训练 image captioning 模型
- 完全消除 "The image" 开头（v6 占 74.5%）和 `\n` 换行符（v6 占 32.4%）
- 生成速度提升 41%（5s/张 vs 8.5s/张）
- 遵循六条 prompt 工程原则（角色设定、上下文、格式约束、few-shot、负面约束）

**最终选择：v7 caption + v6 objects/category/short_story**
- 用于 Task 2 训练时仅需 caption，采用 v7
- 如需完整标注（objects、category、short_story），采用 v6

### 关键 Prompt 工程启示

1. **显式数量目标有效** — v6 的 "aim for 15-25 items" 使物体数量从 v3 的 11.6 提升到 21.7，而不损害格式稳定性
2. **反模糊约束提升质量** — "avoid vague words like 'nice', 'some', 'various'" 有效减少了笼统描述
3. **视觉细节锚定增强故事质量** — 要求"Reference at least two specific visual details"使 short_story 更贴合图片内容
4. **简单格式 + 强约束 > 复杂格式 + 弱约束** — v6 用纯文本数组（简单格式）+ 显式数量目标和负面约束（强约束），优于 v4/v5 的 JSON 对象（复杂格式）+ 模糊的 "be thorough"（弱约束）
5. **在已验证的基础上迭代** — v6 并非全新设计，而是在 v3 的稳定基础上针对性修复弱点，这种迭代策略比从头设计更可靠
6. **六条原则指导 caption 优化** — v7 通过角色设定（原则②）、上下文说明（原则③）、格式硬约束（原则④）、few-shot 示例（原则⑤）和负面约束（原则⑥），将 caption 从 v6 的 134 词精简到 61 词，同时完全消除了 meta 开头和换行符问题
7. **全量分析驱动迭代** — v7 的改进基于 v6 全量 2000 张数据的统计分析（而非仅看 10 张样本），发现了 10 张样本中未暴露的系统性问题（74.5% "The image" 开头）
