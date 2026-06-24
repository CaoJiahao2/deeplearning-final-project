# Attention 模型推理结果（val 前 10 张）

**模型**: attention (BERT tokenizer + BERT embedding init)
**权重**: checkpoint/v7/attention/attention_best.pth
**解码策略**: Greedy

## 结果

| 图片 | 生成描述 |
|------|---------|
| 2002.jpg | a small, tree perches on a grassy ledge, its feathers reflecting the sunlight reflecting the sunlight. the scene is set against a dark backdrop with sunlight filtering through the trees, casting dappled shadows on the ground. |
| 2003.jpg | two prze osski stand in an enclosure of a enclosure, their feathers reaching towards a brown mane and brown. the scene is bathed in natural daylight, highlighting the textures of the rocks and scattered trees. |
| 2004.jpg | a hand holds a clear plastic cup with a clear plastic lid, featuring a hand and white hand sits on a white surface. the top holds a green spoon containing a clear broer with a clear plastic spoon to the scene. |
| 2005.jpg | a tall skyscrapers dominate the city skyline, their silhouettes of orange and white facades. the buildings feature tall residential structures with tall structures and a tall residential structures line the horizon. |
| 2006.jpg | a fluffy cat with a light brown coat sits on a wooden desk, its gaze directed slightly upward. the cat's fur is a mix of light brown and white, with a dark brown coat and it is positioned around the cat. |
| 2007.jpg | a fluffy cat with a red coat sits on a wooden desk, its gaze directed slightly upward. the room is softly lit by natural daylight, highlighting the cat's fur and the gentle ripple. |
| 2008.jpg | a snow-covered landscape is blanketed in snow, with snow-covered trees in the background. the scene is set in a white background with bare trees, with a few people stroll away from the right. |
| 2009.jpg | a traditional chinese pavilion with a red roof stands prominently in a park-like setting, its white structure contrasting with the surrounding greenery. visitors stroll along the path, enjoying the serene atmosphere. |
| 2010.jpg | a magnolia tree in full bloom captures a vibrant pink tree, with clusters of pink flowers and a pink flowers. the blossoms are in full bloom, some pausing to admire the springtime. |
| 2011.jpg | a bustling street scene unfolds under a vast blue sky, where people stroll and ride pink flowering trees. a winding path filled with cars, while people stroll along the road. |

## 观察

1. **仍有重复问题** — 部分描述出现词语重复（如 "reflected the sunlight reflecting the sunlight"）
2. **语义基本合理** — 大部分描述能正确识别图片内容（猫、城市、雪景等）
3. **细节丰富度** — 相比旧版有所提升，但仍有改进空间
4. **Beam search 可能改善** — 当前使用 greedy 解码，beam search 可能减少重复

## 运行方式

```bash
python scripts/infer_val_top10.py
```
