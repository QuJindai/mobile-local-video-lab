# 去噪模型精度诊断

原始 FP16 去噪模型已完成真实权重前向、ONNX 导出和 ORT CPU 执行，但原有
逐元素容差 `atol=0.005, rtol=0.01` 未通过。手机接口需要 FP32 端口；
仅修改端口声明或精度比较的参考值不能解决这两个问题。

本轮使用同一固定上游源码、权重和完整输入，执行三个对照：

1. 保留原始 FP16 模型及未放宽的容差，保存确切输入、参考输出和实际输出。
2. 额外观察嵌入与各个模型块的输出，定位误差积累；同时记录观察前后的
   最终输出差异，避免把增加中间输出导致的优化变化误认为生产图修复。
3. 将已转 FP16 的同一组权重和输入精确扩展到 FP32，作为运算精度候选方案。
   分别比较 FP32 ONNX 与 FP32 PyTorch，以及两者与原始 FP16 PyTorch。
   前者通过不代表后者通过；完整模型包和手机成功标志始终为 false。

每个导出和运行阶段使用独立进程，保留失败证据。CPU 结果只用于诊断；
MNN 转换、实际 Adreno 执行和完整视频效果仍需各自验证。

代码复核发现上游两处密集注意力在 dropout 后强制 `.half()`，直接
`model.float()` 会导致 FP16 概率与 FP32 value 矩阵相乘报错。候选方案明确
将这一行适配为 `.to(v.dtype)`，记录原方法与适配方法哈希；原始 FP16
参考不作修改。14 个 LiteLA 的 RoPE 表保持原来的半精度频率与三角函数
结果，再精确扩展，避免同时改变位置常量。两个密集注意力原有 FP32 RoPE
保持原样。上述适配不代表原始 FP16 精度门槛已经通过。

复核同时确认：增加中间输出只用于观察，不能强迫后续计算使用重新舍入的
张量。另设原始 `t2i_modulate` 表达式的微型对照，将单个 ORT 图与逐个
算子、显式 FP16 边界相比较。这个实验若显示差异，只证明舍入机制存在，
不能单独归因整个去噪网络的误差。

候选代码复核未发现阻断项，6 项针对性检查通过。后续执行复用原始实验
的只读基准，校验原导出脚本、上游源码、权重及每个文件的身份；原始
报告不重写，候选方案不能把旧失败报告改成成功。

## 原始参考实测结果

[运行 34691139577](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34691139577)
对应源码 `eccdf2744ffaf02a19a8d9c70d88aa51016cfc21`。原始 FP16 的完整
前向耗时 612.04 秒，导出和两种 ORT 图均实际执行完成。706,560 个最终
输出值中 78 个超出原有容差，最大绝对误差 0.013671875，RMSE 0.0017619934。
增加观察点前后的最终输出完全相同。在所观察的节点中，`block_2` 首次
超过该容差；这定位了进一步检查的范围，不等于已确定其中某个算子的责任。

该运行随后在未适配的 FP32 注意力处报 Float/Half 类型冲突，与复核发现
一致。原始参考阶段的结果有效并保留；整次运行的结论仍为失败。

- [原始比较报告](evidence/denoiser-precision-original-original-fp16-compare.json)
- [原始导出证据](evidence/denoiser-precision-original-original-fp16-export.json)
- [未适配 FP32 失败报告](evidence/denoiser-precision-original-promoted-fp32-export.json)
- 原始图与确切输入 artifact：`10297535932`，699,175,066 字节；ZIP SHA-256
  `acb8aba107c488a035de54c6c4f54e72b6d652065f8778ee019c097df02d65a7`。

## 适配后的 FP32 实测结果

[运行 34692274739](https://github.com/QuJindai/mobile-local-video-lab/actions/runs/34692274739)
对应源码 `daadc7648d921e4fd9a2babbad203c528f851f7b`。10 项原有检查和 6 项
针对性检查通过。真实权重的 FP32 前向耗时 19.57 秒，导出、ORT 执行和
全部对照完成，没有再次出现 Float/Half 类型冲突。

| 比较 | 容差 atol / rtol | 超标元素 / 706,560 | 最大绝对误差 | RMSE | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| FP32 ONNX 对 FP32 PyTorch | 0.0001 / 0.001 | 36 | 0.0008411855 | 0.00001208484 | 未通过 |
| FP32 PyTorch 对原始 FP16 | 0.005 / 0.01 | 80 | 0.0134997368 | 0.0017354018 | 未通过 |
| FP32 ONNX 对原始 FP16 | 0.005 / 0.01 | 80 | 0.0134987831 | 0.0017355511 | 未通过 |

这次 Actions 的 success 表示实验执行完毕，**不表示模型精度通过**。
报告中的模型资格、完整流程资格、Android 模型包就绪标志均为 false。
当前 FP32 候选不能作为已合格的 MobileI2V 模型包发布。

舍入微型对照也完成：原始 PyTorch 表达式输出 0，单个 ORT 图输出
0.0078125，显式分开的 FP16 算子边界恢复输出 0。这证明该受控输入下的
舍入差异，尚不能解释完整网络全部误差。MNN 固定版本的 `CastOnnx.cpp`
还会将 `Cast(FLOAT16)` 改成 `DT_FLOAT`，因此普通 Cast 不能充当保留半精度
舍入的实现。后续需进一步隔离第 3 个模型块，并评估保留精度语义的执行
适配；不能仅改端口名称、扩大容差或将新参考结果替代原始参考。

- [候选比较报告](evidence/denoiser-precision-adapted-promoted-fp32-compare.json)
- [候选导出证据](evidence/denoiser-precision-adapted-promoted-fp32-export.json)
- [舍入对照](evidence/denoiser-precision-adapted-rounding-control.json)
- 全部图与确切输入 artifact：`10298335461`，1,356,329,530 字节；ZIP SHA-256
  `edeb2d355dea4860a0dbd219461d22c2fa7c3892c06125aeb5247024684d1cdf`。
- 报告 ZIP 下载后校验 SHA-256：
  `40a55e16d1255b9703318264d4c75b21fa14fe15615142b28b1312231e00d82d`。

本轮没有发布新 APK 或模型包。手机截图所示的 Depth 3D 与原始权重下载
状态已记录于 [V0.7.1 手机测试记录](V0.7.1_HANDSET_TEST.md)。
