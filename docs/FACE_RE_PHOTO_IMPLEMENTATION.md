# FACE-RE 照片换脸首个闭环

2026-09-13，承接用户已授权的换脸主线与 APK 交付请求。

## 本次实现

参考照提供身份，目标照提供构图、动作和背景。两张图均需检测到且只检测到
一张可用人脸。使用 SCRFD 2.5G 五点检测、ArcFace W600K R50 身份编码和
INSwapper 128 的 FP32 本机 ONNX Runtime 推理；对齐后换脸并羽化贴回，
保存完整目标照片。此阶段不提供凭提示词生成新动作或新场景的入口。

模型来自 FaceFusion 官方模型资产仓的 models-3.0.0 发布，先在构建侧实跑并
记录 SHA-256、输入输出和身份条件影响，再固定为 Android 下载校验清单。
模型和代码许可分别注明。大权重不随 APK 重复分发，使用校验后原子激活；
已有 MobileI2V 文件和历史结果不迁移、不重下。

## 模块接口与任务

1. 模型与真实执行：`tools/face/` 和独立 qualification workflow，输出固定
   模型清单及 emap，验证三模型推理和换脸对身份输入的响应。
2. 几何与图像核心：`FaceSwapMath`、`FaceSwapPixels`，五点相似变换、逆变换、
   RGB NCHW、SCRFD 输出解码/NMS、身份投影和羽化合成；使用独立已知值测试。
3. Android 执行：`FaceModelStore` 流式下载和完整性校验；`FaceSwapEngine`
   每任务管理模型资源，不并发加载多个换脸任务，失败时关闭资源。
4. 使用闭环：`FaceSwapActivity` 两张照片、模型准备、运行/取消状态、结果
   保存/分享/最近记录、返回原视频实验；后台任务只消费任务开始时的输入。
5. 集成构建：保持 `com.qujindai.localvideo` 与原签名，V0.9.0/code 11；
   保留 RIFE/Depth/MobileI2V，既有测试与 APK 验证继续执行。

## 验证标准

几何与张量的已知值测试；真实模型 CPU 推理；改变源身份影响结果；异常输入
拒绝；APK 编译、包名、签名、模型清单与既有运行时打包。模型 host 测试与
Android 打包不能替代 S24U 的视觉质量、耗时、内存和连续生成验收。
