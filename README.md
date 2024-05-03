# ReFlow-VAE-SVC

安装依赖，数据准备，配置编码器（hubert 或者 contentvec) ，声码器 (nsf-hifigan) 与音高提取器 (RMVPE) 的环节与 DDSP-SVC 项目相同。


（1）预处理：

```bash
python -m ReFlowVaeSVC.preprocess -c configs/reflow-vae-wavenet.yaml
```

（2）训练（无底模）：

```bash
python -m ReFlowVaeSVC.train -c configs/reflow-vae-wavenet.yaml
```
Beta版底模可以在这里下载：https://huggingface.co/OOPPEENN/pretrained_model

（3）非实时推理：

```bash
# 普通模式, 需要语义编码器, 比如 contentvec
python -m ReFlowVaeSVC.main -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -tid <target_speaker_id> -step <infer_step> -method <method>
# VAE 模式, 无需语义编码器, 特化 sid 到 tid 的变声（或者音高编辑，如果sid == tid）
python -m ReFlowVaeSVC.main -i <input.wav> -m <model_ckpt.pt> -o <output.wav> -k <keychange (semitones)> -sid <source_speaker_id> -tid <target_speaker_id> -step <infer_step> -method <method>
```
