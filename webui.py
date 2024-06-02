import gradio as gr
import os,subprocess,yaml




if not os.path.exists("./results"):
    os.makedirs("./results")

class WebUI:
    def __init__(self) -> None:
        self.info=Info()
        self.opt_cfg_pth='configs/conf.yaml'
        self.main_ui()
    
    def main_ui(self):
        with gr.Blocks() as ui:
            gr.Markdown('## 一个reflow vae svc 的推理和训练webui')
            with gr.Tab("训练/Training"):
                gr.Markdown(self.info.general)
                
                with gr.Accordion('数据集说明',open=False):
                    gr.Markdown(self.info.dataset)
                
                gr.Markdown('## 生成配置文件')
                with gr.Row():
                    self.batch_size=gr.Slider(minimum=12,maximum=1024,value=24,label='Batch_size',interactive=True)
                    self.learning_rate=gr.Number(value=0.0005,label='学习率',info='真心不建议超过0.0002')
                    self.f0_extractor=gr.Dropdown(['parselmouth', 'dio', 'harvest', 'crepe','rmvpe'],type='value',value='rmvpe',label='f0提取器种类',interactive=True)
                    self.val=gr.Number(value=1000,label='验证步数',info='检查点，会被覆盖哦',interactive=True)
                    self.save=gr.Number(value=1000,label='强制保存步数',info='被强制保存的模型',interactive=True)
                    self.n_spk=gr.Number(value=1,label='说话人数量',interactive=True)
                with gr.Row():
                    self.amp_dtype=gr.Dropdown(['fp16','fp32',"bf16"],value='fp16',label='训练精度',interactive=True)
                    self.cache_fp16=gr.Checkbox(value=True,label='FP16半精度缓存？',info='减少显存占用',interactive=True)
                    self.decay_step=gr.Number(value=1000,label='decay_step',info='学习率衰减的步数',interactive=True)
                    self.gamma=gr.Number(value=0.5,label='gamma',info='学习率衰减的量',interactive=True)
                    self.gpu_id=gr.Number(value=1,label='训练使用的GPU设备号',info='0...1...2...3',interactive=True)
                    self.use_pitch_aug=gr.Checkbox(value=True,label='启用音域增强？',info='拓宽音域但损失音质',interactive=True)
                with gr.Row():
                    self.device=gr.Dropdown(['cuda','cpu'],value='cuda',label='使用设备',interactive=True)
                    self.num_workers=gr.Number(value=2,label='读取数据进程数',info='如果你的设备性能很好，可以设置为0',interactive=True)
                    self.cache_all_data=gr.Checkbox(value=True,label='启用缓存',info='将数据全部加载以加速训练',interactive=True)
                    self.cache_device=gr.Dropdown(['cuda','cpu'],value='cuda',type='value',label='缓存设备',info='如果你的显存比较大，设置为cuda',interactive=True)
                    self.n_layers=gr.Number(value=20,label='n_layers',info='就是n_layers',interactive=True)
                    self.n_chans=gr.Number(value=768,label=' n_chans',info='就是n_chans',interactive=True)
                self.bt_create_config=gr.Button(value='创建配置文件')
                
                gr.Markdown('## 预处理')
                with gr.Accordion('预训练说明',open=False):
                    gr.Markdown(self.info.preprocess)
                with gr.Row():
                    self.bt_open_data_folder=gr.Button('打开数据集文件夹')
                    self.bt_preprocess=gr.Button('开始预处理')
                gr.Markdown('## 训练')
                with gr.Accordion('训练说明',open=False):
                    gr.Markdown(self.info.train)
                with gr.Row():
                    self.bt_train=gr.Button('开始训练')
                    self.bt_visual=gr.Button('启动可视化')
                    gr.Markdown('启动可视化后[点击打开](http://127.0.0.1:6006)')
                
            with gr.Tab('推理/Inference'):
                with gr.Accordion('推理说明',open=False):
                    gr.Markdown("按照说明用！")
                with gr.Row():
                    self.input_wav=gr.Audio(type='filepath',label='选择待转换音频')
                    self.choose_model=gr.Textbox('exp/model_chino.pt',label='模型路径')
                with gr.Row():
                    self.keychange=gr.Slider(-24,24,value=0,step=1,label='变调')
                    self.id=gr.Number(value=1,label='说话人id')
                    self.infer_step=gr.Number(value=20,label='inferstep',info='采样步数')
                    self.method=gr.Dropdown(['rk4','euler'],value='euler',type='value',label='采样器',interactive=True)
                with gr.Row():
                    self.bt_infer=gr.Button(value='开始转换')
                    self.output_wav=gr.Audio(type='filepath',label='输出音频')
            self.bt_create_config.click(fn=self.create_config)
            self.bt_open_data_folder.click(fn=self.openfolder)
            self.bt_preprocess.click(fn=self.preprocess)
            self.bt_train.click(fn=self.training)
            self.bt_visual.click(fn=self.visualize)
            self.bt_infer.click(fn=self.inference,inputs=[self.input_wav,self.choose_model,self.keychange,self.id,self.infer_step,self.method],outputs=self.output_wav)
        ui.launch(inbrowser=True,server_port=7858)
        
    def openfolder(self):
        try:
            os.startfile('data')
        except:
            print('Fail to open folder!')


    def create_config(self):
        with open('configs/reflow-vae-wavenet.yaml','r',encoding='utf-8') as f:
            conf=yaml.load(f.read(),Loader=yaml.FullLoader)
        conf['data']['f0_extractor']=str(self.f0_extractor.value)
        conf['train']['interval_val']=int(self.val.value)
        conf['train']['interval_force_save']=int(self.save.value)
        conf['train']['batch_size']=int(self.batch_size.value)
        conf['device']=str(self.device.value)
        conf['train']['num_workers']=int(self.num_workers.value)
        conf['train']['cache_all_data']=str(self.cache_all_data.value)
        conf['train']['cache_device']=str(self.cache_device.value)
        conf['train']['amp_dtype']=str(self.amp_dtype.value)
        conf['train']['cache_fp16']=str(self.cache_fp16.value)
        conf['train']['decay_step']=int(self.decay_step.value)
        conf['train']['gamma']=int(self.gamma.value)
        conf['env']['gpu_id']=int(self.gpu_id.value)
        conf['model']['use_pitch_aug']=str(self.use_pitch_aug.value)
        conf['model']['n_layers']=int(self. n_layers.value)
        conf['model']['n_chans']=int(self.n_chans.value)
        print('配置文件信息：'+str(conf))
        with open(self.opt_cfg_pth,'w',encoding='utf-8') as f:
            yaml.dump(conf,f)
        print('成功生成配置文件')

    
    def preprocess(self):
        subprocess.Popen('python -u draw.py',stdout=subprocess.PIPE)
        preprocessing_process=subprocess.Popen('python -u preprocess.py -c '+self.opt_cfg_pth,stdout=subprocess.PIPE)
        while preprocessing_process.poll() is None:
            output=preprocessing_process.stdout.readline().decode('utf-8')
            print(output)
        print('预处理完成')
            
    def training(self):
        train_process=subprocess.Popen('python -u train.py -c '+self.opt_cfg_pth,stdout=subprocess.PIPE)
        while train_process.poll() is None:
            output=train_process.stdout.readline().decode('utf-8')
            print(output)
        
            
    def visualize(self):
        tb_process=subprocess.Popen('tensorboard --logdir=exp --port=6006',stdout=subprocess.PIPE)
        while tb_process.poll() is None:
            output=tb_process.stdout.readline().decode('utf-8')
            print(output)
            
    def inference(self,input_wav:str,model:str,keychange,id,infer_step,method):
        print(input_wav,model)
        output_wav='results/'+ input_wav.replace('\\','/').split('/')[-1]
        cmd='python -u main.py -i '+input_wav+' -m '+model+' -o '+output_wav+' -k '+str(int(keychange))+' -tid '+str(int(id))+' -method '+method+' -step '+str(int(infer_step))
        infer_process=subprocess.Popen(cmd,stdout=subprocess.PIPE)
        while infer_process.poll() is None:
            output=infer_process.stdout.readline().decode('utf-8')
            print(output)
        print('推理完成')
        return output_wav

class Info:
    def __init__(self) -> None:
        self.general='''
### 不看也没事，大致就是  
1.设置好配置之后点击创建配置文件  
2.点击‘打开数据集文件夹’，音频全塞到data\\train\\audio下面  
3.点击‘开始预处理’等待执行完毕  
4.点击‘开始训练’和‘启动可视化’然后点击右侧链接  
'''
        self.dataset="""
### 1. 配置训练数据集和验证数据集

#### 1.1 手动配置：

将所有的训练集数据 (.wav 格式音频切片) 放到 `data/train/audio`。

将所有的验证集数据 (.wav 格式音频切片) 放到 `data/val/audio`。

#### 1.2 程序随机选择：

自动运行`python draw.py`,程序将帮助你挑选验证集数据（可以调整 `draw.py` 中的参数修改抽取文件的数量等参数）。

#### 1.3文件夹结构目录展示：
- 单人物目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    └─ val
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav

 ```
- 多人物目录结构：

```
data
├─ train
│    ├─ audio
│    │    ├─ 1
│    │    │   ├─ aaa.wav
│    │    │   ├─ bbb.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ccc.wav
│    │    │   ├─ ddd.wav
│    │    │   └─ ....wav
│    │    └─ ...
│    └─ val
│    │    ├─ 1
│    │    │   ├─ eee.wav
│    │    │   ├─ fff.wav
│    │    │   └─ ....wav
│    │    ├─ 2
│    │    │   ├─ ggg.wav
│    │    │   ├─ hhh.wav
│    │    │   └─ ....wav
│    │    └─ ...
```
                            """
        self.preprocess='''

### 注意：
1. 请保持所有音频切片的采样率与 yaml 配置文件中的采样率一致！如果不一致，程序可以跑，但训练过程中的重新采样将非常缓慢。（可选：使用Adobe Audition™的响度匹配功能可以一次性完成重采样修改声道和响度匹配。）

2. 所有音频切片的时长不应少于 2 秒。如果音频切片太多，则需要较大的内存，取消勾选启用缓存可以解决此问题。

4. 如果您的数据集质量不是很高，请在配置文件中将 'f0_extractor' 设为 'rmvpe'。rmvpe 算法的抗噪性最好，但代价是会极大增加数据预处理所需的时间。

5. 配置文件中的 ‘n_spk’ 参数将控制是否训练多说话人模型。如果您要训练**多说话人**模型，为了对说话人进行编号，所有音频文件夹的名称必须是**不大于 ‘n_spk’ 的正整数**。
        '''
        self.train='''
## 训练


### 1. 训练：
1. 记得改那个见鬼的说话人
2. 记得改模型深度和通道以匹配你的预训练模式
3. 底模在哪自己找去。
4. 把预训练模型`model_0.pt`解压到`.\exp\reflowvae-test\`中
5. 启动训练。
        '''
        self.visualize='''
## 可视化
```bash
# 使用tensorboard检查训练状态
tensorboard --logdir=exp
```
第一次验证 (validation) 后，在 TensorBoard 中可以看到合成后的测试音频。

        '''




webui=WebUI()
