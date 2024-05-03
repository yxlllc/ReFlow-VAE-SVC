import os
import time
import numpy as np
import torch
try:
    import torch_musa
    use_torch_musa = True
except ImportError:
    use_torch_musa = False
import librosa
from ReFlowVaeSVC.logger.saver import Saver
from ReFlowVaeSVC.logger import utils
from torch import autocast
# from torch.cuda.amp import GradScaler

def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters_with_grad = [p for p in parameters if p.grad is not None]

    torch.nn.utils.clip_grad_value_(parameters_with_grad, clip_value)
    
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters_with_grad]), 2)
    return total_norm
    
def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            mel = model(
                    data['units'], 
                    data['f0'], 
                    data['volume'], 
                    data['spk_id'],
                    vocoder=vocoder,
                    infer=True,
                    return_wav=False,
                    infer_step=args.infer.infer_step, 
                    method=args.infer.method)
            signal = vocoder.infer(mel, data['f0'])
            ed_time = time.time()
                        
            # RTF
            run_time = ed_time - st_time
            song_time = signal.shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss = model(
                data['units'], 
                data['f0'], 
                data['volume'], 
                data['spk_id'],
                vocoder=vocoder,
                gt_spec=data['mel'],
                infer=False)
            test_loss += loss.item()
            
            # log mel
            saver.log_spec(data['name'][0], data['mel'], mel)
            
            # log audio
            path_audio = os.path.join(args.data.valid_path, 'audio', data['name_ext'][0])
            audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
            saver.log_audio({fn+'/gt.wav': audio, fn+'/pred.wav': signal})
            
    # report
    test_loss /= num_batches 
    
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
    # saver
    saver = Saver(args, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    start_epoch = initial_global_step // num_batches
    model.train()
    saver.log_info('======= start training =======')
    if use_torch_musa:
        scaler = torch.musa.amp.GradScaler()
    else:
        scaler = torch.cuda.amp.GradScaler()
    
    if args.train.amp_dtype == 'fp32':
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    for epoch in range(start_epoch, args.train.epochs):
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # forward
            if dtype == torch.float32:
                loss = model(data['units'].float(), data['f0'], data['volume'], data['spk_id'], 
                                aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False)
            else:
                if use_torch_musa:   
                    with torch.musa.amp.autocast(dtype=dtype):
                        loss = model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                        aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False)
                else:
                    with autocast(device_type=args.device, dtype=dtype):
                        loss = model(data['units'], data['f0'], data['volume'], data['spk_id'], 
                                        aug_shift=data['aug_shift'], vocoder=vocoder, gt_spec=data['mel'].float(), infer=False)
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                if dtype == torch.float32:
                    loss.backward()
                    grad_norm = clip_grad_value_(model.parameters(), 1)
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_value_(model.parameters(), 1)
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | loss: {:.3f} | time: {} | step: {} | grad: {:.2f}'.format(
                        epoch,
                        batch_idx,
                        num_batches,
                        args.env.expdir,
                        args.train.interval_log/saver.get_interval_time(),
                        current_lr,
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step,
                        grad_norm
                    )
                )
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/lr': current_lr,
                    'train/grad_norm': grad_norm
                })
            
            # validation
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # run testing set
                test_loss = test(args, model, vocoder, loader_test, saver)
                
                # log loss
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.3f}. '.format(
                        test_loss,
                    )
                )
                
                saver.log_value({
                    'validation/loss': test_loss,
                })
                
                model.train()

                          
