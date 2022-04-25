import argparse
import json
import pathlib
from pathlib import Path

import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch  # monorec model
from evaluater import Evaluater
from utils.parse_config import ConfigParser


def main(config: ConfigParser):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.initialize('data_loader', module_data)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])  # 1个 depth_loss
    # print("loss: ", loss)
    metrics = [getattr(module_metric, met) for met in config['metrics']]  # 7个
    # print("metrics: ", metrics)

    # build model architecture, then print to console

    if "arch" in config.config:
        models = [config.initialize('arch', module_arch)]    # monorec model
    else:
        models = config.initialize_list("models", module_arch)

    results = []
    for i, model in enumerate(models):   # 就一个model
        model_dict = dict(model.__dict__)   # 获取model 参数
        keys = list(model_dict.keys())
        '''
        dict_keys(['training', '_parameters', '_buffers', '_non_persistent_buffers_set', 
        '_backward_hooks', '_is_full_backward_hook', '_forward_hooks', '_forward_pre_hooks', 
        '_state_dict_hooks', '_load_state_dict_pre_hooks', '_modules', 'inv_depth_min_max', 
        'cv_depth_steps', 'use_mono', 'use_stereo', 'use_ssim', 'sfcv_mult_mask', 'pretrain_mode', 
        'pretrain_dropout', 'pretrain_dropout_mode', 'augmentation', 'simple_mask', 'mask_use_cv', 
        'mask_use_feats', 'cv_patch_size', 'no_cv', 'depth_large_model', 'checkpoint_location', 'mask_cp_loc', 
        'depth_cp_loc', 'freeze_module', 'freeze_resnet', 'augmenter'])
        '''
        for k in keys:
            if k.startswith("_"):
                model_dict.__delitem__(k)
            elif type(model_dict[k]) == np.ndarray:
                model_dict[k] = list(model_dict[k])


        dataset_dict = dict(data_loader.dataset.__dict__)   # 获取dataset的参数
        keys = list(dataset_dict.keys())   # 代表的是dataset的init paras
        '''
        ['dataset_dir', 'frame_count', 'target_image_size', 'offset_d', 'nusc', 'pointsensor_channel', 
        'camera_channel', '_offset', 'length', 'dilation', 'use_color_augmentation', 'return_mvobj_mask']
        '''
        for k in keys:
            if k.startswith("_"):
                dataset_dict.__delitem__(k)
            elif type(dataset_dict[k]) == np.ndarray:
                dataset_dict[k] = list(dataset_dict[k])
            elif isinstance(dataset_dict[k], pathlib.PurePath):
                dataset_dict[k] = str(dataset_dict[k])


        logger.info(model_dict)
        logger.info(dataset_dict)
        print("############################ start eval ##########################")
        # 传入了模型，loss，需要记录的metrics，config，测试数据
        evaluater = Evaluater(model, loss, metrics, config=config, data_loader=data_loader)
        print("############################ end eval ##########################")
        result = evaluater.eval(i)  # eval 0th model
        result["metrics"] = result["metrics"]
        del model
        result["metrics_info"] = [metric.__name__ for metric in metrics]
        logger.info(result)
        results.append({
            # "model": model_dict,
            # "dataset": dataset_dict,
            "result": result
        })

    save_file = Path(config.log_dir) / "results.json"
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Finished")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Deeptam Evaluation')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    config = ConfigParser(args)
    print(config.config)
    main(config)
