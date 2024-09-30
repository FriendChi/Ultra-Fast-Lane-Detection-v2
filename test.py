import torch, os
from utils.common import merge_config, get_model
from evaluation.eval_wrapper import eval_lane
import torch
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    cfg.distributed = distributed
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    net = get_model(cfg)

    state_dict = torch.load(cfg.test_model, map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = True)

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    def run_test_tusimple(net,data_root,work_dir,exp_name, distributed, crop_ratio, train_width, train_height, batch_size = 8, row_anchor = None, col_anchor = None):
        output_path = os.path.join(work_dir,exp_name+'.%d.txt'% get_rank())
        fp = open(output_path,'w')
        loader = get_test_loader(batch_size,data_root,'Tusimple', distributed, crop_ratio, train_width, train_height)
        for data in dist_tqdm(loader):
            imgs,names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                pred = net(imgs)
            for b_idx,name in enumerate(names):
                tmp_dict = {}
                tmp_dict['lanes'] = generate_tusimple_lines(pred['loc_row'][b_idx], pred['exist_row'][b_idx], pred['loc_col'][b_idx], pred['exist_col'][b_idx], row_anchor = row_anchor, col_anchor = col_anchor, mode = '4row')
                tmp_dict['h_samples'] = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260,
                 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 
                 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 
                 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
                tmp_dict['raw_file'] = name
                tmp_dict['run_time'] = 10
                json_str = json.dumps(tmp_dict)
    
                fp.write(json_str+'\n')
        fp.close()
