from pathlib import Path
from collections import defaultdict, OrderedDict
import json
import types
import platform
import argparse
import struct
import torch

from lvae.models.registry import get_model
import lvae.utils.coding as coding
from lvae.evaluation import imcoding_evaluate


@torch.no_grad()
def compress_scalable(self, im, lmb=None):
    assert hasattr(self, 'keep_z'), f'keep_z is not defined in {self.__class__.__name__}'

    lmb = lmb or self.default_lmb # if no lmb is provided, use the default one
    lv_block_results = self.forward_end2end(im, lmb=lmb, mode='compress')
    assert len(lv_block_results) == self.num_latents
    lv_block_results = lv_block_results[:self.keep_z]

    assert im.shape[0] == 1, f'Right now only support a single image, got {im.shape=}'
    all_lv_strings = [res['strings'][0] for res in lv_block_results]
    string = coding.pack_byte_strings(all_lv_strings)
    # encode lambda and image shape in the header
    nB, _, imH, imW = im.shape
    header1 = struct.pack('f', lmb)
    header2 = struct.pack('3H', nB, imH//self.max_stride, imW//self.max_stride)
    string = header1 + header2 + string
    return string

@torch.no_grad()
def decompress_scalable(self, string):
    assert hasattr(self, 'keep_z'), f'keep_z is not defined in {self.__class__.__name__}'

    # extract lambda
    _len = 4
    lmb, string = struct.unpack('f', string[:_len])[0], string[_len:]
    # extract shape
    _len = 2 * 3
    (nB, nH, nW), string = struct.unpack('3H', string[:_len]), string[_len:]
    all_lv_strings = coding.unpack_byte_string(string)
    assert len(all_lv_strings) == self.keep_z

    lmb = self.expand_to_tensor(lmb, n=nB)
    lmb_embedding = self._get_lmb_embedding(lmb, n=nB)

    feature = self.get_bias(bhw_repeat=(nB, nH, nW))
    str_i = 0
    for bi, block in enumerate(self.dec_blocks):
        if getattr(block, 'is_latent_block', False):
            if str_i < self.keep_z: # scalable decoding
                strs_batch = [all_lv_strings[str_i],]
                feature, _ = block(feature, lmb_embedding, mode='decompress', strings=strs_batch)
                str_i += 1
            else: # prior mean
                feature, _ = block(feature, lmb_embedding, mode='sampling', t=0.0)
        elif getattr(block, 'requires_embedding', False):
            feature = block(feature, lmb_embedding)
        else:
            feature = block(feature)
    assert str_i == len(all_lv_strings), f'str_i={str_i}, len={len(all_lv_strings)}'
    im_hat = self.process_output(feature)
    return im_hat


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',        type=str,   default='qarv_base')
    parser.add_argument('-a', '--model_args',   type=str,   default='pretrained=True')
    parser.add_argument('-n', '--dataset_name', type=str,   default='kodak')
    parser.add_argument('-d', '--device',       type=str,   default='cuda:3')
    args = parser.parse_args()

    kwargs = eval(f'dict({args.model_args})')
    model = get_model(args.model, **kwargs)

    model = model.to(device=torch.device(args.device))
    model.eval()
    model.compress_mode()

    model.default_lmb = 768
    model.compress = types.MethodType(compress_scalable, model)
    model.decompress = types.MethodType(decompress_scalable, model)

    save_json_path = Path(f'runs/results/{args.dataset_name}-{args.model}-scalable.json')
    if not save_json_path.parent.is_dir():
        print(f'Creating {save_json_path.parent} ...')
        save_json_path.parent.mkdir(parents=True)

    all_lmb_stats = defaultdict(list)
    for keep in range(1, 10):
        model.keep_z = keep

        results = imcoding_evaluate(model, args.dataset_name)
        print(results)

        for k,v in results.items():
            all_lmb_stats[k].append(v)

    # save to json
    json_data = OrderedDict()
    json_data['name'] = args.model
    json_data['description'] = 'Scalable compression results'
    json_data['test-set'] = args.dataset_name
    json_data['platform'] = platform.platform()
    json_data['device']   = str(torch.device(args.device))
    json_data['results']  = all_lmb_stats
    with open(save_json_path, 'w') as f:
        json.dump(json_data, fp=f, indent=4)
    print(f'\nSaved results to {save_json_path} \n')
    # print results
    for k, vlist in all_lmb_stats.items():
        vlist_str = ', '.join([f'{v:.12f}'[:7] for v in vlist])
        print(f'{k:<6s} = [{vlist_str}]')


if __name__ == '__main__':
    main()
