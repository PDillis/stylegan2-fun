# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os

import projector
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots, save_every_step=False, save_final_step=False):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
            # If user wishes to save the dlatent every step
            if save_every_step:
                np.save(dnnlib.make_run_dir_path(png_prefix.split(os.sep)[-1] + 'step%04d.npy' % proj.get_cur_step()), proj.get_dlatents())
    # If the user wishes to only save the final projected dlatent (and it hasn't already been saved)
    if save_final_step and not save_every_step:
        np.save(dnnlib.make_run_dir_path(png_prefix.split(os.sep)[-1] + 'step%04d.npy' % proj.get_cur_step()), proj.get_dlatents())
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def project_generated_images(network_pkl, 
                             seeds, 
                             num_snapshots, 
                             truncation_psi, 
                             num_steps, 
                             save_every_step=False, 
                             save_final_step=False):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector(num_steps=num_steps)
    proj.set_network(Gs)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        project_image(proj, targets=images, 
                      png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), 
                      num_snapshots=num_snapshots,
                      save_every_step=save_every_step,
                      save_final_step=save_final_step)

#----------------------------------------------------------------------------

def project_real_images(network_pkl, 
                        dataset_name, 
                        data_dir, 
                        num_images, 
                        num_snapshots, 
                        num_steps, 
                        save_every_step=False, 
                        save_final_step=False):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector(num_steps=num_steps)
    proj.set_network(Gs)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, 
                      png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), 
                      num_snapshots=num_snapshots,
                      save_every_step=save_every_step, 
                      save_final_step=save_final_step)

#----------------------------------------------------------------------------

# My extended version of this helper function:
def _parse_num_range(s):
    '''
    Input:
        String s, a comma separated list of numbers 'a,b,c', a range 'a-c',
        or even a combination of both 'a,b-c', 'a-b,c', 'a,b-c,d,e-f,...'
    Output:
        nums, a list of ascending ordered ints with repeating values deleted
    '''
    # Sanity check 0:
    # In case there's a space between the numbers (impossible due to argparse,
    # but hey, I am that paranoid):
    s = s.replace(' ', '')
    # Split w.r.t comma:
    str_list = s.split(',')
    nums = []
    for el in str_list:
        if '-' in el:
            # The range will be 'a-b', so we wish to find both a and b using re:
            range_re = re.compile(r'^(\d+)-(\d+)$')
            match = range_re.match(el)
            # We get the two numbers:
            a = int(match.group(1))
            b = int(match.group(2))
            # Sanity check 1: accept 'a-b' or 'b-a', with a<=b:
            if a <= b: r = [n for n in range(a, b + 1)]
            else: r = [n for n in range(b, a + 1)]
            # Use extend since r will also be an array:
            nums.extend(r)
        else:
            # It's a single number, so just append it:
            nums.append(int(el))
    # Sanity check 2: delete repeating numbers:
    nums = list(set(nums))
    # Return the numbers in ascending order:
    return sorted(nums)

#----------------------------------------------------------------------------

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

  # --num-snapshots=N will take N snapshots of the projection process; --num-steps=M will 
  # In both, add the flags --save-every-step if you wish to save the dlatent at every step, or --save-final-step if you only wish to save the final dlatent.
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    project_generated_images_parser = subparsers.add_parser('project-generated-images', help='Project generated images')
    project_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds; input can be a,b,c-d or any combination (default: %(default)s)', dest='seeds', default=range(3))
    project_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', dest='num_snapshots', default=5)
    project_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', dest='truncation_psi', default=1.0)
    project_generated_images_parser.add_argument('--num-steps', type=int, help='Number of steps (default: %(default)s)', dest='num_steps', default=1000)
    project_generated_images_parser.add_argument('--save-every-step', action='store_true', help='Save npy file of the dlatent at every step, including last one (default: %(default)s)', dest='save_every_step')
    project_generated_images_parser.add_argument('--save-final-step', action='store_true', help='Save npy file of the dlatent at the final step, if it hasn`t already been saved (default: %(default)s)', dest='save_final_step')
    project_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', dest='data_dir', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5, dest='num_snapshots')
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3, dest='num_images')
    project_real_images_parser.add_argument('--num-steps', type=int, help='Number of steps (default: %(default)s)', default=1000, dest='num_steps')
    project_real_images_parser.add_argument('--save-every-step', action='store_true', help='Save npy file of the dlatent at every step, including last one (default: %(default)s)', dest='save_every_step')
    project_real_images_parser.add_argument('--save-final-step', action='store_true', help='Save npy file of the dlatent at the final step, if it hasn`t already been saved (default: %(default)s)', dest='save_final_step')
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'project-generated-images': 'run_projector.project_generated_images',
        'project-real-images': 'run_projector.project_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
