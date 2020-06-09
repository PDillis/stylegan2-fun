# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse

import scipy
import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

import re
import sys

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pretrained_networks

#----------------------------------------------------------------------------

def create_image_grid(images, grid_size=None):
    '''
    Args:
        images: np.array, images
        grid_size: tuple(Int), size of grid (grid_w, grid_h)
    Returns:
        grid: np.array, image grid of size grid_size
    '''
    # Some sanity check:
    assert images.ndim == 3 or images.ndim == 4
    num, img_h, img_w = images.shape[0], images.shape[1], images.shape[2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)
    # Get the grid
    grid = np.zeros(
        [grid_h * img_h, grid_w * img_w] + list(images.shape[-1:]), dtype=images.dtype
    )
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y : y + img_h, x : x + img_w, ...] = images[idx]

    return grid

#----------------------------------------------------------------------------

def generate_images(network_pkl, seeds, truncation_psi, grid=False):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    images = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d)...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        image = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        images.append(image[0])
        PIL.Image.fromarray(image[0], 'RGB').save(dnnlib.make_run_dir_path('seed%04d.png' % seed))
    if grid:
        print('Generating image grid...')
        PIL.Image.fromarray(create_image_grid(np.array(images)), 'RGB').save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    # Sanity check: styles are actually possible for generated image size
    max_style = int(2 * np.log2(Gs.output_shape[-1])) - 3
    assert max(col_styles) <= max_style, "Maximum col-style allowed: {}".format(max_style)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W = Gs.output_shape
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
    canvas.save(dnnlib.make_run_dir_path('grid.png'))

#----------------------------------------------------------------------------

def lerp_video(
    network_pkl,                # Path to pretrained model pkl file
    seeds,                      # Random seeds
    grid_w=None,                # Number of columns
    grid_h=None,                # Number of rows
    truncation_psi=1.0,         # Truncation trick
    duration_sec=30.0,          # Duration of video in seconds
    smoothing_sec=3.0,
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="16M",
    minibatch_size=8,
):
    num_frames = int(np.rint(duration_sec * mp4_fps))

    print('Loading network from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    print("Generating latent vectors...")
    if grid_w == grid_h == None and len(seeds) > 1:
        # number of images according to the seeds provided
        num = len(seeds)
        # Get the grid width and height according to num:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)
        grid_size = [grid_w, grid_h]
        # [frame, image, channel, component]:
        shape = [num_frames] + Gs.input_shape[1:]
        # Get the latents:
        all_latents = np.stack([np.random.RandomState(seed).randn(*shape).astype(np.float32) for seed in seeds], axis=1)
    elif None not in (grid_w, grid_h) and len(seeds) == 1:
        # Otherwise, the user gives one seed and the grid width and height:
        grid_size = [grid_w, grid_h]
        # [frame, image, channel, component]:
        shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]
        # Get the latents with the random state:
        random_state = np.random.RandomState(seeds)
        all_latents = random_state.randn(*shape).astype(np.float32)
    else:
        print("Error: wrong combination of arguments!")
        sys.exit()

    all_latents = scipy.ndimage.gaussian_filter(
        all_latents,
        [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape),
        mode="wrap"
    )
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Define the kwargs for the Generator:
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                      nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    # Aux function: Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        # Get the images (with labels = None)
        images = Gs.run(latents, None, **Gs_kwargs)
        # Generate the grid for this timestamp:
        grid = create_image_grid(images, grid_size)
        # grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    import moviepy.editor

    # Generate video using make_frame:
    print('Generating interpolation video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    mp4 = "{}x{}-lerp.mp4".format(*grid_size)
    videoclip.write_videofile(dnnlib.make_run_dir_path(mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

#----------------------------------------------------------------------------

def style_mixing_video(
    network_pkl,
    src_seed,               # Seed of the source image style (row)
    dst_seeds,              # Seeds of the destination image styles (columns)
    col_styles,             # Styles to transfer from top row to source image
    truncation_psi=1.0,
    duration_sec=30.0,
    smoothing_sec=3.0,
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="16M",
    minibatch_size=8,
):
    # Calculate the number of frames:
    num_frames = int(np.rint(duration_sec * mp4_fps))
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    # Sanity check: styles are actually possible for generated image size
    max_style = int(2 * np.log2(Gs.output_shape[-1])) - 3
    assert max(col_styles) <= max_style, "Maximum col-style allowed: {}".format(max_style)

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    # Left col latents
    print('Generating Source W vectors...')
    src_shape = [num_frames] + Gs.input_shape[1:]
    src_z = np.random.RandomState(*src_seed).randn(*src_shape).astype(np.float32) # [frames, src, component]
    src_z = scipy.ndimage.gaussian_filter(
        src_z, [smoothing_sec * mp4_fps] + [0] * (len(Gs.input_shape) - 1), mode="wrap"
    )
    src_z /= np.sqrt(np.mean(np.square(src_z)))
    # Map into the detangled latent space W:
    src_w = Gs.components.mapping.run(src_z, None)
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # Top row latents
    print('Generating Destination W vectors...')
    dst_z = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds])
    dst_w = Gs.components.mapping.run(dst_z, None)
    dst_w = w_avg + (dst_w - w_avg) * truncation_psi
    # Get the width and height of each image:
    _N, _C, H, W = Gs.output_shape
    # Generate our canvas where we will paste all the generated images:
    canvas = PIL.Image.new("RGB", (W * (len(dst_seeds) + 1), H * (len(src_seed) + 1)), "white")

    # Generate ALL the source images:
    src_images = Gs.components.synthesis.run(src_w, **Gs_syn_kwargs)
    # Generate the column images:
    dst_images = Gs.components.synthesis.run(dst_w, **Gs_syn_kwargs)
    # Paste them:
    for col, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, "RGB"), ((col + 1) * H, 0))

    # Aux function: Frame generation func for moviepy.
    def make_frame(t):
        # Get the frame number according to time t:
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        # We wish the image belonging to the frame at time t:
        src_image = src_images[frame_idx]
        # Paste it to the lower left:
        canvas.paste(PIL.Image.fromarray(src_image, "RGB"), (0, H))

        # Now, for each of the column images:
        for col, dst_image in enumerate(list(dst_images)):
            # Select the pertinent latent w column:
            w_col = np.stack([dst_w[col]]) # [18, 512] -> [1, 18, 512]
            # Replace the values defined by col_styles:
            w_col[:, col_styles] = src_w[frame_idx, col_styles]
            # Generate these synthesized images:
            col_images = Gs.components.synthesis.run(w_col, **Gs_syn_kwargs)
            # Paste them in their respective spot:
            for row, image in enumerate(list(col_images)):
                canvas.paste(
                    PIL.Image.fromarray(image, "RGB"),
                    ((col + 1) * H, (row + 1) * W),
                )
        return np.array(canvas)

    import moviepy.editor

    # Generate video using make_frame:
    print('Generating style-mixed video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    grid_size = [len(dst_seeds), len(src_seed)]
    mp4 = "{}x{}-style-mixing.mp4".format(*grid_size)
    videoclip.write_videofile(dnnlib.make_run_dir_path(mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

#----------------------------------------------------------------------------

def sightseeding(
    network_pkl,                # Path to pretrained model pkl file
    seeds,                      # List of random seeds to use
    loop=False,                 # Loop back to the first seed when reaching the end
    truncation_psi=1.0,         # Truncation trick
    seed_sec=5.0,               # Time duration between seeds
    mp4_fps=30,
    mp4_codec="libx264",
    mp4_bitrate="16M",
    minibatch_size=8,
):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    # Number of steps to take between each latent vector
    n_steps = int(np.rint(seed_sec * mp4_fps))
    # If we wish to return to the first latent, then we add it to the end 
    if loop:
        seeds.append(seeds[0])
    # Number of frames in total
    num_frames = int(n_steps * (len(seeds) - 1))
    # Duration in seconds
    duration_sec = num_frames / mp4_fps
    # Generate the random vectors from each seed
    print('Generating W vectors...')
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]

    # Helper function for linear interpolation
    def lerp(w1, w2, n_steps):
        '''
        Input:
            w1, w2: numpy arrays (latent vectors)
            n_steps: number of steps to take between w1 and w2
        Output:
            vectors: numpy array of latent vectors, without including w2 
        '''
        ratios = np.linspace(0, 1, num=n_steps, endpoint=False)
        vectors = list()
        for ratio in ratios:
            w = (1.0 - ratio) * w1 + ratio * w2
            vectors.append(w)
        return np.asarray(vectors)
    
    src_w = np.empty([0] + list(all_w.shape[1:]), dtype=np.float64)
    for i in range(len(all_w) - 1):
        # We interpolate between each pair of latents
        interp = lerp(all_w[i], all_w[i+1], n_steps)
        # append it to our source
        src_w = np.append(src_w, interp, axis=0)
    # Do the truncation trick
    src_w = w_avg + (src_w - w_avg) * truncation_psi

    # Our grid will be 1x1
    grid_size = [1,1]

    # Aux function: Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latent = src_w[frame_idx]
	# Select the pertinent latent w column:
        w = np.stack([latent]) # [18, 512] -> [1, 18, 512]
        image = Gs.components.synthesis.run(w, **Gs_syn_kwargs)
        # Generate the grid for this timestamp:
        grid = create_image_grid(image, grid_size)
        # grayscale => RGB
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)
        return grid

    import moviepy.editor

    # Generate video using make_frame:
    print('Generating sightseeding video...')
    videoclip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    name = '-'
    name = name.join(map(str, seeds))
    mp4 = "{}-sighseeding.mp4".format(name)
    videoclip.write_videofile(dnnlib.make_run_dir_path(mp4),
                              fps=mp4_fps,
                              codec=mp4_codec,
                              bitrate=mp4_bitrate)

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

def _parse_seeds(s):
    '''
    Input:
        String s, a comma separated list of numbers 'a,b,c,...'
    Output:
        nums, an unordered list of ints in s with no deletion of repeated values or sorting
    '''
    # Do the same sanity check as above:
    s = s.replace(' ', '')
    # Split w.r.t. comma:
    str_list = s.split(',')
    nums = [int(el) for el in str_list]
    return nums

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0

  # Generate 30-second-long uncurated 'ffhq' 3x2 interpolation video at 30fps (truncation-psi=1.0)
  python %(prog)s lerp-video --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=1000

  # Generate 30-second-long curated 'ffhq' 3x2 interpolation video at 30fps (truncation-psi=1.0)
  python %(prog)s lerp-video --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=42-47

  # Generate 50-second-long uncurated 'cars' 5x3 interpolation video at 60fps, with truncation-psi=0.7
  python %(prog)s lerp-video --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=1000 --grid-w=5 --grid-h=3 --truncation-psi=0.7 --duration_sec=50 --fps=60

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-video --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seed=85 --col-seeds=55,821,1789,293

  # Generate style mixing example of fine styles layers (64^2-1024^2, as in StyleGAN)
  python %(prog)s style-mixing-video --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seed=85 --col-seeds=55,821,1789,293 --col-styles=8-17 --truncation-psi=1.0
  
  # Generate sightseeding video (1x1), with 5-second interpolation between seeds, looping back to the first seed in the end
  python %(prog)s sightseeding --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=4,9,7,5,4,6,8  --loop=True 
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', dest='seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', dest='truncation_psi', default=0.5)
    parser_generate_images.add_argument('--create-grid', type=bool, help='Save the generated images in a grid (default: %(default)s)', dest='grid', default=False)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', dest='row_seeds', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', dest='col_seeds', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', dest='col_styles', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', dest='truncation_psi', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_lerp_video = subparsers.add_parser('lerp-video', help='Generate interpolation video (lerp) between random vectors')
    parser_lerp_video.add_argument('--network', help='Path to network pickle filename', dest='network_pkl', required=True)
    parser_lerp_video.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', dest='seeds', required=True)
    parser_lerp_video.add_argument('--grid-w', type=int, help='Video grid width/columns (default: %(default)s)', default=None, dest='grid_w')
    parser_lerp_video.add_argument('--grid-h', type=int, help='Video grid height/rows (default: %(default)s)', default=None, dest='grid_h')
    parser_lerp_video.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0, dest='truncation_psi')
    parser_lerp_video.add_argument('--duration-sec', type=float, help='Duration of video (default: %(default)s)', default=30.0, dest='duration_sec')
    parser_lerp_video.add_argument('--fps', type=int, help='FPS of generated video (default: %(default)s)', default=30, dest='mp4_fps')
    parser_lerp_video.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_video = subparsers.add_parser('style-mixing-video', help='Generate style mixing video (lerp)')
    parser_style_mixing_video.add_argument('--network', help='Path to network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_video.add_argument('--row-seed', type=_parse_num_range, help='Random seed to use for image source row', dest='src_seed', required=True)
    parser_style_mixing_video.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns (style)', dest='dst_seeds', required=True)
    parser_style_mixing_video.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6', dest='col_styles')
    parser_style_mixing_video.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.7, dest='truncation_psi')
    parser_style_mixing_video.add_argument('--duration-sec', type=float, help='Duration of video (default: %(default)s)', default=30, dest='duration_sec')
    parser_style_mixing_video.add_argument('--fps', type=int, help='FPS of generated video (default: %(default)s)', default=30, dest='mp4_fps')
    parser_style_mixing_video.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_sightseeding = subparsers.add_parser('sightseeding', help='Generate latent interpolation video between a set of user-fed random seeds.')
    parser_sightseeding.add_argument('--network', help='Path to network pickle filename', dest='network_pkl', required=True)
    parser_sightseeding.add_argument('--seeds', type=_parse_seeds, help='List of random seeds', dest='seeds', required=True)
    parser_sightseeding.add_argument('--loop', type=bool, help='Close the sightseeding loop; after the last seed, return to the first one (default: %(default)s)', default=False, dest='loop')
    parser_sightseeding.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0, dest='truncation_psi')
    parser_sightseeding.add_argument('--seed-sec', type=float, help='Number of seconds between each seed (default: %(default)s)', default=5.0, dest='seed_sec')
    parser_sightseeding.add_argument('--fps', type=int, help='FPS of generated video (default: %(default)s)', default=30, dest='mp4_fps')
    parser_sightseeding.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'style-mixing-example': 'run_generator.style_mixing_example',
        'lerp-video': 'run_generator.lerp_video',
        'style-mixing-video': 'run_generator.style_mixing_video',
        'sightseeding': 'run_generator.sightseeding'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
