
**NOTE: This will be the last push, as I will migrate the code and project to [StyleGAN2-ADA](https://github.com/PDillis/stylegan2-ada).**

# Let's have fun with StyleGAN2!

SOTA GANs can become cumbersome or even downright intimidating, if not daunting. StyleGAN2 is no different, especially when you consider the compute capabilities usually needed to fully train one model from scratch. This is what this repository is here for! My wish is that anyone can enjoy as much their trained models without the hassle of dealing with battling the code *that* much: you still have to put some sweat to create the videos you have envisioned, but it shouldn't be *unnecessary* sweat.

We add some features to StyleGAN2's [official repo](#sgan2), so please get acquainted with the original one first before delving deep into this branch. Take special attention to the  [Requirements](#requirements), in order for the code to run correctly. Should you encounter any errors, check out a possible solution in [Troubleshooting](#tshoot).

In essence,this repo adds two new features:
* [Interpolation videos](#latent)
    * [Random vector interpolation](#lerp)
    * [Style mixing video](#style)
    * [Sightseeding](#sightseeding)
    * [Circular interpolation](#circular)
* [Projection videos](#proj)
    * [Mass Projector](#mass_proj)
    * [Save projected videos](#save_proj)

Two other changes have been made to the original repository, specifically also in `run_generator.py`:
* When generating random images by seed number, you can now save them in a grid by adding `--grid=True`; the grid dimensions will be inferred by the number of `--seeds` you use
* The `_parse_num_range(s)` function has been modified in order to accept any combination of comma separated numbers or ranges, or combination of both, i.e.: `--seeds=1,2,5-100,999-1004,123456` is now also accepted
    * *Caution:* for `style-mixing-video`, for example, using too many `--col-seeds` will result in an OOM!

<a name="tshoot"></a>
## Troubleshooting

But first, some troubleshooting: if, by some reason, you run into the following error while attempting to run the official repo in Ubuntu 18.04:

```
error: #error "C++ versions less than C++11 are not supported."
```

know that you are not alone and that many have suffered as you surely have. The good news is that there's a [quick fix](https://stackoverflow.com/a/59368180) for this: modify the `nvcc` call in [line 64](https://github.com/NVlabs/stylegan2/blob/7d3145d23013607b987db30736f89fb1d3e10fad/dnnlib/tflib/custom_ops.py#L64) in `dnnlib/tflib/custom_ops.py` by replacing it with:

```python
cmd = 'nvcc --std=c++11 -DNDEBUG ' + opts.strip()
```

The bad news is that it doesn't always work, and it's not a *pretty* fix, but at least it let me run the code on my PC. You might not encounter this error, but make sure to always check if you meet the [Requirements](#requirements), especially by running the test `test_nvcc.cu`.

<a name="latent"></a>
## Latent space exploration

What is a trained GAN without a bit of latent space exploration? We will use my [trained model on huipils from Guatemala and Mexico](https://youtu.be/GzHKtcPTKR4), `huipils.pkl`, as well as other models that can be found in [Justin Pinkney's](https://www.justinpinkney.com/) [ncredible repo](https://github.com/justinpinkney/awesome-pretrained-stylegan2). Make sure to visit it if you wish to start using one of the many available models!

We can do the typical interpolation between random latent vectors, without the need to specify *which* random vectors. A new run will be added to the `./results` subdir each time you run the code with the pertinent name of the subcommand used (unless you change this path of course).

<a name="lerp"></a>
### Random interpolation

A linear interpolation or [lerp](https://en.wikipedia.org/wiki/Linear_interpolation) is done between random vectors in ***Z***, which in turn will be mapped into ***W*** by the Generator. There are two options available:

* Give a list of `--seeds` (in the form `a,b,c`, `a-b,c`, or even a combination `'a,b-c,d,e-f,...'`), and the code will infer the best width and height for the generated video. For example, we wish the seeds to be from `42` to `47` (inclusive), then we run:

```
python run_generator.py lerp-video \
    --network=/path/to/huipils.pkl \
    --seeds=42-47
```

![3x2-lerp](./docs/gifs/3x2-lerp.gif)

 (All GIFs in this README are resized/downgraded)

 By default, we will have that `--truncation-psi=1.0`, `--duration-sec=30.0` and `--fps=30`, so modify these as you please.

* Give a single seed in `--seeds`, as well as the grid shape (`--grid-w` and `--grid-h`, respectively) for the generated video. We can then generate a 60fps, 15 second video of size 2x2, with `--truncation-psi=0.7` like so:

```
python run_generator.py lerp-video \
    --network=/path/to/huipils.pkl \
    --seeds=1000 --grid-w=2 --grid-h=2 \
    --truncation-psi=0.7 --fps=60 --duration-sec=15
```

![2x2-lerp](./docs/gifs/2x2-lerp.gif)

In either of these cases, you might have a specific run where you think that a *slower* interpolation video might better show the capabilities of your trained network. That is, you wish to keep the generated path as is, but slowed down when viewing it in the video. This is what `--slowdown` is for.

#### Slowdown

For such cases, you can add `--slowdown=N`, where `N` is a power of 2 (e.g., `1, 2, 4, 8, ...`, default value is `1`). This is why every lerp interpolation video is saved as `{grid-w}_{grid-h}-lerp-{N}xslowdown.mp4` in the respective `./results` subdir. I apologize if the naming becomes confusing, but you can always rename your saved videos to however you like. Quickly generating a 1x1 interpolation video, we then run:

```
python run_generator.py lerp-video \
    --network=/path/to/huipils.pkl \
    --seeds=7 --truncation-psi=0.7 --duration-sec=15\
```

We can see the result of this code if we add the following (remember, `--slowdown=1` is the default value):

| `--slowdown=1` | `--slowdown=2` | `--slowdown=4` | `--slowdown=8` |
| :-: | :-: | :-: | :-: |
| ![1xslowdown](./docs/gifs/lerp-1xslowdown.gif) | ![2xslowdown](./docs/gifs/lerp-2xslowdown.gif) | ![4xslowdown](./docs/gifs/lerp-4xslowdown.gif) | ![8xslowdown](./docs/gifs/lerp-8xslowdown.gif) |

<a name="style"></a>
### Style mixing video

We can recreate the [style mixing video](https://youtu.be/c-NJtV9Jvp0?t=145) by specifying which styles from the source image (`--row-seed`, which will generate a [lerp](#lerp) video using said seed) we want to replace in the destination image (`--col-seeds`, which will fix the images using these seeds per column):

```
python run_generator.py style-mixing-video \
    --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
    --col-seeds=55,293,821,1789,293 --row-seed=85
```

![style-mixing](./docs/gifs/4x1-style-mixing.gif)

 By default, we will have that `--truncation-psi=0.7`, `--duration-sec=30.0`, `--fps=30`, and `--col-styles=0-6`, which indicates that we will use the styles from layers 4x4 up to the first layer of the 32x32 (remember there are two per layer).

As stated before, we will replace the selected `--col-styles` from the `--col-seeds` with the respective styles from the `--row-seed`. So, if you wish to apply the **coarse styles** defined in the [StyleGAN paper](https://arxiv.org/abs/1812.04948), you can use `--col-styles=0-3`; for the **middle styles**, use `--col-styles=4-7`; and finally, for the **fine styles**, use `--col-styles=8-max_style`, where `max_style` will depend on the generated image size of your model. The following table gives a small summary of this value:

| `Gs.ouptut_shape[-1]` | `max_style` |
| --- | --- |
| `1024` | `17` |
| `512` | `15` |
| `256` | `13` |
| `128` | `11` |
| `...` | `...` |

I hope you get the gist of it or, if you wish to make it truly independent of any user input error in the code you develop, you can always calculate it via:

```python
max_style = int(2 * np.log2(Gs.output_shape[-1])) - 3
```

Which is used in the assertions at the beginning of the `style_mixing_example` and `style_mixing_video` functions.

To visualize how these different `--col-styles` are replaced, the following table will be generated with the code:

```
python run_generator.py style-mixing-video \
   --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
   --col-seeds=55,821,1789,293 --row-seed=85
```

We just need to add the styles to replace, which can be found in the first column:

| Styles to replace |  Result |
| :-: | :-: |
| `--col-styles=0-3` | ![style-mixing-coarse](./docs/gifs/4x1-style-mixing-coarse.gif) |
| `--col-styles=4-7` | ![style-mixing-middle](./docs/gifs/4x1-style-mixing-middle.gif) |
| `--col-styles=8-17` | ![style-mixing-fine](./docs/gifs/4x1-style-mixing-fine.gif) |

Of course, more wacky combinations of `--col-styles` (for example, `--col-styles=1,5-8,13`) can be made, so long as the final result is satisfactory to your needs. Be sure to understand *what* the specific styles are doing in your model, as this will be key for you to know what you wish to transfer.

On the other hand, if you only wish to show the result of the style mixing (i.e., the images in the second row, second column onward), then you can add the `--only-stylemix` flag. In the following example, we will transfer only the fine layers, so the resulting image will only be changing color, not the general identity of the generated person:

```
python run_generator.py style-mixing-video \
   --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
   --col-seeds=1769 --row-seed=85 \
   --col-styles=8-17 --only-stylemix
```

![onlymix](./docs/gifs/1x1-style-mixing_onlymix.gif)

<a name="sightseeding"></a>
### Sightseeding

Running `python run_generator.py generate-images`, you might've encountered a set of `--seeds` that you wish to further explore in the latent space. To this end is this code, as we can now visit a list of user-specified `seeds` and they will be visited ***in order*** by interpolating between them.

Using the unofficial [Metfaces model](https://github.com/justinpinkney/awesome-pretrained-stylegan2#painting-faces), we can thus travel between 4 different seeds like so:

```
python run_generator.py sightseeding \
    --network=/path/to/metfaces.pkl \
    --seeds=161-163,83,161 \
    --truncation-psi=0.7 --seed-sec=5.0
```

Note we add the first seed back at the end in order to create a loop. Now, we need to add also which type of interpolation to make (`--interp-type`): a [`linear`](https://en.wikipedia.org/wiki/Linear_interpolation) or [`spherical`](https://en.wikipedia.org/wiki/Slerp) interpolation between the generated latent vectors. Likewise, we can perform this interpolation either in ***Z*** (`--interp-in-z=True`) or ***W*** (`--interp-in-z=False`). Irrespective of these two options, we can add the flag `--smooth` in order to smoothly interpolate between the latent vectors. This is useful if your model is too *jumpy* when reaching the different seeds, so use it as you please.

The results of the previous code along with these different parameters can be seen in the following tables:

| `--interp-in-z=True` |  | `--smooth` |
| :-: | :-: | :-: |
| `--interp-type=linear` | ![sightseeding-z-linear](./docs/gifs/sightseeding-linear_z.gif) | ![sightseeding-z-smooth](./docs/gifs/sightseeding-smooth-linear_z.gif) |
| `--interp-type=spherical` | ![sightseeding-z-spherical](./docs/gifs/sightseeding-spherical_z.gif) | ![sightseeding-z-spherical](./docs/gifs/sightseeding-smooth-spherical_z.gif) |

| `--interp-in-z=False` |  | `--smooth` |
| :-: | :-: | :-: |
| `--interp-type=linear` | ![sightseeding-w-linear](./docs/gifs/sightseeding-linear_w.gif) | ![sightseeding-w-smooth](./docs/gifs/sightseeding-smooth-linear_w.gif) |
| `--interp-type=spherical` | ![sightseeding-w-spherical](./docs/gifs/sightseeding-spherical_w.gif) | ![sightseeding-w-spherical](./docs/gifs/sightseeding-smooth-spherical_w.gif) |

As we can see, the transition between the seeds will be different depending on both of these settings, so you should play with these in order to get exactly what you want to show. In general, I've found that spherical interpolation in ***W*** is already smooth enough, but you can always use the `--smooth` whenever you please.

<a name="circular"></a>
### Circular interpolation

This is a crude version of a circular interpolation video inspired by [Lorenz Diener](https://github.com/halcy/stylegan/blob/master/Stylegan-Generate-Encode.ipynb). We do the following:

* Two dimensions `z1, z2` in ***Z*** will be chosen at random (defined by the `seed`), on which we will define a plane

* On it, we will then define a circle of radius `radius=10.0` centered at the origin. We partition it in equal strides via polar coordinates with the total number of frames `num_frames = int(np.rint(duration_sec * mp4_fps))` as steps. Hence, the longer the video length and higher the FPS, the smaller the step.

* We fill the entire latent space with zeros and then replace the values of the points in the circle with their Cartesian values, starting at `(10.0, 0.0)` and moving counter-clockwise.

    * Note that the value for the `radius` does not really matter, as we are only changing the value of *one* of the `512` dimensions in ***Z*** whilst keeping everything else as zero. I do not recommend changing this value, though other types of trajectories can be made, akin to Mario Klingemann's [Hyperdimensional Attractions](http://www.aiartonline.com/highlights/mario-klingemann-3/).

Using the [Wikiart model](https://github.com/justinpinkney/awesome-pretrained-stylegan2#WikiArt), we run the following:

```
python run_generator.py circular-video \
    --network=/path/to/wikiart.pkl \
    --seed=70 --grid-w=2 --grid-h=1 \
    --truncation-psi=0.7
```
We must also add how long the video will last, `--duration-sec`, though as usual the default value will be 30 seconds. I recommend using a low value first in order to see if the determined path is worthy of exploring. If it is, then using a larger video length would better reveal the details generated, again as long as it suits your needs.

| `--duration-sec=5.0` | `--duration-sec=20.0` | `--duration-sec=60.0` |
| :-: | :-: | :-: |
| ![circular-video1](./docs/gifs/2x1-circular-5s.gif) | ![circular-video2](./docs/gifs/2x1-circular-20s.gif) | ![circular-video3](./docs/gifs/2x1-circular-60s.gif) |

All in all, this is akin to the `--slowdown` parameter in the [lerp](#lerp) interpolation videos above, albeit in a much more controlled way as the path is set from the beginning.

   * **TODO:** add style transfer and optionally let the user decide in which plane to do generate the circle.

<a name="proj"></a>
## Recreating the Projection Videos

To generate your own projection videos [as in the official implementation](https://drive.google.com/open?id=1ZpEiQuUFm4XQRUxoJj3f4O8-f102iXJc), you must of course have to have already a projection in your `results` subdir! As a side note, I must put emphasis on the fact that **you can use either a trained model.pkl from StyleGAN or StyleGAN2**, so this projection code can be used for your *old* StyleGAN models as well, which I found useful as the majority of my work was done with StyleGAN.

For example, to project generated images by your trained model, run:

```
python run_projector.py project-generated-images \
    --network=/path/to/network.pkl \
    --num-snapshots=1000 \
    --seeds=....
```

where, if you know specific seeds that you wish to project, include it in the `--seeds` argument.

To project real images, these must be in a `tfrecord` file, so the easiest thing to do is to use the file you used to train your [StyleGAN](https://github.com/NVlabs/stylegan) or StyleGAN2 model.

Then, to project real images, run:

```
python run_projector.py project-real-images \
    --network=/path/to/network.pkl \
    --data-dir=/path/to/tfrecord/root/dir \
    --dataset=tfrecord_name \
    --num-snapshots=1000 \
    --num-images=N(as many as you wish/have the time)
```

Take heed that, if you run the above code, say, two times, with the first time setting `--num-images=5` and the second time setting `--num-images=10`, then the first run will be contained in the second, as they will have the same seed. As such, if you wish to project a specific set of real images from your training data, then simply convert these to a `tfrecord` file with `dataset_tool.py` and you don't have to worry about when you will get the specific image(s) you want to project.

Note that `--num-snapshots=1000` is required for the bash script below, as the final video will be of length of 20 seconds, and will run at 50 fps. Modifying this and in turn the length of the projection video is not so complicated, so feel free to so, but remember to modify **both** numbers.

So now, in the `stylegan2-fun` dir, you just have to run:

```
./projection_video.sh 1
```

if, for example, your Run 1 was a projection of real or generated images, i.e., there exists either `00001-project-generated-images` or `00001-project-real-images` in `results`. The result of this bash script will be that your images will be sorted in subdirectories in each run by either seed or real image number, like so:

```
./results/00001-project-real-images
├   _finished.txt
├───image0000
│   ├   image0000-projection.mp4
│   ├   image0000-step0001.png
│   ├   ...
│   └   image0000-step0999.png
├───image0001
│   ├   image0001-projection.mp4
│   ├   image0001-step0001.png
│   ├   ...
│   └   image0001-step0999.png
├───image0002
├───...
├───run.txt
├───submit_config.pkl
└───submit_config.txt
```

For each of these, a projection video will be generated which will have, to the right, the *Target image* (be it generated or real), and to the left, the progression of the projection at each step, up to iteration 1000 (hence why 1000 snapshots).

An example of this is the following, where we are projecting a center-cropped image of the [A2D2](https://www.audi-electronics-venture.de/aev/web/de/driving-dataset.html) dataset (right) into the latent space of the StyleGAN2 trained only on the [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset (left):

![projection-video](./docs/gifs/image0001-projection.gif)

Watch the full size video [here](https://youtu.be/9-CUDF07cEE). For my purposes, there was no need to preserve each individual video (the projection and the target image, `left.mp4` and `right.mp4` respectively in `projection_video.sh`), so they are deleted at the [end of the script](https://github.com/PDillis/stylegan2-fun/blob/a6d9840bb23702d9450e1dc0debcba6f4f50b218/projection_video.sh#L74). Remove this line if you wish to keep them. Furthermore, if you don't want any text to appear on the videos, remove the `-vf "drawtext=..."` flag from both lines [62](https://github.com/PDillis/stylegan2-fun/blob/a6d9840bb23702d9450e1dc0debcba6f4f50b218/projection_video.sh#L62) and [67](https://github.com/PDillis/stylegan2-fun/blob/a6d9840bb23702d9450e1dc0debcba6f4f50b218/projection_video.sh#L67).

<a name="mass_proj"></a>
## Mass Projector

Another tool that was useful for my purposes is `mass_projector.sh`. Given a directory with all the model checkpoints you wish to analyze, you can then project as many images as you wish per model checkpoint in order to do a comparison, such as calculating the PSNR, SSIM or MSE between the target final projected image.

***I truly have no idea if this will be useful for someone else, but it was for me, so I add it here.***

Usage:

```
./mass_projector.sh name-of-dataset /dataset/root/path /models/root/path N
```

where `name-of-dataset` will be the name of your dataset in your `/dataset/root/path` (the same terminology we use whilst projecting images or training the model), `/models/root/path` will be the path to the directory containing all the `pkl` files that you wish to project `N` images from the dataset.

Note that, by default, we will have `--num-snapshots=1`, as we are only interested in the final projection. As a side effect, this will speed up the projection by ~3x at least from my experience: from around 12 minutes to 4 minutes per image projected on an [NVIDIA 1080 GTX](https://www.geforce.com/hardware/desktop-gpus/geforce-gtx-1080/specifications).

<a name="save_proj"></a>
## Saving projected disentangled latent vectors

On the other hand, if you wish to save the latent vector that is obtained by the projection, I've modified the code to allow you to do this. You must also decide how many steps to take (`--num-steps`), how many snapshots to take of the process (`--num-snapshots`), and finally if you wish to save the disentangled latent vectors at every step (`--save-every-dlatent`) or just the final one (`--save-final-dlatent`). These disentangled latent vectores will be saved in the [`npy` format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format).

For example, for generated images:

```
python run_projector.py project-generated-images \
    --network=/path/to/network.pkl \
    --num-steps=100 --num-snapshots=5 \
    --seeds=0,3-10,99 --save-every-dlatent
```

This will produce, for each of the selected seeds (in this case, `0,3,4,5,6,7,8,9,10,99`), 5 different snapshots (images) at steps `20`, `40`, `60`, `80`, and `100`, along with saving the disentangled latent vector that produces the projected images in these snapshots. These vectors will be saved in the respective run in the following manner: `seed0000-step0020.npy`, `seed0000-step0040.npy`, etc. If instead of `--save-every-dlatent` you use `--save-final-dlatent`, then in the above case, only the `seed0000-step0100.npy` will be saved. These options are available for both generated and real images, so use them as you please.

For the generated images, you can also save the target disentangled latent vector by adding the `--save-target-dlatent` flag. This is of course not particularly essential, as you can always generate these disentangled vectors by the respective seed, but it's something that was useful for me and I hope it is useful for someone else.

Finally, using the same `pkl` file to load the `Gs` as in the projection, you can use these saved files to generate the image like so:

```python
dlatent = np.load('/results/00001-project-generated-images/seed0000-step0020.npy')

image = Gs.components.synthesis.run(dlatent, **Gs_syn_kwargs)

img = PIL.Image.fromarray(image, "RGB")
```

---

Henceforth it will be the official implementation of StyleGAN2, so please pay close attention to the official author's notes:

---
<a name="sgan2"></a>
## StyleGAN2 &mdash; Official TensorFlow Implementation

![Teaser image](./docs/stylegan2-teaser-1024x256.png)

**Analyzing and Improving the Image Quality of StyleGAN**<br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila<br>

Paper: http://arxiv.org/abs/1912.04958<br>
Video: https://youtu.be/c-NJtV9Jvp0<br>

Abstract: *The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent vectors to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably detect if an image is generated by a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.*

For business inquiries, please contact [researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com)<br>
For press and other inquiries, please contact Hector Marinez at [hmarinez@nvidia.com](mailto:hmarinez@nvidia.com)<br>

| Additional material | &nbsp;
| :--- | :----------
| [StyleGAN2](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7) | Main Google Drive folder
| &boxvr;&nbsp; [stylegan2-paper.pdf](https://drive.google.com/open?id=1fnF-QsiQeKaxF-HbvFiGtzHF_Bf3CzJu) | High-quality version of the paper
| &boxvr;&nbsp; [stylegan2-video.mp4](https://drive.google.com/open?id=1f_gbKW6FUUHKkUxciJ_lQx29mCq_fSBy) | High-quality version of the video
| &boxvr;&nbsp; [images](https://drive.google.com/open?id=1Sak157_DLX84ytqHHqZaH_59HoEWzfB7) | Example images produced using our method
| &boxv;&nbsp; &boxvr;&nbsp;  [curated-images](https://drive.google.com/open?id=1ydWb8xCHzDKMTW9kQ7sL-B1R0zATHVHp) | Hand-picked images showcasing our results
| &boxv;&nbsp; &boxur;&nbsp;  [100k-generated-images](https://drive.google.com/open?id=1BA2OZ1GshdfFZGYZPob5QWOGBuJCdu5q) | Random images with and without truncation
| &boxvr;&nbsp; [videos](https://drive.google.com/open?id=1yXDV96SFXoUiZKU7AyE6DyKgDpIk4wUZ) | Individual clips of the video as high-quality MP4
| &boxur;&nbsp; [networks](https://drive.google.com/open?id=1yanUI9m4b4PWzR0eurKNq6JR1Bbfbh6L) | Pre-trained networks
| &ensp;&ensp; &boxvr;&nbsp;  [stylegan2-ffhq-config-f.pkl](https://drive.google.com/open?id=1Mgh-jglZjgksupF0XLl0KzuOqd1LXcoE) | StyleGAN2 for <span style="font-variant:small-caps">FFHQ</span> dataset at 1024&times;1024
| &ensp;&ensp; &boxvr;&nbsp;  [stylegan2-car-config-f.pkl](https://drive.google.com/open?id=1MutzVf8XjNo6TUg03a6CUU_2Vlc0ltbV) | StyleGAN2 for <span style="font-variant:small-caps">LSUN Car</span> dataset at 512&times;384
| &ensp;&ensp; &boxvr;&nbsp;  [stylegan2-cat-config-f.pkl](https://drive.google.com/open?id=1MyowTZGvMDJCWuT7Yg2e_GnTLIzcSPCy) | StyleGAN2 for <span style="font-variant:small-caps">LSUN Cat</span> dataset at 256&times;256
| &ensp;&ensp; &boxvr;&nbsp;  [stylegan2-church-config-f.pkl](https://drive.google.com/open?id=1N3iaujGpwa6vmKCqRSHcD6GZ2HVV8h1f) | StyleGAN2 for <span style="font-variant:small-caps">LSUN Church</span> dataset at 256&times;256
| &ensp;&ensp; &boxvr;&nbsp;  [stylegan2-horse-config-f.pkl](https://drive.google.com/open?id=1N55ZtBhEyEbDn6uKBjCNAew1phD5ZAh-) | StyleGAN2 for <span style="font-variant:small-caps">LSUN Horse</span> dataset at 256&times;256
| &ensp;&ensp; &boxur;&nbsp;&#x22ef;  | Other training configurations used in the paper

<a name="requirements"></a>
## Requirements

* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* TensorFlow 1.14 or 1.15 with GPU support. The code does not support TensorFlow 2.0.
* On Windows, you need to use TensorFlow 1.14 &mdash; TensorFlow 1.15 will not work.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

StyleGAN2 relies on custom TensorFlow ops that are compiled on the fly using [NVCC](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). To test that your NVCC installation is working correctly, run:

```.bash
nvcc test_nvcc.cu -o test_nvcc -run
| CPU says hello.
| GPU says hello.
```

On Windows, the compilation requires Microsoft Visual Studio to be in `PATH`. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Preparing datasets

Datasets are stored as multi-resolution TFRecords, similar to the [original StyleGAN](https://github.com/NVlabs/stylegan). Each dataset consists of multiple `*.tfrecords` files stored under a common directory, e.g., `~/datasets/ffhq/ffhq-r*.tfrecords`. In the following sections, the datasets are referenced using a combination of `--dataset` and `--data-dir` arguments, e.g., `--dataset=ffhq --data-dir=~/datasets`.

**FFHQ**. To download the [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset) dataset as multi-resolution TFRecords, run:

```.bash
pushd ~
git clone https://github.com/NVlabs/ffhq-dataset.git
cd ffhq-dataset
python download_ffhq.py --tfrecords
popd
python dataset_tool.py display ~/ffhq-dataset/tfrecords/ffhq
```

**LSUN**. Download the desired LSUN categories in LMDB format from the [LSUN project page](https://www.yf.io/p/lsun). To convert the data to multi-resolution TFRecords, run:

```.bash
python dataset_tool.py create_lsun_wide ~/datasets/car ~/lsun/car_lmdb --width=512 --height=384
python dataset_tool.py create_lsun ~/datasets/cat ~/lsun/cat_lmdb --resolution=256
python dataset_tool.py create_lsun ~/datasets/church ~/lsun/church_outdoor_train_lmdb --resolution=256
python dataset_tool.py create_lsun ~/datasets/horse ~/lsun/horse_lmdb --resolution=256
```

**Custom**. Create custom datasets by placing all training images under a single directory. The images must be square-shaped and they must all have the same power-of-two dimensions. To convert the images to multi-resolution TFRecords, run:

```.bash
python dataset_tool.py create_from_images ~/datasets/my-custom-dataset ~/my-custom-images
python dataset_tool.py display ~/datasets/my-custom-dataset
```

## Using pre-trained networks

Pre-trained networks are stored as `*.pkl` files on the [StyleGAN2 Google Drive folder](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7). Below, you can either reference them directly using the syntax `gdrive:networks/<filename>.pkl`, or download them manually and reference by filename.

**Generating images**:

```.bash
# Generate uncurated ffhq images (matches paper Figure 12)
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=6600-6625 --truncation-psi=0.5

# Generate curated ffhq images (matches paper Figure 11)
python run_generator.py generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --seeds=66,230,389,1518 --truncation-psi=1.0

# Generate uncurated car images
python run_generator.py generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --seeds=6000-6025 --truncation-psi=0.5

# Example of style mixing (matches the corresponding video clip)
python run_generator.py style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
```

The results are placed in `results/<RUNNING_ID>/*.png`. You can change the location with `--result-dir`. For example, `--result-dir=~/my-stylegan2-results`.

**Projecting images to latent space**:

```.bash
# Project generated images
python run_projector.py project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --seeds=0,1,5

# Project real images
python run_projector.py project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --dataset=car --data-dir=~/datasets
```

You can import the networks in your own Python code using `pickle.load()`. For this to work, you need to include the `dnnlib` source directory in `PYTHONPATH` and create a default TensorFlow session by calling `dnnlib.tflib.init_tf()`. See [run_generator.py](./run_generator.py) and [pretrained_networks.py](./pretrained_networks.py) for examples.

## Training networks

To reproduce the training runs for config F in Tables 1 and 3, run:

```.bash
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=ffhq --mirror-augment=true
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=car --total-kimg=57000
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=cat --total-kimg=88000
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=church --total-kimg 88000 --gamma=100
python run_training.py --num-gpus=8 --data-dir=~/datasets --config=config-f \
  --dataset=horse --total-kimg 100000 --gamma=100
```

For other configurations, see `python run_training.py --help`.

We have verified that the results match the paper when training with 1, 2, 4, or 8 GPUs. Note that training FFHQ at 1024&times;1024 resolution requires GPU(s) with at least 16 GB of memory. The following table lists typical training times using NVIDIA DGX-1 with 8 Tesla V100 GPUs:

| Configuration | Resolution      | Total kimg | 1 GPU   | 2 GPUs  | 4 GPUs  | 8 GPUs | GPU mem |
| :------------ | :-------------: | :--------: | :-----: | :-----: | :-----: | :----: | :-----: |
| `config-f`    | 1024&times;1024 | 25000      | 69d 23h | 36d 4h  | 18d 14h | 9d 18h | 13.3 GB |
| `config-f`    | 1024&times;1024 | 10000      | 27d 23h | 14d 11h | 7d 10h  | 3d 22h | 13.3 GB |
| `config-e`    | 1024&times;1024 | 25000      | 35d 11h | 18d 15h | 9d 15h  | 5d 6h  | 8.6 GB  |
| `config-e`    | 1024&times;1024 | 10000      | 14d 4h  | 7d 11h  | 3d 20h  | 2d 3h  | 8.6 GB  |
| `config-f`    | 256&times;256   | 25000      | 32d 13h | 16d 23h | 8d 21h  | 4d 18h | 6.4 GB  |
| `config-f`    | 256&times;256   | 10000      | 13d 0h  | 6d 19h  | 3d 13h  | 1d 22h | 6.4 GB  |

Training curves for FFHQ config F (StyleGAN2) compared to original StyleGAN using 8 GPUs:

![Training curves](./docs/stylegan2-training-curves.png)

After training, the resulting networks can be used the same way as the official pre-trained networks:

```.bash
# Generate 1000 random images without truncation
python run_generator.py generate-images --seeds=0-999 --truncation-psi=1.0 \
  --network=results/00006-stylegan2-ffhq-8gpu-config-f/networks-final.pkl
```

## Evaluation metrics

To reproduce the numbers for config F in Tables 1 and 3, run:

```.bash
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-ffhq-config-f.pkl \
  --metrics=fid50k,ppl_wend --dataset=ffhq --mirror-augment=true
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-car-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=car
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-cat-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=cat
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-church-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=church
python run_metrics.py --data-dir=~/datasets --network=gdrive:networks/stylegan2-horse-config-f.pkl \
  --metrics=fid50k,ppl2_wend --dataset=horse
```

For other configurations, see the [StyleGAN2 Google Drive folder](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7).

Note that the metrics are evaluated using a different random seed each time, so the results will vary between runs. In the paper, we reported the average result of running each metric 10 times. The following table lists the available metrics along with their expected runtimes and random variation:

| Metric      | FFHQ config F  | 1 GPU  | 2 GPUs  | 4 GPUs | Description |
| :---------- | :------------: | :----: | :-----: | :----: | :---------- |
| `fid50k`    | 2.84 &pm; 0.03 | 22 min | 14 min  | 10 min | [Fr&eacute;chet Inception Distance](https://arxiv.org/abs/1706.08500)
| `is50k`     | 5.13 &pm; 0.02 | 23 min | 14 min  | 8 min  | [Inception Score](https://arxiv.org/abs/1606.03498)
| `ppl_zfull` | 348.0 &pm; 3.8 | 41 min | 22 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in Z, full paths
| `ppl_wfull` | 126.9 &pm; 0.2 | 42 min | 22 min  | 13 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in W, full paths
| `ppl_zend`  | 348.6 &pm; 3.0 | 41 min | 22 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in Z, path endpoints
| `ppl_wend`  | 129.4 &pm; 0.8 | 40 min | 23 min  | 13 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) in W, path endpoints
| `ppl2_wend` | 145.0 &pm; 0.5 | 41 min | 23 min  | 14 min | [Perceptual Path Length](https://arxiv.org/abs/1812.04948) without center crop
| `ls`        | 154.2 / 4.27   | 10 hrs | 6 hrs   | 4 hrs  | [Linear Separability](https://arxiv.org/abs/1812.04948)
| `pr50k3`    | 0.689 / 0.492  | 26 min | 17 min  | 12 min | [Precision and Recall](https://arxiv.org/abs/1904.06991)

Note that some of the metrics cache dataset-specific data on the disk, and they will take somewhat longer when run for the first time.

## License

Copyright &copy; 2019, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. To view a copy of this license, visit https://nvlabs.github.io/stylegan2/license.html

## Citation

```
@article{Karras2019stylegan2,
  title   = {Analyzing and Improving the Image Quality of {StyleGAN}},
  author  = {Tero Karras and Samuli Laine and Miika Aittala and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  journal = {CoRR},
  volume  = {abs/1912.04958},
  year    = {2019},
}
```

## Acknowledgements

We thank Ming-Yu Liu for an early review, Timo Viitanen for his help with code release, and Tero Kuosmanen for compute infrastructure.
