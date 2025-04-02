import os
import re
import time
import argparse
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import cv2 as cv
from diffusers import AutoencoderTiny
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from quantization import QParam

VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)

@torch.no_grad()
def generation(
        model,
        sampler,
        decoder,
        rank,
        interval,
        frame_path,
        H=512,
        W=512,
        seed=0,
        filter=None
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)
    uc = None

    # Set up metrics.
    loss_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').cuda()
    lpips_list = []

    # Set the result and log paths.
    prompt_dir = os.path.join(frame_path, 'results/rank{}_interval{}/'.format(rank, interval))
    log_output = open(os.path.join(prompt_dir, 'log.txt'), 'a')

    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        def generate(randn, c, gt, idx):
            # Generating a frame from the prompt
            samples_z = sampler(denoiser, randn, cond=c, uc=uc)
            img = decoder(samples_z)

            # Calculate and log the metric.
            img = torch.clamp(img, min=-1.0, max=1.0)
            lpips_value = loss_lpips(img, gt).item()
            print('frame: {}, lpips:{}'.format(idx, lpips_value))
            log_output.write('frame: {}, lpips:{}\n'.format(idx, lpips_value))
            log_output.flush()
            lpips_list.append(lpips_value)

            # Save the generate frame.
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
            img = (
                (255 * img)
                .to(dtype=torch.uint8)
                .permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
            )
            img = img[0][:, :, ::-1]
            cv2.imwrite(os.path.join(prompt_dir, '{:05d}.png'.format(idx)), img)

            return samples_z

        # The random seed needs to be consistent with the inversion.
        rand_noise = seeded_randn(shape, seed)
        sigma = torch.Tensor([0.05]).float().cuda()
        prev_frame = None

        prompts = sorted(glob(os.path.join(prompt_dir, 'frame_*.prompt')))
        for prompt_pair in zip(prompts[::], prompts[1::]):
            prompt_curr = prompt_pair[0]
            id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
            prompt_next = prompt_pair[1]
            id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))
            prompt_curr = torch.load(prompt_curr)
            prompt_next = torch.load(prompt_next)

            # low-rank factors
            U_curr, V_curr, U_next, V_next = prompt_curr['U'], prompt_curr['V'], prompt_next['U'], prompt_next['V']

            # prompt dequantization
            Quant_Param_U_curr = QParam(num_bits=8)
            Quant_Param_U_curr.scale = prompt_curr['U_scale']
            Quant_Param_U_curr.zero_point = prompt_curr['U_zero_point']
            U_curr = Quant_Param_U_curr.dequantize_tensor(U_curr)
            Quant_Param_V_curr = QParam(num_bits=8)
            Quant_Param_V_curr.scale = prompt_curr['V_scale']
            Quant_Param_V_curr.zero_point = prompt_curr['V_zero_point']
            V_curr = Quant_Param_V_curr.dequantize_tensor(V_curr)

            Quant_Param_U_next = QParam(num_bits=8)
            Quant_Param_U_next.scale = prompt_next['U_scale']
            Quant_Param_U_next.zero_point = prompt_next['U_zero_point']
            U_next = Quant_Param_U_next.dequantize_tensor(U_next)
            Quant_Param_V_next = QParam(num_bits=8)
            Quant_Param_V_next.scale = prompt_next['V_scale']
            Quant_Param_V_next.zero_point = prompt_next['V_zero_point']
            V_next = Quant_Param_V_next.dequantize_tensor(V_next)

            if prev_frame is None:
                # initialize for the first frame
                prev_frame = torch.load(os.path.join(prompt_dir, 'init.pth'))
                z = (prev_frame * sigma + rand_noise * (1 - sigma))
                c = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(dim=0)
                prompt = {'crossattn': c}
                gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(id_curr)))
                prev_frame = generate(randn=z, c=prompt, gt=gt, idx=id_curr)

            z = (prev_frame * sigma + rand_noise * (1 - sigma))
            # linear interpolation on keyframe prompts, approximating the intermediate prompts.
            for step in range(1, interval + 1):
                factor = 1 / interval
                u = (1 - step * factor) * U_curr + (step * factor) * U_next
                v = (1 - step * factor) * V_curr + (step * factor) * V_next
                # prompt composition
                c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
                prompt = {'crossattn': c}
                gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(id_curr+step)))
                prev_frame = generate(randn=z, c=prompt, gt=gt, idx=id_curr+step)

        print('mean lpips: {}'.format(np.mean(lpips_list)))
        log_output.write('mean lpips: {}\n'.format(np.mean(lpips_list)))
        log_output.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-rank', type=int, default="8")
    parser.add_argument('-interval', type=int, default="10")
    args = parser.parse_args()

    # Set up and load the models.
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).cuda()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)

    # generating frames from prompts
    generation(
       model, sampler, decoder=taesd.decoder, rank=args.rank, interval=args.interval, frame_path=args.frame_path, H=512, W=512, seed=seed_, filter=state.get("filter")
    )
