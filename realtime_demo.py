import time
import torch
import numpy as np
import cv2 as cv
import os
import re
import glob
import argparse
from quantization import QParam
from polygraphy import cuda
from tensorrt_acceleration import Engine

## Real-time generation engine, from prompt to video.
class Generator():
    def __init__(self, batch=1, device='cuda:0'):
        self.cuda_stream = cuda.Stream()
        self.use_cuda_graph = False
        self.batch = batch # the number of frames generated at once, needs to match the engine file.
        self.device = device
        self.noise = self.seeded_randn(shape=(1,4,64,64), seed=88) # the random seed needs to be consistent with the inversion.
        self.sigma = torch.Tensor([0.05]).float().cuda()
        self.prev_frame = None
        self.timesteps = torch.Tensor([999]).long().to(self.device)
        self.timesteps = self.timesteps.repeat(self.batch)
        self.c_in = torch.Tensor([[[[0.0683]]]]).float().to(self.device)
        self.c_out = torch.Tensor([[[[-14.6146]]]]).float().to(self.device)
        self.denoise_engine= None
        self.decoder_engine = None
        self.denoise_engine_load()
        self.decoder_engine_load()
    def denoise_engine_load(self):
        self.denoise_engine = Engine('engine/denoise_batch_{}.engine'.format(self.batch))
        self.denoise_engine.load()
        self.denoise_engine.activate()
        self.denoise_engine.allocate_buffers(
            shape_dict={
                "x": [self.batch, 4, 64, 64],
                "timesteps": [self.batch],
                "context": [self.batch, 77, 1024],
            },
            device=self.device,
        )

    def decoder_engine_load(self):
        self.decoder_engine = Engine('engine/decoder_batch_{}.engine'.format(self.batch))
        self.decoder_engine.load()
        self.decoder_engine.activate()
        self.decoder_engine.allocate_buffers(
            shape_dict={
                "latent": [self.batch, 4, 64, 64],
            },
            device=self.device,
        )

    def run_denoise_engine(self, x, timesteps, context):
        output = self.denoise_engine.infer(
            {
                "x": x,
                "timesteps": timesteps,
                "context": context,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        return output['out']

    def run_decoder_engine(self, latent):
        output = self.decoder_engine.infer(
            {
                "latent": latent,
            },
            self.cuda_stream,
            use_cuda_graph=self.use_cuda_graph,
        )
        return output['images']

    def normalization(self, images_tensor, to_numpy=False):
        images_tensor = torch.clamp((images_tensor + 1.0) / 2.0, min=0.0, max=1.0)
        images_tensor = images_tensor.permute(0, 2, 3, 1)
        images_tensor = (images_tensor * 255).byte()
        if to_numpy:
            return images_tensor.detach().cpu().numpy()
        else:
            return images_tensor.detach()

    def seeded_randn(self, shape, seed):
        randn = np.random.RandomState(seed).randn(*shape)
        randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
        return randn

    def add_noise(self, prev_frame):
        noised_prev_frame = (prev_frame * self.sigma + self.noise * (1 - self.sigma)).detach()
        noised_prev_frame = noised_prev_frame.repeat(batch,1,1,1)
        return noised_prev_frame

    def generate(self, cond):
        z = self.add_noise(self.prev_frame)
        latent_noise = self.run_denoise_engine(z, self.timesteps, cond)
        latent = latent_noise * self.c_out + z / self.c_in
        images = self.run_decoder_engine(latent)
        self.prev_frame = latent[-1:,...]
        return images

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-prompt_dir', type=str, default="data/sky/results/rank8_interval10")
    parser.add_argument('-batch', type=int, default="10")
    parser.add_argument('-visualize', type=bool, default=True)
    args = parser.parse_args()

    prompt_dir = args.prompt_dir
    batch = args.batch
    Generator_RT = Generator(batch=batch)
    result_dir = prompt_dir
    result_frames = None
    speed_warm_up = False
    generation_speed = []
    prompts = sorted(glob.glob(os.path.join(prompt_dir, 'frame_*.prompt')))
    for prompt_pair in zip(prompts[::], prompts[1::]):
        prompt_curr = prompt_pair[0]
        id_curr = int(re.search(r'frame_(\d{5})\.prompt', prompt_curr).group(1))
        prompt_next = prompt_pair[1]
        id_next = int(re.search(r'frame_(\d{5})\.prompt', prompt_next).group(1))
        # interpolation interval
        interval = id_next - id_curr

        prompt_curr = torch.load(prompt_curr, weights_only=True)
        prompt_next = torch.load(prompt_next, weights_only=True)
        # low-rank factors
        U_curr, V_curr, U_next, V_next = prompt_curr['U'], prompt_curr['V'], prompt_next['U'], prompt_next['V']

        torch.cuda.synchronize()
        t_begin = time.time()
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

        rank = U_curr.shape[1]
        prompt = []
        # linear interpolation on keyframe prompts, approximating the intermediate prompts.
        for step in range(1, interval + 1):
            factor = 1 / interval
            u = (1 - step * factor) * U_curr + (step * factor) * U_next
            v = (1 - step * factor) * V_curr + (step * factor) * V_next
            # prompt composition
            c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
            prompt.append(c)
        prompt = torch.concatenate(prompt, dim=0)
        # generating frames from prompts
        if Generator_RT.prev_frame is None:
            # initialize for the first frame
            Generator_RT.prev_frame = torch.load(os.path.join(prompt_dir, 'init.pth'), weights_only=True)
            c0 = (U_curr @ V_curr / np.sqrt(rank)).unsqueeze(dim=0)
            images = Generator_RT.generate(c0)
            images = Generator_RT.normalization(images, to_numpy=True)
            result_frames = images[0:1,...]
        images = Generator_RT.generate(prompt)
        images = Generator_RT.normalization(images, to_numpy=True)
        torch.cuda.synchronize()
        t_end = time.time()
        if speed_warm_up:
            # generation speed in FPS
            fps = 1 / (t_end - t_begin) * batch
            generation_speed.append(fps)
            print('Generation Speed: {} FPS'.format(fps))
        else:
            # the first batch is used for warming up
            speed_warm_up = True
        result_frames = np.append(result_frames, images, axis=0)

    average_speed = int(np.mean(generation_speed))
    for i in range(result_frames.shape[0]):
        image = result_frames[i,...]
        image = image[:, :, ::-1]
        image = np.ascontiguousarray(image)
        # Save the generated frames
        cv.imwrite(os.path.join(result_dir, '{:05d}.png'.format(i)), image)
        if args.visualize:
            # Visualize the generated frames
            cv.putText(image, 'Generation Speed {} FPS'.format(average_speed), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv.imshow('Real-time Generation', image)
            cv.waitKey(10)



