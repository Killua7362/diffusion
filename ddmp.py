import os 
import torch
import torch.nn as nn 
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import copy
import logging
from torch.utils.tensorboard import SummaryWriter
from utils import setup_logging,get_data
from modules import UNet_Conditional,EMA
import numpy as np

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",level=logging.INFO,datefmt='%I:%M:%S')

class Diffusion:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=0.02,img_size=64,device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end= beta_end
        self.img_size = img_size
        self.device = device
        
        self.beta = self.prepare_noise_schedule().to(device=device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)
    
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)
    
    def noise_images(self,x,t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.-self.alpha_hat[t])[:,None,None,None]
        epsilon = torch.rand_like(x)
        return sqrt_alpha_hat * x +  sqrt_one_minus_alpha_hat * epsilon,epsilon
    
    def sample_time_steps(self,n):
        return torch.randint(low=1.,high=self.noise_steps,size=(n,))
    
    def sample(self,model,n,labels,cfg_scale=3):
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randint((n,3,self.img_size,self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1,self.noise_steps)),position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x,t,labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x,t,None)
                    predicted_noise = torch.lerp(uncond_predicted_noise,predicted_noise,cfg_scale)
                alpha = self.alpha[t][:,None,None,None]
                alpha_hat = self.alpha_hat[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                if i>1:
                    noise = torch.rand_like(x)
                else:
                    noise = torch. zeros_like(x)
                x = 1/torch.sqrt(alpha) * (x-((1-alpha)/(torch.sqrt(1-alpha_hat))) * predicted_noise) +torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1,1)+1)/2
        x = (x*255).type(torch.uint8)
        return x 
    
def train(args):
    setup_logging(args.run_name)
    device = args.device 
    dataloader = get_data(args)
    model = UNet_Conditional(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    mse=nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size,device=device)
    logger = SummaryWriter(os.path.join("runs",args.run_name))
    l = len(dataloader)
    ema = EMA(beta=0.99)
    ema_model = copy.deepcopy(model).eval().requires_grad(False)
    
    for epoch in range(args.epochs):
        logging.info(f'Starting epoch {epoch}')
        pbar = tqdm(dataloader)
        for i ,(images,labels) in enumerate(pbar):
            
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_time_steps(images.shape[0]).to(device)
            x_t,noise = diffusion.noise_images(images,t)
            
            if np.random.random() < 0.1:
                labels = None
                
            predicted_noise = model(x_t,t,labels)
            loss = mse(noise,predicted_noise)
            optimizer.zero_grad()
            loss.bacward()
            optimizer.step()
            ema.step_ema(ema_model,model)