import torch
#from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from IPython.display import display
from utils import compute_ca_loss, Pharse2idx, draw_box, save_image
import hydra
import os
from tqdm import tqdm

def inference(device, unet, vae, tokenizer, text_encoder, cfg):
    
    prompt=cfg.input.prompt
    phrases=cfg.input.phrases
    bboxes=cfg.input.bboxes

    #get obj positions [[1,2],[5]]
    object_positions=Pharse2idx(prompt, phrases)

    #embed uncon
    uncond_input = tokenizer(
            [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    #embed prompt
    input_ids = tokenizer(
            [prompt] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]

    #text embed: concat
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    #seed
    generator = torch.manual_seed(cfg.inference.rand_seed) 

    #init latents
    latents = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    #init scheduler
    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss=torch.tensor(1000)#large enough to overcome the while condition in the 1st iter

    for index, t in enumerate(tqdm(noise_scheduler.timestep)):
        iteration=0
    
        #while loss in range
        while loss.item/cfg.inference.loss_scale > cfg.inference.loss_threshold \
        and iteration < cfg.inference.max_iter \
        and index < cfg.inference.max_index_step:
            
            latents=latents.requires_grad_(True)
            latent_model_input=latents
            latent_model_input=noise_scheduler.scale_model_input(latent_model_input, t)#match the algorithm

            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)#unet(con_embed)
            
            loss=compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes, \
                                 object_positions=object_positions)*cfg.inference.loss_scale
            
            grad_cond=torch.autograd.grad(loss.requires_grad_(True), [latents])[0] ##[latents]
            #the gradient of latents will be returned

            #update latent by grad_cond
            latents=latents-grad_cond * noise_scheduler.sigmas[index]**2###?

            iteration+=1
            torch.cuda.empty_cache()
#update latent end

        with torch.no_grad():
            latent_model_input=torch.cat([latents]*2)#re-assign
            latent_model_input=noise_scheduler.scale_model_input(latent_model_input,t)

            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
            
            noise_pred = noise_pred.sample

            u,p=noise_pred.chunk(2)
            noise_pred=u+cfg.inference.classifier_free_guidance(p-u)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample#based on the noise pred, output prev sample(clearer one)
            torch.cuda.empty_cache()

    with torch.no_grad():
        #decode image
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images



 

@hydra.main(version_base=None, config_path="conf", config_name="base_config")
def main(cfg):
    with open(cfg.general.unet_config) as f:
        unet_config=json.load(f)
    unet=unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    #**pass the dict as input
    
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    ##my input is in cfg
 
    #not save the cfg

    #inference
    pil_images=inference(device, unet, vae, tokenizer, text_encoder, cfg)
    #prompt, bboxes, phrases are all in cfg

    #show example images
    display(*pil_images)

if __name__=="__main__":
    main()