from efficient_sam.build_efficient_sam import build_efficient_sam_vits
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import torch_tensorrt

if __name__ == "__main__":

    model = build_efficient_sam_vits()
    model.to(torch.device("cuda"))
    
    sample_image_np = np.array(Image.open("figs/examples/dogs.jpg"))
    sample_image_tensor = transforms.ToTensor()(sample_image_np).to(torch.device("cuda"))

    input_points = torch.tensor([[[[580, 350], [650, 350]]]], dtype=torch.float32).to(torch.device("cuda"))
    input_labels = torch.tensor([[[1, 1]]], dtype=torch.float32).to(torch.device("cuda"))
    print('Running single inference for testing')
    

    input_points = torch.tensor([[[[580, 350], [650, 350]]]]).to(torch.device("cuda"))
    input_labels = torch.tensor([[[1, 1]]]).to(torch.device("cuda"))

    predicted_logits, predicted_iou = model(
        sample_image_tensor[None, ...],
        input_points,
        input_labels,
    )
    mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()
    masked_image_np = sample_image_np.copy().astype(np.uint8) * mask[:,:,None]
    Image.fromarray(masked_image_np).save(f"figs/examples/dogs_mask.png")
    print('Test image saved successfully')
    
    print('Compiling model...')
    with torch_tensorrt.logging.graphs():
        trt_model = torch_tensorrt.compile(model, 
            inputs= [sample_image_tensor[None, ...],
                     input_points,
                     input_labels],
            enabled_precisions= {torch.float32},
            workspace_size=2000000000,
            truncate_long_and_double=True,                                
        )
        
            # capture_dynamic_output_shape_ops=True,
            # ir="dynamo",
            # dynamic=False
        print('Successfully compiled model', trt_model)
