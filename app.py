# Import the dependencies
import gradio as gr
from PIL import Image
import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt


# Load the SAM model and processor
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-77")
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-77")


# Global variable to store input points
input_points = []

# Helper functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3),
                                np.array([0.6])],
                               axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
# Function to get pixel coordinates
def get_pixel_coordinates(image, evt: gr.SelectData):
    global input_points
    x, y = evt.index[0], evt.index[1]
    input_points = [[[x, y]]]
    return perform_prediction(image)

# Function to perform SAM model prediction
def perform_prediction(image):
    global input_points
    # Preprocess the image
    inputs = processor(images=image, input_points=input_points, return_tensors="pt")
    # Perform prediction
    with torch.no_grad():
        outputs = model(**inputs)
    iou = outputs.iou_scores
    max_iou_index = torch.argmax(iou)

    # Post-process the masks
    predicted_masks = processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )
    predicted_mask = predicted_masks[0]

    # Display the mask on the image
    mask_image = show_mask_on_image(image, predicted_mask[:,max_iou_index], return_image=True)
    return mask_image

# Function to overlay mask on the image
def show_mask_on_image(raw_image, mask, return_image=False):
    if not isinstance(mask, torch.Tensor):
        mask = torch.Tensor(mask)

    if len(mask.shape) == 4:
        mask = mask.squeeze()

    fig, axes = plt.subplots(1, 1, figsize=(15, 15))

    mask = mask.cpu().detach()
    axes.imshow(np.array(raw_image))
    show_mask(mask, axes)
    axes.axis("off")
    plt.show()

    if return_image:
        fig = plt.gcf()
        fig.canvas.draw()
        # Convert plot to image
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = Image.fromarray(img)
        plt.close(fig)
        return img



# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style='text-align: center; font-family: "Times New Roman";'>
            <h1 style='color: #FF6347;'>One Click Image Segmentation App</h1>
            <h3 style='color: #4682B4;'>Model: SlimSAM-uniform-77</h3>
            <h3 style='color: #32CD32;'>Made By: Md. Mahmudun Nabi</h3>
        </div>
        """
    )
    with gr.Row():
          
        img = gr.Image(type="pil", label="Input Image",height=400, width=600)
        output_image = gr.Image(label="Masked Image")

    img.select(get_pixel_coordinates, inputs=[img], outputs=[output_image])


    if __name__ == "__main__":
        demo.launch(share=False)