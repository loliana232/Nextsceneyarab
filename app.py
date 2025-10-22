import gradio as gr
import numpy as np
import random
import torch
import spaces

from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler
from optimization import optimize_pipeline_
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

from huggingface_hub import InferenceClient
import math
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

import os
import base64
import json

# System prompt for VLM to suggest next scene prompts
NEXT_SCENE_SUGGESTION_PROMPT = '''
# Next Scene Prompt Generator
You are a cinematic visual continuity expert. Your task is to analyze the provided image and generate a professional "Next Scene" prompt that maintains narrative coherence while introducing organic transitions.

## Analysis Framework:
1. Identify the current scene's key elements:
   - Subject(s) and their positions
   - Camera angle and framing (close-up, medium, wide, etc.)
   - Environment and setting
   - Lighting and atmosphere
   - Mood and emotional tone

2. Generate a "Next Scene" prompt following these principles:
   - Begin with specific camera direction (dolly, push, pull, track, pan, tilt, etc.)
   - Describe the transition type:
     * Camera movement: Dolly shots, push-ins, pull-backs, tracking moves
     * Framing evolution: Wide to close-up transitions, angle shifts, reframing
     * Environmental reveals: New characters entering frame, expanded scenery, spatial progression
     * Atmospheric shifts: Lighting changes, weather evolution, time-of-day transitions
   - Maintain compositional coherence with the original scene
   - Include specific visual details about what changes and what remains

## Output Format:
Generate a single, concise prompt starting with "Next Scene:" followed by the camera movement and visual description.

## Examples of Good Prompts:
- "Next Scene: The camera pulls back from a tight close-up on the airship to a sweeping aerial view, revealing an entire fleet of vessels soaring through a fantasy landscape."
- "Next Scene: The camera tracks forward and tilts down, bringing the sun and helicopters closer into frame as a strong lens flare intensifies."
- "Next Scene: The camera pans right, removing the dragon and rider from view while revealing more of the floating mountain range in the distance."
- "Next Scene: The camera dollies in slowly while the morning fog begins to lift, revealing hidden architectural details in the background as golden sunlight breaks through."
- "Next Scene: A smooth tracking shot follows the subject as they move left, with the background shifting to reveal a previously unseen cityscape bathed in twilight."

Return only the prompt text, no additional explanation.
'''

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  
Please strictly follow the rewriting rules below:
## 1. General Principles
- Keep the rewritten prompt **concise and comprehensive**. Avoid overly long sentences and unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the scene in the input images.  
- If multiple sub-images are to be generated, describe the content of each sub-image individually.  
## 2. Task-Type Handling Rules
### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  
### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.  
- Both adding new text and replacing existing text are text replacement tasks, For example:  
    - Replace "xx" to "yy"  
    - Replace the mask / bounding box to "yy"  
    - Replace the visual object to "yy"  
- Specify text position, color, and layout only if user has required.  
- If font is specified, keep the original language of the font.  
### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.  
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person's eyebrows with little change, look natural."
### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, vibrant colors"  
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- **Colorization tasks (including old photo restoration) must use the fixed template:**  
  "Restore and colorize the old photo."  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.  
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.
### 5. Material Replacement
- Clearly specify the object and the material. For example: "Change the material of the apple to papercut style."
- For text material replacement, use the fixed template:
    "Change the material of text "xxxx" to laser style"
### 6. Logo/Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible. For example:
   > Original: "Convert to sapphire material"  
   > Rewritten: "Convert the main subject in the image to sapphire material, preserving similar shape and structure"
- When migrating logos/patterns to new scenes, ensure shape and structure consistency. For example:
   > Original: "Migrate the logo in the image to a new scene"  
   > Rewritten: "Migrate the logo in the image to a new scene, preserving similar shape and structure"
### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"  
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.  
## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).
# Output Format Example
```json
{
   "Rewritten": "..."
}
'''

# --- Function to suggest next scene prompts using VLM ---
def suggest_next_scene_prompt(images):
    """
    Generates a cinematic next scene prompt suggestion using the VLM.
    """
    if not images or len(images) == 0:
        return ""
    
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Cannot generate prompt suggestion.")
        return ""
    
    try:
        # Get the first image (or last if using output)
        if isinstance(images[0], tuple):
            img = images[0][0]
        else:
            img = images[0]
        
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            return ""
        
        # Initialize the client
        client = InferenceClient(
            provider="cerebras",
            api_key=api_key,
        )
        
        # Format the messages for the chat completions API
        sys_prompt = "You are a cinematic visual continuity expert. Generate next scene prompts that follow professional cinematography principles."
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [
                {"image": f"data:image/png;base64,{encode_image(img)}"},
                {"text": NEXT_SCENE_SUGGESTION_PROMPT}
            ]}
        ]
        
        # Call the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages,
            temperature=0.7,  # Add some creativity
            max_tokens=150
        )
        
        # Parse the response
        suggestion = completion.choices[0].message.content
        
        # Clean up the suggestion
        suggestion = suggestion.strip()
        
        # Ensure it starts with "Next Scene:" if not already
        if not suggestion.startswith("Next Scene:"):
            suggestion = f"Next Scene: {suggestion}"
        
        return suggestion
        
    except Exception as e:
        print(f"Error generating prompt suggestion: {e}")
        return ""

# --- Prompt Enhancement using Hugging Face InferenceClient ---
def polish_prompt_hf(prompt, img_list):
    """
    Rewrites the prompt using a Hugging Face InferenceClient.
    """
    # Ensure HF_TOKEN is set
    api_key = os.environ.get("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN not set. Falling back to original prompt.")
        return prompt

    try:
        # Initialize the client
        prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {prompt}\n\nRewritten Prompt:"
        client = InferenceClient(
            provider="cerebras",
            api_key=api_key,
        )

        # Format the messages for the chat completions API
        sys_promot = "you are a helpful assistant, you should provide useful answers to users."
        messages = [
            {"role": "system", "content": sys_promot},
            {"role": "user", "content": []}]
        for img in img_list:
            messages[1]["content"].append(
                {"image": f"data:image/png;base64,{encode_image(img)}"})
        messages[1]["content"].append({"text": f"{prompt}"})

        # Call the API
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507",
            messages=messages,
        )
        
        # Parse the response
        result = completion.choices[0].message.content
        
        # Try to extract JSON if present
        if '{"Rewritten"' in result:
            try:
                # Clean up the response
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except:
                polished_prompt = result
        else:
            polished_prompt = result
            
        polished_prompt = polished_prompt.strip().replace("\n", " ")
        return polished_prompt
        
    except Exception as e:
        print(f"Error during API call to Hugging Face: {e}")
        # Fallback to original prompt if enhancement fails
        return prompt
    


def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", 
                                                 # scheduler=scheduler,
                                                 torch_dtype=dtype).to(device)
weights_path = hf_hub_download(
    repo_id="linoyts/Qwen-Image-Edit-Rapid-AIO",
    filename="transformer/transformer_weights.safetensors",
    repo_type="model"
)
state_dict = load_file(weights_path)

# load next scene LoRA 
pipe.transformer.load_state_dict(state_dict, strict=False)
pipe.load_lora_weights(
        "lovis93/next-scene-qwen-image-lora-2509", 
        weight_name="next-scene_lora-v2-3000.safetensors", adapter_name="next-scene"
    )
pipe.set_adapters(["next-scene"], adapter_weights=[1.])
pipe.fuse_lora(adapter_names=["next-scene"], lora_scale=1.)
pipe.unload_lora_weights()


# Apply the same optimizations from the first version
pipe.transformer.__class__ = QwenImageTransformer2DModel
pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# --- Ahead-of-time compilation ---
optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(gallery):
    """Copy the output image to input for iterative editing"""
    if gallery and len(gallery) > 0:
        return gallery
    return None

@spaces.GPU(duration=12)
def infer(
    images,
    prompt,
    seed,
    randomize_seed,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generates an image using the local Qwen-Image diffusers pipeline.
    """
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load input images into PIL Images
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item[0], str):
                    pil_images.append(Image.open(item[0]).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception:
                continue

    if height==256 and width==256:
        height, width = None, None
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    if rewrite_prompt and len(pil_images) > 0:
        prompt = polish_prompt_hf(prompt, pil_images)
        print(f"Rewritten Prompt: {prompt}")
    

    # Generate the image
    image = pipe(
        image=pil_images if len(pil_images) > 0 else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    # Return images, seed, and make button visible
    return image, seed, gr.update(visible=True)

# --- Function to handle prompt suggestion button ---
def on_suggest_prompt(images, current_prompt):
    """
    Generates a next scene prompt suggestion when the button is clicked.
    """
    if not images or len(images) == 0:
        return current_prompt, gr.update(visible=False)
    
    suggestion = suggest_next_scene_prompt(images)
    
    if suggestion:
        return suggestion, gr.update(visible=True, value="✨ Suggestion generated!")
    else:
        return current_prompt, gr.update(visible=True, value="❌ Could not generate suggestion")

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#logo-title {
    text-align: center;
}
#logo-title img {
    width: 400px;
}
#edit_text{margin-top: -62px !important}
.suggest-btn {
    background: linear-gradient(90deg, #5b47d1 0%, #7c6fe1 100%);
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Edit Logo" width="400" style="display: block; margin: 0 auto;">
            <h2 style="font-style: italic;color: #5b47d1;margin-top: -27px !important;margin-left: 96px">[Plus] Fast, 4-steps with Qwen Rapid AIO + Next Scene LoRA</h2>
        </div>
        """)
        gr.Markdown("""
        **Enhanced with Next Scene LoRA for cinematic continuity!** 🎬
        
        [Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. 
        This demo uses [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) with [Next Scene LoRA](https://huggingface.co/lovis93/next-scene-qwen-image-lora-2509) for cinematic storytelling.
        
        **New Feature:** Click "🎬 Suggest Next Scene" to get AI-generated cinematic prompts that maintain visual continuity!
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)
                
                # Add the suggest prompt button below input images
                suggest_btn = gr.Button("🎬 Suggest Next Scene Prompt", 
                                       variant="primary", 
                                       elem_classes=["suggest-btn"])
                suggest_status = gr.Textbox(visible=False, 
                                           label="", 
                                           interactive=False,
                                           max_lines=1)

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil")
                # Add this button right after the result gallery - initially hidden
                use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction (or click 'Suggest Next Scene' for AI suggestions)",
                    container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=4,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                
                rewrite_prompt = gr.Checkbox(label="Enhance prompt (recommended)", value=True)

        gr.Markdown("""
        ### 🎬 Next Scene Tips:
        - **Camera Movements**: Prompts will include dolly shots, push-ins, pull-backs, tracking moves
        - **Framing Evolution**: Wide to close-up transitions, angle shifts, reframing
        - **Environmental Reveals**: New characters, expanded scenery, spatial progression
        - **Atmospheric Shifts**: Lighting changes, weather evolution, time transitions
        
        **Pro tip**: Chain multiple generations to create cinematic storyboards!
        """)

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    # Event handler for suggest prompt button
    suggest_btn.click(
        fn=on_suggest_prompt,
        inputs=[input_images, prompt],
        outputs=[prompt, suggest_status]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[suggest_status],
        _js="() => setTimeout(() => {}, 3000)"  # Hide status after 3 seconds
    )
    
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
            rewrite_prompt,
        ],
        outputs=[result, seed, use_output_btn],  # Added use_output_btn to outputs
    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )

if __name__ == "__main__":
    demo.launch()