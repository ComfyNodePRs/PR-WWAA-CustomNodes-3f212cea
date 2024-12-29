# WWAA-CustomNodes
Custom Nodes for ComfyUI made by the team at [WeirdWonderfulAI.Art](https://weirdwonderfulai.art)
These are developed based on the needs where there was a gap to make our workflows better. You are welcome to use it as you see fit.

[![logo](https://weirdwonderfulai.art/wp-content/uploads/2022/01/WWAA_web_logo.jpg "WeirdWonderfulAI.Art")](https://weirdwonderfulai.art)

## Line Count
Custom node that takes a string list as input and will output text lines found within as Integer. It will remove blank lines from the final count.
![wwaa-Line-count](https://github.com/user-attachments/assets/9117cc3f-63ed-4b1d-9747-47fbc50c2fee)

## Join String
Custom node that can take a string value and add pre & post text in one go to produce a full joined string. I created this to allow me to dynamically cycle through many LoRA and create different images.
![JoinString node](https://github.com/user-attachments/assets/df486621-a12b-4bd9-82f9-cb7cdffac4aa)

## Dithering Node
Taking some of the Dithering Algorithms and made a node that allows you generate the cool effects with bunch of options to tweak.
![dithering node](https://github.com/user-attachments/assets/8f68f4f2-092b-4b4f-80fa-7b60d79bf648)

## Image Batch Loader

A custom node for ComfyUI that enables sequential loading of images from a directory with advanced sorting and control options. The node maintains its state between executions, allowing for automatic incremental loading of images in a controlled manner.

### Features

- **Sequential Image Loading**: Automatically increment through images in a directory
- **Multiple File Type Support**: Filter images by extension (PNG, JPG, JPEG, or ALL)
- **Advanced Sorting Options**:
  - Numerical (natural sort for numbered filenames)
  - Alphabetical
  - Creation time
  - Modification time
- **GPU/CUDA Support**: Automatic GPU acceleration when available
- **Directory Reload Control**: Option to force directory rescanning
- **Index Control**: Reset capability and automatic wraparound
- **State Management**: Maintains state between executions for true sequential processing

### Node Inputs

| Input | Type | Description |
|-------|------|-------------|
| directory_path | STRING | Path to the directory containing images |
| file_extension | ["PNG", "JPG", "JPEG", "ALL"] | Filter for specific file types |
| reset_index | BOOLEAN | When True, resets the counter to 0 |
| sort_method | ["alphabetical", "numerical", "creation_time", "modification_time"] | Method to sort the images |
| reload_directory | BOOLEAN | When True, forces directory rescan and index reset |

### Node Outputs

| Output | Type | Description |
|--------|------|-------------|
| image | TENSOR | The loaded image as a GPU tensor |
| current_index | INT | Current position in the sequence |
| total_images | INT | Total number of images in directory |
| filename | STRING | Name of the current image file |

### Sorting Behavior

#### Numerical Sort (Default)
- Handles numbered filenames intelligently
- Example: `["img1.png", "img2.png", "img10.png"]`
- Instead of: `["img1.png", "img10.png", "img2.png"]`

#### Alphabetical Sort
- Standard alphabetical ordering
- Case-insensitive

#### Time-based Sorting
- Creation time: Orders by file creation timestamp
- Modification time: Orders by last modified timestamp