import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt


@triton.jit
def bilateral_filter_kernel(
    input_ptr,
    img_ptr,  # Input image
    out_ptr,  # Output image
    width,  # Image width
    height,  # Image height
    spatial_sigma,
    range_sigma,
    value_sigma,
    kernel_radius,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the coordinates of the pixel this program is responsible for
    x = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    y = tl.program_id(1)
    mask = (x < width) & (y < height)

    # Load the central pixel intensity
    offset = y * width + x

    center_value = tl.load(input_ptr + offset, mask=mask, other=0.0)
    center_intensity = tl.load(img_ptr + offset, mask=mask, other=0.0)

    # Initialize accumulators
    out_intensity = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    normalization = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop over the kernel window
    for dy in range(-kernel_radius, kernel_radius + 1):
        for dx in range(-kernel_radius, kernel_radius + 1):
            # Compute neighbor coordinates
            xn = x + dx
            yn = y + dy
            neighbor_mask = (xn >= 0) & (xn < width) & (yn >= 0) & (yn < height) & mask

            # Compute spatial weight
            spatial_dist_sq = dx * dx + dy * dy
            spatial_weight = tl.exp(
                -0.5 * spatial_dist_sq / (spatial_sigma * spatial_sigma)
            )

            # Load neighbor intensity
            neighbor_offset = yn * width + xn
            neighbor_intensity = tl.load(
                img_ptr + neighbor_offset, mask=neighbor_mask, other=0.0
            )

            neighbor_value = tl.load(
                input_ptr + neighbor_offset, mask=neighbor_mask, other=0.0
            )
            value_diff = (neighbor_value - center_value)
            
            # Compute range weight
            range_diff = neighbor_intensity - center_intensity
            range_weight = tl.exp(
                -0.5 * (range_diff * range_diff) / (range_sigma * range_sigma) +
                -0.5 * (value_diff * value_diff) / (value_sigma * value_sigma)
            )

            # Compute combined weight
            weight = spatial_weight * range_weight * neighbor_mask.to(tl.float32)

            # Accumulate weighted intensity and normalization factor
            out_intensity += neighbor_intensity * weight
            normalization += weight

    # Avoid division by zero
    normalization = tl.where(normalization == 0, 1.0, normalization)

    # Compute final output intensity
    out_pixel = out_intensity / normalization

    # Store the result
    tl.store(out_ptr + offset, out_pixel, mask=mask)


def bilateral_filter_torch_triton(unary, img, spatial_sigma, range_sigma, kernel_radius):
    # Ensure the image is a 2D tensor
    assert img.ndim == 2, "Input image must be grayscale and 2D"
    height, width = img.shape
    img = img.contiguous()
    unary = unary.contiguous()

    # Allocate output tensor
    out = torch.empty_like(img)

    # Convert to float32 for computation
    img = img.to(torch.float32)
    out = out.to(torch.float32)

    # Define block size
    BLOCK_SIZE = 128  # Adjust based on your GPU's capabilities

    # Calculate grid dimensions
    grid_x = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid_x, height)

    # Launch Triton kernel
    bilateral_filter_kernel[grid](
        input_ptr=unary,
        img_ptr=img,
        out_ptr=out,
        width=width,
        height=height,
        spatial_sigma=spatial_sigma,
        range_sigma=range_sigma,
        value_sigma=1,
        kernel_radius=kernel_radius,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


if __name__ == "__main__":
    # Load a sample grayscale image (e.g., using PyTorch or any image library)
    # For simplicity, let's create a synthetic image
    height, width = 256, 256
    img = torch.linspace(0, 1, steps=width).repeat(height, 1)

    # Add some noise
    noise = torch.randn_like(img) * 0.1
    noisy_img = img + noise
    noisy_img = noisy_img.clamp(0, 1)

    # Move image to GPU
    noisy_img_gpu = noisy_img.cuda()

    # Set parameters
    spatial_sigma = 5.0
    range_sigma = 0.1
    kernel_radius = 5

    # Apply the bilateral filter
    filtered_img_gpu = bilateral_filter_torch_triton(
        noisy_img_gpu, spatial_sigma, range_sigma, kernel_radius
    )

    # Move result back to CPU
    filtered_img = filtered_img_gpu.cpu()

    # Plot the images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img.cpu(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_img.cpu(), cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Filtered Image")
    plt.imshow(filtered_img, cmap="gray")
    plt.axis("off")

    plt.show()
