import torch
import triton
import triton.language as tl


@triton.jit
def convolve(
    input_ptr, output_ptr, kernel_ptr, X, half_kernel_size, BLOCK_SIZE: tl.constexpr
):
    """
    Applies a 1D convolution with a Gaussian kernel to the input using Triton for GPU acceleration.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the result will be stored.
    kernel_ptr : tl.tensor
        Pointer to the kernel tensor (Gaussian) used for convolution.
    X : int
        Length of the input data (number of elements along the 1D dimension).
    half_kernel_size : int
        Half the size of the kernel, used to calculate the kernel range.
    BLOCK_SIZE : tl.constexpr
        Size of the block used in the Triton kernel grid, determines the number of elements processed per block.

    Notes:
    ------
    This function operates on 1D input data, performing convolution over a specified kernel size.
    Handles boundaries by zero-padding the input when necessary.
    """

    batch_id = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets

    mask = idx < X
    result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for k in range(2 * half_kernel_size + 1):
        kernel_val = tl.load(kernel_ptr + k)
        input_idx = idx + k - half_kernel_size
        input_mask = (input_idx >= 0) & (input_idx < X) & mask
        input_val = tl.load(
            input_ptr + batch_id * X + input_idx, mask=input_mask, other=0.0
        )
        result += input_val * kernel_val

    tl.store(output_ptr + batch_id * X + idx, result, mask=mask)


def convolve_1d(input, kernel):
    """
    Performs a 1D convolution on a batch of input sequences using a Gaussian kernel.

    Parameters:
    -----------
    input : torch.Tensor
        A 2D tensor of shape (B, X) where B is the batch size and X is the length of the sequence.
    kernel : torch.Tensor
        A 1D tensor representing the convolution kernel (e.g., a Gaussian kernel).

    Returns:
    --------
    torch.Tensor
        A 2D tensor of shape (B, X) containing the convolved output.

    Notes:
    ------
    The function divides the computation into blocks using Triton, performing convolution efficiently
    on GPU by parallelizing over the batch dimension and the sequence length.
    """

    B, X = input.shape
    output = torch.empty_like(input)

    BLOCK_SIZE = 256
    num_blocks = (X + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (B, num_blocks)
    half_kernel_size = kernel.shape[0] // 2

    device = input.device
    assert device.is_cuda, "Input tensor must be on a CUDA device for triton ops."
    kernel = kernel.to(device)

    convolve[grid](
        input_ptr=input,
        output_ptr=output,
        kernel_ptr=kernel,
        X=X,
        half_kernel_size=half_kernel_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
