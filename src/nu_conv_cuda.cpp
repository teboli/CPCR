// #include <torch/torch.h>
#include <torch/extension.h>

#include <vector>

#include <iostream>

// CUDA forward declarations

at::Tensor conv_motion_cuda_forward(
    at::Tensor input,
    at::Tensor mag,
    at::Tensor ori);

at::Tensor inv_conv_motion_cuda_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias);

at::Tensor conv_cls_cuda_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias);

at::Tensor line_cuda_forward(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor labels);

// CUDA backward declarations

at::Tensor conv_motion_cuda_backward(
    at::Tensor grad_out,
    at::Tensor mag,
    at::Tensor ori,
    at::Tensor mcos,
    at::Tensor msin);

at::Tensor inv_conv_motion_cuda_backward(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight);

at::Tensor conv_cls_cuda_backward1(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight);

at::Tensor conv_cls_cuda_backward2(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor labels_unique);

at::Tensor line_cuda_backward1(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels);

at::Tensor line_cuda_backward2(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor conv_motion_forward(
    at::Tensor input,
    at::Tensor mag,
    at::Tensor ori) {
    CHECK_INPUT(input);
    CHECK_INPUT(mag);
    CHECK_INPUT(ori);

    return conv_motion_cuda_forward(input, mag, ori);
}


at::Tensor inv_conv_motion_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(labels);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    return inv_conv_motion_cuda_forward(input, labels, weight, bias);
}


at::Tensor conv_cls_forward(
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(labels);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    return conv_cls_cuda_forward(input, labels, weight, bias);
}

at::Tensor line_forward(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor labels) {
    CHECK_INPUT(xlab);
    CHECK_INPUT(ylab);
    CHECK_INPUT(xvar);
    CHECK_INPUT(labels);

    return line_cuda_forward(xlab, ylab, xvar, labels);
}

at::Tensor conv_motion_backward(
    at::Tensor grad_out,
    at::Tensor mag,
    at::Tensor ori,
    at::Tensor mcos,
    at::Tensor msin) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(mag);
    CHECK_INPUT(ori);
    CHECK_INPUT(mcos);
    CHECK_INPUT(msin);

    return conv_motion_cuda_backward(grad_out, mag, ori, mcos, msin);
}

at::Tensor inv_conv_motion_backward(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(labels);
    CHECK_INPUT(weight);

    return inv_conv_motion_cuda_backward(grad_out, labels, weight);
}

at::Tensor conv_cls_backward1(
    at::Tensor grad_out,
    at::Tensor labels,
    at::Tensor weight) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(labels);
    CHECK_INPUT(weight);

    return conv_cls_cuda_backward1(grad_out, labels, weight);
}

at::Tensor conv_cls_backward2(
    at::Tensor grad_out,
    at::Tensor input,
    at::Tensor labels,
    at::Tensor weight,
    at::Tensor labels_unique) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(input);
    CHECK_INPUT(labels);
    CHECK_INPUT(weight);
    CHECK_INPUT(labels_unique);

    return conv_cls_cuda_backward2(grad_out, input, labels, weight, labels_unique);
}

at::Tensor line_backward1(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels) {
    CHECK_INPUT(xlab);
    CHECK_INPUT(ylab);
    CHECK_INPUT(xvar);
    CHECK_INPUT(yvar);
    CHECK_INPUT(labels);

    return line_cuda_backward1(xlab, ylab, xvar, yvar, labels);
}

at::Tensor line_backward2(
    at::Tensor xlab,
    at::Tensor ylab,
    at::Tensor xvar,
    at::Tensor yvar,
    at::Tensor labels) {
    CHECK_INPUT(xlab);
    CHECK_INPUT(ylab);
    CHECK_INPUT(xvar);
    CHECK_INPUT(yvar);
    CHECK_INPUT(labels);

    return line_cuda_backward2(xlab, ylab, xvar, yvar, labels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_motion_forward", &conv_motion_forward, "Conv Motion forward (CUDA)");
    m.def("conv_motion_backward", &conv_motion_backward, "Conv Motion backward (CUDA)");
    m.def("inv_conv_motion_forward", &inv_conv_motion_forward, "Inv Conv Motion forward (CUDA)");
    m.def("inv_conv_motion_backward", &inv_conv_motion_backward, "Inv Conv Motion backward (CUDA)");
    m.def("conv_cls_forward", &conv_cls_forward, "Conv Cls forward (CUDA)");
    m.def("conv_cls_backward1", &conv_cls_backward1, "Conv Cls backward1 (CUDA)");
    m.def("conv_cls_backward2", &conv_cls_backward2, "Conv Cls backward2 (CUDA)");
    m.def("line_forward", &line_forward, "Line forward (CUDA)");
    m.def("line_backward1", &line_backward1, "Line backward1 (CUDA)");
    m.def("line_backward2", &line_backward2, "Line backward2 (CUDA)");
}