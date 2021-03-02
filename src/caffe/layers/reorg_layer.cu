#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    template <typename Dtype>
    __global__ void reorg_kernel(const Dtype *srcData,
                             const int in_w, const int in_h,
                             const int in_c, const int batch,
                             const int stride, int forward, Dtype *dstData) {
      int64_t gid = (blockDim.x * blockIdx.x) + threadIdx.x;
      int64_t step = gridDim.x * blockDim.x;
      int out_c = in_c * stride * stride;
      int out_h = in_h / stride;
      int out_w = in_w / stride;
      int out_elements = out_w * out_h * out_c * batch;
      while (gid < out_elements) {
        int remain = gid;
        int out_b_idx = remain / out_c / out_h  / out_w;
        remain = remain % (out_c * out_h * out_w);
        int out_c_idx = remain / out_h  / out_w;
        remain = remain % (out_h * out_w);
        int out_h_idx = remain / out_w;
        int out_w_idx = remain % out_w; 
     
        int in_c_idx = out_c_idx / stride / stride;
        int in_inner_c_idx = out_c_idx % (stride * stride);
        int in_inner_h_idx = in_inner_c_idx / stride;
        int in_inner_w_idx = in_inner_c_idx % stride;

        int in_h_idx = out_h_idx * stride + in_inner_h_idx;
        int in_w_idx = out_w_idx * stride + in_inner_w_idx;

        int in_idx = out_b_idx * in_c * in_w * in_h + in_c_idx * in_w * in_h + in_h_idx * in_w + in_w_idx;
        dstData[gid] = srcData[in_idx];
        gid += step;
      }
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        const Dtype *bottom_data = bottom[0]->gpu_data();
        int count = bottom[0]->count();
        Dtype *top_data = top[0]->mutable_gpu_data();
        reorg_kernel<Dtype>
         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(bottom_data, width_, height_,
                  channels_, batch_num_, stride_, reverse_, top_data);
    }

    template<typename Dtype>
    void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
        if(!propagate_down[0]){
            return;
        }
        int count = diff_.count();
        const Dtype *top_diff = diff_.mutable_gpu_diff();
        Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
        reorg_kernel<Dtype>
         <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(top_diff, width_, height_,
                  channels_, batch_num_, stride_, !reverse_, bottom_diff);
    }

INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);

}  // namespace caffe
