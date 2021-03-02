//***********************************************
#include <vector>
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	inline __device__ int idx(
		const int c,
		const int height,
		const int width,
		const int h,
		const int w	
		) {
		return c * height * width + h * width + w;
	}

	//CUDA hernel for forward
	template <typename Dtype>
	__global__ void UpsampleBilinearForward(
										const int N,const int old_height,const int old_width,
										const int new_height, const int new_width,
										const Dtype* in, Dtype* out) {
		CUDA_KERNEL_LOOP(index, N) {
			int c = index / (new_height * new_width);
			int h = (index % (new_height * new_width)) / new_width;
			int w = index % (new_height * new_width) % new_width;

			const float rheight = (new_height > 1) ? (old_height - 1.f) / (new_height - 1.f) : 0.f;
			const float rwidth = (new_width > 1) ? (old_width - 1.f) / (new_width - 1.f) : 0.f;


			//Compute Y axis lambdas
			const float h1r = rheight * h;
			const int h1 = (int)h1r;
			const int h1p = (h1 < old_height - 1) ? 1 : 0;
			const float h1lambda = h1r - h1;
			const float h0lambda = 1.f - h1lambda;


			//Compute X axis lambdas
			const float w1r = rwidth * w;
			const int w1 = (int)w1r;
			const int w1p = (w1 < old_width - 1) ? 1 : 0;
			const float w1lambda = w1r - w1;
			const float w0lambda = 1.0f - w1lambda;

			float X0 = in[idx(c, old_height, old_width, h1, w1)];
			float X1 = in[idx(c, old_height, old_width, h1, w1 + w1p)];
			float X2 = in[idx(c, old_height, old_width, h1 + h1p, w1)];
			float X3 = in[idx(c, old_height, old_width, h1 + h1p, w1 + w1p)];

			
			out[idx(c, new_height, new_width, h, w)] = 
				h0lambda * (w0lambda * X0 + w1lambda * X1) + 
				h1lambda * (w0lambda * X2 + w1lambda * X3);		
		}
	}

	template <typename Dtype>
	__global__ void UpsampleNearestForward(
										const int N,const int old_height,const int old_width,
										const int new_height, const int new_width,
										const Dtype* in, Dtype* out) {
		CUDA_KERNEL_LOOP(index, N) {
			int c = index / (new_height * new_width);
			int h = (index % (new_height * new_width)) / new_width;
			int w = index % (new_height * new_width) % new_width;
			int old_h = h * old_height / new_height;
			int old_w = w * old_width / new_width;

			out[c * new_height * new_width + h * new_width + w] = 
			in[c * old_height * old_width + old_h * old_width + old_w];
		}
	}


	template <typename Dtype>
	void UpsampleLayer<Dtype>::Forward_gpu(
								const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data_mu = top[0]->mutable_gpu_data();

		const int count = top[0]->count();		
		const int old_height = bottom[0]->height();
		const int old_width = bottom[0]->width();

		if(UpsampleParameter_UpsampleOp_NEAREST == this->layer_param_.upsample_param().mode()) {
			// NOLINT_NEXT_LINE(whitespace/operators)			
			UpsampleNearestForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
				count, old_height, old_width, height_, width_, bottom_data, top_data_mu);

			CUDA_POST_KERNEL_CHECK;
		}
		else if(UpsampleParameter_UpsampleOp_BILINEAR == this->layer_param_.upsample_param().mode()) {
			// NOLINT_NEXT_LINE(whitespace/operators)			
			UpsampleBilinearForward<Dtype><<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>(
				count, old_height, old_width, height_, width_, bottom_data, top_data_mu);
			CUDA_POST_KERNEL_CHECK;
		}

	}

	template <typename Dtype>
	void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
												const vector<bool>& propagate_down,
												const vector<Blob<Dtype>*>& bottom) {
		//暂时不需要
		
	}

	INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);	
}

//***********************************************
