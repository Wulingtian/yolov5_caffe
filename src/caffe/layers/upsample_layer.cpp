
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void UpsampleLayer<Dtype>::Forward_cpu_bilinear(
								 Blob<Dtype>* bottom,
								 Blob<Dtype>* top) {
		const Dtype* bottom_data = bottom->cpu_data();
		Dtype* top_data_mu = top->mutable_cpu_data();

		const int NC = bottom->num()* bottom->channels();
		const int old_height = bottom->height();
		const int old_width = bottom->width();

		const float rheight = (height_ > 1) ? (float)(old_height - 1) / (height_ - 1) : 0.f;
		const float rwidth = (width_ > 1) ? (float)(old_width - 1) / (width_ - 1) : 0.f;

		for (int nc = 0; nc < NC; ++nc) {
			for (int h2 = 0; h2 < height_; ++h2) {
				const float h1r = rheight * h2;
				const int h1 = h1r;
				const int h1p = (h1 < old_height - 1) ? 1 : 0;
				const float h1lambda = h1r - h1;
				const float h0lambda = 1.f - h1lambda;
				for (int w2 = 0; w2 < width_; ++w2) {
					const float w1r = rwidth * w2;
					const int w1 = w1r;
					const int w1p = (w1 < old_width - 1) ? 1 : 0;
					const float w1lambda = w1r - w1;
					const float w0lambda = 1.f - w1lambda;
					const Dtype* Xdata = &bottom_data[h1 * old_width + w1];
					Dtype* Ydata = &top_data_mu[h2 * width_ + w2];

					Ydata[0] = h0lambda * (w0lambda * Xdata[0] + w1lambda * Xdata[w1p]) + 
								h1lambda * (w0lambda * Xdata[h1p * old_width] + 
											w1lambda * Xdata[h1p * old_width + w1p]);
				}
			}
			bottom_data += old_width * old_height;
			top_data_mu += width_ * height_;			
		}									
	}
		
	template <typename Dtype>
	void UpsampleLayer<Dtype>::Forward_cpu_nearest(
								 Blob<Dtype>* bottom,
								 Blob<Dtype>* top) {
		const Dtype* bottom_data = bottom->cpu_data();
		Dtype* top_data_mu = top->mutable_cpu_data();

		const int NC = bottom->num()*bottom->channels();
		const int old_height = bottom->height();
		const int old_width = bottom->width();

		for (int nc = 0; nc < NC; ++nc) 
			for (int h = 0; h < height_; ++h) 
				for (int w = 0; w < width_; ++w) {
					int old_h = h*old_height/height_;
					int old_w = w*old_width/width_;
					top_data_mu[nc * height_ * width_ + h * width_ + w] = 
					bottom_data[nc * old_height * old_width + old_h * old_width + old_w];
				}									
	}

	
	template <typename Dtype>
	void UpsampleLayer<Dtype>::LayerSetUp(
								const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top) {
		UpsampleParameter upsample_param = this->layer_param_.upsample_param();
		if (2 != bottom.size() && 
			!(upsample_param.has_height() && upsample_param.has_width()) && 
			!(upsample_param.has_height_scale() && upsample_param.has_width_scale())) {
			CHECK_GE(0, 1) << "Upsample either has two bottom or use (height and width) or (height_scale and width_scale)!";			
		}
	}

	template <typename Dtype>
	void UpsampleLayer<Dtype>::Reshape(
								const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top) {
		UpsampleParameter upsample_param = this->layer_param_.upsample_param();

		if (2 == bottom.size()) {
			height_ = bottom[1]->height();
			width_ = bottom[1]->width();		
		}
		else if (upsample_param.has_height() && upsample_param.has_width()) {
			height_ = upsample_param.height();
			width_ = upsample_param.width();
			CHECK_GE(height_, 2) << "Upsample height must greater or equal two";
			CHECK_GE(width_, 2) << "Upsample width must greater or equal two";
		}
		else if(upsample_param.has_height_scale() && upsample_param.has_width_scale()) {
			height_scale_ = upsample_param.height_scale();
			width_scale_ = upsample_param.width_scale();
			CHECK_GE(height_scale_, 2) << "Upsample height_scale must greater or equal two";
			CHECK_GE(width_scale_, 2) << "Upsample width_scale must greater or equal two";	

			height_ = bottom[0]->height()*height_scale_;
			width_ = bottom[0]->width()*width_scale_;
		}
		else {
			CHECK_GE(0, 1) << "Upsample either has two bottom or use (height and width) or (height_scale and width_scale)!";			
		}		
		vector<int> out_shape(4,1);
		out_shape[0] = bottom[0]->num();
		out_shape[1] = bottom[0]->channels();
		out_shape[2] = height_;
		out_shape[3] = width_;

		top[0]->Reshape(out_shape);
	}

	template <typename Dtype>
	void UpsampleLayer<Dtype>::Forward_cpu(
								const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top) {
		if(UpsampleParameter_UpsampleOp_NEAREST == this->layer_param_.upsample_param().mode()) {
			Forward_cpu_nearest(bottom[0],top[0]);
		}
		else if(UpsampleParameter_UpsampleOp_BILINEAR == this->layer_param_.upsample_param().mode()) {
			Forward_cpu_bilinear(bottom[0],top[0]);
		}
	}

	template <typename Dtype>
	void UpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
												const vector<bool>& propagate_down,
												const vector<Blob<Dtype>*>& bottom) {
		//暂时不需要
	}

#ifdef CPU_ONLY
	STUB_GPU(UpsampleLayer);
#endif

	INSTANTIATE_CLASS(UpsampleLayer);
	REGISTER_LAYER_CLASS(Upsample);
}// namespace caffe
