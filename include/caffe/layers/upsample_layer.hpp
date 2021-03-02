#ifndef CAFFE_UPSAMPLE_LAYER_HPP
#define CAFFE_UPSAMPLE_LAYER_HPP

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	template <typename Dtype>
	class UpsampleLayer:public Layer<Dtype> {
	public:
		explicit UpsampleLayer(const LayerParameter& param)
			:Layer<Dtype>(param) {}
		virtual inline const char* type() const {return "Upsample";}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
							const vector<Blob<Dtype>*>& top);

		virtual inline int MinBottomBlobs() const {return 1;}
		virtual inline int MaxBottomBlobs() const {return 2;}
		virtual inline int ExactNumTopBlobs() const {return 1;}


	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
								const vector<Blob<Dtype>*>& top);		
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
								const vector<bool>& propagate_down,
								const vector<Blob<Dtype>*>& bottom);		
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
								const vector<bool>& propagate_down,
								const vector<Blob<Dtype>*>& bottom);
	private:
		void Forward_cpu_bilinear(Blob<Dtype>* bottom,
								 Blob<Dtype>* top);
		void Forward_cpu_nearest(Blob<Dtype>* bottom,
								 Blob<Dtype>* top);
	private:		
		int height_;
		int width_;
		float height_scale_;
		float width_scale_;		
	};
} // namespace caffe

#endif // CAFFE_UPSAMPLE_LAYER_HPP
