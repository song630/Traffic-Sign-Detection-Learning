// Caffe ROI Pooling的源码解析
// 来自博客 https://blog.csdn.net/lanran2/article/details/60143861

template <typename Dtype>
void ROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// 输入由两部分组成，data和rois
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_rois = bottom[1]->cpu_data();
	// Number of ROIs
	int num_rois = bottom[1]->num();
	int batch_size = bottom[0]->num();
	int top_count = top[0]->count();
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top_count, Dtype(-FLT_MAX), top_data);
	int* argmax_data = max_idx_.mutable_cpu_data();
	caffe_set(top_count, -1, argmax_data);

	// For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
	// 遍历原图的每一个ROI
	for (int n = 0; n < num_rois; ++n) {
		int roi_batch_ind = bottom_rois[0];
		// spatial_scale_: 指输入图片与刚通过CNN的feature map的比值,
		// 这个feature map指roi pooling层的输入
		// 把原图的坐标通过乘上这个比例映射到feature map(roi pooling的输入)中
		// 得到feature map中的矩形框的4个坐标 即下面4个变量
		int roi_start_w = round(bottom_rois[1] * spatial_scale_);
		int roi_start_h = round(bottom_rois[2] * spatial_scale_);
		int roi_end_w = round(bottom_rois[3] * spatial_scale_);
		int roi_end_h = round(bottom_rois[4] * spatial_scale_);
		// 计算每个roi在feature map(roi pooling的输入)上面的大小
		int roi_height = max(roi_end_h - roi_start_h + 1, 1);
		int roi_width = max(roi_end_w - roi_start_w + 1, 1);
		// pooling之后的feature map的一个值对应于pooling之前的feature map上的大小
		// (由于roi的大小不一致 所以每次都需要计算一次)
		// pooled_height_, pooled_width_: 指roi pooling之后的feature map的固定长宽
		// 下面的2个变量是roi pooling之前的ROI尺寸与之后的ROI尺寸之比
		const Dtype bin_size_h = static_cast<Dtype>(roi_height)
		                         / static_cast<Dtype>(pooled_height_);
		const Dtype bin_size_w = static_cast<Dtype>(roi_width)
		                         / static_cast<Dtype>(pooled_width_);
		// 找到对应的roi的feature map 如果input data的batch size为1
		// 那么roi_batch_ind=0
		const Dtype* batch_data = bottom_data + bottom[0]->offset(roi_batch_ind);
		// pooling的过程是针对每一个channel的 所以需要循环遍历
		for (int c = 0; c < channels_; ++c) {
			// 计算output的每一个值 所以需要遍历output(roi pooling的输出)中的每一个像素 然后求出所有值
			for (int ph = 0; ph < pooled_height_; ++ph) {
				for (int pw = 0; pw < pooled_width_; ++pw) {
					// Compute pooling region for this output unit:
					// 计算output上的一点往回对应于input上面区域的大小[hstart, wstart, hend, wend]
					// output上一个像素乘上: roi pooling之前的ROI尺寸与之后的ROI尺寸之比
					int hstart = static_cast<int>(floor(static_cast<Dtype>(ph)
					                                  * bin_size_h));
					int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) // +1表示是1个像素的范围
					                               * bin_size_h));
					int wstart = static_cast<int>(floor(static_cast<Dtype>(pw)
					                                  * bin_size_w));
					int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) // +1表示是1个像素的范围
					                               * bin_size_w));
					// 将映射后的区域平动到对应的位置[hstart, wstart, hend, wend]
					// 上面得出的只是大小 并不是往回对应的准确位置
					hstart = min(max(hstart + roi_start_h, 0), height_);
					hend = min(max(hend + roi_start_h, 0), height_);
					wstart = min(max(wstart + roi_start_w, 0), width_);
					wend = min(max(wend + roi_start_w, 0), width_);
					// 如果映射后的矩形框不符合
					bool is_empty = (hend <= hstart) || (wend <= wstart);
					// pool_index指的是此时计算的output的值对应于output的位置
					const int pool_index = ph * pooled_width_ + pw;
					// 如果矩形不符合 此处output的值设为0 此处的对应于输入区域的最大值为-1
					if (is_empty) {
						top_data[pool_index] = 0;
						argmax_data[pool_index] = -1;
					}
					// 遍历: output的值往回对应于input的区域块(要做max / average pooling)
					for (int h = hstart; h < hend; ++h) {
						for (int w = wstart; w < wend; ++w) {
							// 对应于input上的准确像素位置
							const int index = h * width_ + w;
							// 保存2个东西
							// 计算区域块的最大值 保存在output对应的位置上 同时记录最大值的索引
							// 这里用的是max 还可以用average
							if (batch_data[index] > top_data[pool_index]) {
								top_data[pool_index] = batch_data[index];
								argmax_data[pool_index] = index;
							}
						}
					}
				}
			}
			// Increment all data pointers by one channel
			batch_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
			argmax_data += max_idx_.offset(0, 1);
		}
		// Increment ROI data pointer
		bottom_rois += bottom[1]->offset(1);
	}
}
