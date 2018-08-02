 // Mask-RCNN中的ROIAlign源码-caffe2
 // 来自博客 https://blog.csdn.net/yiyouxian/article/details/79221830

#include "roi_align_op.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

#ifdef CAFFE2_USE_MKL
#include "caffe2/mkl/operators/operator_fallback_mkl.h"
#endif // CAFFE2_USE_MKL

namespace caffe2 {
namespace {

template <typename T>
struct PreCalc {
	int pos1;
	int pos2;
	int pos3;
	int pos4;
	T w1;
	T w2;
	T w3;
	T w4;
};

/*;
height, 		 // 输入数据的第三个维度长度即h
width, 			 // 输入数据的第四个维度长度即w
pooled_height,   // pooling后的h
pooled_width,    // pooling后的w
iy_upper, 		 // 每个小网格内用于pooling的垂直方向采样点数
ix_upper, 		 // 每个小网格内用于pooling的水平方向采样点数
roi_start_h, 	 // roi在输入图像中的坐标y1变换到roi-align输入feature map的坐标 float型
roi_start_w, 	 // roi在输入图像中的坐标x1变换到roi-align输入feature map的坐标 float型
bin_size_h, 	 // 每个roi分块后每个小块垂直方向包含的bin数量(即尺寸)
bin_size_w, 	 // 每个roi分块后每个小块水平方向包含的bin数量(即尺寸)
roi_bin_grid_h,  // 每个小网格内用于pooling的垂直方向采样点数
roi_bin_grid_w,  // 每个小网格内用于pooling的水平方向采样点数
*/

// 获得双线性插值采样点周围四个坐标的索引以及对应的权重
template <typename T>
void pre_calc_for_bilinear_interpolate(
	const int height,
	const int width,
	const int pooled_height,
	const int pooled_width,
	const int iy_upper,
	const int ix_upper,
	T roi_start_h,
	T roi_start_w,
	T bin_size_h,
	T bin_size_w,
	int roi_bin_grid_h,
	int roi_bin_grid_w,
	std::vector<PreCalc<T>>& pre_calc) {
	int pre_calc_index = 0;
	for (int ph = 0; ph < pooled_height; ph++) {
		for (int pw = 0; pw < pooled_width; pw++) {
			for (int iy = 0; iy < iy_upper; iy++) {
				// 计算采样点垂直坐标，按每个小网格大小进行均匀采样roi_bin_grid_h个值
				const T yy = roi_start_h + ph * bin_size_h + 
					static_cast<T>(iy + .5f) * bin_size_h /
					static_cast<T>(roi_bin_grid_h);  //  e.g., 0.5, 1.5
				// 计算采样点水平坐标，按每个小网格大小进行均匀采样roi_bin_grid_w个值
				for (int ix = 0; ix < ix_upper; ix++) {
					const T xx = roi_start_w + pw * bin_size_w +
						static_cast<T>(ix + .5f) * bin_size_w /
						static_cast<T>(roi_bin_grid_w);
					T x = xx;
					T y = yy;
					// deal with: inverse elements are out of feature map boundary
					// 处理越界
					if (y < -1.0 || y > height || x < -1.0 || x > width) {
						// empty
						PreCalc<T> pc;
						pc.pos1 = 0;
						pc.pos2 = 0;
						pc.pos3 = 0;
						pc.pos4 = 0;
						pc.w1 = 0;
						pc.w2 = 0;
						pc.w3 = 0;
						pc.w4 = 0;
						pre_calc[pre_calc_index] = pc;
						pre_calc_index += 1;
						continue;
					}
					if (y <= 0)
						y = 0;
					if (x <= 0)
						x = 0;

					int y_low = (int)y; // 采样点y向下取整 找其上方最近的整数坐标
					int x_low = (int)x; // 采样点x向下取整 找其左方最近的整数坐标
					int y_high;
					int x_high;
 
					// 计算采样点下方最近的整数坐标
					if (y_low >= height - 1) {
						y_high = y_low = height - 1;
						y = (T)y_low;
					} else {
						y_high = y_low + 1;
					}

					// 计算采样点右方最近的整数坐标
					if (x_low >= width - 1) {
						x_high = x_low = width - 1;
						x = (T)x_low;
					} else {
						x_high = x_low + 1;
					}
 
					// 根据采样点坐标计算双线性插值对应的四个权重
					T ly = y - y_low;
					T lx = x - x_low;
					T hy = 1. - ly, hx = 1. - lx;
					T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
 
					// save weights and indeces
					PreCalc<T> pc;
					// 将坐标换算成在整个输入featuremap中的坐标
					pc.pos1 = y_low * width + x_low;
					pc.pos2 = y_low * width + x_high;
					pc.pos3 = y_high * width + x_low;
					pc.pos4 = y_high * width + x_high;
					pc.w1 = w1;
					pc.w2 = w2;
					pc.w3 = w3;
					pc.w4 = w4;
					pre_calc[pre_calc_index] = pc;

					pre_calc_index += 1;
				}
			}
		}
	}
}

/*
nthreads, 		// 输出总长度
bottom_data, 	// 输入数据
spatial_scale,  // 1 / stride(stride可以理解为感受野大小)
channels, 		// 输入数据的第二个维度长度即channel
height, 		// 输入数据的第三个维度长度即h
width, 			// 输入数据的第四个维度长度即w
pooled_height,  // pooling后的h
pooled_width,   // pooling后的w
sampling_ratio, // pooling中采用的采样点 每个small window采样sampling_ratio_个点然后计算均值作为pooling结果 sampling_ratio_值小于0时采用整个small window所有点求均值
bottom_rois, 	// roi数据 float类型 5列 分别为roi个数 x1, y1, x2, y2
roi_cols, 		// roi列数
top_data, 		// 输出数据
*/

template <typename T>
void ROIAlignForward(
	const int nthreads,
	const T* bottom_data,
	const T& spatial_scale,
	const int channels,
	const int height,
	const int width,
	const int pooled_height,
	const int pooled_width,
	const int sampling_ratio,
	const T* bottom_rois,
	int roi_cols,
	T* top_data,
	StorageOrder order) {
	DCHECK(roi_cols == 4 || roi_cols == 5);

	int n_rois = nthreads / channels / pooled_width / pooled_height; // 根据nthreads的计算公式 最终n_rois就是输入到roialign层roi数量
	// (n, c, ph, pw) is an element in the pooled output
	// can be parallelized using omp
	// #pragma omp parallel for num_threads(32)
	for (int n = 0; n < n_rois; n++) {
		int index_n = n * channels * pooled_width * pooled_height; // 每个roi经过roi-align操作输出的索引(网络输出索引)
	// roi could have 4 or 5 columns
	const T* offset_bottom_rois = bottom_rois + n * roi_cols; // 每个roi信息的索引
	int roi_batch_ind = 0;
	if (roi_cols == 5) {
		roi_batch_ind = offset_bottom_rois[0]; // 获取第一个roi的索引
		offset_bottom_rois++; // offset_bottom_rois指向x1
	}

	// Do not using rounding; this implementation detail is critical
	T roi_start_w = offset_bottom_rois[0] * spatial_scale; // roi在输入图像中的坐标x1变换到roialign输入feature map的坐标 float型
	T roi_start_h = offset_bottom_rois[1] * spatial_scale; // roi在输入图像中的坐标y1变换到roialign输入feature map的坐标 float型
	T roi_end_w = offset_bottom_rois[2] * spatial_scale; // roi在输入图像中的坐标x2变换到roialign输入feature map的坐标 float型
	T roi_end_h = offset_bottom_rois[3] * spatial_scale; // roi在输入图像中的坐标y2变换到roialign输入feature map的坐标 float型
	// 以下操作则为roipooling的做法，取整丢失了精度
	// T roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
	// T roi_start_h = round(offset_bottom_rois[1] * spatial_scale);
	// T roi_end_w = round(offset_bottom_rois[2] * spatial_scale);
	// T roi_end_h = round(offset_bottom_rois[3] * spatial_scale);

	// Force malformed ROIs to be 1x1
	T roi_width = std::max(roi_end_w - roi_start_w, (T)1.); // 当roi在roialign输入的feature map中宽度小于1时强制设置为1
	T roi_height = std::max(roi_end_h - roi_start_h, (T)1.); // 当roi在roialign输入的feature map中高度小于1时强制设置为1
	T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height); // 计算每个roi分块后每个小块垂直方向包含的bin数量(即尺寸)
	T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width); // 计算每个roi分块后每个小块水平方向包含的bin数量(即尺寸)

	// We use roi_bin_grid to sample the grid and mimic integral
	// 根据sampling_ratio设置用于pooling的采样点数
	int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio
		: ceil(roi_height / pooled_height); // e.g., = 2
	int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio 
		: ceil(roi_width / pooled_width);

	// We do average (integral) pooling inside a bin
	const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4 // 采样点总数

	// we want to precalculate indeces and weights shared by all chanels,
	// this is the key point of optimiation
	std::vector<PreCalc<T>> pre_calc(
		roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height); // size为所有分块采样点总数
	// 获得双线性插值采样点周围四个坐标的索引以及对应的权重
	pre_calc_for_bilinear_interpolate(
		height, 		// 输入数据的第三个维度长度即h
		width, 			// 输入数据的第四个维度长度即w
		pooled_height,  // pooling后的h
		pooled_width,   // pooling后的w
		roi_bin_grid_h, // 每个小网格内用于pooling的垂直方向采样点数
		roi_bin_grid_w, // 每个小网格内用于pooling的水平方向采样点数
		roi_start_h, 	// roi在输入图像中的坐标y1变换到roialign输入featuremap的坐标 float型
		roi_start_w, 	// roi在输入图像中的坐标x1变换到roialign输入featuremap的坐标 float型
		bin_size_h, 	// 每个roi分块后每个小块垂直方向包含的bin数量(即尺寸)
		bin_size_w, 	// 每个roi分块后每个小块水平方向包含的bin数量(即尺寸)
		roi_bin_grid_h, // 每个小网格内用于pooling的垂直方向采样点数
		roi_bin_grid_w, // 每个小网格内用于pooling的水平方向采样点数
		pre_calc
	);

	if (order == StorageOrder::NCHW) {
		for (int c = 0; c < channels; c++) {
			int index_n_c = index_n + c * pooled_width * pooled_height; // 每个输出对应channel的索引
			const T* offset_bottom_data =
				bottom_data + (roi_batch_ind * channels + c) * height * width; // 每个输入featuremap对应channel的索引
			int pre_calc_index = 0;
 
			for (int ph = 0; ph < pooled_height; ph++) {
				for (int pw = 0; pw < pooled_width; pw++) {
					int index = index_n_c + ph * pooled_width + pw; // 每个输出坐标的索引
					T output_val = 0.; // 用于统计每个小网格内采样点总数
					// 遍历每个小网格内的采样点 根据双线性插值方法计算采样坐标处的值 用于aver-pooling
					for (int iy = 0; iy < roi_bin_grid_h; iy++) {
						for (int ix = 0; ix < roi_bin_grid_w; ix++) {
							PreCalc<T> pc = pre_calc[pre_calc_index];
							output_val += pc.w1 * offset_bottom_data[pc.pos1] +
								pc.w2 * offset_bottom_data[pc.pos2] +
								pc.w3 * offset_bottom_data[pc.pos3] +
								pc.w4 * offset_bottom_data[pc.pos4];
							pre_calc_index += 1;
						}
					}
					output_val /= count;
					top_data[index] = output_val;
				} // for pw
			} // for ph
		} // for c
    } // if nchw

	if (order == StorageOrder::NHWC) {
		const T* offset_bottom_data = bottom_data + roi_batch_ind * channels * height * width;
		int pre_calc_index = 0;

		for (int ph = 0; ph < pooled_height; ph++) {
			for (int pw = 0; pw < pooled_width; pw++) {
				EVecXf output_vals = EVecXf::Zero(channels);
				for (int iy = 0; iy < roi_bin_grid_h; iy++) {
					for (int ix = 0; ix < roi_bin_grid_w; ix++) {
						PreCalc<T> pc = pre_calc[pre_calc_index];

						ConstEigenVectorMap<T> data_1(offset_bottom_data + channels * pc.pos1, channels);
						ConstEigenVectorMap<T> data_2(offset_bottom_data + channels * pc.pos2, channels);
						ConstEigenVectorMap<T> data_3(offset_bottom_data + channels * pc.pos3, channels);
						ConstEigenVectorMap<T> data_4(offset_bottom_data + channels * pc.pos4, channels);
						output_vals += pc.w1 * data_1 + pc.w2 * data_2 + pc.w3 * data_3 + pc.w4 * data_4;
						pre_calc_index += 1;
					}
				}
				output_vals /= count;
				int index_nhw = index_n + (ph * pooled_width + pw) * channels;
				std::memcpy(top_data + index_nhw, output_vals.data(), channels * sizeof(T));
			} // for pw
		} // for ph
	} // if nhwc
} // for n
}

} // namespace

template <>
bool RoIAlignOp<float, CPUContext>::RunOnDevice() {
	auto& X = Input(0); // Input data to pool, NCHW
	auto& R = Input(1); // RoIs
	auto* Y = Output(0); // RoI pooled data

	if (R.size() == 0) {
		// Handle empty rois
		if (order_ == StorageOrder::NCHW) {
			Y->Resize(0, X.dim32(1), pooled_height_, pooled_width_);
		} else if (order_ == StorageOrder::NHWC) {
			Y->Resize(0, pooled_height_, pooled_width_, X.dim32(3));
		}
		// The following mutable_data calls are needed to allocate the tensors
		Y->mutable_data<float>();
		return true;
	}

	CAFFE_ENFORCE_EQ(R.ndim(), 2);
	// if R has 5 columns, the first column is the index, otherwise 0
	CAFFE_ENFORCE(R.dim32(1) == 4 || R.dim32(1) == 5);

	assert(sampling_ratio_ >= 0);

	if (order_ == StorageOrder::NCHW) {
		Y->Resize(R.dim32(0), X.dim32(1), pooled_height_, pooled_width_); // ROI数量 输入通道数 pooling后的高 pooling后的宽
		int output_size = Y->size(); // 计算输出总长度
		ROIAlignForward<float>(
			output_size, 	 // 输出总长度
			X.data<float>(), // 输入数据
			spatial_scale_,  // 1/stride(stride可以理解为感受野大小)
			X.dim32(1), 	 // 输入数据的第二个维度长度即channel
			X.dim32(2), 	 // 输入数据的第三个维度长度即h
			X.dim32(3), 	 // 输入数据的第四个维度长度即w
			pooled_height_,  // pooling后的h
			pooled_width_,   // pooling后的w
			sampling_ratio_, // pooling中采用的采样点 每个small window采样sampling_ratio_个点然后计算均值作为pooling结果 sampling_ratio_值小于0时采用整个small window所有点求均值
			R.data<float>(), // roi数据 5列 分别为roi个数 x1,y1,x2,y2
			R.dim32(1), 	 // roi列数
			Y->mutable_data<float>(), // 输出数据
			order_
		);
	} else if (order_ == StorageOrder::NHWC) {
		Y->Resize(R.dim32(0), pooled_height_, pooled_width_, X.dim32(3));
		int output_size = Y->size();
		ROIAlignForward<float>(
			output_size,
			X.data<float>(),
			spatial_scale_,
			X.dim32(3),
			X.dim32(1),
			X.dim32(2),
			pooled_height_,
			pooled_width_,
			sampling_ratio_,
			R.data<float>(),
			R.dim32(1),
			Y->mutable_data<float>(),
			order_
		);
	}
	return true;
}

REGISTER_CPU_OPERATOR(RoIAlign, RoIAlignOp<float, CPUContext>);

#ifdef CAFFE2_HAS_MKL_DNN
REGISTER_MKL_OPERATOR(
	RoIAlign,
	mkl::MKLFallbackOp<RoIAlignOp<float, CPUContext>>);
#endif // CAFFE2_HAS_MKL_DNN

// Input: X, rois; Output: Y
OPERATOR_SCHEMA(RoIAlign)
	.NumInputs(2)
	.NumOutputs(1)
	.SetDoc(R"DOC(
Region of Interest (RoI) align operation as used in Mask R-CNN.
)DOC")
	.Arg(
		"spatial_scale",
		"(float) default 1.0; Spatial scale of the input feature map X "
		"relative to the input image. E.g., 0.0625 if X has a stride of 16 "
		"w.r.t. the input image.")
	.Arg("pooled_h", "(int) default 1; Pooled output Y's height.")
	.Arg("pooled_w", "(int) default 1; Pooled output Y's width.")
	.Arg(
		"sampling_ratio",
		"(int) default -1; number of sampling points in the interpolation grid "
		"used to compute the output value of each pooled output bin. If > 0, "
		"then exactly sampling_ratio x sampling_ratio grid points are used. If "
		"<= 0, then an adaptive number of grid points are used (computed as "
		"ceil(roi_width / pooled_w), and likewise for height).")
	.Input(0, "X", "4D feature map input of shape (N, C, H, W).")
	.Input(
		1,
		"RoIs",
		"2D input of shape (R, 5) specifying R RoIs with five columns "
		"representing: batch index in [0, N - 1], x1, y1, x2, y2. The RoI "
		"coordinates are in the coordinate system of the input image.")
	.Output(
		0,
		"Y",
		"4D output of shape (R, C, pooled_h, pooled_w). The r-th batch element "
		"is a pooled feature map cooresponding to the r-th RoI.");
} // namespace caffe2
