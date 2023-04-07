<img src='imgs/rendering.jpg'/>

## Cài đặt môi trường 
```
conda env create -f environment.yml
conda activate autoint
```

## Autoint cho một hàm đơn giản 
Để train autoint cho một hàm 1 biến đơn giản, chạy notebook `autoint_example.ipynb` 

## Autoint cho bài tóan sparse tomography
Có thể huấn luyện một mạng cho bài toán chụp cắt lớp thưa thớt được trình bày trong bài báo với câu lệnh 

```
python train_sparse_tomography.py
```
## Autoint cho bài toán kết xuất thần kinh (neural rendering) 

AutoInt có thể được sử dụng để xấp xỉ phương trình kết xuất khối (volume rendering equation), đây là phương trình tích phân tính tổng độ truyền qua và độ phát xạ dọc theo các tia để hiển thị hình ảnh. Trong khi các trình kết xuất thần kinh thông thường yêu cầu hàng trăm mẫu dọc theo mỗi tia để đánh giá các tích phân này (tương đương với hàng trăm lần forward pass tốn kém), AutoInt cho phép đánh giá các tích phân này với số lần chuyển tiếp ít hơn nhiều.

### Huấn luyện mạng 

Trước khi huấn luyện AutoInt cho bài toán kết xuất thần kinh, cần tải bộ dữ liệu xuống thư mục `data`. Chúng tôi cho phép huấn luyện trên bất kỳ bộ dữ liệu nào trong số ba bộ dữ liệu. Dữ liệu Blender tổng hợp từ [NeRF](https://github.com/bmild/nerf) và [LLFF](https://github.com/Fyusion/LLFF) được lưu trữ [tại đây](https:// drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Dữ liệu [DeepVoxels](https://github.com/vsitzmann/deepvoxels) được lưu trữ [tại đây](https://drive.google.com/open?id=1lUvJWB6oFtT8EQ_NzBrXnmi25BufxRfl).

Các tệp cấu hình được cung cấp trong thư mục `experiment_scripts/configs`. Ví dụ: để huấn luyện trên bộ dữ liệu NeRF Blender, hãy chạy lệnh sau 
```
python train_autoint_radiance_field.py --config ./configs/config_blender_tiny.ini
tenorboard --logdir=../logs/ --port=6006
```

Cấu hình này sẽ huấn luyện một mạng cho kết quả với độ phân giải thấp. Để huấn luyện các cảnh ở độ phân giải cao (cần thời gian huấn luyện lâu hơn), hãy sử dụng các tệp cấu hình `config_blender.ini`, `config_deepvoxels.ini` hoặc `config_llff.ini`.

### Kết xuất

Kết xuất từ một mô hình đã được huấn luyện có thể được thực hiện bằng lệnh sau.
```
python train_autoint_radiance_field.py --config /path/to/config/file --render_model ../logs/path/to/log/directory <epoch number> --render_output /path/to/output/folder
```

Ở đây, tham số `--render_model` cho biết thư mục nơi mô hình và checkpoints được lưu. Ví dụ: nó sẽ là `../logs/blender_lego` cho tập dữ liệu mặc định của Blender. Sau đó, bạn có thể tìm thấy số epoch bằng cách xem số của tên tệp checkpoints đã lưu trong `../logs/blender_lego/checkpoints/`. Cuối cùng, `--render_output` sẽ chỉ định một thư mục nơi các hình ảnh đầu ra sẽ được lưu.
