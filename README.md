  基于st的multizone的tof的ai开发
  先尝试了纯分类的大概能用，dbscan效果最好，如果家具一定是矩形就可以分别，精度不高
  然后尝试了，纯2d卷积，效果一般，中心可以分别但是棱角会扩散开{
  在“2d...”文件夹中有单独的readme
  预处理，转换成depth map，保存数据
python data_loader.py --function wash_and_save

python data_loader.py --function print_pinned_labels

python train.py train --dataset_path washed_labeled_data.pt --epochs 8 --batch_size 32
python train.py evaluate --dataset_path washed_labeled_data.pt --model_path simplified_alexnet_model.pth
python train.py predict --dataset_path washed_labeled_data.pt --model_path simplified_alexnet_model.pth

  }

  然后因为点云图难以标注，默认家具是规则矩形来分类还好，计算每个顶尖所在区域后判断标注就行了，但参考exel文件中视角就能知道如果不是规则立方体就很难了
因此用blender做了sim，得到exr点云

第三部分就是点云的
  
