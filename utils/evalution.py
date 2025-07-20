# 使用鞋带公式（也称为高斯面积公式）来计算多边形的面积
# 这个示例假设四边形的顶点是按照顺时针或逆时针顺序提供的。如果顶点的顺序不正确，计算的面积可能会是负值
import numpy as np

def calculate_bbox(polygon):
    return [
        min(polygon, key=lambda x: x[0])[0],  # 最小经度
        min(polygon, key=lambda x: x[1])[1],  # 最小纬度
        max(polygon, key=lambda x: x[0])[0],  # 最大经度
        max(polygon, key=lambda x: x[1])[1]   # 最大纬度
    ]

# 计算交集和并集的边界框
def calculate_overlap_and_union(bbox1, bbox2):
    overlap_bbox = [
        max(bbox1[0], bbox2[0]),
        max(bbox1[1], bbox2[1]),
        min(bbox1[2], bbox2[2]),
        min(bbox1[3], bbox2[3])
    ]
    union_bbox = [
        min(bbox1[0], bbox2[0]),
        min(bbox1[1], bbox2[1]),
        max(bbox1[2], bbox2[2]),
        max(bbox1[3], bbox2[3])
    ]
    return overlap_bbox, union_bbox

# 计算面积
def calculate_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

# 计算IoU
def calculate_iou(polygon1, polygon2):
    bbox1 = calculate_bbox(polygon1)
    bbox2 = calculate_bbox(polygon2)
    
    overlap_bbox, union_bbox = calculate_overlap_and_union(bbox1, bbox2)
    intersection_area = calculate_area(overlap_bbox) if overlap_bbox[0] < overlap_bbox[2] and overlap_bbox[1] < overlap_bbox[3] else 0
    union_area = calculate_area(union_bbox)
    
    return intersection_area / union_area if union_area else 0

def calculate_iou_dict(estimated_dict, real_dict, write_path):

    # 计算多个估计值和真实值之间的IoU
    ious = {}
    all_temp = 0
    with open(write_path, 'w') as f:
        for key in estimated_dict.keys():
            if key in real_dict:
                if estimated_dict[key] != [None]*8:
                    ious[key] = calculate_iou(estimated_dict[key], real_dict[key])
                    all_temp += ious[key]
                    info = key + ' ' + str(ious[key]) + '\n'
                    f.write(info)
                else:
                    info = key + ' ' + str(0) + '\n'
                    f.write(info)
                    
        
    return ious, all_temp/len(real_dict)

# # 示例：估计的四个点和真实的四个点，每个点是一个 (x, y) 坐标
# estimated_polygon = [(10, 20), (30, 40), (50, 30), (10, 10)]  # 估计的四边形顶点
# real_polygon = [(15, 25), (35, 45), (55, 35), (15, 15)]      # 真实的四边形顶点

# # 计算IoU
# iou = calculate_iou(estimated_polygon, real_polygon)
# print(f"The IoU between the estimated and real polygons is: {iou:.2f}")