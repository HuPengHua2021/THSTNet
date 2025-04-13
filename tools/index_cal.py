def cal_patch_index(patch_size, image_size, stride):

    h_list = [i for i in range(0, image_size[0] - patch_size + 1, stride)]

    # 如果最后一块没有覆盖到边界，添加一块覆盖边界
    if h_list[-1] + patch_size < image_size[0]:
        h_list.append(image_size[0] - patch_size)

    w_list = [i for i in range(0, image_size[1] - patch_size + 1, stride)]

    # 如果最后一块没有覆盖到边界，添加一块覆盖边界
    if w_list[-1] + patch_size < image_size[1]:
        w_list.append(image_size[1] - patch_size)
    return h_list, w_list


def test_fill_index(i, j, h_start, w_start, h_list, w_list, patch_size):
    # 预测时每个patch的预测结果只保留中间一半
    h_end = h_start + patch_size
    w_end = w_start + patch_size
    patch_h_start = 0
    patch_h_end = patch_size
    patch_w_start = 0
    patch_w_end = patch_size

    if i != 0:
        h_start = h_start + patch_size // 4
        patch_h_start = patch_size // 4

    if i != len(h_list) - 1:
        h_end = h_end - patch_size // 4
        patch_h_end = patch_h_end - patch_size // 4

    if j != 0:
        w_start = w_start + patch_size // 4
        patch_w_start = patch_size // 4

    if j != len(w_list) - 1:
        w_end = w_end - patch_size // 4
        patch_w_end = patch_w_end - patch_size // 4

    fill_index = [h_start, h_end, w_start, w_end]
    patch_index = [patch_h_start, patch_h_end, patch_w_start, patch_w_end]

    return fill_index, patch_index
