import numpy as np
def int_to_matrix(integer_representation, rows=3, cols=5):
    """将一个整数解码回一个3x5的矩阵"""
    if not 0 <= integer_representation < cols**rows:
        raise ValueError(f"整数必须在 0 到 {cols**rows - 1} 之间")

    # 1. 创建一个全为0的目标矩阵
    matrix = np.zeros((rows, cols), dtype=int)
    
    temp_int = integer_representation
    
    # 2. 使用取模和整除运算来逐位解码
    for i in range(rows):
        # 使用取模运算找到当前数位的“数字” (即'1'的位置)
        pos = temp_int % cols
        matrix[i, pos] = 1
        
        # 使用整除运算去掉当前数位，为下一位数做准备
        temp_int //= cols
        
    return matrix
def aggregation_choice_to_matrix(aggregation_choice):
            """将 aggregation_choice 整数转换为 (N+1) x (M+1) 的矩阵"""
            num_users = 5
            num_edges = 2
            if not 0 <= aggregation_choice < (num_edges + 1)**(num_users + 1):
                raise ValueError(f"aggregation_choice 必须在 0 到 {(num_edges + 1)**(num_users + 1) - 1} 之间")

            matrix = np.zeros((num_users + 1, num_edges + 1), dtype=int)
            temp_int = aggregation_choice

            # 解码聚合决策 (第0行)
            pos = temp_int % (num_edges + 1)
            matrix[0, pos] = 1
            temp_int //= (num_edges + 1)

            # 解码每个用户的卸载决策
            for i in range(1, num_users + 1):
                pos = temp_int % (num_edges + 1)
                matrix[i, pos] = 1
                temp_int //= (num_edges + 1)

            return matrix
def main():
    # 测试
    test_int = 728
    matrix = aggregation_choice_to_matrix(test_int)
    print(matrix)

if __name__ == "__main__":
    main()
    