a = [1, 2, 2, 2, 2, 3, 4, 4, 4, 6, 7]
b = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# 二分查找第一个是value的index
def bc(a, value):
    la = len(a) - 1
    low = 0
    high = la
    while low <= high:
        # mid = int(low + ( (high - low) >> 1))
        mid = low + ( (high - low) >> 1)
        if a[mid] > value:
            high = mid - 1
        elif a[mid] < value:
            low = mid + 1
        else:
            if mid == 0 or a[mid - 1] != value:
                return mid
            else:
                high = mid - 1
    return -1

print(bc(a,9))
print(bc(a,2))
print(bc(a,4))
mid = 11>>1
print(a[mid])

# 二分查找第一个大于等于value的index
def bc2(a, value):
    la = len(a) - 1
    low = 0
    high = la
    while low <= high:
        mid = low + ( (high - low) >> 1)
        if a[mid] >= value:
            if mid == 0 or a[mid - 1] < value:
                return mid
            else:
                high = mid - 1
        else:
            low = mid + 1
    return -1

print(bc2(a,9))
print(bc2(a,5))
print(bc2(a,4))
