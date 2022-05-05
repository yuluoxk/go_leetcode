package main

import "fmt"

func main() {
	// 169
	a := []int{1, 2, 2, 3, 4, 2, 2, 2}
	b := majorityElement(a)
	fmt.Println(b)
}

// 169 多数元素
func majorityElement(nums []int) int {
	var endNum, count int = nums[0], 1
	for i := 1; i < len(nums); i++ {
		value := nums[i]
		if endNum == value {
			count++
		} else {
			count--
			if count == 0 {
				endNum = value
				count = 1
			}
		}
	}
	return endNum
}
