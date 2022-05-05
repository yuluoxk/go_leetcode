package main

import "fmt"

func main() {
	// 169
	a := []int{1, 2, 2, 3, 4, 2, 2, 2}
	//b := majorityElement(a)
	c := rob(a)
	fmt.Println(c)
}

// 198 打家劫舍
func rob(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	if len(nums) == 1 {
		return nums[0]
	}

	dp := make([]int, len(nums))
	dp[0] = nums[0]
	dp[1] = max(nums[0], nums[1])
	for i := 2; i < len(nums); i++ {
		dp[i] = max(dp[i-2]+nums[i], dp[i-1])
	}
	return dp[len(dp)-1]
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
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
