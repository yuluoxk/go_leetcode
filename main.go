package main

import "fmt"

func main() {
	// 169
	//a := []int{1, 2, 2, 3, 4, 2, 2, 2}
	//b := majorityElement(a)
	//a := [][]byte{{1, 1, 1, 1, 0}, {1, 1, 0, 1, 0}, {1, 1, 0, 0, 0}, {0, 0, 0, 0, 0}}
	//a := [][]byte{{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}}
	//b := numIslands(a)
	//fmt.Println(b)
	l5 := ListNode{Val: 5, Next: nil}
	l4 := ListNode{Val: 4, Next: &l5}
	l3 := ListNode{Val: 3, Next: &l4}
	l2 := ListNode{Val: 2, Next: &l3}
	l1 := ListNode{Val: 1, Next: &l2}
	lp := reverseList(&l1)
	fmt.Println(lp.Val)
	fmt.Println(lp.Next.Val)
	fmt.Println(lp.Next.Next.Val)
	fmt.Println(lp.Next.Next.Next.Val)
	fmt.Println(lp.Next.Next.Next.Next.Val)
	fmt.Println(lp.Next.Next.Next.Next.Val)
}

type ListNode struct {
	Val  int
	Next *ListNode
}

// 206. 反转链表
func reverseList(head *ListNode) *ListNode {
	var pre *ListNode
	curr := head
	for curr != nil {
		var nextNode = curr.Next
		curr.Next = pre
		pre = curr
		curr = nextNode
	}
	return pre
}

func testByte() {
	//a := []byte {1,2,3,1,2,3}
	a := []byte{'1', '2', '3', '1', '2', '3'}
	for _, value := range a {
		if value == 1 {
			fmt.Println(value)
		}
	}
}

// 200. 岛屿数量
//grid [][]byte 没有指定数量是切片
// '1' 和 1 不是一个东西
func numIslands(grid [][]byte) int {
	if grid == nil || len(grid) == 0 {
		return 0
	}
	count := 0
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == '1' {
				count++
				dfs(grid, i, j)

			}
		}
	}
	return count
}

func dfs(grid [][]byte, r int, c int) {
	if grid == nil || len(grid) == 0 {
		return
	}
	if r < 0 || r >= len(grid) || c < 0 || c >= len(grid[0]) {
		return
	}
	if grid[r][c] == '0' {
		return
	}
	if grid[r][c] == '2' {
		return
	}
	grid[r][c] = '2'
	dfs(grid, r-1, c)
	dfs(grid, r+1, c)
	dfs(grid, r, c-1)
	dfs(grid, r, c+1)
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
