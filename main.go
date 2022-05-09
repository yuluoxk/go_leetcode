package main

import (
	"fmt"
	"math"
)

func main() {
	// 169
	//a := []int{1, 2, 2, 3, 4, 2, 2, 2}
	//b := majorityElement(a)
	//a := [][]byte{{1, 1, 1, 1, 0}, {1, 1, 0, 1, 0}, {1, 1, 0, 0, 0}, {0, 0, 0, 0, 0}}
	//a := [][]byte{{'1', '1', '1', '1', '0'}, {'1', '1', '0', '1', '0'}, {'1', '1', '0', '0', '0'}, {'0', '0', '0', '0', '0'}}
	//b := numIslands(a)
	//fmt.Println(b)

	// 206. 反转链表
	//l5 := ListNode{Val: 5, Next: nil}
	//l4 := ListNode{Val: 4, Next: &l5}
	//l3 := ListNode{Val: 3, Next: &l4}
	//l2 := ListNode{Val: 2, Next: &l3}
	//l1 := ListNode{Val: 1, Next: &l2}
	//lp := reverseList(&l1)
	//fmt.Println(lp.Val)
	//fmt.Println(lp.Next.Val)
	//fmt.Println(lp.Next.Next.Val)
	//fmt.Println(lp.Next.Next.Next.Val)
	//fmt.Println(lp.Next.Next.Next.Next.Val)
	//fmt.Println(lp.Next.Next.Next.Next.Val)
	//pre := [][]int{{1, 0}, {1, 2}, {2, 3}, {3, 4}}
	//a := canFinish(5, pre)
	//fmt.Println(a)

	//a := []int{1, 2, 3, 5, 6, 9, 12}
	//b := search(a, 9)
	//fmt.Println(b)

	a := []int{1, 2, 3, 5, 6, 9, 12}
	b := minSubArrayLen(11, a)
	fmt.Println(b)
}

// 209. 长度最小的子数组 _滑动窗口
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	ans := math.MaxInt32
	start, end := 0, 0
	sum := 0
	for end < n {
		sum = sum + nums[end]
		for sum >= target {
			ans = min(ans, end-start+1)
			sum -= nums[start]
			start += 1
		}
		end += 1
	}

	if ans == math.MaxInt32 {
		return 0
	}
	return ans
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

//704. 二分查找
func search(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := (high-low)/2 + low
		num := nums[mid]
		if num == target {
			return mid
		} else if num > target {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}
	return -1
}

// 208. 实现 Trie (前缀树)
type Trie struct {
	children [26]*Trie
	isEnd    bool
}

func Constructor() Trie {
	return Trie{}
}

func (this *Trie) Insert(word string) {
	node := this
	for _, ch := range word {
		ch -= 'a'
		if node.children[ch] == nil {
			node.children[ch] = &Trie{}
		}
		node = node.children[ch]
	}
	node.isEnd = true
}

func (this *Trie) SearchPrefix(word string) *Trie {
	node := this
	for _, ch := range word {
		ch -= 'a'
		if node.children[ch] == nil {
			return nil
		}
		node = node.children[ch]
	}
	return node
}

func (this *Trie) Search(word string) bool {
	node := this.SearchPrefix(word)
	return node != nil && node.isEnd
}

func (this *Trie) StartsWith(prefix string) bool {
	return this.SearchPrefix(prefix) != nil
}

//207. 课程表
func canFinish(numCourses int, prerequisites [][]int) bool {
	var (
		edges  = make([][]int, numCourses)
		indeg  = make([]int, numCourses)
		result []int
	)

	for _, info := range prerequisites {
		edges[info[1]] = append(edges[info[1]], info[0])
		indeg[info[0]]++
	}

	q := []int{}
	for i := 0; i < numCourses; i++ {
		if indeg[i] == 0 {
			q = append(q, i)
		}
	}

	for len(q) > 0 {
		u := q[0]
		q = q[1:]
		result = append(result, u)
		for _, v := range edges[u] {
			indeg[v]--
			if indeg[v] == 0 {
				q = append(q, v)
			}
		}
	}

	return len(result) == numCourses
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
