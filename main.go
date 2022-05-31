package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"time"
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

	//a := []int{2, 3,1,2,4,3}
	//b := minSubArrayLenn(7, a)
	//fmt.Println(b)

	//a := []int{1}
	//b := findKthLargest_(a, 1)
	//fmt.Println(b)

	//a := [][]byte{{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}}
	//b := maximalSquare(a)
	//fmt.Println(b)

	a := [][]int{{5}, {6}}
	b := searchMatrix1(a, 6)
	fmt.Println(b)
}

//240. 搜索二维矩阵 II 二分查找
func searchMatrix1(matrix [][]int, target int) bool {
	for _, row := range matrix {
		fmt.Println(row)
		i := sort.SearchInts(row, target)
		if i < len(row) && row[i] == target {
			return true
		}
	}
	return false
}

//240. 搜索二维矩阵 II Z字形查找
func searchMatrix(matrix [][]int, target int) bool {
	m := len(matrix)
	n := len(matrix[0])
	i, j := 0, n-1
	for i < m && j >= 0 {
		if matrix[i][j] == target {
			return true
		} else if matrix[i][j] > target {
			j--
		} else {
			i++
		}
	}
	return false
}

//238. 除自身以外数组的乘积
func productExceptSelf(nums []int) []int {
	length := len(nums)
	ans := make([]int, length)

	ans[0] = 1
	for i := 1; i < length; i++ {
		ans[i] = ans[i-1] * nums[i-1]
	}
	r := 1
	for i := length - 1; i >= 0; i-- {
		ans[i] = ans[i] * r
		r = r * nums[i]
	}
	return ans
}

//238. 除自身以外数组的乘积 (左右乘积列表)
func productExceptSelf1(nums []int) []int {
	length := len(nums)
	left, right, ans := make([]int, length), make([]int, length), make([]int, length)
	left[0] = 1
	for i := 1; i < length; i++ {
		left[i] = left[i-1] * nums[i-1]
	}

	right[length-1] = 1
	for i := length - 2; i >= 0; i-- {
		right[i] = right[i+1] * nums[i+1]
	}

	for i := 0; i < length; i++ {
		ans[i] = left[i] * right[i]
	}

	return ans
}

//236. 二叉树的最近公共祖先
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	parent := map[int]*TreeNode{}
	visited := map[int]bool{}

	var dfs func(node *TreeNode)
	dfs = func(r *TreeNode) {
		if r == nil {
			return
		}
		if r.Left != nil {
			parent[r.Left.Val] = r
			dfs(r.Left)
		}
		if r.Right != nil {
			parent[r.Right.Val] = r
			dfs(r.Right)
		}
	}
	dfs(root)

	for p != nil {
		visited[p.Val] = true
		p = parent[p.Val]
	}

	for q != nil {
		if visited[q.Val] == true {
			return q
		}
		q = parent[q.Val]
	}
	return nil
}

//236. 二叉树的最近公共祖先(递归)
func lowestCommonAncestor1(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	if root.Val == p.Val || root.Val == q.Val {
		return root
	}
	left := lowestCommonAncestor1(root.Left, p, q)
	right := lowestCommonAncestor1(root.Right, p, q)
	if left != nil && right != nil {
		return root
	}
	if left == nil {
		return right
	}
	return left
}

//206. 反转链表 —迭代
func reverseList1(head *ListNode) *ListNode {
	var prev *ListNode = nil
	curr := head
	for curr != nil {
		next := curr.Next
		curr.Next = prev
		prev = curr
		curr = next
	}
	return prev
}

//206. 反转链表 —递归
func reverseList2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	p := reverseList2(head.Next)
	head.Next.Next = head
	head.Next = nil
	return p
}

//234. 回文链表
func isPalindrome(head *ListNode) bool {
	vals := []int{}
	for head != nil {
		vals = append(vals, head.Val)
		head = head.Next
	}
	start, end := 0, len(vals)-1
	for start < end {
		if vals[start] != vals[end] {
			return false
		}
		start++
		end--
	}
	return true
}

//234. 回文链表
func isPalindrome2(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}

	var prev *ListNode = nil
	fast, slow := head, head
	for fast != nil && fast.Next != nil {
		fast = fast.Next.Next
		curr := slow.Next
		slow.Next = prev
		prev = slow
		slow = curr
	}

	if fast != nil {
		slow = slow.Next
	}

	for slow != nil && prev != nil {
		if slow.Val != prev.Val {
			return false
		}
		slow = slow.Next
		prev = prev.Next
	}
	return true
}

//226. 翻转二叉树
func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left = right
	root.Right = left
	return root
}

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

//1277. 统计全为 1 的正方形子矩阵
func countSquares(matrix [][]int) int {
	dp := make([][]int, len(matrix))
	ans := 0

	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			if i == 0 || j == 0 {
				dp[i][j] = matrix[i][j]
			} else if matrix[i][j] == 0 {
				dp[i][j] = 0
			} else {
				dp[i][j] = min(min(dp[i-1][j], dp[i][j-1]), dp[i-1][j-1]) + 1
			}
			ans += dp[i][j]
		}
	}

	return ans
}

//221. 最大正方形
func maximalSquare(matrix [][]byte) int {
	dp := make([][]int, len(matrix))
	maxSize := 0

	if len(matrix) == 0 {
		return 0
	}

	for i := 0; i < len(matrix); i++ {
		dp[i] = make([]int, len(matrix[i]))
		for j := 0; j < len(matrix[i]); j++ {
			dp[i][j] = int(matrix[i][j] - '0')
			if dp[i][j] == 1 {
				maxSize = 1
			}
		}
	}

	for i := 1; i < len(matrix); i++ {
		for j := 1; j < len(matrix[i]); j++ {
			if dp[i][j] == 1 {
				dp[i][j] = min(min(dp[i][j-1], dp[i-1][j]), dp[i-1][j-1]) + 1
				if dp[i][j] > maxSize {
					maxSize = dp[i][j]
				}
			}
		}
	}

	return maxSize * maxSize
}

// 215. 数组中的第K个最大元素_快速排序随机选
func findKthLargest_(nums []int, k int) int {
	rand.Seed(time.Now().UnixNano())
	return quickSelect(nums, 0, len(nums)-1, len(nums)-k)
}

func quickSelect(nums []int, left, right, index int) int {
	if left == right {
		return nums[left]
	}
	q := randomPartition(nums, left, right)
	if q == index {
		return nums[q]
	} else if q < index {
		return quickSelect(nums, q+1, right, index)

	} else {
		return quickSelect(nums, left, q-1, index)
	}
}

func randomPartition(nums []int, left, right int) int {
	i := rand.Int()%(right-left) + left
	nums[i], nums[right] = nums[right], nums[i]
	return partition(nums, left, right)
}

func partition(nums []int, left, right int) int {
	pivot := nums[right]
	j := left - 1
	for i := left; i < right; i++ {
		if nums[i] <= pivot {
			j++
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
	nums[right], nums[j+1] = nums[j+1], nums[right]
	return j + 1
}

// 215. 数组中的第K个最大元素_堆排
func findKthLargest(nums []int, k int) int {
	heapSize := len(nums)
	makeMaxHeap(nums, heapSize)
	for i := len(nums) - 1; i >= len(nums)-k+1; i-- {
		nums[0], nums[i] = nums[i], nums[0]
		heapSize--
		makeMaxHeap(nums, heapSize)
	}
	return nums[0]
}

func makeMaxHeap(nums []int, heapSize int) {
	for i := heapSize / 2; i >= 0; i-- {
		maxHeapify(nums, i, heapSize)
	}
}

func maxHeapify(nums []int, i int, heapSize int) {
	left, right, largest := 2*i+1, 2*i+2, i
	if left < heapSize && nums[left] > nums[largest] {
		largest = left
	}
	if right < heapSize && nums[right] > nums[largest] {
		largest = right
	}
	if largest != i {
		nums[i], nums[largest] = nums[largest], nums[i]
		maxHeapify(nums, largest, heapSize)
	}
}

// 209. 长度最小的子数组_二分查找
func minSubArrayLenn(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}

	ans := math.MaxInt32
	sums := make([]int, n+1)
	sums[0] = 0
	for i := 1; i < n; i++ {
		sums[i] = sums[i-1] + nums[i-1]
	}

	for i := 0; i <= n; i++ {
		target = target + sums[i]
		bound := sort.SearchInts(sums, target)
		if bound <= n {
			ans = min(ans, bound)
		}
	}

	if ans == math.MaxInt32 {
		return 0
	}
	return ans
}

// 209. 长度最小的子数组 _滑动窗口
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	ans := math.MaxInt32
	sums := make([]int, n+1)
	// 为了方便计算，令 size = n + 1
	// sums[0] = 0 意味着前 0 个元素的前缀和为 0
	// sums[1] = A[0] 前 1 个元素的前缀和为 A[0]
	// 以此类推
	for i := 1; i <= n; i++ {
		sums[i] = sums[i-1] + nums[i-1]
	}
	for i := 1; i <= n; i++ {
		target := target + sums[i-1]
		bound := sort.SearchInts(sums, target)
		fmt.Println(bound)
		if bound < 0 {
			bound = -bound - 1
		}
		if bound <= n {
			ans = min(ans, bound-(i-1))
		}
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
