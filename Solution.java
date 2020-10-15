import javax.swing.tree.TreeNode;
import java.util.*;

public class Solution {

    /**
     * You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes
     * contain a single digit. Add the two numbers and return it as a linked list. Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
     * Output: 7 -> 0 -> 8
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(0);
        ListNode head = dummy;
        int carry = 0;
        while (l1 != null || l2 != null){
            int n1 = l1 == null ? 0 : l1.val;
            int n2 = l2 == null ? 0 : l2.val;
            ListNode node = new ListNode((n1+n2+carry)%10);
            carry = (n1+n2+carry)/10;
            dummy.next = node;
            dummy = dummy.next;
            if(l1 != null){
                l1 = l1.next;
            }
            if(l2 != null){
                l2 = l2.next;
            }
        }
        if(carry != 0){
            dummy.next = new ListNode(carry);
        }
        return head.next;
    }

    /**
     * Given a string, find the length of the longest substring without repeating characters. example: Given "abcabcbb", the answer is "abc", which the length is 3
     * Given "bbbbb", the answer is "b", with the length of 1 and Given "pwwkew", the answer is "wke", with the length of 3.Note that the answer must be a substring
     */
    public int lengthOfLongestSubstring(String s) {
        int start = -1;
        int maxLen = 0;
        Map<Character, Integer> map = new HashMap<>();
        for(int i=0; i<s.length(); i++){
            char ch = s.charAt(i);
            if(map.containsKey(ch)){
                start = Math.max(start, map.get(ch));
            }
            map.put(ch, i);
            maxLen = Math.max(maxLen, i-start);
        }
        return maxLen;
    }

    /**
     * There are two sorted arrays nums1 and nums2 of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
     * Example: nums1 = [1, 3] nums2 = [2] The median is 2.0 and nums1 = [1, 2] nums2 = [3, 4] The median is (2 + 3)/2 = 2.5
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        return 0;
    }

    public String longestPalindrome(String s) {
        return null;
    }

    /**
     * Reverse digits of an integer. Example1: x = 123, return 321 and x = -123, return -321
     */
    public int reverse(int x) {
        int ret = 0;
        while(x != 0){
            if(ret * 10 / 10 != ret){//integer overflow
                return 0;
            }
            ret = ret * 10 + x%10;
            x/=10;
        }
        return ret;
    }

    /**
     * Implement regular expression matching with support for '.' and '*'  '.' Matches any single character. '*' Matches zero or more of the preceding element.
     * The matching should cover the entire input string (not partial). examples: isMatch("aa","aa") → true  isMatch("aaa","aa") → false
     * isMatch("aa", "a*") → true  isMatch("aa", ".*") → true  isMatch("ab", ".*") → true  isMatch("aab", "c*a*b") → true
     */
    public boolean isPatternMatch(String s, String p) {
        if (s == null || p == null) {
            return false;
        }
        boolean[][] dp = new boolean[s.length()+1][p.length()+1];
        dp[0][0] = true;
        for (int i = 0; i < p.length(); i++) {
            if (p.charAt(i) == '*' && dp[0][i-1]) {
                dp[0][i+1] = true;
            }
        }
        for (int i = 0 ; i < s.length(); i++) {
            for (int j = 0; j < p.length(); j++) {
                if (p.charAt(j) == '.') {
                    dp[i+1][j+1] = dp[i][j];
                }
                if (p.charAt(j) == s.charAt(i)) {
                    dp[i+1][j+1] = dp[i][j];
                }
                if (p.charAt(j) == '*') {
                    if (p.charAt(j-1) != s.charAt(i) && p.charAt(j-1) != '.') {
                        dp[i+1][j+1] = dp[i+1][j-1];
                    } else {
                        dp[i+1][j+1] = (dp[i+1][j] || dp[i][j+1] || dp[i+1][j-1]);
                    }
                }
            }
        }
        return dp[s.length()][p.length()];
    }

    public boolean isPatternMatch2(String s, String p) {
        boolean ret = true;
        if(s != null && p != null){
            int i = 0, j = 0;
            int starIdx = -1;
            int match = 0;
            while(i<s.length()){
                char ch1 = s.charAt(i);
                if(j<p.length() && (ch1 == p.charAt(j) || p.charAt(j) == '?')){
                    i++;
                    j++;
                }else if(j<p.length() && p.charAt(j) == '*'){
                    starIdx = j;
                    match = i;
                    j++;
                }else if(starIdx != -1){
                    j = starIdx + 1;
                    match++;
                    i = match;
                }else{
                    return false;
                }
            }
            while(j < p.length() && p.charAt(j) == '*'){
                j++;
            }
            ret = j == p.length();
        }
        return ret;
    }

    /**
     * Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
     * Note: The solution set must not contain duplicate triplets. Example given array S = [-1, 0, 1, 2, -1, -4], A solution set is:
     * [ [-1, 0, 1],  [-1, -1, 2]  ]
     */
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new LinkedList<>();
        Arrays.sort(nums);
        for(int i=0;i<nums.length-2;++i){
            if(i>0 && nums[i]==nums[i-1]) continue;
            int l=i+1, r=nums.length-1;
            while(l<r){
                if(nums[i]+nums[l]+nums[r]==0){
                    res.add(Arrays.asList(nums[i], nums[l], nums[r]));
                    while(++l<r && nums[l]==nums[l-1]);
                    while(--r>l && nums[r]==nums[r+1]);
                }else if(nums[i]+nums[l]+nums[r]>0){
                    while(--r>l && nums[r]==nums[r+1]);
                }else{
                    while(++l<r && nums[l]==nums[l-1]);
                }
            }
        }
        return res;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> ret = new LinkedList<>();
        if(nums.length >= 4){
            Arrays.sort(nums);
            int i=0;
            for(;i<nums.length;){
                for(int j=i+1; j<nums.length;){
                    for(int k=j+1; k<nums.length;){
                        for(int m=k+1; m<nums.length;){
                            int sum = nums[i]+nums[j]+nums[k]+nums[m];
                            if(sum == target){
                                List<Integer> row = new ArrayList<>();
                                row.add(nums[i]);
                                row.add(nums[j]);
                                row.add(nums[k]);
                                row.add(nums[m]);
                                ret.add(row);
                                break;
                            }else if(sum < target){
                                while(++m<nums.length && nums[m]==nums[m-1]);
                            }else{
                                break;
                            }
                        }
                        while(++k<nums.length && nums[k]==nums[k-1]);
                    }
                    while(++j<nums.length && nums[j]==nums[j-1]);
                }
                while(++i<nums.length && nums[i]==nums[i-1]);
            }
        }
        return ret;
    }

    /**
     * Given a digit string, return all possible letter combinations that the number could represent. A mapping of digit to letters (just like on the telephone buttons)
     * Input:Digit string "23" Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     */
    public List<String> letterCombinations(String digits) {
        String[] sets = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> ret = new ArrayList<String>();
        letterCombinationsHelper(ret, sets, digits, 0, "");
        return ret;
    }

    private void letterCombinationsHelper(List<String> ret, String[] set, String digits, int start, String item){
        if(digits.length() == item.length()){
            ret.add(item);
            return;
        }
        int idx = digits.charAt(start) - 50;
        String str = set[idx];
        for(int i=0; i<str.length(); i++){
            letterCombinationsHelper(ret, set, digits, start+1, item+str.charAt(i));
        }
    }

    /**
     * Given a linked list, remove the nth node from the end of list and return its head. Example given 1->2->3->4->5, and n = 2 output 1->2->3->5
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null){
            fast = fast.next;
            if(n-- < 0){
                slow = slow.next;
            }
        }
        if(n == 0){
            head = head.next;
        }else{
            slow.next = slow.next.next;
        }
        return head;
    }

    /**
     * Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses. example given n = 3, a solution set is:
     * [ "((()))", "(()())", "(())()", "()(())", "()()()" ]
     */
    public List<String> generateParenthesis(int n) {
        List<String> ret = new ArrayList<>();
        generateParenthesisDFS(ret, "", n, n);
        return ret;
    }

    private void generateParenthesisDFS(List<String> ret, String str, int left, int right){
        if(left == 0 && right == 0){
            ret.add(str);
            return;
        }
        if(left > 0){
            generateParenthesisDFS(ret, str+"(", left-1, right);
        }
        if(right > left){
            generateParenthesisDFS(ret, str+")", left, right-1);
        }
    }

    /**
     * Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity
     */
    public ListNode mergeKLists(ListNode[] lists) {
        if(lists == null || lists.length == 0){
            return null;
        }
        int start = 0, end = lists.length -1;
        return splitKList(lists, start, end);
    }

    private ListNode splitKList(ListNode[] lists, int start, int end){
        if(start == end){
            return lists[start];
        }
        int m = (start+end)/2;
        return mergeSorts(splitKList(lists, start, m), splitKList(lists, m+1, end));
    }

    private ListNode mergeSorts(ListNode l1, ListNode l2){
        if(l1 == null){
            return l2;
        }
        if(l2 == null){
            return l1;
        }
        if(l1.val < l2.val){
            l1.next = mergeSorts(l1.next, l2);
            return l1;
        }else{
            l2.next = mergeSorts(l2.next, l1);
            return l2;
        }
    }

    /**
     * Given a linked list, swap every two adjacent nodes and return its head. Given 1->2->3->4, you should return the list as 2->1->4->3.
     */
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy;
        ListNode fast = head;
        ListNode next = null;
        while(fast != null && fast.next != null){
            next = fast.next;
            slow.next = next;
            fast.next = next.next;
            next.next = fast;
            slow = fast;
            fast = fast.next;
        }
        return dummy.next;
    }

    /**
     * Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.If the number of nodes is not a multiple of k
     * then left-out nodes in the end should remain as it is. You may not alter the values in the nodes, only nodes itself may be changed. example
     * Given list: 1->2->3->4->5 For k = 2, you should return: 2->1->4->3->5 For k = 3, you should return: 3->2->1->4->5
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if(head == null){
            return head;
        }
        ListNode tail = head;
        for(int i=1; i<k; i++){
            tail = tail.next;
            if(tail == null){
                return head;
            }
        }
        ListNode nextBound = tail.next;
        ListNode slow = head;
        ListNode fast = head.next;
        ListNode next = null;
        for(int i=1; i<k; i++){
            next = fast.next;
            fast.next = slow;
            slow = fast;
            fast = next;
        }
        head.next = reverseKGroup(nextBound, k);
        return tail;
    }

    /**
     * Reverse a linked list from position m to n. Do it in-place and in one-pass. e.g Given 1->2->3->4->5->NULL, m = 2 and n = 4,
     * return 1->4->3->2->5->NULL. Note 1 ≤ m ≤ n ≤ length of list.
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode slow = dummy;
        int idx = 1;
        while(idx < m){
            slow = slow.next;
            idx++;
        }
        ListNode start = slow;
        slow = slow.next;
        ListNode fast = slow.next;
        ListNode tmp;
        while(m < n){
            tmp = fast.next;
            fast.next = slow;
            slow = fast;
            fast = tmp;
            m++;
        }
        start.next.next = fast;
        start.next = slow;
        return dummy.next;
    }

    public ListNode inplaceReverseLinkList(ListNode head){
        ListNode slow = null;
        ListNode fast = head;
        ListNode next = null;
        while(fast != null){
            next = fast.next;
            fast.next = slow;
            slow = fast;
            fast = next;
        }
        return slow;
    }

    /**
     * Given a list, rotate the list to the right by k places, where k is non-negative.Example: Given 1->2->3->4->5->NULL and k = 2, return 4->5->1->2->3->NULL.
     */
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null || head.next == null || k == 0){
            return head;
        }
        ListNode cur = head;
        int len = 1;
        while(cur.next != null){
            len++;
            cur = cur.next;
        }
        cur.next = head;
        for(int i=0; i<len-k%len; i++){
            cur = cur.next;
        }
        ListNode ret = cur.next;
        cur.next = null;
        return ret;
    }

    /**
     * You are given a string, s, and a list of words, words, that are all of the same length. Find all starting indices of substring(s) in s that is a concatenation
     *  of each word in words exactly once and without any intervening characters.Given s: "barfoothefoobarman" words: ["foo", "bar"] return [0,9]
     */
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> ret = new ArrayList<>();
        Map<String, Boolean> map = new HashMap<>();
        findSubstringHelperDFS(s, ret, Arrays.asList(words), map, "");
        return ret;
    }

    private void findSubstringHelperDFS(String s, List<Integer> ret, List<String> words, Map<String, Boolean> map,String item){
        if(words.isEmpty()){
            int idx = s.indexOf(item);
            if(idx > -1 && !map.containsKey(item)){
                while(idx > -1){
                    ret.add(idx);
                    idx = s.indexOf(item, idx+1);
                }
                map.put(item, true);
            }
            return;
        }
        for(int i=0; i<words.size(); i++){
            List<String> copy = new ArrayList<>(words);
            String str = copy.remove(i);
            findSubstringHelperDFS(s, ret, copy, map, item+str);
        }
    }

    /**
     * Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.If such arrangement is not possible,
     * it must rearrange it as the lowest possible order (ie, sorted in ascending order).The replacement must be in-place. Given 1,2,3 → 1,3,2
     * and 3,2,1 → 1,2,3 and 1,1,5 → 1,5,1
     */
    public void nextPermutation(int[] nums) {
        if(nums != null && nums.length >0){
            int k = -1;
            for(int i=nums.length -2; i>=0; i--){
                if(nums[i] < nums[i+1]){
                    k = i;
                    break;
                }
            }
            if(k == -1){
                reverse(nums, 0, nums.length-1);
                return;
            }
            int m = -1;
            for(int i=nums.length - 1; i>k; i--){
                if(nums[i] > nums[k]){
                    m = i;
                    break;
                }
            }
            int tmp = nums[k];
            nums[k]=nums[m];
            nums[m] = tmp;
            reverse(nums, k+1, nums.length-1);
        }
    }

    /**
     * Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.For "(()", the longest valid
     * parentheses substring is "()", which has length = 2.Another example is ")()())", where the longest valid parentheses substring is "()()" and length = 4
     */
    public int longestValidParentheses(String s) {
        int ret = 0;
        if(s != null){
            Stack<Integer> stack = new Stack<>();
            stack.push(-1);
            for(int i=0; i<s.length(); i++){
                if(s.charAt(i) == '('){
                    stack.push(i);
                }else{
                    if(stack.peek() == -1 || s.charAt(stack.peek()) == ')'){
                        stack.push(i);
                    }else{
                        stack.pop();
                        ret = Math.max(ret, i-stack.peek());
                    }
                }
            }
        }
        return ret;
    }

    /**
     * Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
     * Example [1,3,5,6], 5 → 2 [1,3,5,6], 2 → 1, You may assume no duplicates in the array
     */
    public int searchInsert(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        int m = 0;
        while(start < end){
            m = (start+end)/2;
            if(nums[m] == target){
                return m;
            }else if(nums[m] < target){
                start = m+1;
            }else{
                end = m-1;
            }
        }
        if(start == end){
            return target <= nums[start] ? start : start+1;
        }
        return m;
    }

    /**
     * Determine if a Sudoku is valid. for 9x9 grid a valid sudoku is 1: Each row must have the numbers 1-9 occuring just once 2: Each column must have the
     * numbers 1-9 occuring just once. 3: numbers 1-9 must occur just once in each of the 9 sub-boxes of the grid
     */
    public boolean isValidSudoku(char[][] board) {
        int [][] row = new int[9][9];
        int [][] col = new int[9][9];
        int box [][][] = new int[3][3][9];
        for(int i=0; i<9; i++){
            for(int j=0; j<9; j++){
                if(board[i][j] != '.'){
                    int idx = board[i][j] - '0' -1;
                    row[i][idx] += 1;
                    if(row[i][idx] > 1){
                        return false;
                    }
                    col[idx][j] += 1;
                    if(col[idx][j] > 1){
                        return false;
                    }
                    box[i/3][j/3][idx] += 1;
                    if(box[i/3][j/3][idx] > 1){
                        return false;
                    }
                }
            }
        }
        return true;
    }

    /**
     * Write a program to solve a Sudoku puzzle by filling the empty cells indicated by the character '.' You may assume that there will be only one unique solution
     */
    public void solveSudoku(char[][] board) {

    }

    /**
     * Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T. The same repeated number
     * may be chosen from C unlimited number of times. Note: All numbers (including target) will be positive integers. The solution set must not contain duplicate combinations.
     * E.g: input [2, 3, 6, 7] 7, return [ [7], [2, 2, 3] ]
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        combinationSumBacktrack(ret, candidates, target, 0, new ArrayList<>());
        return ret;
    }

    private void combinationSumBacktrack(List<List<Integer>> ret, int[] candidates, int target, int start, List<Integer> row){
        if(target < 0){
            return;
        }
        if(target == 0){
            ret.add(new ArrayList<Integer>(row));
        }else{
            for (int i=start; i<candidates.length; i++){
                row.add(candidates[i]);
                combinationSumBacktrack(ret, candidates, target-candidates[i], i, row);
                row.remove(row.size()-1);
            }
        }
    }

    /**
     * Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T. Each number in C
     * may only be used once in the combination. Given [10, 1, 2, 7, 6, 1, 5] and 8 return [ [1, 7], [1, 2, 5], [2, 6], [1, 1, 6] ]
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        Arrays.sort(candidates);
        combinationSum2Backtrack(ret, candidates, target, 0, new ArrayList<>());
        return ret;
    }

    private void combinationSum2Backtrack(List<List<Integer>> ret, int[] candidates, int target, int start, List<Integer> row){
        if(target < 0){
            return;
        }
        if(target == 0){
            ret.add(new ArrayList<>(row));
        }else{
            for(int i=start; i<candidates.length; i++){
                if(i > start && candidates[i] == candidates[i-1]){
                    continue;
                }
                row.add(candidates[i]);
                combinationSum2Backtrack(ret, candidates, target-candidates[i], i+1, row);
                row.remove(row.size()-1);
            }
        }
    }

    /**
     * Given a collection of distinct numbers, return all possible permutations. Example [1,2,3] returns
     * [ [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1] ]
     */
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        List<Integer> init = new ArrayList<Integer>();
        for(int i: nums){
            init.add(i);
        }
        permute(new ArrayList<Integer>(), init, ret);
        return ret;
    }

    public void permute(List<Integer> prefix, List<Integer> nums, List<List<Integer>> res){
        if(nums.size() == 0){
            res.add(prefix);
        }else{
            for(int i=0; i<nums.size(); i++){
                List<Integer> item = new ArrayList<>(nums);
                List<Integer> newPrefix = new ArrayList<>(prefix);
                newPrefix.add(item.remove(i));
                permute(newPrefix, item, res);
            }
        }
    }

    /**
     * Given a collection of numbers that might contain duplicates, return all possible unique permutations
     */
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        List<Integer> all = new ArrayList<>();
        Arrays.sort(nums);
        for(int i: nums){
            all.add(i);
        }
        permuteUniqueHelper(ret, new ArrayList(), all);
        return ret;
    }

    private void permuteUniqueHelper(List<List<Integer>> ret, List<Integer> item, List<Integer> nums){
        if(nums.size() == 0){
            ret.add(item);
            return;
        }
        for(int i=0; i<nums.size(); i++){

        }
    }

    /**
     * The set [1,2,3,…,n] contains a total of n! unique permutations. By listing and labeling all of the permutations in order,
     if n=3 we will get "123" "132" "213" "231" "312" "321" Given n and k, return the kth permutation sequence. Note (1<=n<=9)
     */
    public String getPermutation(int n, int k){
        int []num = new int[n];
        int permCount = 1;
        for(int i=0;i<n;i++){
            num[i] = i+1;
            permCount*=(i+1);
        }
        k--;
        StringBuilder target = new StringBuilder();
        for(int i=0;i<n;i++){
            permCount = permCount/(n-i);
            int choosed = k/permCount;
            target.append(num[choosed]);
            for(int j=choosed;j<n-i-1;j++){
                num[j] = num[j+1];
            }
            k = k%permCount;
        }
        return target.toString();
    }

    public List<List<String>> solveNQueens(int n) {
        return null;
    }

    public int totalNQueens(int n) {
        return 0;
    }

    /**
     * Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
     * Example: [0,1,0,2,1,0,1,3,2,1,2,1], return 6
     */
    public int trap(int[] height) {
        int sum = 0;
        int i=0, j = height.length -1;
        int h = 0;
        while(i < j){
            if(height[i] < height[j]){
                h = Integer.max(h, height[i]);
                sum = sum + h - height[i];
                i++;
            }else{
                h = Integer.max(h, height[j]);
                sum = sum + h - height[j];
                j--;
            }
        }
        return sum;
    }

    /**
     *  Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints
     *  of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water
     */
    public int maxArea(int[] height) {
        int max = 0,start=0,end=height.length-1;
        while (start < end){
            max = Math.max(max, (end - start) * Math.min(height[start], height[end]));
            if(height[start] < height[end]){
                start ++;
            }else{
                end--;
            }
        }
        return max;
    }

    public int largestRectangleArea(int[] heights) {
        int max = 0;
        for(int i=0; i< heights.length; i++){
            if(i+1 < heights.length && heights[i] <= heights[i+1]){
                continue;
            }
            int minh = heights[i];
            for(int j=i; j>=0; j--){
                minh = Math.min(minh, heights[j]);
                max = Math.max(max, minh * (i-j+1));
            }
        }
        return max;
    }

    /**
     * Given two numbers represented as strings, return multiplication of the numbers as a string. Note: numbers can be arbitrarily large and are non-negative,
     * Converting the input string to integer is NOT allowed and You should NOT use internal library such as BigInteger.
     */
    public String multiply(String num1, String num2) {
        int n1 = num1.length();
        int n2 = num2.length();
        int[] product = new int[n1 + n2];
        for(int i=n1-1; i>=0; i--){
            for(int j=n2 - 1; j>=0; j--){
                int d1 = num1.charAt(i) - '0';
                int d2 = num1.charAt(j) - '0';
                product[i+j+1] += d1 * d2;
            }
        }
        int digit = 0;
        StringBuilder sb = new StringBuilder();
        for (int i = n1 + n2 - 1; i >= 0; i--) {
            int tmp = product[i] + digit;
            sb.append(tmp % 10);
            digit = tmp / 10;
        }
        sb.reverse();
        if (sb.charAt(0) == '0') sb.deleteCharAt(0);
        return sb.toString();
    }

    /**
     * Given an array of non-negative integers, you are initially positioned at the first index of the array. Each element in the array represents your maximum jump
     * length at that position Your goal is to reach the last index in the minimum number of jumps. Given [2,3,1,1,4] The minimum number of jumps to reach the
     * last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index
     */
    public int jump(int[] nums) {
        int count=0,max=0,next=0;
        for(int i=0; i<nums.length; i++){
            if(i > max){
                max = next;
                count++;
            }
            next = Integer.max(next, i+nums[i]);
        }
        return count;
    }

    /**
     * You are given an n x n 2D matrix representing an image. Rotate the image in place by 90 degrees (clockwise)
     */
    public void rotate(int[][] matrix) {
        int len = matrix.length-1;
        int layer = matrix.length / 2;
        for(int i=0; i<layer; i++){
            for(int j=i; j<len-i; j++){
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[len-j][i];
                matrix[len-j][i] = matrix[len-i][len-j];
                matrix[len-i][len-j] = matrix[j][len-i];
                matrix[j][len-i] = tmp;
            }
        }
    }

    /**
     * Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order. Example: [ [ 1, 2, 3 ], [ 4, 5, 6 ], [ 7, 8, 9 ] ]
     * return [1,2,3,6,9,8,7,4,5]
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> ret = new ArrayList<Integer>();
        if(matrix != null && matrix.length > 0){
            int m = matrix.length;
            int n = matrix[0].length;
            int layer = Integer.min(m,n);
            layer = (layer+1)/2;
            for(int i=0; i<layer; i++){
                for(int j = i; j<n-i; j++){//first row
                    ret.add(matrix[i][j]);
                }
                for(int j=i+1; j< m-i-1; j++){//last col
                    ret.add(matrix[j][n-1-i]);
                }
                for(int j=n-i-1; m-1-i != i && j>=i; j--){//bottom row
                    ret.add(matrix[m-1-i][j]);
                }
                for(int j=m-2-i; n-1-i != i && j>i; j--){//first col
                    ret.add(matrix[j][i]);
                }
            }
        }
        return ret;
    }

    /**
     * Implement pow(x, n)
     */
    public double myPow(double x, int n) {
        if(n == 0){
            return 1;
        }else{
            if(n % 2 == 0){
                return myPow(x * x, n/2);
            }else{
                if(n>0){
                    return x* myPow(x, n-1);
                }else{
                    return 1/x*myPow(x, n+1);
                }
            }
        }
    }

    /**
     * Find the contiguous subarray within an array (containing at least one number) which has the largest sum. given the array [−2,1,−3,4,−1,2,1,−5,4],
     * the contiguous subarray [4,−1,2,1] has the largest sum = 6
     */
    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for(int i=1; i<nums.length; i++){
            if(nums[i-1] > 0){
                nums[i] = nums[i-1]+nums[i];
            }
            if(max < nums[i]){
                max = nums[i];
            }
        }
        return max;
    }

    /**
     * Given an array of non-negative integers, you are initially positioned at the first index of the array.Each element in the array represents your maximum
     * jump length at that position. Determine if you are able to reach the last index. example: [2,3,1,1,4], return true. [3,2,1,0,4], return false
     */
    public boolean canJump(int[] nums) {
        int i =0;
        int maxReach = 0;
        for(; i<nums.length && i<= maxReach; i++){
            maxReach = Math.max(i+nums[i], maxReach);
        }
        return i == nums.length;
    }

    /**
     * Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).Intervals were initially sorted according to their start times
     * Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9]
     */
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> list = new ArrayList<Interval>();
        if(intervals == null || intervals.size() == 0){
            list.add(newInterval);
        }else{
            int idx = 0;
            for(int i=0; i<intervals.size(); i++){
                Interval tmp = intervals.get(i);
                idx++;
                if(newInterval.end < tmp.start){
                    list.add(newInterval);
                    break;
                }else if((newInterval.start >= tmp.start && newInterval.start <= tmp.end) || (newInterval.start<= tmp.start && newInterval.end >= tmp.start)){
                    newInterval = new Interval(Integer.min(tmp.start, newInterval.start), Integer.max(tmp.end, newInterval.end));
                }else{
                    list.add(tmp);
                }
            }
            if(idx == intervals.size()){
                list.add(newInterval);
            }else{
                for(int i=idx; i< intervals.size(); i++){
                    list.add(intervals.get(i));
                }
            }
        }
        return list;
    }

    /**
     * A robot is located at the top-left corner of a m x n grid The robot can only move either down or right at any point in time.
     * The robot is trying to reach the bottom-right corner of the grid How many possible unique paths are there?
     */
    public int uniquePaths(int m, int n) {
        int[][] p = new int[m][n];
        for(int i=0; i<m; i++){
            for(int j=0; j<n; j++){
                if(i == 0 || j==0){
                    p[i][j] =1;
                }else{
                    p[i][j] = p[i-1][j]+p[i][j-1];
                }
            }
        }
        return p[m-1][n-1];
    }

    //use one dimension array
    public int uniquePaths2(int m, int n) {
        if (m == 0 || n == 0){
            return 0;
        }
        int[] Num = new int[n];
        for (int i = 0; i < m; i ++){
            for (int j = 0; j < n; j++){
                if (i == 0 || j == 0){
                    Num[j] = 1;
                }
                else {
                    Num[j] += Num[j-1];
                }
            }
        }
        return Num[n-1];
    }

    /**
     * Now consider if some obstacles are added to the grids. How many unique paths would there be?
     * An obstacle and empty space is marked as 1 and 0 respectively in the grid
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if(obstacleGrid.length <= 0 || obstacleGrid[0].length <= 0){
            return 0;
        }
        int rows = obstacleGrid.length;
        int cols = obstacleGrid[0].length;
        int paths[] = new int[cols+1];
        paths[1]=1;
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++){
                if(obstacleGrid[i][j] == 1){
                    paths[j+1]=0;
                }else{
                    paths[j+1] = paths[j] + paths[j+1];
                }
            }
        }
        return paths[cols];
    }

    /**
     * Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
     */
    public int minPathSum(int[][] grid) {
        if(grid.length == 0 || grid[0].length == 0){
            return 0;
        }
        int m = grid.length;
        int n = grid[0].length;
        for(int i=0; i<grid.length; i++){
            for(int j=0; j<n; j++){
                if(i == 0 && j!= 0){
                    grid[i][j] += grid[i][j-1];
                }else if(j == 0 && i!=0){
                    grid[i][j] += grid[i-1][j];
                }else if(i==0 && j==0){
                    grid[i][j] = grid[i][j];
                }else{
                    grid[i][j]+=Math.min(grid[i-1][j], grid[i][j-1]);
                }
            }
        }
        return grid[m-1][n-1];
    }

    public boolean isNumber(String s) {
        if(s == null){
            return false;
        }
        s = s.trim();
        int n = s.length();
        if(n == 0){
            return false;
        }
        int signCount = 0;
        boolean hasNum = false;
        boolean hasE = false;
        boolean hasDot = false;
        for(int i=0; i<n; i++){
            char c = s.charAt(i);
            if(!isValid(c)){
                return false;
            }
            if(c>='0' && c<='9'){
                hasNum = true;
            }
            if(c == 'e' || c== 'E'){
                if(!hasE || !hasNum){
                    return false;
                }
                if(i == n-1){
                    return false;
                }
                hasE = true;
            }
            if(c == '.'){
                if(hasDot || hasE){
                    return false;
                }
                if(i == n-1 && !hasNum){
                    return false;
                }
                hasDot = true;
            }
            if(c == '+' || c == '-'){
                if(signCount == 2){
                    return false;
                }
                if(i == n-1){
                    return false;
                }
                if(i> 0 && !hasE){
                    return false;
                }
                signCount++;
            }
        }
        return true;
    }

    boolean isValid(char c) {
        return c == '.' || c == '+' || c == '-' || c == 'e' || c == 'E' || c >= '0' && c <= '9';
    }

    /**
     * Implement int sqrt(int x).
     */
    public int mySqrt(int x) {
        if(x <=1){
            return x;
        }
        int start=2, end=x;
        while(start < end){
            int m = (start+end)/2;
            if(m == x/m){
                return m;
            }else if(x/m < m){
                end = m;
            }else{
                start = m;
            }
        }
        return start;
    }

    /**
     * DP: You are climbing a stair case. It takes n steps to reach to the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
     */
    public int climbStairs(int n) {
        int[] a = new int[n+1];
        a[0] = 0;
        a[1] = 1;
        a[2] = 2;
        for(int i=3; i<=n; i++){
            a[i] = a[i-1] + a[i-2];
        }
        return a[n];
    }

    /**
     * DP: Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)
     * You have the following 3 operations permitted on a word. Insert/Delete/Replace a character.
     */
    public int minDistance(String word1, String word2) {
        char[] m = word1.toCharArray();
        char[] n = word2.toCharArray();
        int[][] tmp = new int[m.length+1][n.length+1];
        for(int i=0; i<tmp[0].length; i++){
            tmp[0][i] = i;
        }
        for(int j=0; j<tmp.length; j++){
            tmp[j][0] = j;
        }
        for(int i=1; i<tmp.length; i++){
            for(int j=1; j<tmp[0].length; j++){
                if(m[i-1] == n[j-1]){
                    tmp[i][j] = tmp[i-1][j-1];
                }else{
                    tmp[i][j] = Math.min(Math.min(tmp[i-1][j], tmp[i][j-1]), tmp[i-1][j-1])+1;
                }
            }
        }
        return tmp[m.length][n.length];
    }

    /**
     * Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place O(1) space.
     */
    public void setZeroes(int[][] matrix) {
        boolean fr = false, fc = false;
        for(int i=0; i<matrix.length; i++){
            for(int j=0; j<matrix[0].length; j++){
                if(matrix[i][j] == 0){
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                    if(i == 0){
                        fr = true;
                    }
                    if(j == 0){
                        fc = true;
                    }
                }
            }
        }
        for(int i=1; i<matrix.length; i++){
            for(int j=1; j<matrix[0].length; j++){
                if(matrix[0][j] == 0 || matrix[i][0] == 0){
                    matrix[i][j] = 0;
                }
            }
        }
        if(fr){
            for(int i=0; i<matrix[0].length; i++){
                matrix[0][i] = 0;
            }
        }
        if(fc){
            for(int i=0; i<matrix.length; i++){
                matrix[i][0] = 0;
            }
        }
    }

    /**
     * Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
     * Integers in each row are sorted from left to right. The first integer of each row is greater than the last integer of the previous row.
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        int r = matrix.length;
        int c = matrix[0].length;
        int start = 0, end = r*c-1;
        while(start <= end){
            int m = (start+end)/2;
            int tmp = matrix[m/c][m%c];
            if(tmp == target){
                return true;
            }
            if(target > tmp){
                start = m+1;
            }else{
                end = m-1;
            }
        }
        return false;
    }

    /**
     * Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.
     * Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively (no extra space)
     */
    public void sortColors(int[] nums) {
        int red = 0;
        int i = 0;
        int blue = nums.length - 1;
        while(i<=blue){
            if(nums[i] == 0){
                nums[i] = nums[red];
                nums[red] = 0;
                if(i == red){
                    i++;
                }
                red++;
            }else if(nums[i] == 2){
                nums[i] = nums[blue];
                nums[blue] = 2;
                blue--;
            }else{
                i++;
            }
        }
    }

    /**
     * Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
     * For example, S = "abcac" T = "ac" Minimum window is "ac"
     */
    public String minWindow(String s, String t) {
        int[] map = new int[256];
        for(char c: t.toCharArray()){
            map[c]++;
        }
        int counter=t.length(), begin=0, end=0, d=Integer.MAX_VALUE, head=0;
        while(end<s.length()){
            char c = s.charAt(end);
            end++;
            if(map[c] > 0){
                counter--;
                map[c]--;
            }
            while(counter==0){ //valid
                if(end-begin<d){
                    head=begin;
                    d=end-head;
                }
                c = s.charAt(begin);
                begin++;
                if(map[c]==0){
                    map[c]++;
                    counter++;  //make it invalid
                }
            }
        }
        return d==Integer.MAX_VALUE ? "": s.substring(head, head+d);
    }

    /**
     * DFS: Given two integers n and k, return all possible combinations of k numbers out of 1 ... n. Example: If n = 4 and k = 2, a solution is:
     * [ [2,4],  [3,4],  [2,3],  [1,2], [1,3], [1,4] ]
     */
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        List<Integer> item = new ArrayList<Integer>();
        addSub(ret, item, n, 1, k);
        return ret;
    }

    public void addSub(List<List<Integer>> list, List<Integer> insert, int n, int start, int k){
        if(k == 0){
            list.add(insert);
            return;
        }else{
            for(int i=start; i<=n; i++){
                List<Integer> newItem = new ArrayList<Integer>(insert);
                newItem.add(i);
                addSub(list, newItem, n, i + 1, k - 1);
            }
        }
    }

    /**
     * DFS: Given a set of distinct integers, nums, return all possible subsets. The solution set must not contain duplicate subsets.
     * Example: give [1,2,3] return [ [3], [1], [2], [1,2,3], [1,3], [2,3], [1,2], [] ]
     */
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        addSubset(ret, new ArrayList<>(), 0, nums);
        return ret;
    }

    public void addSubset(List<List<Integer>> list, List<Integer> item, int start, int[] nums){
        list.add(item);
        for(int i=start; i<nums.length; i++){
            List<Integer> row = new ArrayList<>(item);
            row.add(nums[i]);
            addSubset(list, row, i+1, nums);
        }
    }

    /**
     * DFS: Given a 2D board and a word, find if the word exists in the grid. The word can be constructed from letters of sequentially adjacent cell,
     * where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once. Given board=[
     * ['A','B','C','E'], ['S','F','C','S'], ['A','D','E','E'] word = "SEE", -> returns true, word = "ABCB", -> returns false
     */
    public boolean exist(char[][] board, String word) {
        boolean[][] visited = new boolean[board.length][board[0].length];
        for(int i=0; i<board.length; i++){
            for (int j=0; j<board.length; j++){
                if(board[i][j] == word.charAt(0) && find(i, j, board, 0, word, visited)){
                    return true;
                }
            }
        }
        return false;
    }

    public boolean find(int i, int j, char[][] board, int idx, String word, boolean[][] visited){
        if(idx >= word.length()){
            return true;
        }
        if(i<0 || i >= board.length || j <0 || j>=board[0].length || board[i][j] != word.charAt(idx) || visited[i][j]){
            return false;
        }
        visited[i][j]=true;
        if(find(i-1, j, board, idx+1, word, visited) ||
                find(i+1, j, board, idx+1, word, visited) ||
                find(i, j-1, board, idx+1, word, visited) ||
                find(i, j+1, board, idx+1, word, visited)){
            return true;
        }
        visited[i][j] = false;
        return false;
    }

    /**
     * Given sorted array nums = [1,1,1,2,2,3],if duplicates are allowed at most twice Your function should return length = 5
     * with the first five elements of nums being 1, 1, 2, 2 and 3. It doesn't matter what you leave beyond the new length
     */
    public int removeDuplicates(int[] nums) {
        int count = 0;
        for(int i=2; i<nums.length; i++){
            if(nums[i] == nums[i-2-count]){
                count ++;
            }else{
                nums[i-count] = nums[i];
            }
        }
        return nums.length - count;
    }

    /**
     * search in rotated sorted array rotated sorted array may look like [3,4,5,6,0,1,2] before rotated is [0,1,2,3,4,5,6]
     */
    public boolean search(int[] nums, int target) {
        int start = 0, end = nums.length -1;
        while(start <= end){
            int middle = (start + end)/2;
            if(target == nums[middle]){
                return true;
            }
            if(nums[middle] < nums[end]){//right side is sorted
                if(target > nums[middle] && target <= nums[end]){
                    start = middle + 1;
                }else{
                    end = middle - 1;
                }
            }else if(nums[middle] > nums[end]){//left side is sorted
                if (target < nums[middle] && target >= nums[start]) {
                    end = middle - 1;
                } else {
                    start = middle + 1;
                }
            }else{
                end --;
            }
        }
        return false;
    }

    /**
     * Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
     * 1->2->3->3->4->4->5, return 1->2->5 and Given 1->1->1->2->3, return 2->3.
     */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode node, node1,node2;
        node1=node2=head;
        node = dummy;
        while(node1 != null){
            while(node2.next != null && node1.val == node2.next.val){
                node2 = node2.next;
            }
            if(node1 == node2){
                node = node1;
            }else{
                node.next = node2.next;
            }
            node1 = node2 = node.next;
        }
        return dummy.next;
    }

    /**
     * Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
     * You should preserve the original relative order of the nodes in each of the two partitions.
     * Given 1->4->3->2->5->2 and x = 3 return 1->2->2->4->3->5.
     */
    public ListNode partition(ListNode head, int x) {
        ListNode hd = new ListNode(0);
        ListNode tail = new ListNode(0);
        ListNode hd1 = hd, t2 = tail;
        while(head != null){
            if(head.val < x){
                hd1.next = new ListNode(head.val);
                hd1 = hd1.next;
            }else{
                t2.next = new ListNode(head.val);
                t2 = t2.next;
            }
            head = head.next;
        }
        hd1.next = tail.next;
        return hd.next;
    }

    /**
     * Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
     * Note: You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.
     * The number of elements initialized in nums1 and nums2 are m and n respectively.
     */
    public void inPlaceMerge(int[] nums1, int m, int[] nums2, int n) {
        int i = m-1, j = n-1, k = m + n -1;
        while(i>=0 && j>=0){
            nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
        }
        while(i>=0){
            nums1[k--] = nums1[i--];
        }
        while(j>=0){
            nums1[k--] = nums2[j--];
        }
    }

    /**
     * Given a collection of integers that might contain duplicates, nums, return all possible subsets. Note:
     * 1: Elements in a subset must be in non-descending order. 2: The solution set must not contain duplicate subsets.
     * If nums = [1,2,2], a solution is: [[],[1],[2],[1,2],[2,2],[1,2,2]]
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        subsetsWithDupHelper(ret, nums, new ArrayList<Integer>(), 0);
        return ret;
    }

    public void subsetsWithDupHelper(List<List<Integer>> ret, int[] nums, List<Integer> item, int p){
        ret.add(item);
        if(p == nums.length){
            return;
        }
        for (int i = p; i < nums.length; i++) {
            if(i == p || nums[i] != nums[i-1]){
                List<Integer> tmp = new ArrayList<Integer>(item);
                tmp.add(nums[i]);
                subsetsWithDupHelper(ret, nums, tmp, i+1);
            }
        }
    }

    /**
     * A message containing letters from A-Z is being encoded to numbers using the following mapping:
     * 'A' -> 1, 'B' -> 2, .... 'Z' -> 26
     * Given an encoded message containing digits, determine the total number of ways to decode it.
     * For example: Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
     */
    public int numDecodings(String s) {
        int n = s.length();
        if (n == 0) return 0;
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) != '0' ? 1 : 0;

        for (int i = 1; i < n; i++) {
            if (s.charAt(i) != '0') {
                dp[i + 1] = dp[i];
            }
            int x = Integer.parseInt(s.substring(i - 1,i + 1));
            if (x >= 10 && x <= 26) dp[i + 1] += dp[i - 1];
        }
        return dp[n];
    }

    /**
     * Given a string containing only digits, restore it by returning all possible valid IP address combinations.
     * e.g Given "25525511135" returns ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
     */
    public List<String> restoreIpAddresses(String s) {
        List<String> result = new ArrayList<String>();
        restoreIpAddressesHelper("",s,1,result);
        return result;
    }

    public void restoreIpAddressesHelper(String ip, String rem, int level, List<String> result){
        if(level == 4){
            if(rem.length() < 4 && Integer.parseInt(rem) <= 255 && (rem.length() == 1 || rem.charAt(0) != '0'))
                result.add(ip + "" + rem);
            return;
        }
        String next = "";
        if(rem.length() > 1){
            next = rem.substring(0,1);
            restoreIpAddressesHelper(ip + next+".",rem.substring(1),level+1,result);
        }
        if(rem.length() > 2 && rem.charAt(0) != '0' ){
            next = rem.substring(0,2);
            restoreIpAddressesHelper(ip + next+".",rem.substring(2),level+1,result);
        }
        if(rem.length() > 3 && rem.charAt(0) != '0' ){
            next = rem.substring(0,3);
            if(Integer.parseInt(next) <= 255)
                restoreIpAddressesHelper(ip + next +".",rem.substring(3),level+1,result);
        }
    }

    //in order tree traversal using iteration
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> ret = new ArrayList<Integer>();
        Stack<TreeNode> stack  = new Stack<TreeNode>();
        while(root != null || !stack.isEmpty()){
            if(root != null){
                stack.push(root);
                root = root.left;
            }else{
                TreeNode n = stack.pop();
                ret.add(n.val);
                root = n.right;
            }
        }
        return ret;
    }

    /**
     * Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
     */
    public List<TreeNode> generateTrees(int n) {
        return generateTreesHelper(1, n);
    }

    public List<TreeNode> generateTreesHelper(int start, int end){
        List<TreeNode> ret = new ArrayList<TreeNode>();
        if(start > end){
            ret.add(null);
            return ret;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> left = generateTreesHelper(start, i-1);
            List<TreeNode> right = generateTreesHelper(i+1, end);
            for (TreeNode l: left){
                for (TreeNode r: right){
                    TreeNode root = new TreeNode(i);
                    root.left = l;
                    root.right = r;
                    ret.add(root);
                }
            }
        }
        return ret;
    }

    /**
     * DP: Given n, how many structurally unique BST's (binary search trees) that store values 1...n
     */
    public int numTrees(int n) {
        int[] nums = new int[n+1];
        nums[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < i; j++) {
                nums[i] += nums[j] * nums[i-j-1];
            }
        }
        return nums[n];
    }

    /**
     * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2. Given: s1 = "aabcc", s2 = "dbbca",
     * When s3 = "aadbbcbcac", return true. When s3 = "aadbbbaccc", return false.
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        int n1 = s1.length(), n2 = s2.length(), n3 = s3.length();
        if(n1 + n2 != n3){
            return false;
        }
        boolean dp[][] = new boolean[n1+1][n2+1];
        for(int i=0; i < dp.length; i++){
            for(int j=0; j < dp[0].length; j++){
                int l = i + j -1;
                if(i == 0 && j == 0){
                    dp[i][j] = true;
                }
                else if(i == 0){
                    if(s3.charAt(l) == s2.charAt(j-1)){
                        dp[i][j] = dp[i][j-1];
                    }
                }
                else if(j == 0){
                    if(s1.charAt(i-1) == s3.charAt(l)){
                        dp[i][j] = dp[i-1][j];
                    }
                }
                else{
                    dp[i][j] = (s1.charAt(i-1) == s3.charAt(l) ? dp[i-1][j] : false) || (s2.charAt(j-1) == s3.charAt(l) ? dp[i][j-1] : false);
                }
            }
        }
        return dp[n1][n2];
    }

    /**
     * Given a binary tree, determine if it is a valid binary search tree (BST). BST is defined as follows:
     * 1: The left subtree of a node contains only nodes with keys less than the node's key.
     * 2: The right subtree of a node contains only nodes with keys greater than the node's key.
     * 3: Both the left and right subtrees must also be binary search trees.
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBSTHelper(root, null, null);
    }

    private boolean isValidBSTHelper(TreeNode node, Integer min, Integer max){
        if(node == null){
            return true;
        }
        return (min == null || node.val > min) && (max == null || node.val < max) && isValidBSTHelper(node.left, min, node.val) && isValidBSTHelper(node.right, node.val, max);
    }

    /**
     * Two elements of a binary search tree (BST) are swapped by mistake. Recover the tree without changing its structure
     */
    public void recoverTree(TreeNode root) {
        recoverTreeHelper(root);
        int tmp = first.val;
        first.val = second.val;
        second.val = tmp;
    }

    public void recoverTreeHelper(TreeNode root) {
        if(root != null){
            recoverTreeHelper(root.left);
            if(last != null && last.val > root.val){
                if(first == null){
                    first = last;
                }
                if(first != null){
                    second = root;
                }
            }
            last = root;
            recoverTreeHelper(root.right);
        }
    }
    private TreeNode first = null, second = null, last = null;

    /**
     * Given two binary trees, write a function to check if they are equal or not.
     * Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == q){
            return true;
        }
        if(p != null && q != null){
            return isSameTree(p.left, q.left) && p.val == q.val && isSameTree(q.right, p.right);
        }
        return false;
    }

    /**
     * Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
     */
    public boolean isSymmetric(TreeNode root) {
        if(root == null){
            return true;
        }
        return isSymmetricRecursive(root.left,root.right);
    }

    public boolean isSymmetricRecursive(TreeNode left, TreeNode right){
        if(left == null && right == null){
            return true;
        }
        if(left == null || right == null){
            return false;
        }
        return left.val == right.val && isSymmetricRecursive(left.left, right.right) && isSymmetricRecursive(left.right, right.left);
    }

    public boolean isSymmetricIterative(TreeNode root) {
        if(root == null){
            return true;
        }
        LinkedList<TreeNode> leftq = new LinkedList<TreeNode>();
        LinkedList<TreeNode> rightq = new LinkedList<TreeNode>();
        leftq.push(root.left);
        rightq.push(root.right);
        while(!leftq.isEmpty() && !rightq.isEmpty()){
            TreeNode left = leftq.pop();
            TreeNode right = rightq.pop();
            if(left == null && right == null){
                continue;
            }
            if(left == null || right == null){
                return false;
            }
            if(left.val != right.val){
                return false;
            }
            leftq.push(left.left);
            leftq.push(left.right);
            rightq.push(right.right);
            rightq.push(right.left);
        }
        return leftq.size() == rightq.size();
    }

    /**
     * Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
     * e.g. Given binary tree {3,9,20,#,#,15,7} returns [[3],[9,20],[15,7]]
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if(root != null){
            LinkedList<TreeNode> q = new LinkedList<TreeNode>();
            q.add(root);
            while(!q.isEmpty()){
                int num = q.size();
                List<Integer> item = new ArrayList<Integer>();
                for (int i = 0; i < num; i++) {
                    TreeNode node = q.remove();
                    item.add(node.val);
                    if(node.left != null){
                        q.add(node.left);
                    }
                    if(node.right != null){
                        q.add(node.right);
                    }
                }
                ret.add(item);
            }
        }
        return ret;
    }

    /**
     * Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
     */
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if(root != null){
            LinkedList<TreeNode> q = new LinkedList<TreeNode>();
            q.add(root);
            boolean forward = true;
            while (!q.isEmpty()){
                int len = q.size();
                List<Integer> item = new ArrayList<>();
                ret.add(item);
                for (int i = 0; i < len; i++) {
                    TreeNode node = q.remove();
                    if(forward){
                        item.add(node.val);
                    }else{
                        item.add(0, node.val);
                    }
                    if(node.left != null){
                        q.add(node.left);
                    }
                    if(node.right != null){
                        q.add(node.right);
                    }
                }
                forward = !forward;
            }
        }
        return ret;
    }

    /**
     * Given a binary tree, find its maximum depth.
     * The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
     */
    public int maxDepth(TreeNode root) {
        int h = 0;
        if(root == null){
            return 0;
        }
        return maxDepthHelper(root, 1);
    }

    public int maxDepthHelper(TreeNode node, int h){
        int hl = h;
        int hr = h;
        if(node.left != null){
            hl = maxDepthHelper(node.left, h+1);
        }
        if(node.right != null){
            hr = maxDepthHelper(node.right, h+1);
        }
        return Integer.max(hr, hl);
    }

    /**
     * Given preorder and inorder traversal of a tree, construct the binary tree. Note: You may assume that duplicates do not exist in the tree.
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder == null || inorder == null || preorder.length == 0 || inorder.length == 0){
            return null;
        }
        return buildTreeHelper(preorder, 0, inorder, 0, inorder.length);
    }

    public TreeNode buildTreeHelper(int[] preorder, int prestart, int[] inorder, int start, int end){
        if(prestart >= preorder.length || start >= end){
            return null;
        }
        TreeNode node = new TreeNode(preorder[prestart]);
        int idx = 0;
        for (int i = start; i < end; i++) {
            if(node.val == inorder[i]){
                idx = i;
            }
        }
        node.left = buildTreeHelper(preorder, prestart+1, inorder, start, idx);
        int rightPrePos = prestart + idx - start + 1;//prestart + number of nodes on the left + one
        node.right = buildTreeHelper(preorder, rightPrePos, inorder, idx+1, end);
        return node;
    }

    /**
     * Given inorder and postorder traversal of a tree, construct the binary tree. Note: You may assume that duplicates do not exist in the tree.
     */
    public TreeNode buildTreePost(int[] inorder, int[] postorder) {
        if(inorder == null || postorder == null || inorder.length == 0 || postorder.length == 0){
            return null;
        }
        return buildTreePostHelper(inorder, postorder, postorder.length - 1, 0, inorder.length);
    }

    public TreeNode buildTreePostHelper(int[] inorder, int[] postorder, int postart, int start, int end){
        if(postart >= postorder.length || start >= end){
            return null;
        }
        TreeNode node = new TreeNode(postorder[postart]);
        int idx = 0;
        for (int i = start; i < end; i++) {
            if(inorder[i] == node.val){
                idx = i;
                break;
            }
        }
        node.right = buildTreePostHelper(inorder, postorder, postart -1, idx+1, end);
        int leftPostart = postart - (end - idx);//postart minus number of node on the right
        node.left = buildTreePostHelper(inorder, postorder, leftPostart, start, idx);
        return node;
    }

    /**
     * Given a binary tree, return the bottom-up level order traversal of its nodes' values.
     * (ie, from left to right, level by level from leaf to root).
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if(root != null){
            Queue<TreeNode> q = new LinkedList<>();
            q.add(root);
            while (!q.isEmpty()){
                int size = q.size();
                List<Integer> item = new ArrayList<>();
                ret.add(0,item);
                for (int i = 0; i < size; i++) {
                    TreeNode node = q.remove();
                    if(node.left != null){
                        q.add(node.left);
                    }
                    if(node.right != null){
                        q.add(node.right);
                    }
                    item.add(node.val);
                }
            }
        }
        return ret;
    }

    /**
     * Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums == null || nums.length == 0){
            return null;
        }
        return sortedArrayToBSTHelper(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayToBSTHelper(int[] nums, int start, int end){
        if(start == end){
            return new TreeNode(nums[start]);
        }
        if(end < start){
            return null;
        }
        int idx = (start + end)/2;
        TreeNode root = new TreeNode(nums[idx]);
        root.left = sortedArrayToBSTHelper(nums, start, idx -1);
        root.right = sortedArrayToBSTHelper(nums, idx+1, end);
        return root;
    }

    /**
     *Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
     */
    public TreeNode sortedListToBST(ListNode head) {
        if(head == null){
            return null;
        }
        ListNode end = head;
        while(end.next != null){
            end = end.next;
        }
        return sortedListToBSTHelper(head, end);
    }

    public TreeNode sortedListToBSTHelper(ListNode start, ListNode end){
        if(start == end){
            return new TreeNode(start.val);
        }
        ListNode prev, slow, fast;
        int count = 0;
        prev = slow = fast = start;
        while(fast != end){
            fast = fast.next;
            count ++;
            if(count%2 == 0){//move two steps
                slow = slow.next;
                if(prev.next != slow){
                    prev = prev.next;
                }
            }
        }
        TreeNode node = new TreeNode(slow.val);
        if(prev != slow){
            node.left = sortedListToBSTHelper(start, prev);
        }
        node.right = sortedListToBSTHelper(slow.next, end);
        return node;
    }

    /**
     * Given a binary tree, determine if it is height-balanced.
     * For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
     */
    public boolean isBalanced(TreeNode root) {
        if(root == null){
            return true;
        }
        int left = height(root.left);
        int right = height(root.right);
        int diff = Math.abs(left - right);
        return diff <=1 && isBalanced(root.left) && isBalanced(root.right);
    }

    /**
     * Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
     * For example given a binary tree [5,4,8,11,#,13,4,7,2,#,#,#,1] and sum = 22, return true, as there exists a root-to-leaf path
     * 5->4->11->2 which sums is 22
     */
    public boolean hasPathSum(TreeNode root, int sum) {
        if(root == null){
            return false;
        }
        if(root.val == sum && root.left == null && root.right == null){
            return true;
        }
        return hasPathSum(root.left, sum-root.val) || hasPathSum(root.right, sum-root.val);
    }

    /**
     * Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
     * For example: Given the binary tree [5,4,8,11,#,13,4,7,2,#,#,5,1] and sum = 22,
     */
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        pathSumHelper(ret, new ArrayList<Integer>(), root, sum);
        return ret;
    }

    public void pathSumHelper(List<List<Integer>> ret, List<Integer> item, TreeNode node, int sum){
        if(node != null){
            item.add(node.val);
            if(node.val == sum && node.left == null && node.right == null){
                ret.add(item);
            }
            if(node.left != null){
                pathSumHelper(ret, new ArrayList<>(item), node.left, sum - node.val);
            }
            if(node.right != null){
                pathSumHelper(ret, new ArrayList<>(item), node.right, sum - node.val);
            }
        }
    }

    /**
     * Given a binary tree, flatten it to a linked list in-place.
     * For example: Given [1,2,5,3,4,#,6] the falttened tree should look like [1,#,2,#,3,#,4,#,5,#,6]
     */
    public void flatten(TreeNode root) {
        if(root != null){
            flatten(root.left);
            TreeNode left = root.left;
            TreeNode right = root.right;
            if(left != null){
                root.left = null;
                root.right = left;
                while(left.right != null){
                    left = left.right;
                }
                left.right = right;
            }
            flatten(root.right);
        }
    }

    public int height(TreeNode node){
        if(node == null){
            return 0;
        }
        return 1 + Integer.max(height(node.left), height(node.right));
    }

    /**
     * Given a binary tree, find its minimum depth.
     * The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
     */
    public int minDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        if(root.left == null || root.right == null){
            return 1 + Integer.max(minDepth(root.left), minDepth(root.right));
        }
        return 1 + Integer.min(minDepth(root.left), minDepth(root.right));
    }

    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }

    public class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;
        TreeLinkNode(int x) { val = x; }
    }

    //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
    public void connect(TreeLinkNode root) {
        if(root != null){
            if(root.left != null){
                root.left.next = root.right;
            }
            if(root.right != null && root.next != null){
                root.right.next = root.next.left;
            }
            connect(root.left);
            connect(root.right);
        }
    }

    public void connect2(TreeLinkNode root) {
        TreeLinkNode dummy = new TreeLinkNode(0);
        TreeLinkNode cur = dummy;
        while(root != null){
            if(root.left != null){
                cur.next = root.left;
                cur = cur.next;
            }
            if(root.right != null){
                cur.next = root.right;
                cur = cur.next;
            }
            root = root.next;
            if(root == null){
                root = dummy.next;
                cur = dummy;
                dummy.next = null;
            }
        }
    }

    /**
     * Given numRows, generate the first numRows of Pascal's triangle. For example, given numRows = 5,
     * Return [
     *[1],
     *[1,1],
     *[1,2,1],
     *[1,3,3,1],
     *[1,4,6,4,1]
     *]
     */
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> allrows = new ArrayList<List<Integer>>();
        ArrayList<Integer> row = new ArrayList<Integer>();
        for(int i=0;i<numRows;i++){
            row.add(0, 1);
            for(int j=1;j<row.size()-1;j++)
                row.set(j, row.get(j)+row.get(j+1));
            allrows.add(new ArrayList<Integer>(row));
        }
        return allrows;
    }

    /**
     * Given an index k, return the kth row of the Pascal's triangle. For example, given k = 3, Return [1,3,3,1]
     */
    public List<Integer> getRow(int rowIndex) {
        ArrayList<Integer> row = new ArrayList<Integer>();
        for(int i=0;i<rowIndex;i++){
            row.add(0, 1);
            for(int j=1;j<row.size()-1;j++)
                row.set(j, row.get(j)+row.get(j+1));
        }
        return row;
    }

    /**
     * Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
     * For example, given the following triangle
     * [ [2],
     *  [3,4],
     * [6,5,7],
     * [4,1,8,3]]  The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
     * Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.
     */
    public int minimumTotal(List<List<Integer>> triangle) {
        for(int i = triangle.size()-1; i>=0; i--){
            for(int j = 0; j < triangle.get(i).size()-1; j++){
                List<Integer> row = triangle.get(i-1);
                int sum = row.get(j) + Integer.min(triangle.get(i).get(j), triangle.get(i).get(j+1));
                row.set(j, sum);
            }
        }
        return triangle.get(0).get(0);
    }

    /**
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock),
     * design an algorithm to find the maximum profit.
     */

    public int maxProfit(int[] prices) {
        if(prices == null || prices.length < 1){
            return 0;
        }
        int lowest = prices[0];
        int max = 0;
        for(int i=1; i<prices.length; i++){
            max = Integer.max(max, prices[i] - lowest);
            lowest = Integer.min(lowest, prices[i]);
        }
        return max;
    }

    /**
     * Say you have an array for which the ith element is the price of a given stock on day i.
     * Design an algorithm to find the maximum profit. You may complete at most two transactions.
     * You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     */
    public int maxProfit2(int[] prices) {
        if(prices == null || prices.length < 1){
            return 0;
        }
        int buy1 = Integer.MIN_VALUE;
        int sell1 = 0;
        int buy2 = Integer.MIN_VALUE;
        int sell2 = 0;
        for(int i=0; i<prices.length; i++){
            if(buy1 < -prices[i]){
                buy1 = -prices[i];
            }
            if(sell1 < prices[i] + buy1){
                sell1 = prices[i] + buy1;
            }
            if(buy2 < sell1 - prices[i]){
                buy2 = sell1 - prices[i];
            }
            if(sell2 < buy2 + prices[i]){
                sell2 = buy2 + prices[i];
            }
        }
        return sell2;
    }

    /**
     * Given a binary tree, find the maximum path sum.For this problem, a path is defined as any sequence of nodes from some starting
     * node to any node in the tree along the parent-child connections. The path does not need to go through the root. For example:
     * Given the below binary tree [1,2,3] return 6
     */

    public int maxPathSum(TreeNode root) {
        int[] max = new int[1];
        max[0] = Integer.MIN_VALUE;
        maxPathSumHelper(root, max);
        return max[0];
    }

    public int maxPathSumHelper(TreeNode root, int[] max){
        if(root == null){
            return 0;
        }
        int left = maxPathSumHelper(root.left, max);
        int right = maxPathSumHelper(root.right, max);
        int sum = left + root.val + right;
        if(sum > max[0]){
            max[0] = sum;
        }
        return Integer.max(0, Integer.max(left + root.val, right + root.val));
    }

    /**
     * Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
     * For example,"A man, a plan, a canal: Panama" is a palindrome."race a car" is not a palindrome.
     */
    public boolean isPalindrome(String s) {
        if(s == null || s.length() == 0){
            return true;
        }
        s = s.toLowerCase();
        int start = 0;
        int end = s.length() - 1;
        while(start < s.length() && end >=0){
            if(inRange(s.charAt(start)) && inRange(s.charAt(end)) && s.charAt(start) != s.charAt(end)){
                return false;
            }else if(inRange(s.charAt(start)) && inRange(s.charAt(end)) && s.charAt(start) == s.charAt(end)){
                start++;
                end--;
            }else if(!inRange(s.charAt(start))){
                start++;
            }else if(!inRange(s.charAt(end))){
                end--;
            }
        }
        return true;
    }

    public boolean inRange(char a){
        return a>= '0' && a <='9' || a>='a' && a<='z';
    }

    /**
     * Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s)
     * from beginWord to endWord, such that: Only one letter can be changed at a time; Each intermediate word must exist in the word list
     * For example, Given: beginWord = "hit" endWord = "cog" wordList = ["hot","dot","dog","lot","log"] Return
     * [ ["hit","hot","dot","dog","cog"], ["hit","hot","lot","log","cog"] ]
     * Note: All words have the same length; All words contain only lowercase alphabetic characters.
     */
    public List<List<String>> findLadders(String beginWord, String endWord, Set<String> wordList) {
        if(beginWord == null || endWord == null || !wordList.contains(beginWord) || !wordList.contains(endWord)){
            return null;
        }
        wordList.remove(beginWord);
        wordList.remove(endWord);
        List<List<String>> ret = new ArrayList<List<String>>();
        List<String> row = new ArrayList<String>();
        row.add(beginWord);
        findLaddersHelper(beginWord, endWord, row, wordList, ret);
        return ret;
    }

    public void findLaddersHelper(String beginWord, String endWord,List<String> row, Set<String> wordList, List<List<String>> ret){
        if(nextWord(endWord, row)){
            row.add(endWord);
            ret.add(row);
        }else{
            for(Iterator<String> iter = wordList.iterator(); iter.hasNext();){
                String word = iter.next();
                if(nextWord(word, row)){
                    List<String> tmp = new ArrayList<String>(row);
                    tmp.add(word);
                    iter.remove();
                    Set<String> words = new HashSet<String>(wordList);
                    findLaddersHelper(beginWord, endWord, tmp, words, ret);
                }
            }
        }
    }

    public boolean nextWord(String w1, List<String> row){
        String w2 = row.get(row.size()-1);
        if(w1 == w2 || w1.equals(w2)){
            return false;
        }
        if(w1.length() != w2.length()){
            return false;
        }
        int count = 0;
        for(int i=0; i<w1.length(); i++){
            if(w1.charAt(i) != w2.charAt(i)){
                count++;
            }
        }
        return count == 1;
    }

    /**
     * Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
     * For example,Given [100, 4, 200, 1, 3, 2],The longest consecutive elements sequence is [1, 2, 3, 4].
     * Return its length: 4. Note: Your algorithm should run in O(n) complexity.
     */
    public int longestConsecutive(int[] nums) {
        if(nums == null){
            return 0;
        }
        int max = 0;
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int n: nums){
            if(!map.containsKey(n)){
                int left = map.containsKey(n-1) ? map.get(n-1) : 0;
                int right = map.containsKey(n+1) ? map.get(n+1) : 0;
                int sum = left + right + 1;
                max = Math.max(max, sum);
                map.put(n, sum);
                map.put(n-left, sum);
                map.put(n+right, sum);
            }
        }
        return max;
    }

    /**
     * Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
     * An example is the root-to-leaf path 1->2->3 which represents the number 123. Find the total sum of all root-to-leaf numbers.
     * For example, [1,2,3] The root-to-leaf path 1->2 represents the number 12.
     * The root-to-leaf path 1->3 represents the number 13. Return the sum = 12 + 13 = 25.
     */
    public int sumNumbers(TreeNode root) {
        int[] sum = {0};
        if(root != null){
            sumNumbersHelper(root, sum, 0);
        }
        return sum[0];
    }

    public void sumNumbersHelper(TreeNode node, int[] sum, int val){
        if(node.left == null && node.right == null){
            sum[0] += val * 10 + node.val;
        }else{
            if(node.left != null){
                int pathValue = val * 10 + node.val;
                sumNumbersHelper(node.left, sum, pathValue);
            }
            if(node.right != null){
                int pathValue = val * 10 + node.val;
                sumNumbersHelper(node.right, sum, pathValue);
            }
        }
    }

    /**
     * Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
     * A region is captured by flipping all 'O's into 'X's in that surrounded region. For example,
     * X X X X        X X X X
     * X O O X        X X X X
     * X X O X   =>   X X X X
     * X O X X        X 0 X X
     */
    public void solve(char[][] board) {
        if(board == null || board.length <= 2 || board[0].length <=2) {
            return;
        }
        int row = board.length;
        int col = board[0].length;
        //top and bottom rows first
        for(int i=0; i<col; i++){
            if(board[0][i] == 'O'){
                solveHelperDFS(board, 0, i);
            }
            if(board[row-1][i] == 'O'){
                solveHelperDFS(board, row-1, i);
            }
        }
        //left and right most cols
        for(int j=0; j<row; j++){
            if(board[j][0] == 'O'){
                solveHelperDFS(board, j, 0);
            }
            if(board[j][col-1] == 'O'){
                solveHelperDFS(board, j, col-1);
            }
        }
        for(int i=1; i<row; i++){
            for (int j=1; j<col; j++){
                if(board[i][j] == 'O'){
                    board[i][j] = 'X';
                }
            }
        }
        for(int i=0; i<row; i++){
            for (int j=0; j<col; j++){
                if(board[i][j] == '#'){
                    board[i][j] = 'O';
                }
            }
        }
    }

    public void solveHelperDFS(char[][] board, int i, int j){
        if(board[i][j] == 'O'){
            board[i][j] = '#';
            if(i>1 && board[i-1][j] == 'O'){
                solveHelperDFS(board, i-1, j);
            }
            if(i+1 <board.length && board[i+1][j] == 'O'){
                solveHelperDFS(board, i+1, j);
            }
            if(j>1 && board[i][j-1] == 'O'){
                solveHelperDFS(board, i, j-1);
            }
            if(j+1 < board[0].length && board[i][j+1] == 'O'){
                solveHelperDFS(board, i, j+1);
            }
        }
    }

    /**
     * Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.
     * For example, given s = "aab", Return [ ["aa","b"], ["a","a","b"] ]
     */
    public List<List<String>> partition(String s) {
        if(s == null || s.length() == 0){
            return null;
        }
        List<List<String>> ret = new ArrayList<List<String>>();
        List<String> subs = new ArrayList<>();
        partitionHelper(s,0,ret,subs);
        return ret;
    }

    public void partitionHelper(String s, int start, List<List<String>> ret, List<String> subs){
        for(int i=start; i < s.length(); i++){
            if(isPalindrome(s, start, i)){
                String tmp = s.substring(start, i+1);
                List<String> item = new ArrayList<>(subs);
                item.add(tmp);
                if(i == s.length() -1){
                    ret.add(item);
                }else{
                    partitionHelper(s, i+1, ret, item);
                }
            }
        }
    }

    /**
     * REVIEW: Given a string s, partition s such that every substring of the partition is a palindrome. Return the minimum cuts needed for a palindrome partitioning of s
     * https://discuss.leetcode.com/topic/32575/easiest-java-dp-solution-97-36/2
     */
    public int minCut(String s) {
        char[] c = s.toCharArray();
        int n = c.length;
        int[] cut = new int[n];
        boolean[][] pal = new boolean[n][n];

        for(int i = 0; i < n; i++) {
            int min = i;
            for(int j = 0; j <= i; j++) {
                if(c[j] == c[i] && (j + 1 > i - 1 || pal[j + 1][i - 1])) {
                    pal[j][i] = true;
                    min = j == 0 ? 0 : Math.min(min, cut[j - 1] + 1);
                }
            }
            cut[i] = min;
        }
        return cut[n - 1];
    }

    /**
     * Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors
     */
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        UndirectedGraphNode ret = null;
        if(node != null){
            HashMap<Integer, UndirectedGraphNode> map = new HashMap<>();
            ret = cloneGraphelper(node, map);
        }
        return ret;
    }

    public UndirectedGraphNode cloneGraphelper(UndirectedGraphNode node, HashMap<Integer, UndirectedGraphNode> map){
        UndirectedGraphNode tmp = new UndirectedGraphNode(node.label);
        map.put(tmp.label, tmp);
        for(UndirectedGraphNode item: node.neighbors){
            UndirectedGraphNode child = map.get(item.label);
            if(child == null){
                child = cloneGraphelper(item, map);
            }
            tmp.neighbors.add(child);
        }
        return tmp;
    }

    /**
     * There are N gas stations along a circular route, where the amount of gas at station i is gas[i].You have a car with an unlimited gas tank
     * and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.
     * Return the starting gas station's index if you can travel around the circuit once, otherwise return -1. Note: The solution is guaranteed to be unique.
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int sum = gas[0] - cost[0];
        int min = gas[0] - cost[0];
        int idx = 0;
        for(int i=1; i<n; i++){
            sum += gas[i] - cost[i];
            /*compute the accumulation sum of gas[i]-cost[i], when it takes minimum, the starting point should be the next station*/
            if(sum < min){
                idx = i;
                min = sum;
            }
        }
        return sum < 0 ? -1 : (idx+1)%n;
    }


    /**
     * There are N children standing in a line. Each child is assigned a rating value.You are giving candies to these children subjected to the following requirements:
     * 1: Each child must have at least one candy. 2: Children with a higher rating get more candies than their neighbors. What is the minimum candies you must give?
     */
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] candies = new int[n];
        for (int i = 0; i < n; i++) {
            candies[i] = 1;
        }
        for(int i=1; i<n; i++){
            if(ratings[i] > ratings[i-1]){
                candies[i] = candies[i-1] + 1;
            }
        }
        for(int i=n-1; i>0; i--){
            if(ratings[i-1] > ratings[i] && candies[i-1] <= candies[i]){
                candies[i-1] = candies[i]+1;
            }
        }
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += candies[i];
        }
        return sum;
    }

    /**
     * Given an array of integers, every element appears twice except for one. Find that single one.
     * Note:Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
     */
    public int singleNumber(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int sum = nums[0];
        for(int i = 1; i < nums.length; i++) {
            sum ^= nums[i];
        }
        return sum;
    }

    /**
     * Given an array of integers, every element appears three times except for one. Find that single one.
     * Note:Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
     */
    public int singleNumber2(int[] nums) {
        int x1 = 0;
        int x2 = 0;
        int mask = 0;
        for (int i : nums) {
            x2 ^= x1 & i;
            x1 ^= i;
            mask = ~(x1 & x2);
            x2 &= mask;
            x1 &= mask;
        }
        return x1;  // p = 1, in binary form p = '01', then p1 = 1, so we should return x1 if p = 2, in binary form p = '10', then p2 = 1, so we should return x2.
    }

    /**
     * A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null. Return a deep copy of the list
     */
    public RandomListNode copyRandomList(RandomListNode head) {
        Map<RandomListNode, RandomListNode> cache = new HashMap<>();
        RandomListNode dummy = new RandomListNode(0);
        RandomListNode pre = dummy;
        while (head != null){
            if(cache.get(head) == null){
                cache.put(head, new RandomListNode(head.label));
            }
            if(head.random != null && cache.get(head.random) == null){
                cache.put(head.random, new RandomListNode(head.random.label));
            }
            pre.next = cache.get(head);
            pre.next.random = cache.get(head.random);
            head = head.next;
            pre = pre.next;
        }
        return dummy.next;
    }

    /**
     * REVIEW: Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.
     * For example, given s = "leetcode", dict = ["leet", "code"]. Return true because "leetcode" can be segmented as "leet code".
     * https://helloacm.com/cc-coding-exercise-word-break-dp-bfs-dfs/ && https://leetcode.com/discuss/92903/java-dfs-and-dp-comparision
     */
    public boolean wordBreak(String s, Set<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        int n = s.length();
        boolean[] breakable = new boolean[n + 1];
        breakable[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (breakable[j] && wordDict.contains(s.substring(j, i))) {
                    breakable[i] = true;
                    break;
                }
            }
        }
        return breakable[n];
    }

    public boolean wordBreakDFS(String s, Set<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return false;
        }
        boolean[] visited = new boolean[s.length() + 1];
        return wordBreakHelper(s, visited, 0, wordDict);
    }

    private boolean wordBreakHelper(String s, boolean[] visited, int pos, Set<String> wordDict) {
        if (pos == s.length()) {
            return true;
        }
        visited[pos] = true;
        for (int i = pos + 1; i <= s.length(); i++) {
            if (!visited[i] && wordDict.contains(s.substring(pos, i)) && wordBreakHelper(s, visited, i, wordDict)) {
                return true;
            }
        }
        return false;
    }
    /**
     * REVIEW: Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.
     * Return all such possible sentences.For example, given s = "catsanddog", dict = ["cat", "cats", "and", "sand", "dog"].
     * A solution is ["cats and dog", "cat sand dog"].
     */
    public List<String> wordBreak2(String s, Set<String> wordDict) {
        return DFS(s, wordDict, new HashMap<String, List<String>>());
    }

    List<String> DFS(String s, Set<String> wordDict, HashMap<String, List<String>>map) {
        if (map.containsKey(s))
            return map.get(s);

        List<String> res = new ArrayList<>();
        if (s.length() == 0) {
            res.add("");
            return res;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String>sublist = DFS(s.substring(word.length()), wordDict, map);
                for (String sub : sublist)
                    res.add(word + (sub.isEmpty() ? "" : " ") + sub);
            }
        }
        map.put(s, res);
        return res;
    }

    /**
     * Given a linked list, determine if it has a cycle in it. Follow up: Can you solve it without using extra space?
     */
    public boolean hasCycle(ListNode head) {
        if(head == null){
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while(fast != null && fast.next != null){
            fast = fast.next.next;
            slow = slow.next;
            if(slow == fast){
                return true;
            }
        }
        return false;
    }

    /**
     * Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
     * Note: Do not modify the linked list. Follow up: Can you solve it without using extra space?
     */
    public ListNode detectCycle(ListNode head) {
        if(head == null){
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while(slow != null){
            slow = slow.next;
            if(fast != null && fast.next != null){
                fast = fast.next.next;
            }else{
                return null;
            }
            if(slow == fast){
                break;
            }
        }
        slow = head;
        while(slow != fast){
            slow = slow.next;
            fast = fast.next;
        }
        return fast;
    }

    /**
     * Given a singly linked list L: L0→L1→…→Ln-1→Ln, reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
     * You must do this in-place without altering the nodes' values. For example, Given {1,2,3,4}, reorder it to {1,4,2,3}.
     */
    public void reorderList(ListNode head) {
        if(head != null){
            ListNode slow = head;
            ListNode fast = head;
            while(fast != null && fast.next != null){
                fast = fast.next.next;
                slow = slow.next;
            }
            //reverse the list after middle
            ListNode tail = reverse(slow.next);
            slow.next = null;
            ListNode dummy = new ListNode(0);
            //merge the list (head is equal or one more longer than tail
            while(head != null || tail != null){
                dummy.next = head;
                dummy = dummy.next;
                head = head.next;
                if(tail != null){
                    dummy.next = tail;
                    dummy = dummy.next;
                    tail = tail.next;
                }
            }
        }
    }

    private ListNode reverse(ListNode head){
        if(head == null){
            return null;
        }
        ListNode pre = null;
        ListNode cur = head;
        ListNode forward = head.next;
        while(forward != null){
            cur.next = pre;
            pre = cur;
            cur = forward;
            forward = forward.next;
        }
        cur.next = pre;
        return cur;
    }

    /**
     * Given a binary tree, return the preorder traversal of its nodes' values. For example: Given binary tree {1,#,2,3}, return [1,2,3].
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        if(root != null){
            Stack<TreeNode> s = new Stack<>();
            s.push(root);
            while(!s.isEmpty()){
                TreeNode node = s.pop();
                ret.add(node.val);
                if(node.right != null){
                    s.push(node.right);
                }
                if(node.left != null){
                    s.push(node.left);
                }
            }
        }
        return ret;
    }

    /**
     * Given a binary tree, return the postorder traversal of its nodes' values. For example: Given binary tree {1,#,2,3}, return [3,2,1].
     */
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        if(root != null){
            Stack<TreeNode> s = new Stack<>();
            s.push(root);
            while (!s.isEmpty()){
                TreeNode node = s.pop();
                ret.add(0, node.val);
                if(node.left != null){
                    s.push(node.left);
                }
                if(node.right != null){
                    s.push(node.right);
                }
            }
        }
        return ret;
    }

    /**
     * REVIEW: Sort a linked list using insertion sort.
     */
    public ListNode insertionSortList(ListNode head) {
        if(head != null){
            ListNode fast = head.next;
            while(fast != null){
                ListNode slow = head;
                while(slow != fast){
                    int val = slow.val;
                    if(fast.val < slow.val){
                        slow.val = fast.val;
                        fast.val = val;
                    }
                    slow = slow.next;
                }
                fast = fast.next;
            }
        }
        return head;
    }

    /**
     * REVIEW: Sort a linked list in O(n log n) time using constant space complexity. follow up: how about merge sort
     */
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        ListNode dummyLeft = new ListNode(0);
        ListNode dummyRight = new ListNode(0);
        ListNode preLeft = dummyLeft;
        ListNode preRight = dummyRight;
        ListNode cur = head.next;
        ListNode pivot = head;
        while (cur != null){
            if(cur.val < head.val){
                preLeft.next = cur;
                preLeft = preLeft.next;
            }else if(cur.val > head.val){
                preRight.next = cur;
                preRight = preRight.next;
            }else{
                pivot.next = cur;
                pivot = pivot.next;
            }
            cur = cur.next;
        }
        preLeft.next = preRight.next = pivot.next = null;
        dummyLeft.next = sortList(dummyLeft.next);
        dummyRight.next = sortList(dummyRight.next);
        preLeft = dummyLeft;
        while(preLeft.next != null){
            preLeft = preLeft.next;
        }
        preLeft.next = head;
        pivot.next = dummyRight.next;
        return dummyLeft.next;
    }

    /**
     * Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
     */
    public int maxPoints(Point[] points) {
        int l = points.length;
        if (l == 0) return 0;
        if (l <= 2) return l;
        int res = 0;
        for (int i = 0; i < l - 1; i++) {
            Map<String, Integer> map = new HashMap<>();
            int overlap = 0;
            int lineMax = 0;
            for (int j = i + 1; j < l; j++) {
                int x = points[i].x - points[j].x;
                int y = points[i].y - points[j].y;
                if (x == 0 && y == 0) {
                    overlap++;
                    continue;
                }
                int gcd = generateGcd(x, y);
                x /= gcd;
                y /= gcd;
                String slope = String.valueOf(x) + String.valueOf(y);
                int count = map.getOrDefault(slope, 0);
                count++;
                map.put(slope, count);
                lineMax = Math.max(lineMax, count);
            }
            res = Math.max(res, lineMax + overlap + 1);
        }
        return res;
    }

    public int generateGcd(int x, int y) {
        if (y == 0) return x;
        return generateGcd(y, x % y);
    }

    /**
     * Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, /. Each operand may be an integer or another expression.
     * Some examples: ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
     * ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
     */
    public int evalRPN(String[] tokens) {
        if(tokens == null || tokens.length == 0){
            return 0;
        }
        Stack<Integer> s = new Stack<>();
        for (int i = 0; i < tokens.length; i++) {
            String ch = tokens[i];
            if(ch.equals("+") || ch.equals("-") || ch.equals("*") || ch.equals("/")){
                int num1 = s.pop();
                int num2 = s.pop();
                if(ch.equals("+")){
                    s.push(num1+num2);
                }else if(ch.equals("-")){
                    s.push(num2-num1);
                }else if(ch.equals("*")){
                    s.push(num1*num2);
                }else{
                    s.push(num2/num1);
                }
            }else{
                s.push(Integer.parseInt(ch));
            }
        }
        return s.pop();
    }

    /**
     *Find the contiguous subarray within an array (containing at least one number) which has the largest product.
     * For example, given the array [2,3,-2,4],the contiguous subarray [2,3] has the largest product = 6.
     */
    public int maxProduct(int[] nums) {
        int max = nums[0],min = max, ret = max;
        for(int i=1; i<nums.length; i++){
            if(nums[i] < 0){
                int tmp = max;
                max = min;
                min = tmp;
            }
            max = Integer.max(max*nums[i], nums[i]);
            min = Integer.min(min*nums[i], nums[i]);
            ret = Integer.max(max, ret);
        }
        return ret;
    }

    /**
     * BINARY-SEARCH: Suppose a sorted array is rotated at some pivot unknown to you beforehand. (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     * Find the minimum element. You may assume no duplicate exists in the array.
     * What if duplicates are allowed? Would this affect the run-time complexity? How and why?
     */
    public int findMin(int[] nums) {
        int start = 0;
        int end = nums.length - 1;
        if(nums[start] < nums[end]){
            return nums[start];
        }
        while(end > start+1){
            int middle = (start + end)/2;
            if(nums[start] < nums[middle]){
                start = middle;
            }else {
                end = middle;
            }
        }
        return nums[end];
    }
    //what is duplicates allowed
    public int findMin2(int[] nums) {
        int start = 0;
        int end = nums.length - 1;
        if(nums[start] < nums[end]){
            return nums[start];
        }
        while(end > start+1){
            int m = (start + end)/2;
            if(nums[start] <= nums[m] && nums[start] <= nums[m-1]){
                start = m;
            }else {
                end = m;
            }
        }
        return nums[end];
    }

    /**
     * Write a program to find the node at which the intersection of two singly linked lists begins.
     * Notes: 1: If the two linked lists have no intersection at all, return null.
     * 2:The linked lists must retain their original structure after the function returns.
     * 3: You may assume there are no cycles anywhere in the entire linked structure.
     * 4:Your code should preferably run in O(n) time and use only O(1) memory.
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if(headA == null || headB == null){
            return null;
        }
        ListNode a = headA;
        ListNode b = headB;
        while (a != b){
            a = a == null ? headB : a.next;
            b = b == null ? headA : b.next;
        }
        return a;
    }

    /**
     * A peak element is an element that is greater than its neighbors. Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
     * The array may contain multiple peaks, in that case return the index to any one of the peaks is fine. You may imagine that num[-1] = num[n] = -∞.
     * For example, in array [1, 2, 3, 1], 3 is a peak element and your function should return the index number 2
     */
    public int findPeakElement(int[] nums) {
        int l = 0, r = nums.length - 1;
        while(l < r) {
            int mid = (l + r) / 2;
            if(mid == 0 || nums[mid] > nums[mid - 1]) {
                if(nums[mid] < nums[mid + 1]) l = mid + 1;
                else r = mid;
            }
            else r = mid - 1;
        }
        return l;
    }

    /**
     * REVIEW: Given an unsorted array, find the maximum difference between the successive elements in its sorted form.
     * Try to solve it in linear time/space (o(t) < o(n), bucket sort or radix sort) Return 0 if the array contains less than 2 elements.
     * You may assume all elements in the array are non-negative integers and fit in the 32-bit signed integer range.
     */
    public int maximumGap(int[] nums) {
        if(nums == null || nums.length < 2){
            return 0;
        }
        int n = nums.length;
        int max = nums[0];
        for (int i = 1; i < n; i++) {
            max = Math.max(max, nums[i]);
        }
        int exp = 1;
        int N = 10;
        int[] aux = new int[n];
        while(max/exp > 0){
            int[] count = new int[N];
            for(int i=0; i<n; i++){
                count[(nums[i]/exp)%10]++;
            }
            for(int i=1; i<N; i++){
                count[i] += count[i-1];
            }
            for(int i=n-1; i>=0; i--){
                aux[--count[(nums[i]/exp)%10]] = nums[i];
            }
            for(int i=0; i<n; i++){
                nums[i] = aux[i];
            }
            exp = exp * 10;
        }
        max = 0;
        for(int i =1; i<n; i++){
            max = Math.max(max, aux[i] - aux[i - 1]);
        }
        return max;
    }

    public int compareVersion(String version1, String version2) {
        if(version1 == null || version2 == null){
            return 0;
        }
        String[] va = version1.split("\\.");
        String[] vb = version2.split("\\.");

        for(int i=0, j=0; i<va.length||j<vb.length; i++,j++){
            int a = i<va.length ? Integer.valueOf(va[i]) : 0;
            int b = j<vb.length ? Integer.valueOf(vb[j]) : 0;
            if(a>b) return 1;
            if(a<b) return -1;
        }
        return 0;
    }

    /**
     * Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
     * If the fractional part is repeating, enclose the repeating part in parentheses. For example,
     * Given numerator = 1, denominator = 2, return "0.5".
     * Given numerator = 2, denominator = 1, return "2".
     * Given numerator = 2, denominator = 3, return "0.(6)".
     */
    public String fractionToDecimal(int numerator, int denominator) {
        if(denominator == 0){
            return null;
        }
        long num = (long)numerator;
        long den = (long)denominator;
        StringBuffer str = new StringBuffer("");
        if(num * den < 0){
            str.append("-");
        }
        num = Math.abs(num);
        den = Math.abs(den);
        Map<Long, Integer> cache = new HashMap<>();
        str.append(num/den);
        num = num%den;
        if(num == 0){
            return str.toString();
        }
        str.append(".");
        cache.put(num, str.length());
        while(num != 0){
            num = num *10;
            str.append(num/den);
            num = num%den;
            if(cache.containsKey(num)){
                str.insert(cache.get(num), "(");
                str.append(")");
                return str.toString();
            }else{
                cache.put(num, str.length());
            }
        }
        return str.toString();
    }

    /**
     * Given a positive integer, return its corresponding column title as appear in an Excel sheet. For example:
     * 1 -> A 2 -> B 3 -> C  ... 26 -> Z 27 -> AA 28 -> AB
     */
    public String convertToTitle(int n) {
        String ret = "";
        while(n >0){
            n--;
            ret = (char)('A'+n%26) + ret;
            n = n/26;
        }
        return ret;
    }

    public int titleToNumber(String s) {
        char[] c = s.toCharArray();
        int sum=0;
        int n=0;
        while (n<c.length){
            sum = sum * 26 + c[n] - 64;
            n++;
        }
        return sum;
    }

    /**
     * Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
     * The function should return indices of the two numbers such that they add up to the target, where index1 must be less than index2.
     * Please note that your returned answers (both index1 and index2) are not zero-based: Give numbers={2, 7, 11, 15}, target=9 return index1=1, index2=2
     */
    public int[] twoSum(int[] numbers, int target) {
        int i=0,j=numbers.length-1;
        while(i<j){
            if(numbers[i]+numbers[j] == target){
                break;
            }else if(numbers[i]+numbers[j] <target){
                i++;
            }else{
                j--;
            }
        }
        if(i>j){
            return new int[]{-1, -1};
        }else{
            return new int[]{i+1, j+1};
        }
    }

    /**
     * Given an integer n, return the number of trailing zeroes in n!. Note: Your solution should be in logarithmic time complexity.
     * https://en.wikipedia.org/wiki/Trailing_zero
     */
    public int trailingZeroes(int n) {
        int ret = 0;
        while(n!=0){
            ret+=n/5;
            n/=5;
        }
        return ret;
    }

    /**
     * DP: programming example https://leetcode.com/problems/dungeon-game/
     */
    public int calculateMinimumHP(int[][] dungeon) {
        int m = dungeon.length;
        int n = dungeon[0].length;
        //reverse calculation from last to first one
        dungeon[m-1][n-1] = Math.max(1, 1-dungeon[m-1][n-1]);
        //last columns
        for(int i=m-2; i>=0; i--){
            dungeon[i][n-1] = Math.max(1, dungeon[i+1][n-1] - dungeon[i][n-1]);
        }
        //last row
        for(int i=n-2; i>=0; i--){
            dungeon[m-1][i] = Math.max(1, dungeon[m-1][i+1]-dungeon[m-1][i]);
        }
        for(int i=m-2; i>=0; i--){
            for(int j= n-2; j>=0; j--){
                dungeon[i][j] = Math.min(Math.max(1, dungeon[i][j+1] - dungeon[i][j]), Math.max(1, dungeon[i+1][j]-dungeon[i][j]));
            }
        }
        return dungeon[0][0];
    }

    /**
     * Given a list of non negative integers, arrange them such that they form the largest number.
     * For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
     * Note: The result may be very large, so you need to return a string instead of an integer.
     */
    public String largestNumber(int[] nums) {
        String ret = "";
        String [] str = new String[nums.length];
        for(int i=0; i<nums.length; i++){
            str[i]=String.valueOf(nums[i]);
        }
        Arrays.sort(str, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return (o1+o2).compareTo(o2+o1);
            }
        });
        for (int i = nums.length-1; i >=0 ; i--) {
            ret+=str[i];
        }
        if(str[nums.length-1].equals("0")){
            return "0";
        }
        return ret;
    }

    /**
     * All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA,
     * it is sometimes useful to identify repeated sequences within the DNA. Write a function to find all the 10-letter-long sequences (substrings)
     * that occur more than once in a DNA molecule.Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT", returns: ["AAAAACCCCC", "CCCCCAAAAA"]
     */
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> ret = new ArrayList();
        Set<String> words = new HashSet<>();
        Map<String, Integer> map = new HashMap();
        if(s != null && s.length() > 10){
            for(int i=0; i<s.length()-9; i++){
                String first = s.substring(i, i+10);
                if(map.containsKey(first) && words.add(first)){
                    ret.add(first);
                }else{
                    map.put(first, 1);
                }
            }
        }
        return ret;
    }

    /**
     * Say you have an array for which the ith element is the price of a given stock on day i.Design an algorithm to find the maximum profit. You may complete at most k transactions.
     * Note: You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
     */
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        if (k >= n / 2) {
            int max = 0;
            for (int i = 1; i < n; i++)
                max += Math.max(0, prices[i] - prices[i - 1]);
            return max;
        }
        int[] buy = new int[k + 1];
        int[] sell = new int[k + 1];
        Arrays.fill(buy, Integer.MIN_VALUE);
        Arrays.fill(sell, 0);
        for (int price : prices) {
            for (int i = k; i > 0; i--) {
                sell[i] = Math.max(sell[i], buy[i] + price);
                buy[i] = Math.max(buy[i], sell[i - 1] - price);
            }
        }
        return sell[k];
    }

    /**
     * Rotate an array of n elements to the right by k steps (Could you do it in-place with O(1) extra space?)
     * For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
     */
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k=k%n;
        reverse(nums, 0, nums.length-1);
        reverse(nums, 0, k-1);
        reverse(nums, k, nums.length-1);
    }

    private void reverse(int[] nums, int i, int j){
        while(i<j){
            int tmp = nums[i];
            nums[i] = nums[j];
            nums[j]=tmp;
            i++;
            j--;
        }
    }

    /**
     *Reverse bits of a given 32 bits unsigned integer.For example, given input 43261596
     * (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000)
     */
    public int reverseBits(int n) {
        if(n ==0){
            return n;
        }
        int ret = 0;
        for(int i=0; i<32; i++){
            ret <<=1;
            if((n&1) == 1){
                ret += 1;
            }
            n>>=1;
        }
        return ret;
    }

    /**
     * Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
     * For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3
     * bit hacks  http://www.catonmat.net/blog/low-level-bit-hacks-you-absolutely-must-know/
     */
    public int hammingWeight(int n) {
        int ret = 0;
        for(int i=0; i<32; i++){
            if((n&1) == 1){
                ret+=1;
            }
            n>>=1;
        }
        return ret;
        //event better solution
        //return (n == 0)?0:(hammingWeight(n - (n&(-n))) + 1);
    }

    /**
     * Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
     * For example, given the range [5, 7], you should return 4
     */
    public int rangeBitwiseAnd(int m, int n) {
        while(n > m){
            n = n & (n-1);
        }
        return n;
    }

    /**
     * Divide two integers without using multiplication, division and mod operator. If it is overflow, return MAX_INT and time complexity should be o(log(n))
     */
    public int divide(int dividend, int divisor) {
        if(divisor == 0 || (dividend == Integer.MIN_VALUE && divisor == -1)){
            return Integer.MAX_VALUE;
        }
        int sign = (dividend > 0) ^ (divisor > 0) ? -1 : 1;
        int ret = 0;
        long div = Math.abs((long)dividend);
        long dis = Math.abs((long)divisor);
        while(div >= dis){
            int multiple = 1;
            long tmp = dis;
            while(div >= (tmp<<1)){
                tmp<<=1;
                multiple<<=1;
            }
            ret+=multiple;
            div-=tmp;
        }
        return ret*sign;
    }

    /**
     * You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed,
     * the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically
     * contact the police if two adjacent houses were broken into on the same night.Given a list of non-negative integers representing the amount of money of each house,
     * determine the maximum amount of money you can rob tonight without alerting the police.
     */
    public int rob(int[] nums) {
        if(nums.length == 0){
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        if(nums.length == 2){
            dp[1] = Math.max(nums[0], nums[1]);
        }
        for(int i=2; i<nums.length; i++){
            dp[i] = Math.max(dp[i-1], nums[i]+dp[i-2]);
        }
        return dp[nums.length-1];
    }

    /**
     *Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
     * For example: Given the following binary tree, [1,2,3,#,5,#,4], you should return [1,3,4]
     */
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        pushRight(ret, root, 0);
        return ret;
    }

    private void pushRight(List<Integer> list, TreeNode root, int h){
        if(root != null){
            if(list.size() == h){
                list.add(root.val);
            }
            pushRight(list, root.right, h+1);
            pushRight(list, root.left, h+1);
        }
    }

    //iterative version
    public List<Integer> rightSideView2(TreeNode root) {
        List<Integer> ret = new ArrayList<>();
        if(root != null){
            LinkedList<TreeNode> q = new LinkedList<>();
            q.push(root);
            TreeNode last = null;
            while (!q.isEmpty()){
                int n = q.size();
                for(int i=0; i<n; i++){
                    TreeNode node = q.peek();
                    if(node.left != null){
                        q.add(node.left);
                    }
                    if(node.right != null){
                        q.add(node.right);
                    }
                    last = q.pop();
                }
                ret.add(last.val);
            }
        }
        return ret;
    }

    /**
     * Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting
     * adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
     * Example1: [11110, 11010, 11000, 00000] return 1 and Example2 [11000, 11000, 00100, 00011] return 3
     */
    public int numIslands(char[][] grid) {
        if(grid == null){
            return 0;
        }
        int count = 0;
        for(int i=0; i<grid.length; i++){
            for(int j=0; j<grid[0].length; j++){
                if(grid[i][j] == '1'){
                    numislandsDFS(i, j, grid);
                    count++;
                }
            }
        }
        return count;
    }

    private void numislandsDFS(int i, int j, char[][] grid){
        if(grid[i][j] == '1'){
            grid[i][j] = '#';
            if(i < grid.length -1){
                numislandsDFS(i+1, j, grid);
            }
            if(i > 0){
                numislandsDFS(i-1, j, grid);
            }
            if(j < grid[0].length -1){
                numislandsDFS(i, j+1, grid);
            }
            if(j> 0){
                numislandsDFS(i, j-1, grid);
            }
        }
    }

    /**
     * Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
     * Calling next() will return the next smallest number in the BST.
     * Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
     */
    public class BSTIterator {
        Stack<TreeNode> stack;
        public BSTIterator(TreeNode root) {
            stack = new Stack<>();
            pushLeft(root);
        }
        private void pushLeft(TreeNode node){
            while (node != null){
                stack.push(node);
                node = node.left;
            }
        }
        public boolean hasNext() {
            return !stack.isEmpty();
        }
        public int next() {
            TreeNode node = stack.pop();
            if(node.right != null){
                pushLeft(node.right);
            }
            return node.val;
        }
    }

    /**
     * Design a stack that supports push, pop, top, and retrieving the minimum element in on(1) time.
     * push(x) -- Push element x onto stack. pop() -- Removes the element on top of the stack.
     * top() -- Get the top element. getMin() -- Retrieve the minimum element in the stack.
     */
    public class MinStack {
        Stack<int[]> stack;
        public MinStack() {
            stack = new Stack();
        }

        public void push(int x) {
            int[] item = new int[2];
            item[0] = x;
            if(stack.isEmpty()){
                item[1] = x;
            }else{
                int[] top = stack.peek();
                item[1] = top[1] < x ? top[1] : x;
            }
            stack.push(item);
        }

        public void pop() {
            stack.pop();
        }

        public int top() {
            int[] top = stack.peek();
            return top[0];
        }

        public int getMin() {
            int[] top = stack.peek();
            return top[1];
        }
    }

    class Point {
        int x;
        int y;
        Point() { x = 0; y = 0; }
        Point(int a, int b) { x = a; y = b; }
    }

    static class LRUCache {
        private int capacity;
        private Map<Integer, Node> cache = new HashMap<>();
        Node head;
        Node tail;
        public LRUCache(int capacity) {
            this.capacity = capacity;
        }

        public int get(int key) {
            if(cache.containsKey(key)){
                Node node = cache.get(key);
                this.removeNode(node);
                this.addNode(node);
                return node.value;
            }else{
                return -1;
            }
        }

        public void set(int key, int value) {
            Node insert = new Node(key, value);
            if(cache.containsKey(key)){
                Node node = cache.get(key);
                this.removeNode(node);
                this.addNode(insert);
            }else{
                if(cache.size() == this.capacity){
                    cache.remove(this.tail.key);
                    this.removeNode(this.tail);
                }
                this.addNode(insert);
            }
            cache.put(key, insert);
        }

        private void addNode(Node node){
            node.next = this.head;
            node.prev = null;
            if(this.head != null){
                this.head.prev = node;
            }else{
                this.tail = node;
            }
            this.head = node;
        }

        private void removeNode(Node node){
            Node pre = node.prev;
            Node next = node.next;
            if(pre != null){
                pre.next = next;
            }else{
                this.head = next;
            }
            if(next != null){
                next.prev = pre;
            }else{
                this.tail = pre;
            }
        }

        class Node{
            int key;
            int value;
            Node prev;
            Node next;
            public Node(int key, int value){
                this.key = key;
                this.value = value;
            }
        }
    }

    public static class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
        }
    }

    public boolean isPalindrome(String s, int start, int end){
        while(start < end){
            if(s.charAt(start) != s.charAt(end)){
                return false;
            }
            start++;
            end--;
        }
        return true;
    }

    class RandomListNode {
        int label;
        RandomListNode next, random;
        RandomListNode(int x) { this.label = x; }
    }

     static class UndirectedGraphNode {
         int label;
         List<UndirectedGraphNode> neighbors;
         UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
     }

    public static void main(String[] args) {
        Main t = new Main();
        int[][] n = {{1}};
        //System.out.println(t.maximumGap(new int[]{28,51,48,101}));
        System.out.println(t.twoSum(new int[]{2,3,4}, 6));
    }
}
